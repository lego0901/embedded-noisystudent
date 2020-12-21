import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer, required

import numpy as np
from PIL import Image
import pandas as pd

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from torch.utils.data import Dataset, DataLoader

from model import StoDepth_ResNet


############################################################
####################### For training #######################
############################################################

def make_model(num_layers=20, width=1, prob_0_L=(1, 0.8), dropout_prob=0.5, num_classes=10):
    return StoDepth_ResNet(
        num_layers,
        width=width,
        prob_0_L=prob_0_L,
        dropout_prob=dropout_prob,
        num_classes=num_classes,
    )


def train_model(
    model,
    dataloader_train,
    dataloader_test,
    device,
    optimizer,
    lr_scheduler=None,
    save_path=None,
    log_path=None,
    epochs=100,
    onehot=True,
    mixup=False,
    mixup_alpha=1.0,
):
    start_time = time.time()

    if onehot or mixup:
        criterion = lambda output, target: torch.mean(
            torch.sum(-target * F.log_softmax(output, dim=1), dim=1)
        )
        to_target = lambda target: target.argmax(dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
        to_target = lambda target: target

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    if log_path is not None:
        dirname = os.path.dirname(log_path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    
    logs = []
    best_test_acc = 0.0
    train_dataset_len = len(dataloader_train.dataset)

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in dataloader_train:
            if mixup:
                if not onehot:
                    target = to_onehot(target)
                data, target = mixup_batch(data, target, alpha=mixup_alpha)

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(to_target(target)).sum().item()
        
            passed_time = time.time() - start_time
            hour, rem = divmod(passed_time, 3600)
            minute, second = divmod(rem, 60)

            print("Epoch {:0>3}/{:0>3}: ({:0>5}/{:0>5}) train loss = {:.4f}, acc = {:.4f} ({:0>2}:{:0>2}:{:05.2f})\r"
                .format(
                    epoch,
                    epochs,
                    train_total,
                    train_dataset_len,
                    train_loss / train_total,
                    train_correct / train_total,
                    int(hour), int(minute), second
                ), end="")

        train_loss /= train_total
        train_acc = train_correct / train_total

        test_loss, test_acc = test_model(
            model, dataloader_test, device, onehot=False
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        passed_time = time.time() - start_time
        hour, rem = divmod(passed_time, 3600)
        minute, second = divmod(rem, 60)

        print("Epoch {:0>3}/{:0>3}: train loss = {:.4f}, acc = {:.4f}, test loss = {:.4f}, acc = {:.4f} ({:0>2}:{:0>2}:{:05.2f})"
            .format(
                epoch,
                epochs,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                int(hour), int(minute), second
            ))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        
        logs.append((train_loss, train_acc, test_loss, test_acc))

        if log_path is not None:
            df = pd.DataFrame(
                logs,
                columns=["train_loss", "train_acc", "test_loss", "test_acc"],
                index=range(1, epoch + 1)
            )
            df.to_csv(log_path, encoding="UTF-8")


class DatasetFromTeacher(torch.utils.data.Dataset):
    def __init__(
        self,
        teacher_model,
        dataset_labeled,
        dataset_unlabeled,
        transform_test,
        transform_noisy,
        device,
        args,
        num_classes=10,
        generated_batch_size=128,
    ):
        assert args.label_type in ["hard", "soft", "smooth"]
        super(DatasetFromTeacher, self).__init__()
        self.transform_test = transform_test
        self.transform_noisy = transform_noisy
        self.data, self.label = [], []
        self.device = device
        self.num_classes = num_classes
        self.args = args

        self.smoothing_big, self.smoothing_small = (
            1 - args.label_smoothing_epsilon,
            args.label_smoothing_epsilon / (num_classes - 1),
        )

        self.images_per_class, self.onehot_labels_per_class = {}, {}
        for y in range(num_classes):
            self.images_per_class[y], self.onehot_labels_per_class[y] = [], []

        # add labeled image
        for image, y in dataset_labeled:
            if args.label_type == "hard" or args.label_type == "soft":
                onehot_label = np.zeros(num_classes)
                onehot_label[y] = 1.0
            elif args.label_type == "smooth":
                onehot_label = np.full(num_classes, self.smoothing_small)
                onehot_label[y] = self.smoothing_big

            self.images_per_class[y].append(np.array(transforms.ToPILImage()(image)))
            self.onehot_labels_per_class[y].append(onehot_label)

        # add unlabeled image with pseudo-labels from the teacher model
        teacher_model.eval()
        generated_batch = []
        for image, _ in dataset_unlabeled:
            generated_batch.append(image)
            if len(generated_batch) == generated_batch_size:
                self._add_label_from_generated_batch(teacher_model, generated_batch)
                generated_batch = []

        if len(generated_batch) > 0:
            self._add_label_from_generated_batch(teacher_model, generated_batch)
            generated_batch = []
        
        # to make collected data into numpy array
        for y in range(num_classes):
            self.images_per_class[y] = np.array(self.images_per_class[y])
            self.onehot_labels_per_class[y] = np.array(self.onehot_labels_per_class[y])

        # dataset generation
        for y in range(num_classes):
            self.data.append(self.images_per_class[y])
            self.label.append(self.onehot_labels_per_class[y])

        self.data, self.label = np.concatenate(self.data), np.concatenate(self.label)

    def _add_label_from_generated_batch(self, teacher_model, generated_batch):
        images = np.array(
            [np.array(transforms.ToPILImage()(b)) for b in generated_batch]
        )

        input_batch = [self.transform_test(b).unsqueeze(0) for b in generated_batch]
        input_batch = torch.Tensor(torch.cat(input_batch)).to(self.device)

        with torch.no_grad():
            teacher_model.eval()
            probs = torch.softmax(teacher_model(input_batch), dim=1).cpu().numpy()

        confidences = probs.max(axis=1)
        survival_indices = confidences > self.args.confidence_threshold

        probs = probs[survival_indices, :]
        images = images[survival_indices, :, :, :]

        n = len(probs)

        # why 0.99? to make confidence priority to the labeled one
        if self.args.label_type == "hard":
            mxindex = probs.argmax(axis=1)
            onehot_labels = np.zeros((n, self.num_classes))
            onehot_labels[np.arange(n), mxindex] = 0.99
        elif self.args.label_type == "soft":
            onehot_labels = np.minimum(probs, 0.99)
        elif self.args.label_type == "smooth":
            mxindex = probs.argmax(axis=1)
            onehot_labels = np.full((n, self.num_classes), self.smoothing_small)
            onehot_labels[np.arange(n), mxindex] = self.smoothing_big * 0.99

        for image, onehot_label in zip(images, onehot_labels):
            y = onehot_label.argmax().item()
            self.images_per_class[y].append(image)
            self.onehot_labels_per_class[y].append(onehot_label)
    
    def num_images_per_label(self):
        return [len(self.images_per_class[i]) for i in range(self.num_classes)]

    def balance_data(self, min_images_per_class=4000, max_gap_num_images_between_classes=1000):
        # all classes will have >= min_images_per_classes images
        min_num_images = 1e9
        for y in range(self.num_classes):
            n = len(self.images_per_class[y])
            while n < min_images_per_class:
                k = min(n, min_images_per_class - n)

                confidences = self.onehot_labels_per_class[y].max(axis=1)
                add_indices = (-confidences).argsort()[:k]
                images, labels = list(self.images_per_class[y]), list(self.onehot_labels_per_class[y])
                for idx in add_indices:
                    images.append(self.images_per_class[y][idx])
                    labels.append(self.onehot_labels_per_class[y][idx])

                self.images_per_class[y] = np.array(images)
                self.onehot_labels_per_class[y] = np.array(labels)
                n += k

            min_num_images = min(min_num_images, n)

        # all classes will have <= min_num_images + max_gap_num_images_between_classes images
        max_allowed_num_images = min_num_images + max_gap_num_images_between_classes
        for y in range(self.num_classes):
            n = len(self.images_per_class[y])
            if n > max_allowed_num_images:
                confidences = self.onehot_labels_per_class[y].max(axis=1)
                survived_indices = (-confidences).argsort()[:max_allowed_num_images]
                self.images_per_class[y] = self.images_per_class[y][survived_indices]
                self.onehot_labels_per_class[y] = self.onehot_labels_per_class[y][survived_indices]
        
        # dataset generation
        self.data = []
        self.label = []

        for y in range(self.num_classes):
            self.data.append(self.images_per_class[y])
            self.label.append(self.onehot_labels_per_class[y])

        self.data, self.label = np.concatenate(self.data), np.concatenate(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform_noisy(self.data[index]),
            self.label[index],
        )


def to_onehot(label, num_classes=10):
    return torch.eye(num_classes)[label]


def mixup_batch(data, label, alpha=1.0):
    batch_size = data.shape[0]

    if alpha > 0:
        beta = torch.distributions.beta.Beta(alpha, alpha)
        lam = beta.sample(sample_shape=(batch_size,))
    else:
        lam = 1
    
    index = torch.randperm(batch_size)

    lam_data, lam_label = lam.view(batch_size, 1, 1, 1), lam.view(batch_size, 1)

    mixup_data = lam_data * data + (1 - lam_data) * data[index, :]
    mixup_label = lam_label * label + (1 - lam_label) * label[index, :]

    return mixup_data, mixup_label



############################################################
#################### For test, analysis ####################
############################################################

def test_model(model, dataloader_test, device, onehot=False):
    if onehot:
        criterion = lambda output, target: torch.mean(
            torch.sum(-target * F.log_softmax(output, dim=1), dim=1)
        )
        to_target = lambda target: target.argmax(dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
        to_target = lambda target: target

    model.eval()

    test_loss = 0.0
    test_total = 0
    test_correct = 0

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        test_loss += loss.item() * target.size(0)
        _, predicted = output.max(1)
        test_total += target.size(0)
        test_correct += predicted.eq(to_target(target)).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total

    return test_loss, test_acc


def predict_model(model, dataloader_test, device):
    model.eval()
    predicts = []

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        _, predicted = output.max(1)
        predicts.append(predicted)

    predicts = torch.cat(predicts).cpu()
    return np.array(predicts).astype(np.long)


class DatasetApplyTransform(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_labeled,
        transform,
        num_classes=10,
    ):
        super(DatasetApplyTransform, self).__init__()
        self.transform= transform
        self.data, self.label = [], []
        self.num_classes = num_classes

        for image, y in dataset_labeled:
            self.data.append(np.array(transforms.ToPILImage()(image)))
            self.label.append(y)

        self.data, self.label = np.array(self.data), np.array(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform(self.data[index]),
            self.label[index],
        )