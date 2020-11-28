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

from model import StoDepth_ResNet


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
        model.training = False

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


class DatasetFromTeacher(torch.utils.data.Dataset):
    def __init__(
        self,
        teacher_model,
        dataset_labeled,
        dataset_unlabeled,
        transform_test,
        transform_noisy,
        device,
        num_classes=10,
        label_type="hard",
        label_smoothing_epsilon=0.1,
        confidence_threshold=0.8,
        generated_batch_size=128,
    ):
        assert label_type in ["hard", "soft", "smooth"]
        super(DatasetFromTeacher, self).__init__()
        self.label_type = label_type
        self.transform_test = transform_test
        self.transform_noisy = transform_noisy
        self.data, self.label = [], []
        self.device = device
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        self.smoothing_big, self.smoothing_small = (
            1 - label_smoothing_epsilon,
            label_smoothing_epsilon / (num_classes - 1),
        )

        for image, y in dataset_labeled:
            self.data.append(np.array(transforms.ToPILImage()(image)))
            if label_type == "hard" or label_type == "soft":
                label = np.zeros(num_classes)
                label[y] = 1.0
                self.label.append(label)
            elif label_type == "smooth":
                label = np.full(num_classes, self.smoothing_small)
                label[y] = self.smoothing_big
                self.label.append(label)

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

        self.data, self.label = np.array(self.data), np.array(self.label)

    def _add_label_from_generated_batch(self, teacher_model, generated_batch):
        input_batch = [self.transform_test(b).unsqueeze(0) for b in generated_batch]
        input_batch = torch.Tensor(torch.cat(input_batch)).to(self.device)

        generated_batch = np.array(
            [np.array(transforms.ToPILImage()(b)) for b in generated_batch]
        )

        with torch.no_grad():
            teacher_model.eval()
            labels = torch.softmax(teacher_model(input_batch), dim=1).cpu().numpy()

        survival_indices = labels.max(axis=1) > self.confidence_threshold

        labels = labels[survival_indices, :]
        generated_batch = generated_batch[survival_indices, :, :]

        n = len(labels)

        if self.label_type == "hard":
            mxindex = labels.argmax(axis=1)
            labels = np.zeros((n, self.num_classes))
            labels[np.arange(n), mxindex] = 1.0
        elif self.label_type == "soft":
            # do nothing on the softmax label
            pass
        elif self.label_type == "smooth":
            mxindex = labels.argmax(axis=1)
            labels = np.full((n, self.num_classes), self.smoothing_small)
            labels[np.arange(n), mxindex] = self.smoothing_big

        for image, label in zip(generated_batch, labels):
            self.data.append(image)
            self.label.append(label)

        del generated_batch
        del labels

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


class SGD_with_lars(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum)."""

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, trust_coef=1.): # need to add trust coef
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if trust_coef < 0.0:
            raise ValueError("Invalid trust_coef value: {}".format(trust_coef))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)

        super(SGD_with_lars, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_with_lars, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            global_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)
                d_p_norm = torch.norm(d_p, p=2).add_(momentum, p_norm)
                lr = torch.div(p_norm, d_p_norm).mul_(trust_coef)

                lr.mul_(global_lr)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                d_p.mul_(lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(-1, d_p)

        return loss

