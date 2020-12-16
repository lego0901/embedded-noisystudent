import os
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from torch.utils.data import Dataset, DataLoader

from utils import *


class DatasetFromNumpy(torch.utils.data.Dataset):
    def __init__(self, data, label, transform):
        super(DatasetFromNumpy, self).__init__()
        self.data = data
        self.label = label.astype(np.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform(self.data[index]),
            self.label[index],
        )


def analysis_model_top1_accuracy(model, dataloader_test, device, log=None):
    model.eval()

    test_total = 0
    test_correct = 0

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(data)

        _, predicted = output.max(1)
        test_total += target.size(0)
        test_correct += predicted.eq(target).sum().item()

    test_acc = test_correct / test_total
    
    if log is not None:
        log["top1_acc"] = test_acc

    return test_acc


def analysis_model_top5_accuracy(model, dataloader_test, device, log=None):
    model.eval()

    test_total = 0
    test_correct = 0

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(data)

        predicted_top5 = output.topk(5, dim=1).indices
        test_total += target.size(0)
        test_correct += target.view(-1, 1) \
            .expand_as(predicted_top5).eq(predicted_top5).sum().item()

    test_acc = test_correct / test_total
    
    if log is not None:
        log["top5_acc"] = test_acc

    return test_acc


def analysis_model_fgsm_attack(
    model,
    dataloader_test,
    normalize,
    epsilon,
    device,
    log=None
):
    model.eval()

    test_total = 0
    test_correct = 0

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        data.requires_grad = True
        model.requires_graph = False

        output = model(normalize(data))
        init_pred = output.max(1).indices

        loss = F.nll_loss(output, target)
        loss.backward()
        data_grad = data.grad.data

        fgsm_data = torch.clamp(data + epsilon * data_grad.sign(), 0, 1)
        output = model(normalize(fgsm_data))
        after_pred = output.max(1).indices

        test_total += target.shape[0]
        test_correct += (init_pred.eq(target) * init_pred.eq(after_pred)).sum().item()

    test_acc = test_correct / test_total

    if log is not None:
        log["fgsm_{}".format(epsilon)] = test_acc

    return test_acc


def analysis_model_calibration_error(
    model,
    dataloader_test,
    device,
    confidence_intervals=10,
    log=None,
):
    totals = np.zeros((confidence_intervals))
    corrects = np.zeros((confidence_intervals,))
    confidences = np.zeros((confidence_intervals,))

    model.eval()

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(data)

        prob = torch.softmax(output, dim=1).cpu().numpy()
        prob_scaled = np.clip(np.floor(prob * confidence_intervals), 0, confidence_intervals-1)

        target_onehot = np.zeros((data.shape[0], output.shape[-1])).astype(np.bool)
        target_onehot[np.arange(data.shape[0]), target.cpu().numpy()] = True

        for c in range(confidence_intervals):
            totals[c] += (prob_scaled == c).sum()
            corrects[c] += ((prob_scaled == c) * target_onehot).sum()
            confidences[c] += prob[prob_scaled == c].sum()
    
    accuracies = corrects / totals
    confidences /= totals

    ece = np.abs(accuracies - confidences).mean()
    mce = np.max(np.abs(accuracies - confidences))

    if log is not None:
        log["calib_tot"] = totals
        log["calib_acc"] = accuracies
        log["calib_conf"] = confidences
        log["calib_ece"] = ece
        log["calib_mce"] = mce


def analysis_model_corrupted(
    model,
    cifar10_c_path,
    transform_test,
    device,
    batch_size=256,
    num_workers=4,
    log=None,
):
    if not os.path.exists(cifar10_c_path):
        print('CIFAR-10-C dataset not found')
        exit(1)

    corruptions = os.listdir(cifar10_c_path)
    corruptions.sort()

    if "labels.npy" in corruptions:
        corruptions.remove("labels.npy")

    label = np.load(os.path.join(cifar10_c_path, "labels.npy"))

    model.eval()

    for corruption in corruptions:
        data = np.load(os.path.join(cifar10_c_path, corruption))
        dataset_name = corruption[:-4]

        dataset = DatasetFromNumpy(data, label, transform_test)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        predictions = predict_model(model, dataloader, device)
        corrects = predictions == label

        for idx in range(1, 5+1):
            test_acc = corrects[(idx - 1) * 10000 : idx * 10000].mean()

            if log is not None:
                log["ce_{}_{}".format(dataset_name, idx)] = 1 - test_acc


def analysis_model_perturbated(
    model,
    cifar10_p_path,
    transform_test,
    device,
    batch_size=256,
    num_workers=4,
    log=None,
):
    if not os.path.exists(cifar10_p_path):
        print('CIFAR-10-P dataset not found')
        exit(1)

    perturbations = os.listdir(cifar10_p_path)
    perturbations.sort()

    model.eval()
    
    for perturbation in perturbations:
        data = np.load(os.path.join(cifar10_p_path, perturbation)).transpose(1, 0, 2, 3, 4)
        n, m = data.shape[0], data.shape[1]
        data = data.reshape((-1, 32, 32, 3))

        dataset_name = perturbation[:-4]
        dataset = DatasetFromNumpy(data, np.zeros((len(data),)), transform_test)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        predicted = predict_model(model, dataloader, device)

        num_strengths = predicted.shape[0] // 10000
        
        fp_sum = 0.
        for j in range(1, num_strengths):
            predicted_1 = predicted[0 : 10000]
            predicted_j = predicted[j * 10000 : (j + 1) * 10000]
            fp_sum += (predicted_1 == predicted_j).sum()
        
        fp_corr = fp_sum / ((n - 1) * m)
        fp = 1 - fp_corr

        if log is not None:
            log["fp_{}".format(dataset_name)] = fp


def analysis_model_confusion_matrix(
    model,
    dataloader_test,
    device,
    num_classes=10,
    log=None,
):
    model.eval()

    confusion_matrix = np.zeros((num_classes, num_classes))

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(data)

        _, predicted = output.max(1)

        target = target.cpu().numpy()
        predicted = predicted.cpu().numpy()

        for t, p in zip(target, predicted):
            confusion_matrix[t, p] += 1
    
    if log is not None:
        log["cm"] = confusion_matrix

    return confusion_matrix


def plot_confusion_matrix(
    cm,
    target_names=None,
    cmap=None,
    normalize=True,
    labels=True,
    title='Confusion matrix',
    save_path=None,
    show=True,
):
    # from: https://medium.com/@djin31/how-to-plot-wholesome-confusion-matrix-40134fd402a8
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10, 8.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.rcParams["figure.figsize"] = (30, 50)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()


def flatten_log_labels(log):
    log_labels = []
    for analysis in log:
        if analysis in ["calib_tot", "calib_acc", "calib_conf"]:
            calibration_intervals = log[analysis].shape[0]
            calibration_cut = 1. / calibration_intervals
            for i in range(calibration_intervals):
                log_labels.append("{}_{:0.2f}-{:0.2f}"
                    .format(analysis, calibration_cut * i, calibration_cut * (i+1)))
        else:
            log_labels.append(analysis)
    return log_labels


def flatten_logs(log):
    log_list = []
    for analysis in log:
        cur_analysis = log[analysis]
        if analysis in ["calib_tot", "calib_acc", "calib_conf"]:
            calibration_intervals = cur_analysis.shape[0]
            for i in range(calibration_intervals):
                log_list.append(cur_analysis[i])
        else:
            log_list.append(cur_analysis)
    return log_list


def save_logs_into_csv(
    logs,
    csv_file,
):
    log_labels = None
    to_write = []
    for studies in logs:
        if log_labels is None:
            log_labels = ["study", "model"] + flatten_log_labels(logs[studies])
        
        split_idx = studies.find('/')
        study_name = studies[:split_idx]
        model_name =studies[split_idx+1:]

        to_write.append([study_name, model_name] + flatten_logs(logs[studies]))
    
    df = pd.DataFrame(
        to_write,
        columns=log_labels,
    )
    df.to_csv(csv_file, encoding="UTF-8", index=False)
        