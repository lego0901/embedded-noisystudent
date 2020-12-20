import os
from datetime import datetime
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import augmentations

import numpy as np
from PIL import Image
import json

from utils import *
from analysis import *

import argparse


""" Basic preparation """
parser = argparse.ArgumentParser(description="CIFAR10 noisy student ST model test")
parser.add_argument("--batch_size_test", default=512, help="batch size for testing", type=int)
parser.add_argument("--batch_size_fgsm", default=256, help="batch size for testing", type=int)
parser.add_argument("--num_workers", default=4, help="number of cpu workers", type=int)
parser.add_argument("--test_path", default="./test", help="test file path")
parser.add_argument("--analysis_path", default="./analysis", help="test file path")
parser.add_argument("--model_width", default=1, help="resnet width of model", type=int)
parser.add_argument("--not_analyze_top5", default=False, help="not to test top5 accuracy", type=bool)
parser.add_argument("--not_analyze_fgsm", default=False, help="not to test CIFAR-10-C", type=bool)
parser.add_argument("--not_analyze_c", default=False, help="not to test CIFAR-10-C", type=bool)
parser.add_argument("--not_analyze_p", default=False, help="not to test CIFAR-10-P", type=bool)
parser.add_argument("--not_analyze_calib", default=False, help="not to test calibration errors", type=bool)
parser.add_argument("--not_analyze_cm", default=False, help="not to test confusion matrix", type=bool)
parser.add_argument("--stochastic_depth_0_prob", default=1.0, help="stochastic depth prob of the first resnet layer", type=float)
parser.add_argument("--stochastic_depth_L_prob", default=0.8, help="stochastic depth prob of the final resnet layer", type=float)
parser.add_argument("--dropout_prob", default=0.5, help="dropout probability for fc", type=float)
parser.add_argument("--device", default="auto", help="device to run the model", type=str)

args = parser.parse_args()

no_rewrite_keys = ["batch_size_test", "batch_size_fgsm", "num_workers", "device"]

if not os.path.exists(args.test_path):
    print("Please let me know your test file path")
    exit(1)

if not os.path.exists(args.analysis_path):
    os.mkdir(args.analysis_path)

if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device {} has found".format(device))
else:
    device = args.device


""" Datasets Setting """
labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

cifar10_mean, cifar10_std = [0.4913, 0.4821, 0.4465], [0.2470, 0.2434, 0.2615]

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ]
)

print("Loading labeled test image for CIFAR10 dataset...")
dataset_test = CIFAR10(root="./data", train=False, transform=transform_test, download=True)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=args.batch_size_test,
    shuffle=False,
    num_workers=args.num_workers,
)

if not args.not_analyze_fgsm:
    dataset_raw = CIFAR10(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    dataloader_raw = DataLoader(
        dataset_raw,
        batch_size=args.batch_size_fgsm,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mean = torch.tensor(cifar10_mean).to(device).view(1, -1, 1, 1)
    std = torch.tensor(cifar10_std).to(device).view(1, -1, 1, 1)
    normalize = lambda data : (data - mean) / std


""" Analysis models on test path """
test_files = os.listdir(args.test_path)
test_files.sort()

#test_files = ["noisy_label_smoothing", "mixup_noisy_hard", "noisy_hard", "full_labeled_resnet38"]
#test_files = ["ablation_no_model_noise", "mixup_noisy_hard", "noisy_hard", "full_labeled_resnet38"]

logs = OrderedDict()

for studies in test_files:
    if studies[0] == "_":
        # temporary results
        continue

    config_filename = os.path.join(args.test_path, studies, "config.json")
    model_dir = os.path.join(args.test_path, studies, "model")

    model_args = copy.deepcopy(args)

    with open(config_filename) as json_file:
        model_json = json.load(json_file)
        for key in model_json.keys():
            if key in no_rewrite_keys:
                continue
            model_args.__setattr__(key, model_json[key])

    model_names = os.listdir(model_dir)
    model_names.sort()

    for model_name in model_names:
        model_path = os.path.join(model_dir, model_name)

        print("Loading model from {}, {}".format(studies, model_name))

        query_string_student = "student_resnet"
        query_string_teacher = "teacher_resnet"

        find_student = model_name.find(query_string_student)
        find_teacher = model_name.find(query_string_teacher)

        if find_student != -1:
            # from another noisy student: add model noise
            model_layer = int(model_name[find_student + len(query_string_student) : -4])
            model = make_model(
                model_layer,
                width=model_args.model_width,
                prob_0_L=(model_args.stochastic_depth_0_prob, model_args.stochastic_depth_L_prob),
                dropout_prob=model_args.dropout_prob,
                num_classes=10,
            ).to(device)
        elif find_teacher != -1:
            # from pure teacher model: no model noise
            model_layer = int(model_name[find_teacher + len(query_string_teacher) : -4])
            model = make_model(
                model_layer,
                width=model_args.model_width,
                prob_0_L=(1.0, 1.0),
                dropout_prob=0.0,
                num_classes=10,
            ).to(device)
        else:
            print("Invalid model pth name")
            exit(1)
        
        model.load_state_dict(torch.load(model_path))

        log = OrderedDict()

        print(">>> analyzing top1 accuracy")
        analysis_model_top1_accuracy(
            model,
            dataloader_test,
            device,
            log,
        )

        if not model_args.not_analyze_top5:
            print(">>> analyzing top5 accuracy")
            analysis_model_top5_accuracy(
                model,
                dataloader_test,
                device,
                log
            )
        
        if not model_args.not_analyze_fgsm:
            print(">>> analyzing robustness againest FGSM adversary attack")
            for epsilon in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, \
                            0.06, 0.07, 0.08, 0.09, 0.10]:
                analysis_model_fgsm_attack(
                    model,
                    dataloader_raw,
                    normalize,
                    epsilon,
                    device,
                    log,
                )
        
        if not model_args.not_analyze_calib:
            print(">>> analyzing calibration error")
            analysis_model_calibration_error(
                model,
                dataloader_test,
                device,
                confidence_intervals=10,
                log=log,
            )
        
        if not model_args.not_analyze_c:
            print(">>> analyzing robustness on CIFAR-10-C")
            cifar10_c_path = './data/CIFAR-10-C'
            analysis_model_corrupted(
                model,
                cifar10_c_path,
                transform_test,
                device,
                batch_size=model_args.batch_size_test,
                num_workers=model_args.num_workers,
                log=log,
            )
        
        if not model_args.not_analyze_p:
            print(">>> analyzing robustness on CIFAR-10-P")
            cifar10_p_path = './data/CIFAR-10-P'
            analysis_model_perturbated(
                model,
                cifar10_p_path,
                transform_test,
                device,
                batch_size=model_args.batch_size_test,
                num_workers=model_args.num_workers,
                log=log,
            )
        
        if not model_args.not_analyze_cm:
            print(">>> drawing confusion matrix")
            cm = analysis_model_confusion_matrix(
                model,
                dataloader_test,
                device,
            )

            cm_path = os.path.join(args.analysis_path, "confusion_matrices")
            if not os.path.exists(cm_path):
                os.mkdir(cm_path)

            save_path = os.path.join(cm_path, "./{}_{}.png".format(studies, model_name))
            plot_confusion_matrix(
                cm,
                target_names=labels,
                title="Confusion matrix for \"{}\" - \"{}\"".format(studies, model_name),
                save_path=save_path,
                show=False
            )

        logs["{}/{}".format(studies, model_name)] = log

        save_logs_into_csv(logs, csv_file=os.path.join(args.analysis_path, "logs.csv"))

        del model
