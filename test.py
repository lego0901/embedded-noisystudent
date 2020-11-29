import os
from datetime import datetime

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import augmentations

from utils import *

import argparse


""" Basic preparation """
parser = argparse.ArgumentParser(description="CIFAR10 noisy student ST model test")
parser.add_argument("--batch_size_test", default=512, help="batch size for testing", type=int)
parser.add_argument("--num_workers", default=4, help="number of cpu workers", type=int)
parser.add_argument("--model", default=None, help="load pretrained model")
parser.add_argument("--model_layer", default=-1, help="number of layer of pretrained model", type=int)
parser.add_argument("--model_width", default=1, help="resnet width of model", type=int)
parser.add_argument("--stochastic_depth_0_prob", default=1.0, help="stochastic depth prob of the first resnet layer", type=float)
parser.add_argument("--stochastic_depth_L_prob", default=0.8, help="stochastic depth prob of the final resnet layer", type=float)
parser.add_argument("--dropout_prob", default=0.5, help="dropout probability for fc", type=float)
parser.add_argument("--device", default="auto", help="device to run the model", type=str)

args = parser.parse_args()

if args.model is None or args.model_layer == -1:
    print("Model path and number of layers should be passed")
    exit(1)

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


""" Test model preparation """
print("Loading model from {}".format(args.model))
if args.model.find("model/student_resnet") != -1:
    # from another noisy student: add model noise
    model = make_model(
        args.model_layer,
        width=args.model_width,
        prob_0_L=(args.stochastic_depth_0_prob, args.stochastic_depth_L_prob),
        dropout_prob=args.dropout_prob,
        num_classes=10,
    ).to(device)
else:
    # from pure teacher model: no model noise
    model = make_model(
        args.model_layer,
        width=args.model_width,
        prob_0_L=(1.0, 1.0),
        dropout_prob=0.0,
        num_classes=10,
    ).to(device)

model.load_state_dict(torch.load(args.model))

model.eval()
test_loss, test_acc = test_model(model, dataloader_test, device, onehot=False)
print("Test loss and accuarcy for the model: {}, {}"
    .format(round(test_loss, 4), round(test_acc, 4)))

# TODO: add several tests related to the model