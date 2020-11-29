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
from math import ceil

import argparse
import json


""" Basic preparation """
parser = argparse.ArgumentParser(description="CIFAR10 noisy student ST model")
parser.add_argument("--lr", default=0.1, help="learning rate for training step", type=float)
parser.add_argument("--momentum", default=0.9, help="factor for training step", type=float)
parser.add_argument("--weight_decay", default=1e-4, help="factor for training step", type=float)
parser.add_argument("--batch_size", default=512, help="batch size for training", type=int)
parser.add_argument("--batch_size_test", default=512, help="batch size for testing", type=int)
parser.add_argument("--num_workers", default=4, help="number of cpu workers", type=int)
parser.add_argument("--log_path", default=None, help="directory to save logs")
parser.add_argument("--randaugment_magnitude", default=27, help="magnitude of randaugment", type=int)
parser.add_argument("--no_randaugment", default=False, help="to use randaugment or not", type=bool)
parser.add_argument("--stochastic_depth_0_prob", default=1.0, help="stochastic depth prob of the first resnet layer", type=float)
parser.add_argument("--stochastic_depth_L_prob", default=0.8, help="stochastic depth prob of the final resnet layer", type=float)
parser.add_argument("--dropout_prob", default=0.5, help="dropout probability for fc", type=float)
parser.add_argument("--device", default="auto", help="device to run the model", type=str)
parser.add_argument("--ratio_labeled", default=0.1, help="ratio of labeled training data", type=float)
parser.add_argument("--label_type", default="hard", help="label type for teacher generated dataset", type=str)
parser.add_argument("--label_smoothing_epsilon", default=0.1, help="for epsilon value of label smoothing", type=float)
parser.add_argument("--confidence_threshold", default=0.8, help="minimum confidence level of unlabeled data from the teacher model", type=float)
parser.add_argument("--min_images_per_class", default=4000, help="minimum number of images per each class when generating dataset", type=int)
parser.add_argument("--max_gap_num_images_between_classes", default=1000, help="maximum allowed gap among the number of images per class", type=int)
parser.add_argument("--teacher", default=None, help="load pretrained teacher model")
parser.add_argument("--teacher_layer", default=20, help="teacher initial layer", type=int)
parser.add_argument("--teacher_width", default=1, help="resnet width of teacher model", type=int)
parser.add_argument("--teacher_num_learning_images", default=5000000, help="the number of images required to train teacher", type=int)
parser.add_argument("--student_layer", default=38, help="final student layer number", type=int)
parser.add_argument("--student_width", default=1, help="resnet width of student model", type=int)
parser.add_argument("--student_num_learning_images", default=15000000, help="the number of images required to train student", type=int)
parser.add_argument("--teacher_mixup", default=False, help="apply mixup for teacher model", type=bool)
parser.add_argument("--student_mixup", default=False, help="apply mixup for teacher model", type=bool)
parser.add_argument("--mixup_alpha", default=1.0, help="alpha for beta distrubution of mixup", type=float)
parser.add_argument("--only_train_teacher", default=False, help="only trains the teacher model w/o ST", type=bool)

args = parser.parse_args()

if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device {} has found".format(device))
else:
    device = args.device

if not os.path.exists("test"):
    os.mkdir("test")

if args.log_path is None:
    log_info = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = "./test/run_" + log_info
    if os.path.exists(log_path):
        print("! Warning: path {} already exists".format(log_path))
    else:
        os.mkdir(log_path)
else:
    log_path = os.path.join("./test", args.log_path)
    if os.path.exists(log_path):
        print("! Warning: path {} already exists".format(log_path))
    else:
        os.mkdir(log_path)

with open(os.path.join(log_path, "config.json"), "wt") as f:
    json.dump(vars(args), f, indent=4)


""" Datasets Setting """
labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

cifar10_mean, cifar10_std = [0.4913, 0.4821, 0.4465], [0.2470, 0.2434, 0.2615]

transform_common = transforms.ToTensor()
transform_noisy = transforms.Compose(
    [
        transforms.ToPILImage(),
        augmentations.RandAugment(2, args.randaugment_magnitude),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ]
)

if args.no_randaugment:
    transform_noisy.transforms.pop(1)
    print(transform_noisy)

print("Loading CIFAR10 dataset...")
dataset_train_full = CIFAR10(root="./data", train=True, transform=transform_common, download=True)

print("Separating labeled & unlabeled training set...")
torch.manual_seed(0)  # to deterministically divide splits
num_labeled = int(args.ratio_labeled * len(dataset_train_full))
dataset_train_labeled, dataset_train_unlabeled = torch.utils.data.random_split(
    dataset_train_full,
    [
        num_labeled,
        len(dataset_train_full) - num_labeled,
    ],
)
print("> Number of labeled training images: {}".format(len(dataset_train_labeled)))
print("> Number of unlabeled training images: {}".format(len(dataset_train_unlabeled)))
torch.manual_seed(torch.initial_seed())

print("Loading labeled test image for CIFAR10 dataset...")
dataset_test = CIFAR10(root="./data", train=False, transform=transform_test, download=True)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=args.batch_size_test,
    shuffle=False,
    num_workers=args.num_workers,
)


""" Teacher model preparation """
if args.teacher is None:
    print("Creating and begining to train the teacher model with resnet {}".format(args.teacher_layer))
    
    teacher_model = make_model(
        args.teacher_layer,
        width=args.teacher_width,
        prob_0_L=(1.0, 1.0),
        dropout_prob=0.0,
        num_classes=10,
    ).to(device)
    teacher_model.train()

    dataset_train_teacher = DatasetApplyTransform(
        dataset_train_labeled, transform_noisy, device
    )
    dataloader_train_teacher = DataLoader(
        dataset_train_teacher,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    ys = np.eye(10)[dataset_train_teacher.label].sum(axis=0)
    for i in range(len(labels)):
        print("> Number of image {}: {}".format(labels[i].ljust(10), int(ys[i])))

    teacher_epochs = ceil(args.teacher_num_learning_images / len(dataset_train_teacher))
    print("To learn {} images while learning, {} epochs are required".format(args.teacher_num_learning_images, teacher_epochs))

    optimizer = optim.SGD(
        teacher_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * teacher_epochs), int(0.75 * teacher_epochs)],
        gamma=0.1,
    )
    
    teacher_save_path = os.path.join(log_path, "model/teacher_resnet{}.pth".format(args.teacher_layer))
    teacher_log_path = os.path.join(log_path, "log/teacher_resnet{}.csv".format(args.teacher_layer))

    train_model(
        teacher_model,
        dataloader_train=dataloader_train_teacher,
        dataloader_test=dataloader_test,
        device=device,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        save_path=teacher_save_path,
        log_path=teacher_log_path,
        epochs=teacher_epochs,
        onehot=False,
        mixup=args.teacher_mixup,
        mixup_alpha=args.mixup_alpha,
    )
    
    teacher_model.load_state_dict(torch.load(teacher_save_path))
else:
    print("Loading teacher model from {}".format(args.teacher))
    if args.teacher.find("model/student_resnet") != -1:
        # from another noisy student: add model noise
        teacher_model = make_model(
            args.teacher_layer,
            width=args.teacher_width,
            prob_0_L=(args.stochastic_depth_0_prob, args.stochastic_depth_L_prob),
            dropout_prob=args.dropout_prob,
            num_classes=10,
        ).to(device)
    else:
        # from pure teacher model: no model noise
        teacher_model = make_model(
            args.teacher_layer,
            width=args.teacher_width,
            prob_0_L=(1.0, 1.0),
            dropout_prob=0.0,
            num_classes=10,
        ).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher))

test_loss, test_acc = test_model(teacher_model, dataloader_test, device, onehot=False)
print("Test loss and accuarcy for the teacher: {}, {}"
    .format(round(test_loss, 4), round(test_acc, 4)))

if args.only_train_teacher:
    exit(0)


""" Teacher-noisy student optimizer """
teacher_layer, student_layer = args.teacher_layer, args.teacher_layer + 6

while student_layer <= args.student_layer:
    print("Generating dataset from the teacher model with {} type...".format(args.label_type))
    dataset_train_student = DatasetFromTeacher(
        teacher_model,
        dataset_labeled=dataset_train_labeled,
        dataset_unlabeled=dataset_train_unlabeled,
        transform_test=transforms.Compose([transforms.ToPILImage(), transform_test]),
        transform_noisy=transform_noisy,
        device=device,
        args=args,
    )
    print("Generated {} datasets with the confidence threshold {} (minimum {} images per class)"
        .format(len(dataset_train_student), args.confidence_threshold, args.min_images_per_class))

    ys_before = dataset_train_student.num_images_per_label()
    for i in range(len(labels)):
        print("> Number of image {}: {}".format(labels[i].ljust(10), int(ys_before[i])))
    
    print("Balancing the data...")
    dataset_train_student.balance_data(args.min_images_per_class, args.max_gap_num_images_between_classes)
    ys_after = dataset_train_student.num_images_per_label()
    for i in range(len(labels)):
        print("> Number of image {}: {}".format(labels[i].ljust(10), int(ys_after[i])))
        
    dataset_info = {
        'generated by teacher': ys_before,
        'data balanced': ys_after
    }

    if not os.path.exists(os.path.join(log_path, "log")):
        os.mkdir(os.path.join(log_path, "log"))
    with open(os.path.join(log_path, "log/dataset_student_resnet{}.json".format(student_layer)), "wt") as f:
        json.dump(dataset_info, f, indent=4)

    del teacher_model

    print("Creating and begining to train the student model with resnet {}".format(student_layer))
    student_model = make_model(
        student_layer,
        width=args.student_width,
        prob_0_L=(args.stochastic_depth_0_prob, args.stochastic_depth_L_prob),
        dropout_prob=args.dropout_prob, num_classes=10
    ).to(device)

    dataloader_train_student = DataLoader(
        dataset_train_student,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    student_epochs = ceil(args.student_num_learning_images / len(dataset_train_student))
    print("To learn {} images while learning, {} epochs are required".format(args.student_num_learning_images, student_epochs))
    
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * student_epochs), int(0.75 * student_epochs)],
        gamma=0.1,
    )

    student_save_path = os.path.join(log_path, "model/student_resnet{}.pth".format(student_layer))
    student_log_path = os.path.join(log_path, "log/student_resnet{}.csv".format(student_layer))

    train_model(
        student_model,
        dataloader_train=dataloader_train_student,
        dataloader_test=dataloader_test,
        device=device,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        save_path=student_save_path,
        log_path=student_log_path,
        epochs=student_epochs,
        onehot=True,
        mixup=args.student_mixup,
        mixup_alpha=args.mixup_alpha,
    )

    student_model.load_state_dict(torch.load(student_save_path))

    test_loss, test_acc = test_model(student_model, dataloader_test, device, onehot=False)
    print("Test loss and accuarcy for the student {}: {}, {}"
        .format(student_layer, round(test_loss, 4), round(test_acc, 4)))

    teacher_model = student_model
    student_layer += 6
