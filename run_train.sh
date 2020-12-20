#!/bin/bash

# Basic options about training phase. More options are available in train.py.
OPT="--batch_size=512 --batch_size_test=512 --num_workers=4 --device=cuda"


echo "================================================================"
echo "              full labeled training for comparisons             "
echo "================================================================"

echo "RESNET-20 with full labeled data"
python3 train.py --log_path=full_labeled_resnet20 \
    --only_train_teacher=True --teacher_layer=20 \
    --teacher_num_learning_images=10000000 \
    --ratio_labeled=1.0 \
    $OPT

echo "RESNET-26 with full labeled data"
python3 train.py --log_path=full_labeled_resnet26 \
    --only_train_teacher=True --teacher_layer=26 \
    --teacher_num_learning_images=10000000 \
    --ratio_labeled=1.0 \
    $OPT

echo "RESNET-32 with full labeled data"
python3 train.py --log_path=full_labeled_resnet32 \
    --only_train_teacher=True --teacher_layer=32 \
    --teacher_num_learning_images=10000000 \
    --ratio_labeled=1.0 \
    $OPT

echo "RESNET-38 with full labeled data"
python3 train.py --log_path=full_labeled_resnet38 \
    --only_train_teacher=True --teacher_layer=38 \
    --teacher_num_learning_images=10000000 \
    --ratio_labeled=1.0 \
    $OPT


# run without noisy student ST, only using a single model with 5000 labels
# for ResNet-20, it is used as a shared teacher model
echo "================================================================"
echo "run without noisy student ST only with ratio 0.1 of labeled data"
echo "================================================================"

echo "RESNET-20 without noisy student"
python3 train.py --log_path=without_student_resnet20 \
    --teacher_num_learning_images=5000000 \
    --only_train_teacher=True --teacher_layer=20 \
    $OPT

echo "RESNET-26 without noisy student"
python3 train.py --log_path=without_student_resnet26 \
    --teacher_num_learning_images=5000000 \
    --only_train_teacher=True --teacher_layer=26 \
    $OPT

echo "RESNET-32 without noisy student"
python3 train.py --log_path=without_student_resnet32 \
    --teacher_num_learning_images=5000000 \
    --only_train_teacher=True --teacher_layer=32 \
    $OPT

echo "RESNET-38 without noisy student"
python3 train.py --log_path=without_student_resnet38 \
    --teacher_num_learning_images=5000000 \
    --only_train_teacher=True --teacher_layer=38 \
    $OPT


echo "================================================================"
echo "    run with noisy student ST from ratio 0.1 of labeled data    "
echo "================================================================"

echo "Noisy student SL with hard label"
python3 train.py --log_path=noisy_hard \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    $OPT

echo "Noisy student SL with soft label"
python3 train.py --log_path=noisy_soft \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    $OPT

echo "Noisy student SL with label smoothing"
python3 train.py --log_path=noisy_label_smoothing \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    $OPT


echo "================================================================"
echo "             run with noisy student ST with mixup               "
echo "================================================================"

echo "Noisy student SL with hard label + mixup"
python3 train.py --log_path=mixup_noisy_hard \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True \
    $OPT

echo "Noisy student SL with soft label + mixup"
python3 train.py --log_path=mixup_noisy_soft \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    --student_mixup=True \
    $OPT

echo "Noisy student SL with label smoothing + mixup"
python3 train.py --log_path=mixup_noisy_label_smoothing \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    --student_mixup=True \
    $OPT


echo "================================================================"
echo "                  ablation study with noises                    "
echo "================================================================"

echo "Noisy student SL without stochastic depth"
python3 train.py --log_path=ablation_no_stochastic_depth \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --label_type=hard \
    $OPT

echo "Noisy student SL without dropout"
python3 train.py --log_path=ablation_no_dropout \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --dropout_prob=0.0 \
    --label_type=hard \
    $OPT

echo "Noisy student SL without model noise"
python3 train.py --log_path=ablation_no_model_noise \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --dropout_prob=0.0 \
    --label_type=hard \
    $OPT

echo "Noisy student SL with small dropout(=0.2)"
python3 train.py --log_path=ablation_dropout0.2 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --dropout_prob=0.2 \
    --label_type=hard \
    $OPT

echo "Noisy student SL with RandAugment magnitude of 9"
python3 train.py --log_path=ablation_randaugment9 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --randaugment_magnitude=9 \
    --label_type=hard \
    $OPT

echo "Noisy student SL with RandAugment magnitude of 3"
python3 train.py --log_path=ablation_randaugment3 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --randaugment_magnitude=3 \
    --label_type=hard \
    $OPT

echo "Noisy student SL without RandAugment"
python3 train.py --log_path=ablation_without_randaugment \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --no_randaugment=True \
    --label_type=hard \
    $OPT

echo "Noisy student SL without any noise"
python3 train.py --log_path=ablation_no_noise \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --dropout_prob=0.0 \
    --no_randaugment=True \
    --label_type=hard \
    $OPT


echo "================================================================"
echo "        ablation study with teacher dataset generation          "
echo "================================================================"

echo "Noisy student SL by generating dataset with confidence threshold 0.9"
python3 train.py --log_path=noisy_hard_confidence0.9 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --confidence_threshold=0.9 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.9, soft label"
python3 train.py --log_path=noisy_soft_confidence0.9 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    --confidence_threshold=0.9 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.9, label smoothing"
python3 train.py --log_path=noisy_label_smoothing_confidence0.9 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    --confidence_threshold=0.9 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.9, mixup"
python3 train.py --log_path=mixup_noisy_hard_confidence0.9 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True \
    --confidence_threshold=0.9 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.7"
python3 train.py --log_path=noisy_hard_confidence0.7 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --confidence_threshold=0.7 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.7, soft label"
python3 train.py --log_path=noisy_soft_confidence0.7 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    --confidence_threshold=0.7 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.7, label smoothing"
python3 train.py --log_path=noisy_label_smoothing_confidence0.7 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    --confidence_threshold=0.7 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.7, mixup"
python3 train.py --log_path=mixup_noisy_hard_confidence0.7 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True \
    --confidence_threshold=0.7 \
    $OPT

echo "Noisy student SL by generating dataset with confidence threshold 0.6"
python3 train.py --log_path=noisy_hard_confidence0.6 \
    --teacher=./test/without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --confidence_threshold=0.6 \
    $OPT


echo "================================================================"
echo "       ablation study with the ratio 0.2 of labeled data        "
echo "================================================================"

echo "RESNET-20 without noisy student with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_without_student_resnet20 \
    --only_train_teacher=True --teacher_layer=20 \
    --ratio_labeled=0.2 \
    $OPT

echo "RESNET-26 without noisy student with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_without_student_resnet26 \
    --only_train_teacher=True --teacher_layer=26 \
    --ratio_labeled=0.2 \
    $OPT

echo "RESNET-32 without noisy student with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_without_student_resnet32 \
    --only_train_teacher=True --teacher_layer=32 \
    --ratio_labeled=0.2 \
    $OPT

echo "RESNET-38 without noisy student with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_without_student_resnet38 \
    --only_train_teacher=True --teacher_layer=38 \
    --ratio_labeled=0.2 \
    $OPT

echo "Noisy student SL with hard label with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_noisy_hard \
    --teacher=./test/ratio0.2_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --ratio_labeled=0.2 \
    $OPT

echo "Noisy student SL with soft label with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_noisy_soft \
    --teacher=./test/ratio0.2_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    --ratio_labeled=0.2 \
    $OPT

echo "Noisy student SL with label smoothing with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_noisy_label_smoothing \
    --teacher=./test/ratio0.2_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    --ratio_labeled=0.2 \
    $OPT

echo "Noisy student SL with hard label + mixup with 0.2 of labeled data"
python3 train.py --log_path=ratio0.2_mixup_noisy_hard \
    --teacher=./test/ratio0.2_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True \
    --ratio_labeled=0.2 \
    $OPT


echo "================================================================"
echo "       ablation study with the ratio 0.05 of labeled data       "
echo "================================================================"

echo "RESNET-20 without noisy student with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_without_student_resnet20 \
    --only_train_teacher=True --teacher_layer=20 \
    --ratio_labeled=0.05 \
    $OPT

echo "RESNET-26 without noisy student with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_without_student_resnet26 \
    --only_train_teacher=True --teacher_layer=26 \
    --ratio_labeled=0.05 \
    $OPT

echo "RESNET-32 without noisy student with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_without_student_resnet32 \
    --only_train_teacher=True --teacher_layer=32 \
    --ratio_labeled=0.05 \
    $OPT

echo "RESNET-38 without noisy student with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_without_student_resnet38 \
    --only_train_teacher=True --teacher_layer=38 \
    --ratio_labeled=0.05 \
    $OPT

echo "Noisy student SL with hard label with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_noisy_hard \
    --teacher=./test/ratio0.05_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --ratio_labeled=0.05 \
    $OPT

echo "Noisy student SL with soft label with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_noisy_soft \
    --teacher=./test/ratio0.05_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=soft \
    --ratio_labeled=0.05 \
    $OPT

echo "Noisy student SL with label smoothing with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_noisy_label_smoothing \
    --teacher=./test/ratio0.05_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=smooth \
    --ratio_labeled=0.05 \
    $OPT

echo "Noisy student SL with hard label + mixup with 0.05 of labeled data"
python3 train.py --log_path=ratio0.05_mixup_noisy_hard \
    --teacher=./test/ratio0.05_without_student_resnet20/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True \
    --ratio_labeled=0.05 \
    $OPT
