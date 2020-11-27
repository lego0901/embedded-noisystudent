#!/bin/bash

# run without noisy student ST, only using a single model with 5000 labels
# for ResNet-20, it is used as a shared teacher model
python3 main.py --log_path=shared_teacher \
    --only_train_teacher=True --teacher_layer=20

# for further sizes, they are used to compare with the result of ST
python3 main.py --log_path=without_student_resnet26 \
    --only_train_teacher=True --teacher_layer=26
python3 main.py --log_path=without_student_resnet32 \
    --only_train_teacher=True --teacher_layer=32
python3 main.py --log_path=without_student_resnet38 \
    --only_train_teacher=True --teacher_layer=38


# run noisy student ST with hard label with confidence 0.8
python3 main.py --log_path=noisy_hard \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=hard

# run noisy student ST with soft label
python3 main.py --log_path=noisy_soft \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=soft

# run noisy student ST with label smoothing
python3 main.py --log_path=noisy_label_smoothing \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=smooth


# run noisy student ST with hard label with mixup
python3 main.py --log_path=mixup_noisy_hard \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=hard \
    --student_mixup=True

# run noisy student ST with soft label with mixup
python3 main.py --log_path=mixup_noisy_soft \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=soft \
    --student_mixup=True

# run noisy student ST with label smoothing with mixup
python3 main.py --log_path=mixup_noisy_label_smoothing \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=smooth \
    --student_mixup=True


# ablation study of excluding stochastic depth
python3 main.py --log_path=ablation_no_stochastic_depth \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --label_type=hard

# ablation study of excluding dropout
python3 main.py --log_path=ablation_no_dropout \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --dropout_prob=0.0 \
    --label_type=hard

# ablation study of excluding both stochastic depth and dropout
python3 main.py --log_path=ablation_no_model_noise \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --dropout_prob=0.0 \
    --label_type=hard

# ablation study of reducing RandAugment magnitude by 9
python3 main.py --log_path=ablation_randaugment9 \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --randaugment_magnitude=9 \
    --label_type=hard

# ablation study of reducing RandAugment magnitude by 3
python3 main.py --log_path=ablation_randaugment3 \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --randaugment_magnitude=3 \
    --label_type=hard

# ablation study of reducing RandAugment magnitude by 0
python3 main.py --log_path=ablation_randaugment0 \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --randaugment_magnitude=0 \
    --label_type=hard

# ablation study of noise on the students
python3 main.py --log_path=ablation_no_noise \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --stochastic_depth_0_prob=1.0 --stochastic_depth_L_prob=1.0 \
    --dropout_prob=0.0 \
    --randaugment_magnitude=0 \
    --label_type=hard

# ablation study with confidence 0.6
python3 main.py --log_path=noisy_hard_confidence0.6 \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=hard \
    --confidence_threshold=0.6

# ablation study with confidence 0.4
python3 main.py --log_path=noisy_hard_confidence0.4 \
    --teacher=./test/shared_teacher/model/teacher_resnet20.pth \
    --label_type=hard \
    --confidence_threshold=0.4
