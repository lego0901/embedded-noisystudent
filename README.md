# Noisy Student Self-Training on CIFAR-10

*Term project for Embedded System and Application (SNU 2020 Fall)*

### Basic Training Scheme

I applied noisy students self-training methods on CIFAR-10, with 10/90% of labeled/unlabeled training data. I used teacher-noisy student pipelines as

- Train ResNet-20 as a teacher with only labeled training data.
- ResNet-20 as a teacher and ResNet-26 as a noisy student.
- ResNet-26 as a teacher and ResNet-32 as a noisy student.
- ResNet-32 as a teacher and ResNet-38 as a noisy student.

I used RandAugment of magnitude 27, stochastic depth decays to 0.8, and dropout with 0.5, as in [[2]](https://arxiv.org/abs/1911.04252).

### How to Run

I recommend you to install the same python runtime environment to me.

```bash
pip install -r requirements.txt
```

You can clone and run training scripts via:

```bash
python3 train.py   # for train
python3 test.py    # for test and analysis
```

There are several available options for training/testing, such as batch size or device selection. See `train.py`, `test.py`, `run_train.sh`, `run_test.sh`. The explanation of the implementation is written in the [report](https://github.com/lego0901/embedded-noisystudent/blob/main/document/noisy_student_cifar_10_kor.pdf).

You can run all training and testing scripts include ablations studies on the report via:

```bash
bash ./run_train.sh
bash ./run_test.sh
```

The `run_train.sh` will create a directory named `test`, which contains models and training information(loss and accuracy). The `run_test.sh` will create a directory named `analysis`, which contains `logs.csv` that has Top-1 Acc., FGSM, ECE, CE, FR, ... on each model inside the `test` directory.

Please note that CIFAR-10-C(2.9 GB) and CIFAR-10-P(18.3 GB) are quite heavy. If you don't want to analyze these results, please modify inside of `run_tesh.sh` with

```bash
NOT_ANALYZE="--not_analyze_c=True --not_analyze_p=True"
```

### Results

I used pseudo label with hard, soft, and label smoothing while training. Also, I applied mixup method on hard label type. (Mixup+soft label or +label smoothing were bad in results)

<p align="center">
<img height="360" src=document/resource/results.png>
</p>


You can download all [training models](https://drive.google.com/file/d/16E8sqJv-k6hgZjzSSxlIy_WyJjZIIU55/view?usp=sharing) and [analysis results](https://drive.google.com/file/d/1o4iAi4Cx75iBMAjK3pfPL276fTt8sJyz/view?usp=sharing) from the links. The results for ablation studies are written in the [report](https://github.com/lego0901/embedded-noisystudent/blob/main/document/noisy_student_cifar_10_kor.pdf).

### References

1. Xiaojin Zhu. Semi-supervised learning. tutorial, 2007.
2. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification, 2020.
3. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015.
4. Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space, 2019.
5. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Weinberger. Deep networks with stochastic depth, 2016.
6. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision, 2015.
7. Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization, 2018.
8. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks, 2017.
9. Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples, 2015.
10. Morgane Goibert and Elvis Dohmatob. Adversarial robustness via label-smoothing, 2019.
11. Dan Hendrycks and Thomas G. Dietterich. Benchmarking neural network robustness to common corruptions and surface variations, 2019.

#### Code References & Data

- RandAugment: [ildoonet/pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)
- Stochastic Depth on ResNet: [shamangary/Pytorch-Stochastic-Depth-Resnet](https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet)
- Mixup: [facebookresearch/mixup-cifar-10](https://github.com/facebookresearch/mixup-cifar10)
- CIFAR-10-C and CIFAR-10-P: [hendrycks/robustness](https://github.com/hendrycks/robustness)
- Confusion Matrix: [How to plot wholesome confusion matrix?](https://medium.com/@djin31/how-to-plot-wholesome-confusion-matrix-40134fd402a8)

