"""
Resnet-X model with stochastic depth feature.

Mostly followed the SimCLR code practice.

Naming convension and implementation is from
shamangary/Pytorch-Stochastic-Depth-Resnet with github link:
https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet/blob/master/TYY_stodepth_lineardecay.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Downsample(nn.Module):
    """shrink width/height with stride and add zero padding to new channels"""

    def __init__(self, in_channels, out_channels, stride):
        super(Downsample, self).__init__()
        self.pooling = nn.AvgPool2d(stride)
        self.add_channel = out_channels - in_channels

    def forward(self, x):
        x = self.pooling(x)
        x = F.pad(x, (0, 0, 0, 0, 0, self.add_channel))
        return x


class StoDepth_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, prob, norm_layer, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.norm_layer = norm_layer

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()

        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]

        if self.training:
            sample = self.m.sample(sample_shape=torch.Size([batch_size, 1, 1])).to(x.device)
            sample.requires_grad = False
            residual = residual.mul(sample)

            residual = self.conv1(residual)
            residual = self.bn1(residual)
            residual = self.relu1(residual)

            residual = self.conv2(residual)
            residual = self.bn2(residual)

            if self.downsample is not None:
                x = self.downsample(x)
            residual = residual + x
        else:
            residual = self.conv1(residual)
            residual = self.bn1(residual)
            residual = self.relu1(residual)

            residual = self.conv2(residual)
            residual = self.bn2(residual)

            if self.downsample is not None:
                x = self.downsample(x)
            residual = residual * self.prob + x

        residual = self.relu2(residual)
        return residual


class StoDepth_ResNet(nn.Module):
    def __init__(
        self,
        num_layers,
        prob_0_L=(1.0, 0.8),
        dropout_prob=0.5,
        num_classes=10,
        norm_layer=nn.BatchNorm2d,
    ):
        super(StoDepth_ResNet, self).__init__()

        self.num_layers = num_layers
        assert (num_layers - 2) % 6 == 0
        self.N = (self.num_layers - 2) // 6

        self.prob = prob_0_L[0]
        self.prob_step = (prob_0_L[0] - prob_0_L[1]) / (self.N * 3)

        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layers(planes=16, stride=1)
        self.layer2 = self._make_layers(planes=32, stride=2)
        self.layer3 = self._make_layers(planes=64, stride=2)
        self.bn = norm_layer(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(8)
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(64, num_classes)

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, torch.sqrt(torch.Tensor([2.0 / n])).item())

    def _make_layers(self, planes, stride):
        if stride == 2:
            down_sample = Downsample(self.inplanes, planes, stride)
        else:
            down_sample = None

        layers_list = nn.ModuleList(
            [
                StoDepth_BasicBlock(
                    self.inplanes,
                    planes,
                    self.prob,
                    self.norm_layer,
                    stride,
                    down_sample,
                )
            ]
        )
        self.inplanes = planes
        self.prob -= self.prob_step

        for _ in range(self.N - 1):
            layers_list.append(StoDepth_BasicBlock(planes, planes, self.prob, self.norm_layer))
            self.prob -= self.prob_step

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        x = self.fc(x)
        return x
