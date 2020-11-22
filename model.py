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
    def __init__(self, prob, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.norm_layer = norm_layer

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)

        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        residual = x

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                residual = self.bn1(residual)
                residual = self.relu1(residual)
                residual = self.conv1(residual)

                residual = self.bn2(residual)
                residual = self.relu2(residual)
                residual = self.conv2(residual)

                if self.downsample is not None:
                    x = self.downsample(x)
                residual += x
            else:
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    x = self.downsample(x)
                residual = x
        else:
            residual = self.bn1(residual)
            residual = self.relu1(residual)
            residual = self.conv1(residual)

            residual = self.bn2(residual)
            residual = self.relu2(residual)
            residual = self.conv2(residual)

            if self.downsample is not None:
                x = self.downsample(x)
            residual += self.prob * x

        return residual


class StoDepth_ResNet(nn.Module):
    def __init__(self, prob_0_L, num_layers, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(StoDepth_ResNet, self).__init__()

        self.prob = prob_0_L[0]
        self.prob_step = (prob_0_L[0] - prob_0_L[1]) / (num_layers - 1)

        self.num_layers = num_layers
        assert (num_layers - 2) % 6 == 0
        self.N = (self.num_layers - 2) // 6

        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.inplanes = 16
        self.layer1 = self._make_layers(planes=16, stride=1)
        self.layer2 = self._make_layers(planes=32, stride=2)
        self.layer3 = self._make_layers(planes=64, stride=2)
        self.bn = norm_layer(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(
                    0, torch.sqrt(torch.Tensor([2.0 / n])).item()
                )

    def _make_layers(self, planes, stride):
        if stride == 2:
            down_sample = Downsample(self.inplanes, planes, stride)
        else:
            down_sample = None

        layers_list = nn.ModuleList(
            [
                StoDepth_BasicBlock(
                    self.prob,
                    self.inplanes,
                    planes,
                    self.norm_layer,
                    stride,
                    down_sample,
                )
            ]
        )
        self.inplanes = planes
        self.prob -= self.prob_step

        for _ in range(self.N - 1):
            layers_list.append(
                StoDepth_BasicBlock(self.prob, planes, planes, self.norm_layer)
            )
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
        x = self.fc(x)

        return x
