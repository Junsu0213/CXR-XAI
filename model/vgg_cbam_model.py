# -*- coding:utf-8 -*-
from model.cbam import CBAM
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class VGG19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         conv_block(64, 64),
                                         CBAM(64, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         conv_block(128, 128),
                                         CBAM(128, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         *[conv_block(256, 256) for _ in range(3)],
                                         CBAM(256, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(),
                                         *[conv_block(512, 512) for _ in range(3)],
                                         CBAM(512, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block5 = nn.Sequential(*[conv_block(512, 512) for _ in range(4)],
                                         CBAM(512, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.linear1 = nn.Sequential(nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear3 = nn.Linear(in_features=4096, out_features=self.out_channels, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        return x
