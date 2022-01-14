#encoding:utf-8

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
from torch.autograd import Function
import torch.nn.functional as F

#入力波形をperiod個に分割し、それぞれについて真贋を判定するDiscriminator
class PeriodicDiscriminator(torch.nn.Module):
    def __init__(self, period):
        super(PeriodicDiscriminator, self).__init__()
        self.period = period #入力波形を何分割するか
        self.kernel_size = 5
        self.stride = 3
        self.padding = 2

        self.conv2ds = nn.ModuleList([
            torch.nn.utils.weight_norm(nn.Conv2d(1, 32, (self.kernel_size, 1), (self.stride, 1), padding=(self.padding, 0))),
            torch.nn.utils.weight_norm(nn.Conv2d(32, 128, (self.kernel_size, 1), (self.stride, 1), padding=(self.padding, 0))),
            torch.nn.utils.weight_norm(nn.Conv2d(128, 512, (self.kernel_size, 1), (self.stride, 1), padding=(self.padding, 0))),
            torch.nn.utils.weight_norm(nn.Conv2d(512, 1024, (self.kernel_size, 1), (self.stride, 1), padding=(self.padding, 0))),
            torch.nn.utils.weight_norm(nn.Conv2d(1024, 1024, (self.kernel_size, 1), 1, padding=self.padding)),
        ])
        self.conv2d_post = torch.nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        #feature matching loss(GANの学習安定化用)を取るため各特徴量を保持する
        feature_maps = []

        #1Dの入力波形をperiod個に分割、2Dの特徴量へと変換
        batch_size, channel, length = x.shape
        if length % self.period != 0: #足りない分をpadding
            n_pad = self.period - (length % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            length = length + n_pad
        x = x.view(batch_size, channel, length // self.period, self.period)

        #conv2dとleaky_reluの適用
        for conv2d in self.conv2ds:
            x = conv2d(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)
        x = self.conv2d_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps

#入力波形についてそのまま真贋を判定するDiscriminator
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1ds = nn.ModuleList([
            torch.nn.utils.weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            torch.nn.utils.weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            torch.nn.utils.weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            torch.nn.utils.weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            torch.nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            torch.nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv1d_post = torch.nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        #feature matching loss(GANの学習安定化用)を取るため各特徴量を保持する
        feature_maps = []

        #conv1dとleaky_reluの適用
        for conv1d in self.conv1ds:
            x = conv1d(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)
        x = self.conv1d_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps

class VitsDiscriminator(torch.nn.Module):
    def __init__(self):
        super(VitsDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(),#DiscriminatorSは通常のdiscriminator
            PeriodicDiscriminator(period=2),#PeriodicDiscriminatorは波形をperiod個のchunkに分け、それぞれについて真贋判定を行う
            PeriodicDiscriminator(period=3),
            PeriodicDiscriminator(period=5),
            PeriodicDiscriminator(period=7),
            PeriodicDiscriminator(period=11)
        ])

    def forward(self, input_wave):
        authenticities = []#各discriminatorによるinput_waveの真贋判定結果
        feature_maps_list = []#feature matching loss(GANの学習安定化用)を取るため各層から出力される特徴量を保持する
        
        #各discriminatorに対しinput_waveを入力、真贋判定を実行
        for i, discriminator in enumerate(self.discriminators):
            authenticity, feature_maps = discriminator(input_wave)
            authenticities.append(authenticity)
            feature_maps_list.append(feature_maps)

        return authenticities, feature_maps_list