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

def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int(((kernel_size - 1)*dilation)/2)

class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResNetBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)

            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

class Decoder(torch.nn.Module):
    def __init__(self, speaker_id_embedding_dim):
        super(Decoder, self).__init__()

        self.initial_channel = 192
        self.resblock = 1
        self.resblock_kernel_sizes = [3, 7, 11] 
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_initial_channel = 512
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        self.speaker_id_embedding_dim = speaker_id_embedding_dim

        self.num_kernels = len(self.resblock_kernel_sizes) #3
        self.num_upsamples = len(self.upsample_rates) #4
        self.conv1d_pre = nn.Conv1d(self.initial_channel, self.upsample_initial_channel, 7, 1, padding=3)
        resblock = ResNetBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(torch.nn.utils.weight_norm(
                nn.ConvTranspose1d(self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):#4
            ch = self.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        #ch = 32
        self.conv1d_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        self.cond = nn.Conv1d(self.speaker_id_embedding_dim, self.upsample_initial_channel, 1)

    def forward(self, feature_map, speaker_id_embedded):
        #feature_map.size() : torch.Size([batch_size, 192, 32])
        #feature_map.size() : torch.Size([64, 192, 32])
        x = self.conv1d_pre(feature_map) + self.cond(speaker_id_embedded)
        #x.size() : torch.Size([64, 512, 32])

        for i in range(self.num_upsamples):#4
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):#3
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv1d_post(x)
        x = torch.tanh(x)

        return x
