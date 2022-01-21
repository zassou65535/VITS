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

#z, speaker_id_embeddedを入力にとり音声を生成するネットワーク
class Decoder(torch.nn.Module):
    def __init__(self,
        speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
        in_z_channel = 192,#入力するzのchannel数
        upsample_initial_channel = 512,#入力されたzと埋め込み済み話者idの両者のチャネル数を、まず最初にconvを適用させることによってupsample_initial_channelに揃える
        deconv_strides = [8, 8, 2, 2],#各Deconv1d層のstride
        deconv_kernel_sizes = [16, 16, 4, 4],#各Deconv1d層のカーネルサイズ
        resblock_kernel_sizes = [3, 7, 11],#各ResnetBlockのカーネルサイズ
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]):#各ResnetBlockのdilation
        super(Decoder, self).__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim#話者idの埋め込み先のベクトルの大きさ
        self.in_z_channel = in_z_channel#入力するzのchannel数
        self.upsample_initial_channel = upsample_initial_channel#入力されたzと埋め込み済み話者idのチャネル数を、まず最初にconvを適用させることによってupsample_initial_channelに揃える
        self.deconv_strides = deconv_strides#各Deconv1d層のstride
        self.deconv_kernel_sizes = deconv_kernel_sizes#各Deconv1d層のカーネルサイズ
        self.resblock_kernel_sizes = resblock_kernel_sizes#各ResnetBlockのカーネルサイズ
        self.resblock_dilation_sizes = resblock_dilation_sizes#各ResnetBlockのdilation

        #Deconv1d層をいくつ生成するか
        self.num_deconvs = len(self.deconv_strides)
        #Deconv1d層1つにつき、self.num_resnet_blocks個resnetblockを生成する
        self.num_resnet_blocks = len(self.resblock_kernel_sizes)

        #入力されたzに最初に適用するネットワーク
        self.conv1d_pre = nn.Conv1d(self.in_z_channel, self.upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        #入力された埋め込み済み話者idに最初に適用するネットワーク
        self.cond = nn.Conv1d(self.speaker_id_embedding_dim, self.upsample_initial_channel, 1)

        #各Deconv1d層を作成する
        self.ups = nn.ModuleList()
        for i, (stride, kernel) in enumerate(zip(self.deconv_strides, self.deconv_kernel_sizes)):
            self.ups.append(torch.nn.utils.weight_norm(
                nn.ConvTranspose1d(self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)), kernel_size=kernel, stride=stride, padding=(kernel-stride)//2)))

        #Deconv1d層1つに対し、ResnetBlockをself.num_resnet_blocks個生成する
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resnet_blocks_channels = self.upsample_initial_channel//(2**(i+1))
            for j, (kernel, dilation) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(ResNetBlock(channels=resnet_blocks_channels, kernel_size=kernel, dilation=dilation))

        #resnet_blocks_channels = 32
        self.conv1d_post = nn.Conv1d(resnet_blocks_channels, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

    def forward(self, z, speaker_id_embedded):
        #z, speaker_id_embedded両者のchannel数をconv1dによって揃える
        x = self.conv1d_pre(z) + self.cond(speaker_id_embedded)
        #各Deconv1d層の適用
        for i in range(self.num_deconvs):
            x = F.leaky_relu(x, 0.1)
            #各Deconv1d層の適用
            x = self.ups[i](x)
            #ResnetBlockをself.num_resnet_blocks個ずつ適用、（出力の総和/self.num_resnet_blocks）をxsとする
            xs = None
            for j in range(self.num_resnet_blocks):
                if xs is None:
                    xs = self.resblocks[i*self.num_resnet_blocks+j](x)
                else:
                    xs += self.resblocks[i*self.num_resnet_blocks+j](x)
            x = xs / self.num_resnet_blocks
        x = F.leaky_relu(x)
        #出力音声はchannel数1
        x = self.conv1d_post(x)
        wav_fake = torch.tanh(x)
        #生成された音声の出力
        return wav_fake
