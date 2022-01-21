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

from .wn import WN

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x

class ResidualCouplingLayer(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        speaker_id_embedding_dim=0,
        mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.in_z_channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        #WNを用いて特徴量の抽出を行う　WNの詳細はwn.py参照
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, speaker_id_embedding_dim=speaker_id_embedding_dim)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, speaker_id_embedded=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0) * x_mask
        h = self.wn(h, x_mask, speaker_id_embedded=speaker_id_embedded)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

#zと埋め込み済み話者idを入力にとり、Monotonic Alignment Searchで用いる変数z_pを出力するネットワーク
class Flow(nn.Module):
    def __init__(self,
        speaker_id_embedding_dim=256,#話者idの埋め込み先のベクトルの大きさ
        in_z_channels=192,#入力するzのchannel数
        phoneme_embedding_dim=192,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4):
        super().__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim#話者idの埋め込み先のベクトルの大きさ
        self.in_z_channels = in_z_channels#入力するzのchannel数
        self.phoneme_embedding_dim = phoneme_embedding_dim#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(self.in_z_channels, self.phoneme_embedding_dim, self.kernel_size, self.dilation_rate, self.n_layers, speaker_id_embedding_dim=self.speaker_id_embedding_dim, mean_only=True))
            self.flows.append(Flip())

    def forward(self, z, z_mask, speaker_id_embedded, reverse=False):
        z_p = z
        z_p_mask = z_mask
        #順方向
        if not reverse:
            for flow in self.flows:
                z_p, _ = flow(z_p, z_p_mask, speaker_id_embedded=speaker_id_embedded, reverse=reverse)
        #逆方向
        else:
            for flow in reversed(self.flows):
                z_p = flow(z_p, z_p_mask, speaker_id_embedded=speaker_id_embedded, reverse=reverse)
        return z_p