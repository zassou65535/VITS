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

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                      dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask

#linear spectrogramを入力にとりEncodeを実行するモデル
class PosteriorEncoder(nn.Module):
    def __init__(self):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = 513
        self.out_channels = 192
        self.hidden_channels = 192
        self.kernel_size = 5
        self.dilation_rate = 1
        self.n_layers = 16
        self.gin_channels = 256

        self.pre = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        self.enc = modules.WN(self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, gin_channels=self.gin_channels)
        self.proj = nn.Conv1d(self.hidden_channels, self.out_channels * 2, 1)

    def forward(self, spectrogram, spectrogram_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(spectrogram_lengths, spectrogram.size(2)), 1).to(spectrogram.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask