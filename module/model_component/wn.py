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

@torch.jit.script
def gated_activation_unit(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]#hidden_channels
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])#特徴量の前半はtanh
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])#特徴量の後半はsigmoid
    acts = t_act * s_act
    return acts

#音声を生成するモデル"WaveGlow"(https://arxiv.org/pdf/1811.00002.pdf)でWNという名前で言及されているモジュール
#PosteriorEncoderやFlow内で特徴量の抽出にこのネットワークを用いる
class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_resblocks, speaker_id_embedding_dim):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,#ResidualBlock内のConv1dのカーネルサイズ
        self.dilation_rate = dilation_rate#ResidualBlock内のConv1dのdilationを決めるための数値
        self.n_resblocks = n_resblocks#ResidualBlockをいくつ重ねるか
        self.speaker_id_embedding_dim = speaker_id_embedding_dim#話者idの埋め込み先のベクトルの大きさ

        #n_resblocks個あるResidualBlockの構成要素を保持するModuleList
        self.in_resblocks = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        #embed済み話者idを入力にとり、条件付けを行うための特徴量を出力するネットワーク
        condition_layer = nn.Conv1d(speaker_id_embedding_dim, 2*hidden_channels*n_resblocks, 1)
        self.condition_layer = nn.utils.weight_norm(condition_layer, name='weight')

        #ResidualBlockをn_resblocks個生成
        for i in range(n_resblocks):
            #in_layerに畳み込み層を追加
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_resblocks.append(in_layer)
            #res_skip_layerに畳み込み層を追加
            if i < n_resblocks - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, speaker_id_embedded):
        #x.size(), x_mask.size() : torch.Size([batch_size, 192, length(可変)]) torch.Size([batch_size, 1, length])
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        #embed済み話者idを入力にとり、条件付けを行うための特徴量を出力するネットワークを適用
        speaker_fmap = self.condition_layer(speaker_id_embedded)

        #n_resblocks個のResidualBlockに通す
        for i in range(self.n_resblocks):
            x_in = self.in_resblocks[i](x)
            #speaker_fmapから特徴量を選択
            cond_offset = i * 2 * self.hidden_channels
            speaker_fmap_selected = speaker_fmap[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            #gated_activation_unitの適用　通す情報と通さない情報をフィルタリングすることで学習効率を改善する役割を果たすとされる
            acts = gated_activation_unit(x_in, speaker_fmap_selected, n_channels_tensor)
            #convに通す
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_resblocks - 1:
                #特徴量の前半はxと加算
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                #特徴量の後半は出力用tensorに加算
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask
        