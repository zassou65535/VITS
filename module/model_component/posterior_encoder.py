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

#linear spectrogramを入力にとりEncodeを実行するモデル
class PosteriorEncoder(nn.Module):
    def __init__(self,
        speaker_id_embedding_dim,
        in_channels = 513,
        out_channels = 192,
        hidden_channels = 192,
        kernel_size = 5,
        dilation_rate = 1,
        n_layers = 16,
        ):
        super(PosteriorEncoder, self).__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        #入力スペクトログラムに対し前処理を行うネットワーク
        self.preprocess = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        #WNを用いて特徴量の抽出を行う　WNの詳細はwn.py参照
        self.wn = WN(self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, speaker_id_embedding_dim=self.speaker_id_embedding_dim)
        #ガウス分布の平均と分散を生成するネットワーク
        self.projection = nn.Conv1d(self.hidden_channels, self.out_channels * 2, 1)

    def forward(self, spectrogram, spectrogram_lengths, speaker_id_embedded):
        #maskの生成
        max_length = spectrogram.size(2)
        progression = torch.arange(max_length, dtype=spectrogram_lengths.dtype, device=spectrogram_lengths.device)
        spectrogram_mask = (progression.unsqueeze(0) < spectrogram_lengths.unsqueeze(1))
        spectrogram_mask = torch.unsqueeze(spectrogram_mask, 1).to(spectrogram.dtype)
        #入力スペクトログラムに対しConvを用いて前処理を行う
        x = self.preprocess(spectrogram) * spectrogram_mask
        #WNを用いて特徴量の抽出を行う
        x = self.wn(x, spectrogram_mask, speaker_id_embedded=speaker_id_embedded)
        #出力された特徴マップをもとに統計量を生成
        statistics = self.projection(x) * spectrogram_mask
        gauss_mean, gauss_log_variance = torch.split(statistics, self.out_channels, dim=1)
        #平均gauss_mean, 分散exp(gauss_log_variance)の正規分布から値をサンプリング
        z = (gauss_mean + torch.randn_like(gauss_mean) * torch.exp(gauss_log_variance)) * spectrogram_mask
        return z, gauss_mean, gauss_log_variance, spectrogram_mask