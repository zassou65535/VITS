#encoding:utf-8

import random
import numpy as np
import glob
import os
import itertools
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F

import torchaudio

from module.dataset_util import *
from module.vits_generator import VitsGenerator
from module.vits_discriminator import VitsDiscriminator
from module.loss_function import *

#乱数のシードを設定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

###以下は推論に必要なパラメーター###
#学習済みパラメーターへのパス
trained_weight_path = "./output/vits/train/iteration140000/netG_cpu.pth"
#変換対象としたいwavファイルへのパス
source_wav_path = "./dataset/jvs_preprocessed/jvs_wav_preprocessed/jvs099/VOICEACTRESS100_011.wav"
#変換元の話者id
source_speaker_id = 98
#変換先の話者id
target_speaker_id = 9
#結果を出力するためのディレクトリ
output_dir = "./output/vits/inference/voice_conversion/"
#使用するデバイス
device = "cuda:0"
#学習に使用した音素の種類数
n_phoneme = 40
#話者の数
n_speakers = 100

#生成するor切り出す音声波形の大きさ
segment_size = 8192

###以下は音声処理に必要なパラメーター###
#扱う音声のサンプリングレート
sampling_rate = 22050
#スペクトログラムの計算時に何サンプル単位でSTFTを行うか
filter_length = 1024
#スペクトログラムの計算時に適用する窓の大きさ
win_length = 1024
#ホップ数　何サンプルずらしながらSTFTを行うか
hop_length = 256

#出力用ディレクトリがなければ作る
os.makedirs(output_dir, exist_ok=True)

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

#Generatorのインスタンスを生成
netG = VitsGenerator(n_phoneme=n_phoneme, n_speakers=n_speakers)
#学習済みパラメーターを読み込む
netG.load_state_dict(torch.load(trained_weight_path))
#ネットワークをデバイスに移動
netG = netG.to(device)
#ネットワークを推論モードにする
netG.eval()

###変換対象とするwavファイルの読み込み###
#wavファイルをロード
loaded_wav, _ = torchaudio.load(source_wav_path)
#スペクトログラムを生成
pad_size = int((filter_length-hop_length)/2)
wav_padded = torch.nn.functional.pad(loaded_wav, (pad_size, pad_size), mode='reflect')
spec = torchaudio.functional.spectrogram(
					waveform=wav_padded,
					pad=0,#torchaudio.functional.spectrogram内で使われているtorch.nn.functional.padはmode='constant'となっているが、今回はmode='reflect'としたいため手動でpaddingする
					window=torch.hann_window(win_length),
					n_fft=filter_length,
					hop_length=hop_length,
					win_length=win_length,
					power=2,
					normalized=False,
					center=False
				)
spec = spec.to(device)
spec_lengths = torch.tensor([spec.size(2)], dtype=torch.long).to(device)
source_speaker_id = torch.tensor([source_speaker_id], dtype=torch.long).to(device)
target_speaker_id = torch.tensor([target_speaker_id], dtype=torch.long).to(device)
#推論(音声変換)を実行
output_wav = netG.voice_conversion(spec, spec_lengths, source_speaker_id=source_speaker_id, target_speaker_id=target_speaker_id)[0].data.cpu()
#結果と元音声を出力
torchaudio.save(os.path.join(output_dir, "output.wav"), output_wav, sample_rate=sampling_rate)
torchaudio.save(os.path.join(output_dir, "input.wav"), loaded_wav, sample_rate=sampling_rate)
