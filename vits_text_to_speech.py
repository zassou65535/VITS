#encoding:utf-8

import random
import numpy as np
import glob
import os
import itertools
import time
import sys
import re
import pyopenjtalk

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
trained_weight_path = "./output/vits/train/iteration1999999/netG_cpu.pth"
#音声合成の対象とするテキスト
source_text = "これはテスト音声です"
#対象とする話者id
target_speaker_id = 9
#結果を出力するためのディレクトリ
output_dir = "./output/vits/inference/text_to_speech/"
#使用するデバイス
device = "cuda:0"
#扱う音声のサンプリングレート
sampling_rate = 22050

#学習に使用した音素を列挙
phoneme_list = [' ', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']
#音素とindexを対応付け
phoneme2index = {p : i for i, p in enumerate(phoneme_list, 0)}
#学習に使用した音素の種類数
n_phoneme = len(phoneme_list)
#学習に使用した話者の数
n_speakers = 100

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

##########音声合成の対象とするテキストを音素列に変換、前処理を施す##########
source_text = source_text.strip()#改行コードを削除
source_text = re.sub('・|・|「|」|』', '', source_text)#発音とは無関係な記号を削除
source_text = re.split('、|,|，|。|『', source_text)#句読点、もしくは『で分割
#分割した各文字列について音素列への変換を実行
source_phoneme = [pyopenjtalk.g2p(element) for element in source_text if(not element=="")]

#分割した各文字列についてスペースをカンマに変換
source_phoneme = [element.replace(" ",",") for element in source_phoneme]
#各発話(音素列)をスペース区切りで接合
source_phoneme = ', ,'.join(source_phoneme)
#文字列にpauが含まれている(解釈に失敗した記号)が含まれていれば処理を飛ばす　
if("pau" in source_phoneme):
	print(f"\"pau\" is included:{source_phoneme}")
	sys.exit()

#音素を数値に変換
source_phoneme = source_phoneme.replace("\n", "").split(",")
source_phoneme_converted_into_index = [phoneme2index[p] for p in source_phoneme]
#各音素の間に0を挿入する
text_norm = [0] * (len(source_phoneme_converted_into_index) * 2 + 1)
text_norm[1::2] = source_phoneme_converted_into_index

#音素をtensorへと変換
source_phoneme = torch.LongTensor(text_norm).to(device)
source_phoneme_lengths = torch.tensor([source_phoneme.size()[-1]], dtype=torch.long).to(device)
#対象とする話者idを数値に変換
target_speaker_id = torch.tensor([target_speaker_id], dtype=torch.long).to(device)
#Text to Speechの推論を実行
output_wav = netG.text_to_speech(text_padded=source_phoneme.unsqueeze(0), text_lengths=source_phoneme_lengths, speaker_id=target_speaker_id)[0].data.cpu()
#結果を出力
torchaudio.save(os.path.join(output_dir, "output.wav"), output_wav, sample_rate=sampling_rate)
