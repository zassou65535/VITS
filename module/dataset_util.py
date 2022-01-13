#encoding:utf-8

import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')# AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt
import os

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

#wavファイル、話者id、テキスト(音素列)の3つを読み込むためのDatasetクラス
class AudioSpeakerTextLoader(torch.utils.data.Dataset):
	"""
		1) 前処理によって作成されたtxtファイルに書かれたwavファイル、話者id、テキスト(音素列)の3つを読み込む
		2) テキストを正規化し整数へと変換
		3) wavファイルからスペクトログラムを計算
	"""
	def __init__(self, dataset_txtfile_path, phoneme_list):
		#dataset_txtfile_path : 前処理によって作成されたtxtファイルへのパス
		#phoneme_list : 学習に用いる音素のlist
		self.sampling_rate = 22050
		self.filter_length = 1024
		self.hop_length = 256
		self.win_length = 1024
		self.phoneme_list = phoneme_list
		#音素とindexを対応付け　対応を前計算しておくことでバッチ作成時の処理を高速化する
		self.phoneme2index = {p : i for i, p in enumerate(self.phoneme_list, 0)}

		###前処理によって作成されたtxtファイルの読み込み###
		#一行につき
		#wavファイルへのパス|話者id|音素列
		#というフォーマットで記述されている
		with open(dataset_txtfile_path, "r") as f:
			self.wavfilepath_speakerid_text = [line.split("|") for line in f.readlines()]
		#各行をランダムにシャッフル
		random.seed(1234)
		random.shuffle(self.wavfilepath_speakerid_text)

	def get_audio_text_speaker_pair(self, audiopath_sid_text):
		# separate filename, speaker_id and text
		audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
		wav, spec = self.get_audio(audiopath)
		text = self.get_text(text)
		sid = self.get_sid(sid)
		return (wav, spec, text, sid)

	def get_audio(self, wavfile_path):
		#wavファイルの読み込み
		wav, _ = torchaudio.load(wavfile_path)
		wav = wav.squeeze(dim=0)
		#wavからspectrogramを計算
		#計算結果はファイルに保存しておき、2回目以降はそれを読み込むだけにする
		spec_filename = wavfile_path.replace(".wav", ".spec.pt")
		if os.path.exists(spec_filename):
			spec = torch.load(spec_filename)
		else:
			spec = torchaudio.functional.spectrogram(
									waveform=wav,
									pad=0,
									window=torch.hann_window(self.win_length),
									n_fft=self.filter_length,
									hop_length=self.hop_length,
									win_length=self.win_length,
									power=2,
									normalized=False
								)
			spec = torch.squeeze(spec, 0)
			torch.save(spec, spec_filename)
		return wav, spec

	def get_sid(self, sid):
		sid = torch.LongTensor([int(sid)])
		return sid
	
	def get_text(self, text):
		#Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
		text_splitted = text.replace("\n", "").split(",")
		text_converted_into_index = [self.phoneme2index[p] for p in text_splitted]#音素を数値に変換
		#各音素の間に0を挿入する
		text_norm = [0] * (len(text_converted_into_index) * 2 + 1)
		text_norm[1::2] = text_converted_into_index
		#tensorへと変換
		text_norm = torch.LongTensor(text_norm)
		return text_norm

	def __getitem__(self, index):
		line = self.wavfilepath_speakerid_text[index]
		wavfilepath, speakerid, text = line[0], line[1], line[2]
		wav, spec = self.get_audio(wavfilepath)
		speaker_id = self.get_sid(speakerid)
		text = self.get_text(text)
		return (wav, spec, speaker_id, text)

	def __len__(self):
		return len(self.wavfilepath_speakerid_text)

#AudioSpeakerTextLoaderの__getitem__により取得されたデータをバッチへと固める関数
def collate_fn(batch):
	# batch = [
	# 	(wav, spec, speaker_id, text),
	# 	(wav, spec, speaker_id, text),
	# 	....
	# ]
	max_wav_len = max([x[0].size(0) for x in batch])#wavの最大の長さを算出
	max_spec_len = max([x[1].size(1) for x in batch])#spectrogramの最大の長さを算出
	max_text_len = max([x[3].size(0) for x in batch])#textの最大の長さを算出

	wav_lengths = torch.LongTensor(len(batch))#torch.size([len(batch)])
	spec_lengths = torch.LongTensor(len(batch))
	speaker_id = torch.LongTensor(len(batch))
	text_lengths = torch.LongTensor(len(batch))

	wav_padded = torch.zeros(len(batch), max_wav_len, dtype=torch.float32)
	spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec_len, dtype=torch.float32)
	text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)

	#text_padded, spec_padded, wav_paddedは全ての要素が0で初期化されているが、
	#左詰めで元のtext, spec, wavで上書きすることによりzero-paddingされたtensorを取得できる
	for i, (wav_row, spec_row, speaker_id_row, text_row) in enumerate(batch, 0):
		wav_padded[i, :wav_row.size(0)] = wav_row
		wav_lengths[i] = wav_row.size(0)

		spec_padded[i, :, :spec_row.size(1)] = spec_row
		spec_lengths[i] = spec_row.size(1)

		speaker_id[i] = speaker_id_row

		text_padded[i, :text_row.size(0)] = text_row
		text_lengths[i] = text_row.size(0)

	return  wav_padded, wav_lengths, \
			spec_padded, spec_lengths, \
			speaker_id, \
			text_padded, text_lengths

#batch内の各tensorについて、start_indices[i]で指定されたindexから長さsegment_sizeの箇所を取り出す関数
#学習時、スペクトログラムや音声波形について、時間軸に沿って指定した長さだけ切り取るのに用いる
def slice_segments(input_tensor, start_indices, segment_size):
	output_tensor = torch.zeros_like(input_tensor[:, ..., :segment_size])
	batch_size = input_tensor.size(0)
	for batch_index in range(batch_size):
		index_start = start_indices[batch_index]
		index_end = index_start + segment_size
		output_tensor[batch_index] = input_tensor[batch_index, ..., index_start:index_end]
	return output_tensor
