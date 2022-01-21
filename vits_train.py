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

###以下は学習に必要なパラメーター###
#前処理用スクリプトによって出力された、データセットに関するtxtファイルへのパス
train_dataset_txtfile_path = "./dataset/jvs_preprocessed/jvs_preprocessed_for_train.txt"#学習用
validation_dataset_txtfile_path = "./dataset/jvs_preprocessed/jvs_preprocessed_for_validation.txt"#推論用
#結果を出力するためのディレクトリ
output_dir = "./output/vits/train/"
#使用するデバイス
device = "cuda:0"
#バッチサイズ
batch_size = 16
#イテレーション数
total_iterations = 800000
#学習率
lr = 0.0002
#学習率の減衰率　各epoch終了時に減衰する処理を入れる
lr_decay = 0.999875
#何イテレーションごとに学習結果を出力するか
output_iter = 5000
#学習に使用する音素を列挙
phoneme_list = [' ', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']
#音素の種類数
n_phoneme = len(phoneme_list)
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
#メルスペクトログラムの縦軸(周波数領域)の次元
melspec_freq_dim = 80

#出力用ディレクトリがなければ作る
os.makedirs(output_dir, exist_ok=True)

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

###データセットの読み込み、データセット作成###
#wavファイル、話者id、テキスト(音素列)の3つを読み込むためのDatasetクラス(学習用)
train_dataset = AudioSpeakerTextLoader(
								dataset_txtfile_path=train_dataset_txtfile_path,
								phoneme_list = phoneme_list
							)
train_loader = torch.utils.data.DataLoader(
								train_dataset, 
								batch_size=batch_size,
								collate_fn=collate_fn, 
								num_workers=os.cpu_count(),
								shuffle=False, 
								pin_memory=True,
								#num_workerごとにシードを設定　これがないと各num_workerにおいて乱数が似たような値を返してしまう
    							worker_init_fn=lambda worker_id: torch.manual_seed(manualSeed + worker_id)
							)
print("train dataset size: {}".format(len(train_dataset)))
#推論用
validation_dataset = AudioSpeakerTextLoader(
								dataset_txtfile_path=validation_dataset_txtfile_path,
								phoneme_list = phoneme_list
							)
validation_loader = torch.utils.data.DataLoader(
								validation_dataset, 
								batch_size=1,
								collate_fn=collate_fn, 
								num_workers=0,
								shuffle=False, 
								pin_memory=True,
								#num_workerごとにシードを設定　これがないと各num_workerにおいて乱数が似たような値を返してしまう
    							worker_init_fn=lambda worker_id: torch.manual_seed(manualSeed + worker_id)
							)
print("validation dataset size: {}".format(len(validation_dataset)))

#Generatorのインスタンスを生成
netG = VitsGenerator(n_vocab=n_phoneme, n_speakers=100)
#ネットワークをデバイスに移動
netG = netG.to(device)

#Discriminatorのインスタンスを生成
netD = VitsDiscriminator()
#ネットワークをデバイスに移動
netD = netD.to(device)

#optimizerをGeneratorとDiscriminatorに適用
beta1 = 0.8
beta2 = 0.99
optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=0.01)
optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=0.01)
#Schedulerによって学習中に徐々に学習率を減衰させる　ExponentialLRでは学習率を指数的に減衰させることができる
schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=lr_decay)
schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=lr_decay)

#学習開始
#lossを記録することで学習過程を追うための変数　学習が安定しているかをグラフから確認できるようにする
losses_recorded = {
	"adversarial_loss/D" : [],
	"adversarial_loss/G" : [],
	"duration_loss/G" : [],
	"mel_reconstruction_loss/G" : [],
	"kl_loss/G" : [],
	"feature_matching_loss/G" : []
}
#現在のイテレーション回数
now_iteration = 0

print("Start Training")

#学習開始時刻を保存
t_epoch_start = time.time()

#ネットワークを学習モードにする
netG.train()
netD.train()

#エポックごとのループ　itertools.count()でカウンターを伴う無限ループを実装可能
for epoch in itertools.count():
	#データセットA, Bからbatch_size枚ずつ取り出し学習
	for data in train_loader:
		#deviceに転送
		wav_real, wav_real_length = data[0].to(device), data[1].to(device)
		spec_real, spec_real_length = data[2].to(device), data[3].to(device)
		speaker_id = data[4].to(device)
		text, text_length = data[5].to(device), data[6].to(device)
		#wav_real.size(), wav_real_length.size() : torch.Size([batch_size, 音声のサンプル数]) torch.Size([batch_size])
		#spec_real.size(), spec_real_length.size() : torch.Size([batch_size, 513(周波数領域), STFT後のサンプル数]) torch.Size([batch_size])
		#speaker_id.size() : torch.Size([batch_size])
		#text.size(), text_length.size() : torch.Size([batch_size, 音素列の長さ]) torch.Size([batch_size])

		###Generatorによる生成###
		wav_fake, wav_fake_predicted_length, attn, id_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = netG(text, text_length, spec_real, spec_real_length, speaker_id)
		#wav_fake : torch.Size([64, 1, 8192])　生成された波形
		#wav_fake_predicted_length : torch.Size([64])
		#attn : torch.Size([64, 1, 400, 181])
		#ids_slice : torch.Size([64])
		#x_mask : torch.Size([64, 1, 181])
		#z_mask : torch.Size([64, 1, 400])

		#z : torch.Size([64, 192, 400])
		#z_p : torch.Size([64, 192, 400])
		#m_p : torch.Size([64, 192, 400])
		#logs_p : torch.Size([64, 192, 400])
		#m_q : torch.Size([64, 192, 400])
		#logs_q : torch.Size([64, 192, 400])

		#データセット中のスペクトログラムからメルスペクトログラムを計算
		fbanks = torchaudio.functional.melscale_fbanks(n_freqs=filter_length//2 + 1, f_min=0, f_max=sampling_rate//2, n_mels=melspec_freq_dim, sample_rate=sampling_rate).to(device)
		mel_spec_real = torch.matmul(spec_real.clone().transpose(-1, -2), fbanks).transpose(-1, -2)
		#batch内の各メルスペクトログラム(上で計算したもの)について、id_sliceで指定されたindexから時間軸に沿って(segment_size//hop_length)サンプル分取り出す
		mel_spec_real = slice_segments(input_tensor=mel_spec_real, start_indices=id_slice, segment_size=segment_size//hop_length)
		
		#Generatorによって生成された波形からメルスペクトログラムを計算
		pad_size = int((filter_length-hop_length)/2)
		wav_fake_padded = torch.nn.functional.pad(wav_fake, (pad_size, pad_size), mode='reflect')
		spec_fake = torchaudio.functional.spectrogram(
									waveform=wav_fake_padded,
									pad=0,#torchaudio.functional.spectrogram内で使われているtorch.nn.functional.padはmode='constant'となっているが、今回はmode='reflect'としたいため手動でpaddingする
									window=torch.hann_window(win_length).to(device),
									n_fft=filter_length,
									hop_length=hop_length,
									win_length=win_length,
									power=2,
									normalized=False,
									center=False
								).squeeze(1)
		mel_spec_fake = torch.matmul(spec_fake.clone().transpose(-1, -2), fbanks).transpose(-1, -2)
		
		#データセット中の波形「wav_real」について、batch内の各波形について、id_slice*hop_lengthで指定されたindexから時間軸に沿ってsegment_sizeサンプル分取り出す
		wav_real = slice_segments(input_tensor=wav_real, start_indices=id_slice*hop_length, segment_size=segment_size)

		#####Discriminatorの学習#####
		# wav_real.size() : torch.Size([64, 1, 8192])　本物波形
		# wav_fake.size() : torch.Size([64, 1, 8192])　生成された波形
		authenticity_real, _ = netD(wav_real)
		authenticity_fake, _ = netD(wav_fake.detach())

		#lossを計算
		adversarial_loss_D, _, _ = discriminator_adversarial_loss(authenticity_real, authenticity_fake)#adversarial loss

		#Discriminatorのlossの総計
		lossD = adversarial_loss_D

		#勾配をリセット
		optimizerD.zero_grad()
		#勾配を計算
		lossD.backward()
		#gradient explosionを避けるため勾配を制限
		nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0, norm_type=2.0)
		#パラメーターの更新
		optimizerD.step()

		#####Generatorの学習#####
		authenticity_real, d_feature_map_real = netD(wav_real)
		authenticity_fake, d_feature_map_fake = netD(wav_fake)

		#lossを計算
		duration_loss = torch.sum(wav_fake_predicted_length.float())#duration loss
		mel_reconstruction_loss = F.l1_loss(mel_spec_real, mel_spec_fake)*45#reconstruction loss
		kl_loss = kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask)#KL divergence
		feature_matching_loss = feature_loss(d_feature_map_real, d_feature_map_fake)#feature matching loss(Discriminatorの中間層の出力分布の統計量を, realとfakeの場合それぞれにおいて互いの分布間で近づける)
		adversarial_loss_G, _ = generator_adversarial_loss(authenticity_fake)#adversarial loss

		#Generatorのlossの総計
		lossG = duration_loss + mel_reconstruction_loss + kl_loss + feature_matching_loss + adversarial_loss_G

		sys.exit()

		#勾配をリセット
		optimizerG.zero_grad()
		#勾配を計算
		lossG.backward()
		#gradient explosionを避けるため勾配を制限
		nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0, norm_type=2.0)
		#パラメーターの更新
		optimizerG.step()

		#####stdoutへlossを出力する#####
		loss_stdout = {
			"adversarial_loss/D" : adversarial_loss_D.item(),
			"adversarial_loss/G" : adversarial_loss_G.item(),
			"duration_loss/G" : duration_loss.item(),
			"mel_reconstruction_loss/G" : mel_reconstruction_loss.item(),
			"kl_loss/G" : kl_loss.item(),
			"feature_matching_loss/G" : feature_matching_loss.item()
		}
		if now_iteration % 10 == 0:
			print(f"[{now_iteration}/{total_iterations}]", end="")
			for key, value in loss_stdout.items():
				print(f" {key}:{value:.5f}", end="")
			print("")
		#lossを記録
		for key, value in loss_stdout.items():
			losses_recorded[key].append(value)

		#####学習状況をファイルに出力#####
		if((now_iteration%output_iter==0) or (now_iteration+1>=total_iterations)):
			out_dir = os.path.join(output_dir, f"iteration{now_iteration}")
			#出力用ディレクトリがなければ作る
			os.makedirs(out_dir, exist_ok=True)

			#ここまでの学習にかかった時間を出力
			t_epoch_finish = time.time()
			total_time = t_epoch_finish - t_epoch_start
			with open(os.path.join(out_dir,"time.txt"), mode='w') as f:
				f.write("total_time: {:.4f} sec.\n".format(total_time))

			#####学習済みモデル（CPU向け）を出力#####
			#generatorを出力
			netG.eval()
			torch.save(netG.to('cpu').state_dict(), os.path.join(out_dir, "netG_cpu.pth"))
			netG.to(device)
			netG.train()
			#discriminatorを出力
			netD.eval()
			torch.save(netD.to('cpu').state_dict(), os.path.join(out_dir, "netD_cpu.pth"))
			netD.to(device)
			netD.train()

			#####lossのグラフを出力#####
			plt.clf()
			plt.figure(figsize=(16, 6))
			plt.subplots_adjust(wspace=0.4, hspace=0.6)
			for i, (loss_name, loss_list) in enumerate(losses_recorded.items(), 0):
				plt.subplot(2, 3, i+1)
				plt.title(loss_name)
				plt.plot(loss_list, label="loss")
				plt.xlabel("iterations")
				plt.ylabel("loss")
				plt.legend()
				plt.grid()
			plt.savefig(os.path.join(out_dir, "loss.png"))
			plt.close()

		now_iteration += 1
		#イテレーション数が上限に達したらループを抜ける
		if(now_iteration>=total_iterations):
			break
	#学習率を更新
	schedulerG.step()
	schedulerD.step()
	#イテレーション数が上限に達したらループを抜ける
	if(now_iteration>=total_iterations):
		break
