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
from module.vits_generator import *
from module.vits_discriminator import *

#乱数のシードを設定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#前処理用スクリプトによって出力された、データセットに関するtxtファイルへのパス
train_dataset_txtfile_path = "./dataset/jvs_preprocessed/jvs_preprocessed_for_train.txt"#学習用
validation_dataset_txtfile_path = "./dataset/jvs_preprocessed/jvs_preprocessed_validation.txt"#推論用
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
#何イテレーションごとに学習結果を出力するか
output_iter = 5000
#学習に使用する音素を列挙
phoneme_list = [' ', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']

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
								num_workers=os.cpu_count(),
								shuffle=False, 
								pin_memory=True,
								#num_workerごとにシードを設定　これがないと各num_workerにおいて乱数が似たような値を返してしまう
    							worker_init_fn=lambda worker_id: torch.manual_seed(manualSeed + worker_id)
							)
print("validation dataset size: {}".format(len(validation_dataset)))

#Generatorのインスタンスを生成
netG = VitsGenerator(n_vocab=len(phoneme_list))
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

# #学習開始
# #学習過程を追うための変数　GeneratorとDiscriminatorが拮抗しているかどうかをグラフによって確認できるようにする
# adversarial_losses_netG_A2B = []
# adversarial_losses_netG_B2A = []
# adversarial_losses_netD_A = []
# adversarial_losses_netD_B = []
# #現在のイテレーション回数
# now_iteration = 0

print("Start Training")

#学習開始時刻を保存
t_epoch_start = time.time()

# #エポックごとのループ　itertools.count()でカウンターを伴う無限ループを実装可能
# for epoch in itertools.count():
# 	#ネットワークを学習モードにする
# 	netG_A2B.train(), netG_B2A.train()
# 	netD_A.train(), netD_B.train()
# 	#データセットA, Bからbatch_size枚ずつ取り出し学習
# 	for (real_A, real_B) in zip(dataloader_A, dataloader_B):
# 		#real_A.size() : torch.Size([batch_size, 128, 160])
# 		#real_B.size() : torch.Size([batch_size, 128, 160])
# 		#deviceに転送
# 		real_A = real_A.to(device)
# 		real_B = real_B.to(device)

# 		#-------------------------
#  		#Generatorの学習
# 		#-------------------------
# 		#Generator adversarial losses: hinge loss (from Scyclone paper eq.1)
# 		fake_B = netG_A2B(real_A)
# 		#fake_Bの中央128フレームを切り取ったものをDiscriminatorへの入力とする
# 		pred_fake_B = netD_B(torch.narrow(fake_B, dim=2, start=16, length=128))
# 		loss_adv_G_A2B = torch.mean(F.relu(-1.0 * pred_fake_B))
# 		fake_A = netG_B2A(real_B)
# 		pred_fake_A = netD_A(torch.narrow(fake_A, dim=2, start=16, length=128))
# 		loss_adv_G_B2A = torch.mean(F.relu(-1.0 * pred_fake_A))

# 		#cycle consistency losses: L1 loss (from Scyclone paper eq.1)
# 		cycled_A = netG_B2A(fake_B)
# 		loss_cycle_ABA = F.l1_loss(cycled_A, real_A)
# 		cycled_B = netG_A2B(fake_A)
# 		loss_cycle_BAB = F.l1_loss(cycled_B, real_B)

# 		#identity mapping losses: L1 loss (from Scyclone paper eq.1)
# 		same_B = netG_A2B(real_B)
# 		loss_identity_B = F.l1_loss(same_B, real_B)
# 		same_A = netG_B2A(real_A)
# 		loss_identity_A = F.l1_loss(same_A, real_A)

# 		#Total loss
# 		loss_G = (
# 			loss_adv_G_A2B
# 			+ loss_adv_G_B2A
# 			+ loss_cycle_ABA * weight_cycle_loss
# 			+ loss_cycle_BAB * weight_cycle_loss
# 			+ loss_identity_A * weight_identity_loss
# 			+ loss_identity_B * weight_identity_loss
# 		)
# 		#溜まった勾配をリセット
# 		optimizerG.zero_grad()
# 		#傾きを計算
# 		loss_G.backward()
# 		#gradient explosionを避けるため勾配を制限
# 		nn.utils.clip_grad_norm_(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), max_norm=1.0, norm_type=2.0)
# 		#Generatorのパラメーターを更新
# 		optimizerG.step()

# 		#stdoutへの出力用
# 		log_G = {
# 			"Loss/G_total": loss_G,
# 			"Loss/Adv/G_B2A": loss_adv_G_B2A,
# 			"Loss/Adv/G_A2B": loss_adv_G_A2B,
# 			"Loss/Cyc/A2B2A": loss_cycle_ABA * weight_cycle_loss,
# 			"Loss/Cyc/B2A2B": loss_cycle_BAB * weight_cycle_loss,
# 			"Loss/Id/A2A": loss_identity_A * weight_identity_loss,
# 			"Loss/Id/B2B": loss_identity_B * weight_identity_loss,
# 		}
# 		#グラフへの出力用
# 		adversarial_losses_netG_A2B.append(loss_adv_G_A2B.item())
# 		adversarial_losses_netG_B2A.append(loss_adv_G_B2A.item())

# 		#-------------------------
#  		#discriminatorの学習
# 		#-------------------------
# 		#Adversarial loss: hinge loss (from Scyclone paper eq.1)
# 		#netD_Aの学習
# 		##本物音声によるLoss
# 		#スペクトログラムの中央128フレームを切り取ったものをDiscriminatorへの入力とする
# 		pred_A_real = netD_A(torch.narrow(real_A, dim=2, start=16, length=128))
# 		loss_D_A_real = torch.mean(F.relu(0.5 - pred_A_real))
# 		##偽物音声によるLoss
# 		with torch.no_grad():
# 			fake_A = netG_B2A(real_B)
# 		pred_A_fake = netD_A(torch.narrow(fake_A, dim=2, start=16, length=128))
# 		loss_D_A_fake = torch.mean(F.relu(0.5 + pred_A_fake))
# 		##netD_A total loss
# 		loss_D_A = loss_D_A_real + loss_D_A_fake

# 		#netD_Bの学習
# 		##本物音声によるLoss
# 		pred_B_real = netD_B(torch.narrow(real_B, dim=2, start=16, length=128))
# 		loss_D_B_real = torch.mean(F.relu(0.5 - pred_B_real))
# 		##偽物音声によるLoss
# 		with torch.no_grad():
# 			fake_B = netG_A2B(real_A)
# 		pred_B_fake = netD_B(torch.narrow(fake_B, dim=2, start=16, length=128))
# 		loss_D_B_fake = torch.mean(F.relu(0.5 + pred_B_fake))
# 		##netD_B total loss
# 		loss_D_B = loss_D_B_real + loss_D_B_fake

# 		#Total
# 		loss_D = loss_D_A + loss_D_B

# 		#溜まった勾配をリセット
# 		optimizerD.zero_grad()
# 		#傾きを計算
# 		loss_D.backward()
# 		#gradient explosionを避けるため勾配を制限
# 		nn.utils.clip_grad_norm_(itertools.chain(netD_A.parameters(), netD_B.parameters()), max_norm=1.0, norm_type=2.0)
# 		#Generatorのパラメーターを更新
# 		optimizerD.step()

# 		#stdoutへの出力用
# 		log_D = {
# 			"Loss/D_total": loss_D,
# 			"Loss/D_A": loss_D_A,
# 			"Loss/D_B": loss_D_B,
# 		}
# 		#グラフへの出力用
# 		adversarial_losses_netD_A.append(loss_D_A.item())
# 		adversarial_losses_netD_B.append(loss_D_B.item())

# 		#学習状況をstdoutに出力
# 		if now_iteration % 10 == 0:
# 			print(f"[{now_iteration}/{total_iterations}]", end="")
# 			for key, value in log_G.items():
# 				print(f" {key}:{value:.5f}", end="")
# 			for key, value in log_D.items():
# 				print(f" {key}:{value:.5f}", end="")
# 			print("")

# 		#学習状況をファイルに出力
# 		if((now_iteration%output_iter==0) or (now_iteration+1>=total_iterations)):
# 			out_dir = os.path.join(output_dir, f"iteration{now_iteration}")
# 			#出力用ディレクトリがなければ作る
# 			os.makedirs(out_dir, exist_ok=True)

# 			#ここまでの学習にかかった時間を出力
# 			t_epoch_finish = time.time()
# 			total_time = t_epoch_finish - t_epoch_start
# 			with open(os.path.join(out_dir,"time.txt"), mode='w') as f:
# 				f.write("total_time: {:.4f} sec.\n".format(total_time))

# 			#学習済みモデル（CPU向け）を出力
# 			netG_A2B.eval()
# 			torch.save(netG_A2B.to('cpu').state_dict(), os.path.join(out_dir, "generator_A2B_trained_model_cpu.pth"))
# 			netG_A2B.to(device)
# 			netG_A2B.train()

# 			netG_B2A.eval()
# 			torch.save(netG_B2A.to('cpu').state_dict(), os.path.join(out_dir, "generator_B2A_trained_model_cpu.pth"))
# 			netG_B2A.to(device)
# 			netG_B2A.train()

# 			#lossのグラフを出力
# 			plt.clf()
# 			plt.figure(figsize=(10, 5))
# 			plt.title("Generator_A2B and Discriminator_B Adversarial Loss During Training")
# 			plt.plot(adversarial_losses_netG_A2B, label="netG_A2B")
# 			plt.plot(adversarial_losses_netD_B, label="netD_B")
# 			plt.xlabel("iterations")
# 			plt.ylabel("Loss")
# 			plt.legend()
# 			plt.grid()
# 			plt.savefig(os.path.join(out_dir, "loss_netG_A2B_netD_B.png"))
# 			plt.close()

# 			plt.clf()
# 			plt.figure(figsize=(10, 5))
# 			plt.title("Generator_B2A and Discriminator_A Adversarial Loss During Training")
# 			plt.plot(adversarial_losses_netG_B2A, label="netG_B2A")
# 			plt.plot(adversarial_losses_netD_A, label="netD_A")
# 			plt.xlabel("iterations")
# 			plt.ylabel("Loss")
# 			plt.legend()
# 			plt.grid()
# 			plt.savefig(os.path.join(out_dir, "loss_netG_B2A_netD_A.png"))
# 			plt.close()

# 		now_iteration += 1
# 		#イテレーション数が上限に達したらループを抜ける
# 		if(now_iteration>=total_iterations):
# 			break
# 	#イテレーション数が上限に達したらループを抜ける
# 	if(now_iteration>=total_iterations):
# 		break
