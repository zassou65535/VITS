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

def discriminator_adversarial_loss(discriminator_real_outputs, discriminator_fake_outputs):
	loss = 0
	real_losses = []
	fake_losses = []
	for dr, df in zip(discriminator_real_outputs, discriminator_fake_outputs):
		dr = dr.float()
		df = df.float()
		real_loss = torch.mean((1-dr)**2)
		fake_loss = torch.mean(df**2)
		loss += (real_loss + fake_loss)
		real_losses.append(real_loss.item())
		fake_losses.append(fake_loss.item())
	return loss, real_losses, fake_losses

def generator_adversarial_loss(discriminator_fake_outputs):
	loss = 0
	generator_losses = []
	for dg in discriminator_fake_outputs:
		dg = dg.float()
		l = torch.mean((1-dg)**2)
		generator_losses.append(l)
		loss += l
	return loss, generator_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
	z_p = z_p.float()
	logs_q = logs_q.float()
	m_p = m_p.float()
	logs_p = logs_p.float()
	z_mask = z_mask.float()

	kl = logs_p - logs_q - 0.5
	kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
	kl = torch.sum(kl * z_mask)
	l = kl / torch.sum(z_mask)
	return l

def feature_loss(feature_map_real, feature_map_fake):
	loss = 0
	for fmap_real, fmap_fake in zip(feature_map_real, feature_map_fake):
		for fmreal, fmfake in zip(fmap_real, fmap_fake):
			fmreal = fmreal.float().detach()
			fmfake = fmfake.float()
			loss += torch.mean(torch.abs(fmreal - fmfake))
	return loss * 2 
