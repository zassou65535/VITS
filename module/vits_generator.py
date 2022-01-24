#encoding:utf-8

import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
from torch.autograd import Function
import torch.nn.functional as F

#学習用モデルを構成するための各部品
from .model_component import monotonic_align
from .model_component.decoder import Decoder
from .model_component.flow import Flow
from .model_component.posterior_encoder import PosteriorEncoder
from .model_component.stochastic_duration_predictor import StochasticDurationPredictor
from .model_component.text_encoder import TextEncoder

def slice_segments(x, ids_str, segment_size):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths, segment_size):
    b, d, t = x.size()
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device
    
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path

#モデルの学習を行うためのクラス
class VitsGenerator(nn.Module):
  def __init__(self, n_phoneme, n_speakers):
    super().__init__()
    self.n_phoneme = n_phoneme#入力する音素の種類数
    self.phoneme_embedding_dim = 192#各音素の埋め込み先のベクトルの大きさ
    self.spec_channels = 513#入力する線形スペクトログラムの縦軸(周波数)の次元
    self.z_channels = 192#PosteriorEncoderから出力されるzのchannel数
    self.text_encoders_dropout_during_train = 0.1#学習時のtext_encoderのdropoutの割合
    self.segment_size = 32#decoderによる音声の生成時、潜在変数zから何要素切り出してdecodeするか
    self.n_speakers = n_speakers#話者の種類数
    self.speaker_id_embedding_dim = 256#話者idの埋め込み先のベクトルの大きさ

    #transformerに似た構造のモジュールを用い、音素の列をencodeするネットワーク
    self.text_encoder = TextEncoder(
                      n_phoneme=self.n_phoneme,#音素の種類数
                      phoneme_embedding_dim=self.phoneme_embedding_dim,#各音素の埋め込み先のベクトルの大きさ
                      out_channels=self.z_channels,#出力するmとlogsのchannel数
                      p_dropout=self.text_encoders_dropout_during_train#学習時のdropoutの割合
                    )

    #話者埋め込み用ネットワーク
    self.speaker_embedding = nn.Embedding(
                      num_embeddings=self.n_speakers,#話者の種類数
                      embedding_dim=self.speaker_id_embedding_dim#話者idの埋め込み先のベクトルの大きさ
                    )

    #linear spectrogramと埋め込み済み話者idを入力にとりEncodeを実行、zを出力するモデル
    self.posterior_encoder = PosteriorEncoder(
                      speaker_id_embedding_dim=self.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      in_spec_channels = self.spec_channels,#入力する線形スペクトログラムの縦軸(周波数)の次元
                      out_z_channels = self.z_channels,#PosteriorEncoderから出力されるzのchannel数
                      phoneme_embedding_dim = self.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                    )

    #z, speaker_id_embeddedを入力にとり音声を生成するネットワーク
    self.decoder = Decoder(
                      speaker_id_embedding_dim=self.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      in_z_channel=self.z_channels#入力するzのchannel数
                    )
    
    #Flowとは入出力が可逆なネットワーク
    #zと埋め込み済み話者idを入力にとり、Monotonic Alignment Searchで用いる変数z_pを出力するネットワーク　逆も可能
    #音声変換時は、話者間の変換を実行する役割を果たす
    self.flow = Flow(
                      speaker_id_embedding_dim=self.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      in_z_channels=self.z_channels,#入力するzのchannel数
                      phoneme_embedding_dim=self.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                    )

    #Text-to-Speechの推論時にはAlignmentを自前で作る必要があるため、StochasticDurationPredictorを用いて音素列の情報から音素継続長を予測する必要がある。
    #音声変換の推論時には用いない
    self.stochastic_duration_predictor = StochasticDurationPredictor(
                      speaker_id_embedding_dim=self.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      phoneme_embedding_dim=self.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                      filter_channels=192,
                      kernel_size=3,
                      p_dropout=0.5,
                      n_flows=4
                    )
                    
  def forward(self, text_padded, text_lengths, spec_padded, spec_lengths, speaker_id):
    #text(音素)の内容をTextEncoderに通す
    text_encoded, m_p, logs_p, text_mask = self.text_encoder(text_padded, text_lengths)

    #話者idを埋め込み
    speaker_id_embedded = self.speaker_embedding(speaker_id).unsqueeze(-1)
    #linear spectrogramと埋め込み済み話者idを入力にとりEncodeを実行、zを出力する
    z, m_q, logs_q, spec_mask = self.posterior_encoder(spec_padded, spec_lengths, speaker_id_embedded)
    #zと埋め込み済み話者idを入力にとり、Monotonic Alignment Searchで用いる変数z_pを出力する
    z_p = self.flow(z, spec_mask, speaker_id_embedded=speaker_id_embedded)

    #Monotonic Alignment Search(MAS)の実行　音素の情報と音声の情報を関連付ける役割を果たす
    #MASによって、KL Divergenceを最小にするようなalignmentを求める
    with torch.no_grad():
        #DPで用いる、各ノードのKL Divergenceを前計算しておく
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
        neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
        #不要なノードにマスクをかけた上でDPを実行
        MAS_node_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(spec_mask, -1)
        MAS_path = monotonic_align.maximum_path(neg_cent, MAS_node_mask.squeeze(1)).unsqueeze(1).detach()

    #text(音素)の各要素ごとに、音素長を計算(各音素長は整数)
    duration_of_each_phoneme = MAS_path.sum(2)
    #StochasticDurationPredictorを、音素列の情報から音素継続長を予測できるよう学習させる
    predicted_duration_of_each_phoneme = self.stochastic_duration_predictor(text_encoded, text_mask, duration_of_each_phoneme, speaker_id_embedded=speaker_id_embedded)
    predicted_duration_of_each_phoneme = predicted_duration_of_each_phoneme / torch.sum(text_mask)

    m_p = torch.matmul(MAS_path.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    #z.size() : torch.Size([64, 192, 400])
    #spec_lengths.size() : torch.Size([64])
    #self.segment_size = 32
    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)

    #z_slice.size() : torch.Size([64, 192, 32])
    wav_fake = self.decoder(z_slice, speaker_id_embedded=speaker_id_embedded)
    #wav_fake.size() : torch.Size([64, 1, 8192])

    return wav_fake, predicted_duration_of_each_phoneme, MAS_path, ids_slice, text_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def text_to_speech(self, text_padded, text_lengths, speaker_id, noise_scale=.667, length_scale=1, noise_scale_w=0.8, max_len=None):
    text_encoded, m_p, logs_p, text_mask = self.text_encoder(text_padded, text_lengths)
    speaker_id_embedded = self.speaker_embedding(speaker_id).unsqueeze(-1) #話者埋め込み用ネットワーク

    logw = self.stochastic_duration_predictor(text_encoded, text_mask, speaker_id_embedded=speaker_id_embedded, reverse=True, noise_scale=noise_scale_w)

    w = torch.exp(logw) * text_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    spec_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(text_mask.dtype)
    MAS_node_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(spec_mask, -1)
    MAS_path = generate_path(w_ceil, MAS_node_mask)

    m_p = torch.matmul(MAS_path.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, spec_mask, speaker_id_embedded=speaker_id_embedded, reverse=True)
    o = self.decoder((z * spec_mask)[:,:,:max_len], speaker_id_embedded=speaker_id_embedded)
    return o

  def voice_conversion(self, spec_padded, spec_lengths, source_speaker_id, target_speaker_id):
    assert self.n_speakers > 0
    emb_source = self.speaker_embedding(source_speaker_id).unsqueeze(-1) #話者埋め込み用ネットワーク
    emb_target = self.speaker_embedding(target_speaker_id).unsqueeze(-1) #話者埋め込み用ネットワーク
    z, m_q, logs_q, spec_mask = self.posterior_encoder(spec_padded, spec_lengths, speaker_id_embedded=emb_source)
    z_p = self.flow(z, spec_mask, speaker_id_embedded=emb_source)
    z_hat = self.flow(z_p, spec_mask, speaker_id_embedded=emb_target, reverse=True)
    wav_fake = self.decoder(z_hat * spec_mask, speaker_id_embedded=emb_target)
    return wav_fake
