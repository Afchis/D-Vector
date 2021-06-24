import math
import numpy as np

import torch
import torch.nn as nn


class SincConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=80, kernel_size=251, sample_rate=16000):
        super(SincConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        # set minimum cutoff and bandwidth:
        self.min_freq = 50.0
        self.min_band = 50.0
        # calculate the band params:
        f_cos = self._get_mel_points(sample_rate, out_channels)
        b1, b2 = self._get_bands(f_cos, sample_rate)
        # learnable params:
        filt_b1 = torch.from_numpy(b1 / self.sample_rate).float()
        filt_band = torch.from_numpy((b2 - b1) / self.sample_rate).float()
        self.filt_b1 = nn.Parameter(filt_b1)
        self.filt_band = nn.Parameter(filt_band)
        # define the window function
        self.hamming_window = torch.from_numpy(np.hamming(kernel_size)).float()
        # define the linspace for the sinc function here
        self.t_right = torch.linspace(1, (kernel_size - 1) / 2, steps=int((kernel_size - 1) / 2)) / sample_rate

    def _get_mel_points(self, fs, n_filt, fmin=80):
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
        mel_points = np.linspace(fmin, high_freq_mel, n_filt)  # equally spaced in mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))
        return f_cos

    def _get_bands(self, f_cos, fs):
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100
        return b1, b2

    def _sinc(self, band, t_right):
        t_right = t_right.to(band.device)
        n_filt = band.size(0)
        band = band[:, None]  # (n_filt, 1)
        t_right = t_right[None, :]  # (1, K)
        y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)  # (n_filt, K)
        y_left = torch.flip(y_right, [1])
        y = torch.cat([y_left, torch.ones([n_filt, 1], device=band.device), y_right], dim=1)  # (n_filt, filt_dim)
        return y

    def forward(self, x):
        filt_beg_freq = torch.abs(self.filt_b1) + self.min_freq / self.sample_rate
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + self.min_band / self.sample_rate)
        low_pass1 = 2 * filt_beg_freq[:, None] * self._sinc(filt_beg_freq * self.sample_rate, self.t_right)
        low_pass2 = 2 * filt_end_freq[:, None] * self._sinc(filt_end_freq * self.sample_rate, self.t_right)
        band_pass = (low_pass2 - low_pass1)
        max_band, _ = torch.max(band_pass, dim=1, keepdim=True)
        c = band_pass / max_band  # (n_filt, filt_dim)
        filters = band_pass * self.hamming_window[None, ].to(band_pass.device)  # (n_filt, filt_dim)
        padding = (self.kernel_size - 1) // 2
        out = torch.nn.functional.conv1d(x, filters.view(self.out_channels, 1, self.kernel_size), padding=padding)
        return out


class SpeechEmbedder(nn.Module):
    def __init__(self, in_channels=40):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(in_channels,
                            768,
                            num_layers=3,
                            batch_first=True)
        self.proj = nn.Linear(768, 256)
        self.loss_w = nn.Parameter(torch.tensor([10.]))
        self.loss_b = nn.Parameter(torch.tensor([-5.]))
        self.loss_w.requires_grad = True
        self.loss_b.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def _neg_centroids(self, dvec):
        return torch.mean(dvec, dim=1, keepdim=True)

    def _pos_centroid(self, dvec, sp_idx, utt_idx):
        pos_cent = torch.cat([dvec[sp_idx, :utt_idx], dvec[sp_idx, utt_idx+1:]], dim=0)
        return torch.mean(pos_cent, dim=0, keepdim=True)

    def _sim_matrix(self, dvec):
        neg_centroids = self._neg_centroids(dvec)
        '''
        dvec.shape          --> [speaker_idx, utterance_idx, emb_dim]
        neg_cintroids.shape --> [speaker_idx, 1, emb_dim]
        pos_centroid.shape --> [1, emb_dim]
        '''
        pos_sim = list()
        neg_sim = list()
        for sp_idx in range(dvec.size(0)):
            pos_sim_speaker = list()
            neg_sim_speaker = list()
            for utt_idx in range(dvec.size(1)):
                # pos sim:
                pos_centroid = self._pos_centroid(dvec, sp_idx, utt_idx)
                pos_sim_utt = self.loss_w * F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), pos_centroid, dim=1, eps=1e-6) + self.loss_b # [1]
                pos_sim_speaker.append(pos_sim_utt)
                # neg sim:
                neg_sim_utt = self.loss_w * F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), torch.cat([neg_centroids[:sp_idx], neg_centroids[sp_idx+1:]], dim=0).squeeze(), dim=1) + self.loss_b # [speaker_idx-1]
                neg_sim_speaker.append(neg_sim_utt)
            pos_sim_speaker = torch.stack(pos_sim_speaker, dim=0)
            pos_sim.append(pos_sim_speaker)
            neg_sim_speaker = torch.stack(neg_sim_speaker, dim=0)
            neg_sim.append(neg_sim_speaker)
        pos_sim = torch.stack(pos_sim, dim=0) # [speaker_idx, utterance_idx, 1]
        neg_sim = torch.stack(neg_sim, dim=0) # [speaker_idx, utterance_idx, speaker_idx-1]
        return pos_sim, neg_sim
 
    def _contrast_loss(self, pos_sim, neg_sim):
        loss =  1 - self.sigmoid(pos_sim.squeeze()) + self.sigmoid(neg_sim).max(2)[0]
        return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item()

    def _softmax_loss(self, pos_sim, neg_sim):
        loss = - pos_sim.squeeze() + torch.log(torch.exp(neg_sim).sum(dim=2))
        return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item() #, torch.log(torch.exp(neg_sim).sum(dim=2)).mean().item()

    def _ge2e_loss(self, dvec):
        dvec = dvec.reshape(self.speaker_num, self.utterance_num, -1)
        torch.clamp(self.loss_w, 1e-6)
        pos_sim, neg_sim = self._sim_matrix(dvec)
        loss, pos_sim, neg_sim = self._softmax_loss(pos_sim, neg_sim)
        return loss, pos_sim, neg_sim

    def forward(self, mel):
        # mel.shape --> (speaker_idx, utterance_idx, b, num_mels, T)
        self.speaker_num, self.utterance_num = mel.size(0), mel.size(1)
        mel = mel.reshape(self.speaker_num*self.utterance_num, mel.size(3), mel.size(4))
        # (b, c, num_mels, T) b --> [speaker_idx*utterance_idx] time = 94
        # (b, num_mels, T)   c == 1
        mel = mel.permute(0, 2, 1) # (b, T, num_mels)
        dvec, _ = self.lstm(mel) # (b, T, lstm_hidden)
        if self.train:
            dvec = dvec[:, -1, :]
            dvec = self.proj(dvec) # (b, emb_dim)
            dvec = dvec / dvec.norm(p=2, dim=1, keepdim=True)
            loss, pos_sim, neg_sim = self._ge2e_loss(dvec)
            return loss, pos_sim, neg_sim
        else:
            dvec = dvec.sum(1) / dvec.size(1)
            dvec = self.proj(dvec) # (b, emb_dim)
            dvec = dvec / dvec.norm(p=2, dim=1, keepdim=True)
            return dvec