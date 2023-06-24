import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from speechbrain.lobes.features import Fbank, MFCC
from utils.Helpers import shuffle, randomcut, getTimeDimension, shuffleGrouped
import torchaudio
class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)

class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=154, n_mels=80, coef=0.97, requires_grad=False,
                transpose=False, seconds=2, shuffle_type=None,freqm=20, timem=20, mask_type=1):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop
        self.transpose = transpose
        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(torch.FloatTensor(window), requires_grad=False)
        self.seconds = seconds
        if shuffle_type == None:
            shuffle_type = 'OS'
        self.shuffle_type = shuffle_type.upper()
        #for pretraining random shuffling does not improve shuffle indexes by cache
        if self.shuffle_type == 'SSC':
            timeDim = getTimeDimension(seconds)
            fileName = 'ss_'+ str(timeDim) +'.pt'
            if os.path.exists(fileName):
                self.ids_shuffle = torch.load(fileName)
            else:
                noise = torch.rand(timeDim)  # noise in [0, 1]
                self.ids_shuffle = torch.argsort(noise, dim=0)
                torch.save(self.ids_shuffle, fileName)
        if self.shuffle_type == 'SUC':
            timeDim = 1039 #for 10seconds
            fileName = 'su_'+ str(timeDim) +'.pt'
            if os.path.exists(fileName):
                self.ids_shuffle = torch.load(fileName)
            else:
                noise = torch.rand(timeDim)  # noise in [0, 1]
                self.ids_shuffle = torch.argsort(noise, dim=0)
                torch.save(self.ids_shuffle, fileName)

        print('shuffle type:', shuffle_type)

        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            if mask_type == 2:
                freqm = 10
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)
        self.mask_type = mask_type
        self.mask_freqTime = mask_type > 0
        print('freq and time masking: ', self.mask_freqTime)

    def forward(self, x, train=False):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        x = self.instance_norm(x)

        if self.shuffle_type.upper() == 'SS': #total random shuffled
            x = randomcut(x, self.seconds, self.hop)
            x = shuffle(x)
        elif self.shuffle_type.upper() == 'SSG': #total shuffle by group
            x = randomcut(x, self.seconds)
            x = shuffleGrouped(x, self.seconds, self.hop)
        elif self.shuffle_type.upper() == 'SSC': #shuffle by ids saved in txt
            x = randomcut(x, self.seconds)
            x = x[:,:,self.ids_shuffle]
        elif self.shuffle_type.upper() == 'SU':
            x = shuffle(x)
            x = randomcut(x, self.seconds, self.hop)
        elif self.shuffle_type.upper() == 'SUC':
            x = x[:,:,self.ids_shuffle]
            x = randomcut(x, self.seconds, self.hop)
        else:
            x = randomcut(x, self.seconds, self.hop)

        if train and self.mask_freqTime:
            x = self.freqm(x)
            x = self.timem(x)
            if self.mask_type == 2:
                x = self.freqm(x)
                x = self.timem(x)
        if self.transpose:
            x = x.transpose(-2,-1)
        return x


class _MFCC(MFCC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class _Fbank(Fbank):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

