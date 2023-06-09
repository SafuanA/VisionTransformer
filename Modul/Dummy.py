import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
from utils.features import Mel_Spectrogram, _MFCC, _Fbank

def CreateDummy():
    return Dummy()

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = Mel_Spectrogram()
        self.lin = nn.Linear(208, 10)
        self.loss = nn.MSELoss()

    def forward_loss(self, x, target):
         _, h, w = target.shape
         target = target[:,:, w-10:]
         return self.loss(x, target)
    
    def forward(self, batch):
        wav, _, spk_id_encoded, _ = batch
        mel = self.mel_spec(wav)
        x = mel
        x = self.lin(x)
        loss = self.forward_loss(x, mel)
        return loss, x, mel