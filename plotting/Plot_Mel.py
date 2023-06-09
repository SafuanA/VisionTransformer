import torch

from utils.Helpers import load_audio
from utils.features import Mel_Spectrogram
import matplotlib.pyplot as plt

path_wav = "../../DATA/train_dir/train/wav/id10001/1zcIwhmdeo4/00001.wav"

mel = Mel_Spectrogram(shuffle_type='NO')
waveform = load_audio(path_wav, second=2.00)
waveform = torch.Tensor(waveform)
waveform = waveform[None, :]
waveform = waveform

spectrogram = mel(waveform, False)
spectrogram = spectrogram.detach().numpy()

plt.figure('Mel Spectrogram')
plt.title('Mel-Spektrogramm')
plt.xlabel('Zeit')
plt.ylabel('Frequenz')
plt.imshow(spectrogram[0], origin='lower')
plt.savefig("../plots/melspk.png", transparent=True)

fig = plt.figure(1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.99,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
plt.subplot(121)
plt.title('Pre-Emphasis=0.9')
mel = Mel_Spectrogram(shuffle_type='NO', coef=0.9)
spectrogram = mel(waveform, False)
spectrogram = spectrogram.detach().numpy()
plt.xlabel('Zeit')
plt.ylabel('Frequenz')
plt.imshow(spectrogram[0], origin='lower')

plt.subplot(122)
plt.title('Pre-Emphasis=1')
mel = Mel_Spectrogram(shuffle_type='NO', coef=1)
spectrogram = mel(waveform, False)
spectrogram = spectrogram.detach().numpy()
plt.xlabel('Zeit')
plt.ylabel('Frequenz')
plt.imshow(spectrogram[0], origin='lower')

plt.savefig("../plots/preemp.png", transparent=True)
