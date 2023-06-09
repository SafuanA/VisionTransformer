import numpy as np
import torch

from utils.Helpers import load_audio
from utils.features import Mel_Spectrogram
import matplotlib.pyplot as plt

path_wav = "../../DATA/train_dir/train/wav/id10001/1zcIwhmdeo4/00001.wav"

mel = Mel_Spectrogram(shuffle_type='NO')
waveform = load_audio(path_wav, second=5.00)
waveform = torch.Tensor(waveform)
waveform = waveform[None, :]
waveform = waveform

spectrogram = mel(waveform, False)
spectrogram = spectrogram.detach().numpy()

plt.figure(figsize=(12, 6))

fig, axes = plt.subplot_mosaic(
    [["top", "top"],
     ["bottom l", "bottom r"]]
)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.99,
                    top=0.9,
                    wspace=0.8,
                    hspace=0.5)

axes["top"].set_title('Orginalaussage')
axes["top"].set_xlabel('Zeit')
axes["top"].set_ylabel('Frequenz')
axes["top"].imshow(spectrogram[0], origin='lower')

axes["bottom l"].set_title('OS')
axes["bottom l"].set_xlabel('Zeit')
axes["bottom l"].set_ylabel('Frequenz')
axes["bottom l"].imshow(spectrogram[0][0:80, 100:300], origin='lower')

ss = spectrogram[0][0:80, 100:300]
rng = np.random.default_rng()
rng.shuffle(ss, axis=1)
axes["bottom r"].set_title('SS')
axes["bottom r"].set_xlabel('Zeit')
axes["bottom r"].set_ylabel('Frequenz')
axes["bottom r"].imshow(ss, origin='lower')

plt.show()
