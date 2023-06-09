import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(left=0.12,
                    bottom=0.1,
                    right=0.99,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
plt.subplot(211)
plt.title('Mel Scale')
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Frequenz (Mel)')
f = np.arange(0, 10000, 1)
x = 2595 * np.log10(1 + f / 1000)
plt.plot(x)

plt.subplot(212)
plt.title('Mel-Filterbank')
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Amplitude')
import librosa
import matplotlib.pyplot as plt

sr = 16000
mels = librosa.filters.mel(sr=sr, n_fft=512, n_mels=10, fmin=0, fmax=sr / 2)
mels /= np.max(mels, axis=-1)[:, None]
plt.plot(mels.T)
plt.plot(mels)
plt.show()
