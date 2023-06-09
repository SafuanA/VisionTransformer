from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from utils.Helpers import load_audio

path_wav = "../../DATA/train_dir/train/wav/id10001/1zcIwhmdeo4/00001.wav"

input_data = read(path_wav)
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:208])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Raw Waveform")
# display the plot
plt.show()

import numpy
import scipy.io.wavfile

sample_rate, signal = scipy.io.wavfile.read(path_wav)  # File assumed to be in the same directory
signal = signal[0:int(2 * sample_rate)]
pre_emphasis = 0
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

plt.plot(signal)
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)

plt.title("Vergleich Pre-Emphasis")
plt.xlabel('Anzahl Samples')
plt.ylabel('Frequenz')
plt.plot(emphasized_signal)
plt.legend(['ohne Pre-Emphasis', 'Pre-Emphasis = 0.97'])
plt.show()
