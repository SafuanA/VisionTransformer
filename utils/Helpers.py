import pandas as pd
from scipy.io import wavfile
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sb
#%matplotlib inline
import numpy as np
import random
import torch

def readCSV(train_csv_path, valid_csv_path, test_csv_path, train2_csv_path):
    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(valid_csv_path)

    if train2_csv_path is not None:
        df_train2 = pd.read_csv(train2_csv_path)
        df_train = pd.concat([df_train, df_train2])

    if test_csv_path is not None:
        df_test = pd.read_csv(test_csv_path)
        return df_train, df_valid, df_test

    return df_train, df_valid, None

def load_audio(filename, second=2.0, fixed_audio_start=False, start=0, stop=0):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if fixed_audio_start:
        length = int(sample_rate * 10) #get 10sec at max for unified batch sizes, optimal would be bigger but performance
    else:
        length = int(sample_rate * second)

    if second <= 0:
        return waveform.astype(np.float64).copy()

    if stop > start:
        waveform = waveform[start:stop]

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        #random start for segment length
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    return waveform.copy()

def load_audioFixedStart(filename, second=2.0, fixed_audio_start=False, start=0, stop=0):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if fixed_audio_start:
        length = int(sample_rate * 10) #get 10sec at max for unified batch sizes, optimal would be bigger but performance
    else:
        length = int(sample_rate * second)

    if second <= 0:
        return waveform.astype(np.float64).copy()

    if stop > start:
        waveform = waveform[start:stop]


    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        #no random start for debug purposes
        waveform =  waveform[0:length].astype(np.float64) #use this for us
    return waveform.copy()

def getTimeDimension(seconds):
    #window = 400, hop = 154 --> 2sec = 208 timedim, 0.4sec = 42, 1sec = 104, 2.5sec = 260
    #window = 25, hop = 10 --> 0.4sec = 641 (in t), 
    #window = 25, hop = 15 --> 0.4sec = 427 (in t),
    timeDim = 0
    if seconds == 2.5:
        timeDim = 260
    elif seconds == 2:
        timeDim = 208
    elif seconds == 1:
        timeDim = 104
    elif seconds == 0.4:
        timeDim = 42
    else:
        raise 'uknown time dimension'
    return timeDim

def shuffle(data:torch.Tensor):
    idx = torch.randperm(data.shape[-1])
    data = data[:,:,idx]
    #plot_features(data, title='fbfeatures', name='shuffleTest')
    return data

def shuffleGrouped(data:torch.Tensor, seconds, group_size = 8):
    N,H,W = data.shape
    #1,3,2,0  -> 5, 15, 10, 0
    timeDim = getTimeDimension(seconds) // group_size 
    idx = torch.randperm(timeDim)
    data = data.view(N, H, timeDim, -1)
    data = data[:,:,idx, :]
    data = data.view(N, H, W)
    #plot_features(data[0], title='fbfeatures', name='shuffleTest')
    return data

def randomcut(data:torch.Tensor, seconds):
    _,H,W = data.shape
    timeDim = getTimeDimension(seconds)
    start = int(torch.rand(1).item()*(W-timeDim))
    data = data[:, :,start:start+timeDim]
    #plot_features(data, title='fbfeatures', name='randomcutTest')
    return data

def build_tsne(embeddings, spkrid, spkr_encoder):
    return
    emb = []
    spkrids = []
    for i in range(len(embeddings)):
        emb.append(embeddings[i][0])
        emb.append(embeddings[i][1])
        spkrids.append(spkrid[i][0])
        spkrids.append(spkrid[i][1])
    emb = np.array(emb)
    spkrids = np.array(spkrids)
    emb = emb.squeeze()
    spkrids = spkrids.squeeze()
    unique_speakers, spkr_count_a = np.unique(spkrids, return_counts=True)
    spkr_count = spkr_count_a.size
    print('trial test speaker count', spkr_count)
    np.savetxt('embeddings.txt', emb, delimiter=',')
    np.savetxt('spkrids.txt', emb, delimiter=',')

    X = torch.Tensor(emb)
    Y = tsne(X, 2, 80, 30.0)
    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")
    colors = []
    for val in spkrids:
        i = np.where(unique_speakers == val)
        colors.append(i)
    colors = np.array(colors)
    palette = np.array(sb.color_palette("hls", spkr_count))
    palette = palette[colors.astype(np.int)].squeeze()
    plt.scatter(Y[:, 0], Y[:, 1], s=spkr_count, c=palette)
    plt.show()
    plt.savefig("tsne.png")
    return

    x = TSNE(perplexity=30).fit_transform(emb)
    colors = []
    for val in spkrids:
        i = np.where(unique_speakers == val)
        colors.append(i)
    colors = np.array(colors)

    palette = np.array(sb.color_palette("hls", spkr_count))  #Choosing color palette 
    palette = palette[colors.astype(np.int)].squeeze()
    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette)
    # Add the labels for each digit.
    sp = []
    colors = colors.squeeze()
    for i in range(len(spkrids)):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=9)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        sp.append(txt)
    plt.savefig("tsne.png")
    #return f, ax, txts