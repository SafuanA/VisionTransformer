import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.features import Mel_Spectrogram
from utils.Helpers import load_audio, readCSV
import pandas as pd


def get_mean_std(n_mel=128, sample_number=10000):
    print('Start...')

    df1, df2, df3 = readCSV('wincsv/train.csv', 'wincsv/valid.csv', 'wincsv/test.csv', None)
    frames = [df1, df2]
    df = pd.concat(frames)
    df = df.sample(sample_number)
    Mel = Mel_Spectrogram()
    all_mean = 0.
    all_std = 0.
    count = 0
    frames = []

    for index, row in df.iterrows():
        count += 1
        if count % 200 == 0:
            print(count)
        waveform = load_audio(row['wav'])
        waveform = torch.Tensor(waveform)
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
        #print(row['c1'], row['c2'])
        
    all_mean = all_mean / sample_number
    all_std = all_std / sample_number
    print('all mean', all_mean)
    print('all std', all_std)
    # frequencies = np.concatenate(frames, 0)
    frames = np.concatenate(frames, 1)
    frame_mean = np.mean(frames, 1)
    frame_std = np.std(frames, 1)
    print('frame mean', frame_mean)
    print('frame std', frame_std)
    print(frame_mean.shape, frames.shape)
    dict = {
        'all_mean':all_mean,'all_std':all_std, 
        'frame_mean':frame_mean, 'frame_std':frame_std}
    np.save('./mean_std_'+str(n_mel)+'.npy', dict)
    return {"done": True}

def test(n_mel=128):
    mean_std_file = np.load('./mean_std_'+str(n_mel)+'.npy',allow_pickle=True).item()
    frame_mean = torch.Tensor(mean_std_file['frame_mean'])
    frame_std = torch.Tensor(mean_std_file['frame_std'])
    all_mean = mean_std_file['all_mean']
    all_std = mean_std_file['all_std']
    print(all_mean, all_std)

if __name__ == '__main__':
    get_mean_std(n_mel=80) # for vit
    test(n_mel=80)

