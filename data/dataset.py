import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from utils.augment import WavAugment
from utils.Helpers import load_audio,load_audioFixedStart

infoPrinted = 0
def printInfo(name, n_speakers, n_utterances, n_realutterances):
    global infoPrinted
    if infoPrinted < 2:
        print(name+" Dataset load {} speakers".format(n_speakers))
        print(name+" Dataset load {} utterance".format(n_utterances))
        print(name+" Dataset load {} total size of cutted utterances".format(n_realutterances))
        infoPrinted += 1

class TrainDataset(Dataset):
    def __init__(self, df, speaker_encoder, second=2.0, pairs=False, aug=False, top_n_rows=None, 
                 rir_csv_path=None, noise_csv_path=None, trial_path=None, dsName="Train", full_utterance=False, fixed_audio_start=False, **kwargs):
        self.second = second
        self.pairs = pairs
        self.top_n_rows = top_n_rows
        #---pandas dataframe
        #"ID","duration","wav","start","stop","spk_id"
        if full_utterance:
            df = df.drop_duplicates(subset=['wav'])
            df = df.assign(start=0)
            df = df.assign(stop=0)
        if not top_n_rows:
            df = shuffle(df) #remove shuffling for future work, for reproducability and shuffle with pytorch/pl
        df = df.reset_index(drop=True)
        self.df = df#.T.to_dict('list')

        self.speaker_encoder = speaker_encoder

        #wav readings
        self.fixed_audio_start = fixed_audio_start
        self.trial_path = trial_path
        if self.trial_path:
            df_trial = pd.read_csv(self.trial_path, sep=" ", header=None)
            self.trials = df_trial[0], df_trial[1].str.removesuffix(".wav"), df_trial[2].str.removesuffix(".wav") 
        #augmentation
        self.aug = aug
        if self.aug:
            self.wav_aug = WavAugment(noise_csv_path, rir_csv_path)
        #speechbrain breaks each utterance in subutterances
        self.real_length = len(set(df['ID'].values))
        printInfo(dsName, len(set(df['spk_id'].values)), len(set(df['wav'].values)), self.real_length)
        #self.getRow(0)
    def __getitem__(self, index):
        return self.getRow(index)
    
    def getRow(self, index):
        row = self.df.iloc[index]
        path = row['wav']
        start = row['start']
        stop = row['stop']
        waveform = load_audio(path, self.second, self.fixed_audio_start, start, stop) 
        if self.aug:
            waveform = self.wav_aug(waveform)
        label = row['spk_id']
        spk_id_encoded = self.speaker_encoder.get_speaker_label_encoded(label)
        return torch.FloatTensor(waveform), label, spk_id_encoded, path # return float tensor since double tensor is the default

    def getRowDebug(self, index):
        row = self.df.iloc[index]
        path = row['wav']
        start = row['start']
        stop = row['stop']
        waveform = load_audioFixedStart(path, self.second, self.fixed_audio_start, start, stop) 
        if self.aug:
            waveform = self.wav_aug(waveform)
        label = row['spk_id']
        spk_id_encoded = self.speaker_encoder.get_speaker_label_encoded(label)
        return torch.FloatTensor(waveform), label, spk_id_encoded, path # return float tensor since double tensor is the default

    def __len__(self):
        if self.top_n_rows:
            return self.top_n_rows
        return self.real_length

class ValidDataset(TrainDataset):
    def __init__(self,df_valid,  speaker_encoder, second=2.0, pairs=True, aug=False, top_n_rows=None, 
                 rir_csv_path=None,  noise_csv_path=None, full_utterance=False, fixed_audio_start=False, **kwargs):
        super().__init__(df_valid, speaker_encoder, second, pairs, aug, top_n_rows, 
                         rir_csv_path, noise_csv_path, dsName="Valid", full_utterance=full_utterance, fixed_audio_start=fixed_audio_start, **kwargs)

class TestDataset(TrainDataset):
    def __init__(self,df_test,  speaker_encoder, trial_path, second=2.0, pairs=True, aug=False, top_n_rows=None, fixed_audio_start=False,  **kwargs):
        super().__init__(df_test, speaker_encoder, second, pairs, aug, top_n_rows, 
                         trial_path=trial_path, dsName="Test", fixed_audio_start=fixed_audio_start, **kwargs)
        self.ids = self.df['ID']
        #self.__getitem__(1)


    def __getitem__(self, index):
        trial_res, tri1, tri2 = self.trials
        index1 = np.argmax(self.ids == tri1[index])
        index2 = np.argmax(self.ids == tri2[index])
        return trial_res[index], self.getRow(index1), self.getRow(index2)

        #row = self.df[index]
        #trial_res, tri1, tri2 = self.trials
        #tri1 = tri1[index]
        #tri2 = trix2[index]
        #index1 = np.argmax(tri1 == tri1[index])
        #index2 = np.argmax(tri2 == tri2[index])
        ##index1 = row[index1]
        ##index1 = row[index2]
        #return trial_res[index], self.getRow(index1), self.getRow(index2)

    def __len__(self):
        if self.top_n_rows:
            return self.top_n_rows
        return len(self.trials[0])