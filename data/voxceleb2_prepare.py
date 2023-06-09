import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from scipy.io import wavfile
from scipy import signal
import soundfile
import tqdm
import os
import glob
import argparse
import torchaudio

from voxceleb_prepare import _get_chunks

def buildVox2csv(searchdir, savedir):
    csv_dict = []
    spk_ids = []
    wavs = []

    from sys import platform
    if platform == "linux" or platform == "linux2":
        splitter = '/'
    elif platform == "win32":
        splitter = '\\'

    SAMPLERATE=16000
    my_sep = "--"
    seg_dur=3.0
    amp_th=5e-04

    for wav in tqdm.tqdm(glob.glob(searchdir+'/*/*/*.wav', recursive=True), dynamic_ncols=True):
        try:
            [spk_id, sess_id, utt_id] = wav.split(splitter)[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])
        if spk_id not in spk_ids:
            spk_ids.append(spk_id)
        signal, fs = torchaudio.load(wav)
        signal = signal.squeeze(0)
        audio_duration = signal.shape[0] / SAMPLERATE
        uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
        for chunk in uniq_chunks_list:
            s, e = chunk.split("_")[-2:]
            start_sample = int(float(s) * SAMPLERATE)
            end_sample = int(float(e) * SAMPLERATE)

            #  Avoid chunks with very small energy
            mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
            if mean_sig < amp_th:
                continue

            # Composition of the csv_line
            csv_line = [
                chunk,
                str(audio_duration),
                wav,
                start_sample,
                end_sample,
                spk_id,
            ]
            csv_dict.append(csv_line)

    df = pd.DataFrame(data=csv_dict)
    try:
        df.to_csv('train2.csv')
        print(f'Saved data list file at {savedir}')
    except OSError as err:
        print(f'Ran in an error while saving {savefile}: {err}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--searchdir', help='searchdir dir',
                        type=str, default=None)
    parser.add_argument('--savedir',
                        help='savedir', type=str, default=None)
    args = parser.parse_args()
    buildVox2csv(args.searchdir,args.savedir)