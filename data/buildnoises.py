import torch.nn as nn
import pandas as pd

import tqdm
import os
import argparse

def buildrirCSV(searchdir, savedir):
    noise_id = []
    noises = []

    for root, dirs, filenames in tqdm.tqdm(os.walk(searchdir, followlinks=False)):
        files = [f for f in filenames if f.endswith('.wav')]
        for name in tqdm.tqdm(files):
            noise_id.append(name)
            noises.append(root + '/' + name)

    csv_dict = {"noise_id": noise_id,
                "noise_path": noises
                }
    df = pd.DataFrame(data=csv_dict)
    try:
        df.to_csv('rir.csv')
        print(f'Saved data list file at {savedir}')
    except OSError as err:
        print(f'Ran in an error while saving rir.csv: {err}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--searchdir', help='dataset dir',
                        type=str, default="data")
    parser.add_argument('--savedir',
                        help='list save path', type=str, default="data_list")
    args = parser.parse_args()
    buildrirCSV(args.searchdir,args.savedir)
