import os

DATA_PATH = '/data1/msd/msd'

TMP_PATH = '/nas/msd/tmp_data_16'
#TMP_PATH = '/data1/msd/msd/tmp_data_repr'

import json
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation
import random
# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from multiprocessing import Pool
from audio_extract import analyze_mel, cut_audio, melspectrogram


def create_spectrograms(filepath):
    sr = 16000
    #filepath = '/data1/msd/msd/audio/{}/{}/{}/{}.mp3'.format(filepath[2], filepath[3],filepath[4],filepath)
    try:
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            npz_path = os.path.join(TMP_PATH,'{}/{}/{}'.format(filename[0],filename[1],filename[2]))
            npz_spec_file = os.path.join(npz_path,'{}.npz'.format(filename))
            os.makedirs(npz_path, exist_ok=True)
            if not os.path.exists(npz_spec_file):
                print ("Processing file", filepath)
                audio = cut_audio(filepath, sampleRate=sr, segment_duration=29.1)
                frames = melspectrogram(audio, sampleRate=sr, frameSize=512, hopSize=256,
                         warpingFormula='slaneyMel', window='hann', normalize='unit_tri')

                np.savez(npz_spec_file,
                    mel8=frames['mel_8_db'].T,
                    mel16=frames['mel_16_db'].T,
                    mel24=frames['mel_24_db'].T,
                    mel32=frames['mel_32_db'].T,
                    mel48=frames['mel_48_db'].T,
                    mel96=frames['mel_96_db'].T,
                    mel128=frames['mel_128_db'].T,
                    mel8_log=frames['mel_8_log1+10kx'].T,
                    mel16_log=frames['mel_16_log1+10kx'].T,
                    mel24_log=frames['mel_24_log1+10kx'].T,
                    mel32_log=frames['mel_32_log1+10kx'].T,
                    mel48_log=frames['mel_48_log1+10kx'].T,
                    mel96_log=frames['mel_96_log1+10kx'].T,
                    mel128_log=frames['mel_128_log1+10kx'].T
                )
                """frames = analyze_mel(filepath, segment_duration=29, maxFrequency=8000, replaygain=False)

                np.savez(npz_spec_file,
                    mel8=frames['mel8'].T,
                    mel16=frames['mel16'].T,
                    mel24=frames['mel24'].T,
                    mel32=frames['mel32'].T,
                    mel48=frames['mel48'].T,
                    mel96=frames['mel96'].T,
                    mel128=frames['mel128'].T
                )"""
    except ValueError:
        print ("Failed processing file", filepath)
    except RuntimeError:
        print ("Failed processing file, RuntimeError, ", filepath)


def create_spectrograms_multithread(filelist):
    current_track = 0
    pool = Pool(20)
    results = pool.map(create_spectrograms, filelist)
    print("done.")

if __name__ == "__main__":
    audios = os.path.join(DATA_PATH, 'audio/**/*.mp3')
    filelist = glob.iglob(audios, recursive=True)
    #filelist = json.load(open('./missing.json'))
    create_spectrograms_multithread(filelist)

