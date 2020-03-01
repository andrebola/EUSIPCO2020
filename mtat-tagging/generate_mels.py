import os
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd
import random

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from multiprocessing import Pool
from audio_extract import analyze_mel, cut_audio, melspectrogram

#Parent folder where the input audio is located
DATA_PATH = '/path/mtat'
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
META_PATH = os.path.join(DATA_PATH, 'metadata')

#Folder where the spectrograms will be saved
TMP_PATH = '/path/mtat/tmp_data_mels_16'


def load_metadata():
    csv_file = os.path.join(META_PATH,'annotations_final.csv')

    # we select the last column (-1) as the index column (= filename)
    metadata = pd.read_csv(csv_file, index_col=0, sep='\t')
    return metadata

def create_spectrograms(file_class_pair):
    sr = 16000 # 12000
    filename, classes = file_class_pair
    filepath = os.path.join(AUDIO_PATH, filename)
    if os.path.exists(filepath):
        npz_spec_file = os.path.join(TMP_PATH, filename+'.npz')
        if not os.path.exists(npz_spec_file):
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
                mel128_log=frames['mel_128_log1+10kx'].T,
                classes=classes
            )


def create_spectrograms_multithread(filelist, classes):
    current_track = 0
    pool = Pool(20)
    results = pool.map(create_spectrograms, zip(filelist, classes))
    print("done.")

if __name__ == "__main__":

    genres = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast',
          'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male',
          'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal',
          'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age',
          'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird',
          'country', 'metal', 'female voice', 'choral']


    metadata = load_metadata()
    classes = metadata[genres].values
    filelist = metadata['mp3_path'].values.flatten().tolist()

    create_spectrograms_multithread(filelist, classes)

