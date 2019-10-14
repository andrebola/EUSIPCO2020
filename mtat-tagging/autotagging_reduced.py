import os

#Parent folder where the input audio is located
DATA_PATH = '/path/mtat'
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
META_PATH = os.path.join(DATA_PATH, 'metadata')

MODELS_PATH = '/path/mtat/models'
#Folder where the spectrograms are located
TMP_PATH = '/path/mtat/tmp_data_mels_repr'

# SET GPUs to use:
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"

# General Imports

import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation
import random

# Deep Learning

import keras
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, l1
from keras import regularizers

# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from audio_extract import analyze
from scipy.io import wavfile
from multiprocessing import Pool
import tensorflow as tf

def load_metadata():
    csv_file = os.path.join(META_PATH,'annotations_final.csv')

    # we select the last column (-1) as the index column (= filename)
    metadata = pd.read_csv(csv_file, index_col=0, sep='\t')

    genres = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast',
          'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male',
          'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal',
          'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age',
          'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird',
          'country', 'metal', 'female voice', 'choral']

    return metadata


def load_spectrograms(filelist, melbands):
    list_spectrograms = [[],[],[]]
    list_classes=[[],[],[]]
    random.shuffle(filelist)
    for filename in filelist:
        folder = int(filename[filename.rfind("/") - 1:filename.rfind("/")], 16)
        npz_spec_file = os.path.join(TMP_PATH, filename+'.npz')
        if os.path.exists(npz_spec_file):
            with np.load(npz_spec_file) as npz:
                classes = npz['classes']

                spec_c = npz[melbands]
                if folder < 12:
                    list_spectrograms[0].append(spec_c)  # 0,1,2,3,4,5,6,7,8,9,a,b
                    list_classes[0].append(classes)
                elif folder < 13:
                    list_spectrograms[1].append(spec_c)  # 0,1,2,3,4,5,6,7,8,9,a,b
                    list_classes[1].append(classes)
                else:
                    list_spectrograms[2].append(spec_c)  # 0,1,2,3,4,5,6,7,8,9,a,b
                    list_classes[2].append(classes)

    data_train = np.array(list_spectrograms[0], dtype=K.floatx())
    classes_train = np.array(list_classes[0], dtype=int)
    data_val = np.array(list_spectrograms[1], dtype=K.floatx())
    classes_val = np.array(list_classes[1], dtype=int)
    data_test = np.array(list_spectrograms[2], dtype=K.floatx())
    classes_test = np.array(list_classes[2], dtype=int)

    # replace Inf values:
    # as in our preprocessing some files generated an Inf value in the log10 computation, we replace those by 0:
    data_train[np.isinf(data_train)] = 0
    data_val[np.isinf(data_val)] = 0
    data_test[np.isinf(data_test)] = 0
    print("Loaded features successfully: " + str(len(filelist)), "files, dimensions:", data_train.shape, data_val.shape, data_test.shape)
    return data_train, classes_train, data_val, classes_val, data_test, classes_test


def add_channel(data, n_channels=1):
    # n_channels: 1 for grey-scale, 3 for RGB, but usually already present in the data

    N, ydim, xdim = data.shape

    if keras.backend.image_data_format() == 'channels_last':  # TENSORFLOW
        # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
        data = data.reshape(N, ydim, xdim, n_channels)
    else: # THEANO
        # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
        data = data.reshape(N, n_channels, ydim, xdim)

    return data

def CompactCNN(input_shape, nb_conv, nb_filters, n_mels, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout, multiple_segments=False, graph_model=False, input_tensor=None):

    melgram_input = Input(shape=input_shape)

    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 5), (4, 8), (4, 7), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (4, 5), (3, 8), (4, 7), (4, 3)] #(2, 8), (4, 3)]
    elif n_mels >= 72:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 64:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 48:
        poolings = [(2, 4), (4, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 32:
        poolings = [(2, 4), (2, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 24:
        poolings = [(2, 4), (2, 4), (3, 8), (2, 8), (4, 4)]
    elif n_mels >= 16:
        poolings = [(2, 4), (2, 5), (2, 8), (2, 7), (4, 4)]
    elif n_mels >= 8:
        poolings = [(2, 4), (2, 4), (2, 8), (1, 8), (4, 4)]

    # Determine input axis
    if keras.backend.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    #x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)
    x = BatchNormalization(axis=channel_axis, name='bn_0_freq')(melgram_input)

    if normalize == 'batch':
        pass
        #x = BatchNormalization(name='bn_0_freq')(melgram_input)
        #x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)
    elif normalize in ('data_sample', 'time', 'freq', 'channel'):
        x = Normalization2D(normalize, name='nomalization')(melgram_input)
    elif normalize in ('no', 'False'):
        x = melgram_input

    # Conv block 1
    x = Convolution2D(nb_filters[0], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[0], name='pool1')(x)

    # Conv block 2
    x = Convolution2D(nb_filters[1], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[1], name='pool2')(x)

    # Conv block 3
    x = Convolution2D(nb_filters[2], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[2], name='pool3')(x)

    # Conv block 4
    if nb_conv > 3:
        x = Convolution2D(nb_filters[3], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn4')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=poolings[3], name='pool4')(x)

    # Conv block 5
    if nb_conv == 5:
        x = Convolution2D(nb_filters[4], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn5')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=poolings[4], name='pool5')(x)

    # Flatten the outout of the last Conv Layer
    x = Flatten()(x)

    if nb_hidden == 1:
        x = Dropout(dropout)(x)
        x = Dense(dense_units, activation='relu')(x)
    elif nb_hidden == 2:
        x = Dropout(dropout)(x)
        x = Dense(dense_units[0], activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(dense_units[1], activation='relu')(x)
    else:
        pass

    # Output Layer
    x = Dense(output_shape, activation=activation, name = 'output')(x)

    # Create model
    model = Model(melgram_input, x)

    return model


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data, test_classes, val_set, val_classes, train_set, train_classes, name):
        self.test_data = test_data
        self.test_classes = test_classes
        self.val_data = val_set
        self.val_classes = val_classes
        self.train_data = train_set
        self.train_classes = train_classes
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        #if (epoch+1) % 5 ==0:
        test_pred_prob = self.model.predict(self.test_data)
        roc_auc = 0
        pr_auc = 0
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.test_classes[:, i], test_pred_prob[:, i])
            pr_auc += average_precision_score(self.test_classes[:, i], test_pred_prob[:, i])
        print('Test:')
        print('Epoch: '+str(epoch)+', ROC-AUC '+str(roc_auc/50)+', PR-AUC '+str(pr_auc/50)+', model'+self.name)

        roc_auc = 0
        pr_auc = 0
        val_pred_prob = self.model.predict(self.val_data)
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.val_classes[:, i], val_pred_prob[:, i])
            pr_auc += average_precision_score(self.val_classes[:, i], val_pred_prob[:, i])
        print('Validation:')
        print('Epoch: '+str(epoch)+', ROC-AUC '+str(roc_auc/50)+', PR-AUC '+str(pr_auc/50)+', model'+self.name)

        roc_auc = 0
        pr_auc = 0
        train_pred_prob = self.model.predict(self.train_data)
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.train_classes[:, i], train_pred_prob[:, i])
            pr_auc += average_precision_score(self.train_classes[:, i], train_pred_prob[:, i])
        print('Train:')
        print('Epoch: '+str(epoch)+', ROC-AUC '+str(roc_auc/50)+', PR-AUC '+str(pr_auc/50)+', model'+self.name)

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

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

    # the loss for a single label classification task is CATEGORICAL crossentropy
    # the loss for a MULTI label classification task is BINARY crossentropy
    loss = 'binary_crossentropy'

    # number of Convolutional Layers
    nb_conv_layers = 4

    # number of Filters in each layer
    nb_filters = [128,384,768,2048]
    #nb_filters = [64,192,384,1024]

    # number of hidden layers at the end of the model
    nb_hidden = 0
    dense_units = 200

    # how many output units
    # IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
    output_shape = len(genres)

    # which activation function to use for OUTPUT layer
    # IN A MULTI-LABEL TASK with N classes we use SIGMOID activation same as with a BINARY task
    # as EACH of the classes can be 0 or 1
    output_activation = 'sigmoid'

    # which type of normalization
    normalization = 'batch'

    # droupout
    dropout = 0

    metrics = ['categorical_accuracy']

    batch_size = 64
    epochs = 13

    random_seed = 0

    embeddings_train = []
    embeddings_val = []
    embeddings_test = []

    models = []
    for melbands in ['mel128_log', 'mel96_log', 'mel48_log', 'mel32_log', 'mel24_log', 'mel16_log', 'mel8_log', 'mel128', 'mel96', 'mel48', 'mel32', 'mel24', 'mel16', 'mel8']:
        data_train, classes_train, data_val, classes_val, data_test, classes_test = load_spectrograms(filelist, melbands)

        data_train = add_channel(data_train, n_channels=1)
        data_val = add_channel(data_val, n_channels=1)
        data_test = add_channel(data_test, n_channels=1)

        input_shape = data_train.shape[1:]

        model = CompactCNN(input_shape, nb_conv = nb_conv_layers, nb_filters= nb_filters, n_mels = input_shape[0],
                               normalize=normalization,
                               nb_hidden = nb_hidden, dense_units = dense_units,
                               output_shape = output_shape, activation = output_activation,
                               dropout = dropout)
        model.summary()

        # Optimizer
        adam = optimizers.Adam(lr=0.001)
        optimizer = adam

        # COMPILE MODEL
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        # past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
        past_epochs = 0

        #monitor_metric = 'val_loss'
        #monitor_metric = 'val_categorical_accuracy'
        #early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=4)

        callbacks = []

        # START TRAINING
        history = model.fit(data_train, classes_train,
                     #validation_split=validation_split,
                     validation_data=(data_val, classes_val),
                     verbose=2,
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=callbacks+[TestCallback(data_test, classes_test, data_val, classes_val, data_train, classes_train, melbands)]
                     )

        model.save_weights(os.path.join(MODELS_PATH, "model_repr_"+melbands+".h5"))

    
