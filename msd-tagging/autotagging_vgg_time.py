import os

#Parent folder where the input audio is located
DATA_PATH = '/path/msd/msd'
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
META_PATH = os.path.join(DATA_PATH, 'metadata')

#Folder where the spectrograms are located
TMP_PATH = '/path/msd/msd/tmp_data_repr'
MODELS_PATH = '/path/msd/msd/models'

# SET GPUs to use:
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0,1,2,3"

# General Imports
import pickle
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
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from audio_extract import analyze, analyze_misc, analyze_hp
from scipy.io import wavfile
from multiprocessing import Pool

missing = ['TRBGJIZ128F92E42BC',
  'TREGWSL128F92C9D42',
  'TRJKZDA12903CFBA43',
  'TRVWNOH128E0788B78',
  'TRCGBRK128F146A901']

def load_metadata():
    train_file = os.path.join(META_PATH,'filtered_list_train.cP')
    train_list = pickle.load(open(train_file,'rb'), encoding='bytes')
    valid_list = train_list[201680:]
    train_list = train_list[0:201680]

    test_file = os.path.join(META_PATH,'filtered_list_test.cP')
    test_list = pickle.load(open(test_file,'rb'), encoding='bytes')

    id2tag_file = os.path.join(META_PATH,'msd_id_to_tag_vector.cP')
    id2tag_list = pickle.load(open(id2tag_file,'rb'), encoding='bytes')

    return train_list, valid_list, test_list, id2tag_list

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

def load_spectrograms(msdid_block, id2tag, bins=96):
    y_block = []
    list_spectrograms = []
    for msdid in msdid_block:
        msdid = msdid.decode()
        filename = '{}/{}/{}/{}.mp3.npz'.format(msdid[0], msdid[1], msdid[2], msdid)
        npz_spec_file = os.path.join(TMP_PATH, filename)
        if os.path.exists(npz_spec_file):
            with np.load(npz_spec_file) as npz:
                try:
                    melspec = npz['mel{}'.format(bins)]
                    #melspec = npz['mel48']
                    list_spectrograms.append(melspec)
                    y_block.append(id2tag[msdid.encode()].flatten())
                except KeyError:
                    print ("ERROR", filename)
    item_list = np.array(list_spectrograms, dtype=K.floatx())
    item_list[np.isinf(item_list)] = 0
    item_list = add_channel(item_list)
    y_block= np.array(y_block, dtype='bool')
    return item_list, y_block


def batch_block_generator(train_set, id2tag, batch_size=32, bins=96, time_use=1):
    block_step = 50000
    n_train = len(train_set)
    randomize = True
    while 1:
        for i in range(0, n_train, block_step):
            npy_train_mtrx_x = os.path.join(MODELS_PATH, 'repr_{}bin_{}_x.npy'.format(bins,i))
            npy_train_mtrx_y = os.path.join(MODELS_PATH, 'repr_{}bin_{}_y.npy'.format(bins,i))
            if os.path.exists(npy_train_mtrx_x):
                x_block = np.load(npy_train_mtrx_x)
                y_block = np.load(npy_train_mtrx_y)
            else:
                msdid_block = train_set[i:min(n_train, i+block_step)]
                x_block, y_block = load_spectrograms(msdid_block, id2tag, bins)
                np.save(npy_train_mtrx_x, x_block)
                np.save(npy_train_mtrx_y, y_block)
            items_list = list(range(x_block.shape[0]))
            if randomize:
                random.shuffle(items_list)
            for j in range(0, len(items_list), batch_size):
                if j+batch_size <= x_block.shape[0]:
                    items_in_batch = items_list[j:j+batch_size]
                    x_batch = x_block[items_in_batch,:,::time_use,:]
                    y_batch = y_block[items_in_batch]
                    yield (x_batch, y_batch)

def CompactCNN(input_shape, nb_conv, nb_filters, n_mels, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout, multiple_segments=False, graph_model=False, input_tensor=None):

    melgram_input = Input(shape=input_shape)

    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 5), (4, 8), (4, 4), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (4, 5), (3, 8), (4, 4), (4, 3)] #(2, 8), (4, 3)]
    elif n_mels >= 48:
        poolings = [(2, 4), (4, 5), (3, 8), (2, 4), (4, 4)]
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
    def __init__(self, test_data, test_classes, val_set, val_classes):
        self.test_data = test_data
        self.test_classes = test_classes
        self.val_data = val_set
        self.val_classes = val_classes

    def on_epoch_end(self, epoch, logs={}):
        #if (epoch+1) % 5 ==0:
        test_pred_prob = self.model.predict(self.test_data)
        roc_auc = 0
        pr_auc = 0
        for i in range(50):
            roc_auc += roc_auc_score(self.test_classes[:, i], test_pred_prob[:, i])
            pr_auc += average_precision_score(self.test_classes[:, i], test_pred_prob[:, i])
        print('Test:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

        val_pred_prob = self.model.predict(self.val_data)
        roc_auc = 0
        pr_auc = 0
        for i in range(50):
            roc_auc += roc_auc_score(self.val_classes[:, i], val_pred_prob[:, i])
            pr_auc += average_precision_score(self.val_classes[:, i], val_pred_prob[:, i])
        print('Validation:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    return lrate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b',
                        '--bins',
                        dest="bins",
                        help='Number of bins',
                        default=False)
    parser.add_argument('-m',
                        '--meluse',
                        dest="meluse",
                        help='Use one bin every N melbands',
                        type=int,
                        default=1)

    args = parser.parse_args()

    train_list, valid_list, test_list, id2tag = load_metadata()
    print ("Loaded Metadata")
    data_val, classes_val = load_spectrograms(valid_list, id2tag, args.bins)
    print ("Loaded Validation set, ", data_val.shape, classes_val.shape)
    data_test, classes_test = load_spectrograms(test_list, id2tag, args.bins)
    print ("Loaded Test set, ", data_test.shape, classes_test.shape)

    input_shape = data_val[0,:,::args.meluse,:].shape

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

    # which activation function to use for OUTPUT layer
    # IN A MULTI-LABEL TASK with N classes we use SIGMOID activation same as with a BINARY task
    # as EACH of the classes can be 0 or 1
    output_activation = 'sigmoid'

    # which type of normalization
    normalization = 'batch'

    # droupout
    dropout = 0


    # the loss for a single label classification task is CATEGORICAL crossentropy
    # the loss for a MULTI label classification task is BINARY crossentropy
    loss = 'binary_crossentropy'

    # how many output units
    # IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
    output_shape = 50

    # Optimizers

    # simple case:
    # Stochastic Gradient Descent
    #optimizer = 'sgd'

    # advanced:
    sgd = optimizers.SGD(momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.01)#lr=0.001 decay = 0.03
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

    # We use mostly ADAM
    adam = optimizers.Adam(lr=0.001) #0.001
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)

    metrics = ['accuracy']

    # Optimizer
    optimizer = adam

    batch_size = 32

    epochs = 300

    random_seed = 0

    import tensorflow as tf
    model = CompactCNN(input_shape, nb_conv = nb_conv_layers, nb_filters= nb_filters, n_mels = input_shape[0],
                               normalize=normalization,
                               nb_hidden = nb_hidden, dense_units = dense_units,
                               output_shape = output_shape, activation = output_activation,
                               dropout = dropout)
    model.summary()

    # COMPILE MODEL
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
    past_epochs = 0

    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(MODELS_PATH, "model_vgg_time_repr_{}_{}.h5".format(args.bins, args.meluse)), monitor='val_loss', save_best_only=True, mode='max')
    callbacks = [model_checkpoint]
    callbacks = [keras.callbacks.LearningRateScheduler(step_decay), model_checkpoint]
    import random
    epochs_iter = list(range(18))*40
    random.shuffle(epochs_iter)
    curr_epochs = 1
    # START TRAINING
    history = model.fit_generator(batch_block_generator(train_list, id2tag, batch_size, bins=args.bins, time_use=args.meluse),
                         steps_per_epoch = 6300,
                         validation_data=(data_val[:,:,::args.meluse,:], classes_val),
                         epochs=epochs,
                         verbose=2,
                         initial_epoch=past_epochs,
                         callbacks=callbacks+[TestCallback(data_test[:,:,::args.meluse,:], classes_test, data_val[:,:,::args.meluse,:], classes_val)]
                         )



