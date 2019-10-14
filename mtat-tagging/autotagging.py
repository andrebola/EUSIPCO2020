import os

#Parent folder where the input audio is located
DATA_PATH = '/path/mtat'
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
META_PATH = os.path.join(DATA_PATH, 'metadata')

NPZ_FILE = '/path/mtat/processed/mel_spectrogram_segments_96x1024.npz'
#Folder where the spectrograms are located
TMP_PATH = '/path/mtat/tmp_data_mels_repr'

# SET GPUs to use:
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0,1,2,3"

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

def create_spectrograms(file_class_pair):
    filename, classes = file_class_pair
    filepath = os.path.join(AUDIO_PATH, filename)
    if os.path.exists(filepath):
        npz_spec_file = os.path.join(TMP_PATH, filename+'.npz')
        if not os.path.exists(npz_spec_file):
            frames = analyze(filepath, segment_duration=29)
            misc = analyze_misc(filepath, segment_duration=29)
            hp = analyze_hp(filepath, segment_duration=29)

            np.savez(npz_spec_file, hpcp=frames['hpcp'].T,
                melspec=frames['melbands'].T,
                melbands_harmonic=hp['melbands_harmonic'].T,
                melbands_percussive=hp['melbands_percussive'].T,
                ebu_momentary=misc['ebu_momentary'][:, None].T,
                rms=misc['rms'][:, None].T,
                rms_spectrum=misc['rms_spectrum'][:, None].T,
                hfc=misc['hfc'][:, None].T,
                spectral_centroid=misc['spectral_centroid'][:, None].T,
                zcr=misc['zcr'][:, None].T,
                classes=classes
            )


def create_spectrograms_multithread(filelist, classes):
    current_track = 0
    pool = Pool(20)
    results = pool.map(create_spectrograms, zip(filelist, classes))
    print("done.")


def load_spectrograms(filelist):
    list_spectrograms = [[],[],[]]
    list_classes=[[],[],[]]
    random.shuffle(filelist)
    for filename in filelist:
        folder = int(filename[filename.rfind("/") - 1:filename.rfind("/")], 16)
        npz_spec_file = os.path.join(TMP_PATH, filename+'.npz')
        if os.path.exists(npz_spec_file):
            with np.load(npz_spec_file) as npz:
                classes = npz['classes']
                melspec = npz['mel96']
                """
                melspec = npz['melspec']
                hpcp = npz['hpcp']
                spec_c = np.concatenate((hpcp, melspec))
                """
                spec_c = melspec
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

from keras.initializers import VarianceScaling

def JordiNet(input_shape, nb_filters, output_shape, activation, dropout, y_input=96, num_units = 200):

    melgram_input = Input(shape=input_shape)

    # Front-end

    input_pad_7 = keras.layers.Lambda(lambda _x: tf.pad(_x, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(melgram_input)
    input_pad_3 = keras.layers.Lambda(lambda _x: tf.pad(_x, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT"))(melgram_input)

    # [TIMBRE]
    all_nb_filters = [nb_filters, nb_filters*2, nb_filters*4, nb_filters, nb_filters*2, nb_filters*4]
    filter_shapes = [((7, int(0.9 * y_input)), input_pad_7), ((3, int(0.9 * y_input)), input_pad_3),
                     ((1, int(0.9 * y_input)), melgram_input), ((7, int(0.4 * y_input)), input_pad_7),
                     ((3, int(0.4 * y_input)), input_pad_3), ((1, int(0.4 * y_input)), melgram_input),
                    ]

    outputs = []
    for i, (nb_filter, (filter_shape, input_x)) in enumerate(zip(all_nb_filters, filter_shapes)):
        x = Convolution2D(nb_filter*2, filter_shape, padding='valid',
                          activation='relu', kernel_initializer=VarianceScaling())(input_x)
        # channel_axis is 3
        x = BatchNormalization(axis=3, name='bn%d'%i)(x)
        x = MaxPooling2D(pool_size=[1, x._keras_shape[2]], strides=[1, x._keras_shape[2]], name='pool%d'%i)(x)
        #x = keras.backend.squeeze(x,2)
        x = keras.layers.Lambda(lambda _x: keras.backend.squeeze(_x, 2))(x)
        outputs.append(x)

    # [TEMPORAL-FEATURES]
    kernels = [165, 128, 64, 32]

    for i, kernel in enumerate(kernels):
        avg_pool = keras.layers.AveragePooling2D(pool_size=[1, y_input],strides=[1, y_input], name='avg_pool%d'%i)(melgram_input)
        #avg_pool = keras.backend.squeeze(avg_pool, 3)
        avg_pool = keras.layers.Lambda(lambda _x: keras.backend.squeeze(_x, 3))(avg_pool)
        avg_pool = keras.layers.Convolution1D(nb_filters, kernel, padding='same',
                                              activation='relu', kernel_initializer=VarianceScaling())(avg_pool)
        avg_pool = BatchNormalization(name='bn_t%d'%i)(avg_pool)
        outputs.append(avg_pool)

    # concatenate all feature maps
    pool = keras.layers.concatenate(outputs, 2)
    pool = keras.layers.Lambda(lambda _x: keras.backend.expand_dims(_x, 3))(pool)

    # Back-end
    x = Convolution2D(512, (7,int(pool.shape[2])), padding='valid',
                      activation='relu', kernel_initializer=VarianceScaling())(pool)
    x = BatchNormalization(name='1cnnOut')(x)
    x = keras.layers.Lambda(lambda _x: keras.backend.permute_dimensions(_x, [0, 1, 3, 2]))(x)
    bn_conv1_pad = keras.layers.Lambda(lambda _x: tf.pad(_x, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(x)

    y = Convolution2D(512, (7,int(bn_conv1_pad.shape[2])), padding='valid',
                      activation='relu', kernel_initializer=VarianceScaling())(bn_conv1_pad)
    y = keras.layers.Lambda(lambda _x: keras.backend.permute_dimensions(_x, [0, 1, 3, 2]))(y)
    y = BatchNormalization(name='2cnnOut')(y)

    z = keras.layers.add([x, y])

    # Temporal pooling
    z = MaxPooling2D(pool_size=[2, 1], strides=[2, 1], name='poolOut')(z)
    bn_conv4_pad = keras.layers.Lambda(lambda _x: tf.pad(_x, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(z)

    x = Convolution2D(512, (7, int(bn_conv4_pad.shape[2])), padding='valid',
                      activation='relu', kernel_initializer=VarianceScaling())(bn_conv4_pad)
    x = keras.layers.Lambda(lambda _x: keras.backend.permute_dimensions(_x, [0, 1, 3, 2]))(x)
    x = BatchNormalization(name='3cnnOut')(x)

    z = keras.layers.add([x, z])
    z_max = keras.layers.Lambda(lambda _x: K.max(_x, axis=2))(z)
    z_mean = keras.layers.Lambda(lambda _x: K.mean(_x, axis=2))(z)
    pool2 = keras.layers.concatenate([z_mean, z_max])

    x = Flatten()(pool2)
    x = Dropout(dropout)(x)
    x = Dense(num_units, activation='relu', kernel_initializer=VarianceScaling())(x)
    x = BatchNormalization(name='b')(x)
    x = Dropout(dropout)(x)

    # Output Layer
    x = Dense(output_shape, activation=activation, name='output', kernel_initializer=VarianceScaling())(x)

    # Create model
    model = Model(inputs=melgram_input, outputs=x)

    return model

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data, test_classes, val_set, val_classes, train_set, train_classes):
        self.test_data = test_data
        self.test_classes = test_classes
        self.val_data = val_set
        self.val_classes = val_classes
        self.train_data = train_set
        self.train_classes = train_classes

    def on_epoch_end(self, epoch, logs={}):
        #if (epoch+1) % 5 ==0:
        test_pred_prob = np.zeros(shape=(self.test_data.shape[0], int(1024/128), 50))
        for i in range(0,1024,128):
            test_pred_prob[:,int(i/128), :] = self.model.predict(self.test_data[:,:,i:i+128,:])
        test_pred_prob = np.mean(test_pred_prob, axis=1)
        roc_auc = 0
        pr_auc = 0
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.test_classes[:, i], test_pred_prob[:, i])
            pr_auc += average_precision_score(self.test_classes[:, i], test_pred_prob[:, i])
        print('Test:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

        val_pred_prob = np.zeros(shape=(self.val_data.shape[0], int(1024/128), 50))
        for i in range(0,1024,128):
            val_pred_prob[:,int(i/128),:] = self.model.predict(self.val_data[:,:,i:i+128,:])
        val_pred_prob = np.mean(val_pred_prob, axis=1)
        roc_auc = 0
        pr_auc = 0
        #val_pred_prob = self.model.predict(self.val_data)
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.val_classes[:, i], val_pred_prob[:, i])
            pr_auc += average_precision_score(self.val_classes[:, i], val_pred_prob[:, i])
        print('Validation:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

        roc_auc = 0
        pr_auc = 0
        i = random.randint(0, (1024-128))
        train_pred_prob = self.model.predict(self.train_data[:,:,i:i+128,:])
        for i, label in enumerate(genres):
            roc_auc += roc_auc_score(self.train_classes[:, i], train_pred_prob[:, i])
            pr_auc += average_precision_score(self.train_classes[:, i], train_pred_prob[:, i])
        print('Train:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    return lrate

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

    # didn't created the audio spectrograms before, we can create and save them
    create_features = False

    # if we saved the audio spectrograms before, we try to load them
    load_features_in_matrix = True

    # if not, we store audio features for faster reload the next time
    save_full_matrix = False


    if create_features:
        create_spectrograms_multithread(filelist, classes)
    if load_features_in_matrix:
        data_train, classes_train, data_val, classes_val, data_test, classes_test = load_spectrograms(filelist)
        if save_full_matrix:
            np.savez(NPZ_FILE, data_train=data_train, classes_train=classes_train, data_val=data_val, classes_val=classes_val, data_test=data_test, classes_test=classes_test, filenames=filelist)
            print("Features stored to " + NPZ_FILE)
    else:
        if os.path.exists(NPZ_FILE):
            with np.load(NPZ_FILE) as npz:
                data_train = npz['data_train']
                data_val = npz['data_val']
                data_test = npz['data_test']
                filelist = npz['filenames']
                classes_train = npz['classes_train']
                classes_val = npz['classes_val']
                classes_test = npz['classes_test']
            print("Loaded features successfully: " + str(len(filelist)), "files, dimensions:", data_train.shape, data_val.shape, data_test.shape)
        else:
            load_features = False

    data_train = add_channel(data_train, n_channels=1)
    data_val = add_channel(data_val, n_channels=1)
    data_test = add_channel(data_test, n_channels=1)

    input_shape = data_train[0,:,:128,:].shape

    # the loss for a single label classification task is CATEGORICAL crossentropy
    # the loss for a MULTI label classification task is BINARY crossentropy
    loss = 'binary_crossentropy'

    # number of Filters in each layer
    #nb_filters = 16
    nb_filters = 32

    # how many neurons in each hidden layer
    dense_units = 500
    #dense_units = 200

    # how many output units
    # IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
    output_shape = len(genres)

    # which activation function to use for OUTPUT layer
    # IN A MULTI-LABEL TASK with N classes we use SIGMOID activation same as with a BINARY task
    # as EACH of the classes can be 0 or 1

    output_activation = 'sigmoid'

    # droupout
    dropout = 0.5

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

    validation_split=0.1

    random_seed = 0

    import tensorflow as tf

    model = JordiNet(input_shape, nb_filters= nb_filters, output_shape = output_shape,
                 activation = output_activation, dropout = dropout, y_input = 96, num_units=dense_units)

    model.summary()

    # COMPILE MODEL
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
    past_epochs = 0

    callbacks = [keras.callbacks.LearningRateScheduler(step_decay)]
    import random
    epochs_iter = list(range(18))*40
    random.shuffle(epochs_iter)
    curr_epochs = 1
    # START TRAINING
    for i in epochs_iter:
        pos = i*64
        history = model.fit(data_train[:,:,pos:pos+128,:], classes_train,
                         validation_data=(data_val[:,:,pos:pos+128,:], classes_val),
                         epochs=past_epochs+1,
                         verbose=2,
                         initial_epoch=past_epochs,
                         batch_size=batch_size,
                         callbacks=callbacks+[TestCallback(data_test, classes_test, data_val, classes_val, data_train, classes_train)]
                         )
        past_epochs += 1




