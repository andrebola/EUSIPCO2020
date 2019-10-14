import os

DATA_PATH = '/data1/msd/msd'
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
META_PATH = os.path.join(DATA_PATH, 'metadata')

TMP_PATH = '/data1/msd/msd/tmp_data_repr'
MODELS_PATH = '/data1/msd/msd/models'

# SET GPUs to use:
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def batch_block_generator(train_set, id2tag, pos, samples=128, batch_size=32, bins=96):
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
                    x_batch = x_block[items_in_batch,:,pos:pos+128,:]
                    y_batch = y_block[items_in_batch]
                    yield (x_batch, y_batch)

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
    def __init__(self, test_data, test_classes, val_set, val_classes):
        self.test_data = test_data
        self.test_classes = test_classes
        self.val_data = val_set
        self.val_classes = val_classes

    def on_epoch_end(self, epoch, logs={}):
        #if (epoch+1) % 5 ==0:
        test_pred_prob = np.zeros(shape=(self.test_data.shape[0], int(1024/128), 50))
        for i in range(0,1024,128):
            test_pred_prob[:,int(i/128), :] = self.model.predict(self.test_data[:,:,i:i+128,:])
        test_pred_prob = np.mean(test_pred_prob, axis=1)
        roc_auc = 0
        pr_auc = 0
        for i in range(50):
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
    args = parser.parse_args()

    train_list, valid_list, test_list, id2tag = load_metadata()
    print ("Loaded Metadata")
    data_val, classes_val = load_spectrograms(valid_list, id2tag, args.bins)
    print ("Loaded Validation set, ", data_val.shape, classes_val.shape)
    data_test, classes_test = load_spectrograms(test_list, id2tag, args.bins)
    print ("Loaded Test set, ", data_test.shape, classes_test.shape)

    input_shape = data_val[0,:,:128,:].shape

    # the loss for a single label classification task is CATEGORICAL crossentropy
    # the loss for a MULTI label classification task is BINARY crossentropy
    loss = 'binary_crossentropy'

    # number of Filters in each layer
    nb_filters = 16
    #nb_filters = 8

    # how many neurons in each hidden layer
    dense_units = 500
    #dense_units = 200

    # how many output units
    # IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
    output_shape = 50

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

    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(MODELS_PATH, "model_repr_{}.h5".format(args.bins)), monitor='val_loss', save_best_only=True, mode='max')
    callbacks = [keras.callbacks.LearningRateScheduler(step_decay), model_checkpoint]
    import random
    epochs_iter = list(range(18))*40
    random.shuffle(epochs_iter)
    curr_epochs = 1
    # START TRAINING
    for i in epochs_iter:
        pos = i*64
        n_train = len(train_list)
        history = model.fit_generator(batch_block_generator(train_list, id2tag, pos, 128, batch_size, bins=args.bins),
                         #steps_per_epoch = int(n_train / batch_size),
                         steps_per_epoch = 6300,
                         validation_data=(data_val[:,:,pos:pos+128,:], classes_val),
                         epochs=past_epochs+1,
                         verbose=2,
                         initial_epoch=past_epochs,
                         callbacks=callbacks+[TestCallback(data_test, classes_test, data_val, classes_val)]
                         )
        past_epochs += 1



