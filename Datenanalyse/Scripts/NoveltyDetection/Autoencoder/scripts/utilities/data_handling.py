# read data
# in major parts based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
# licensed under Apache 2.0 by The TensorFlow Authors. All Rights Reserved.
# modified class Dataset for use without images, dropped some arguments

# in Anlehnung an https://github.com/healthDataScience/deep-learning-HAR  und
# https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

# Copyright 2018 Thomas Hemmert-Pottmann
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np  # misc
import pandas as pd
import NoveltyDetection.Autoencoder.scripts.framework as frwk
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_file(name, path, file_format='csv'):
    """ read from comma-separated file or numpy file at given path
    :param name specify the file's name (include file extension)
    :param path specify the path to the file (include '/' at the end)
    :param file_format the format of the file which is loaded, e.g. 'csv', 'npy'
    :type name str
    :type path str
    :type file_format str
    :returns numpy array containing the data
    :rtype ndarray
    """
    if file_format == 'npy':
        try:
            data = np.load(path + name + '.npy')
        except:
            data = pd.read_csv(path + name + '.csv', engine='c', header=None).values
    elif file_format == 'csv':
        # data = np.genfromtxt(path + name + '.csv', delimiter=',')
        # improved csv import (6x faster than np.gentext)
        data = pd.read_csv(path + name + '.csv', engine='c', header=None).values
    else:
        raise ValueError('Specified file format not available, check spelling')
    return data


def read_infotxt(datafolderpath):
    """read the Info.txt file written by Matlab
    :param datafolderpath
    :type datafolderpath str
    :returns the sequence length, number of sensors or channels, list of the sensors
    """
    path = datafolderpath + 'Info.txt'
    with open(path) as f:
        content = f.readlines()
    datapoints = int(content[2].split()[-1])
    sensors = content[4].split()
    n_sensors = len(sensors)
    return datapoints, n_sensors, sensors

def select_sensors(data, avail_sensors, sel_sensors, seq_length):

    n_sensors = len(sel_sensors)

    # preallocate for faster calculation
    reduced_data = np.empty([data.shape[0], seq_length*n_sensors])

    for cnt, sensor in enumerate(sel_sensors):
        if sensor not in sel_sensors: raise ValueError('Sensors not in data found')
        idx = avail_sensors.index(sensor)
        reduced_data[:, cnt*seq_length:(cnt+1)*seq_length] = data[:, idx*seq_length:(idx+1)*seq_length]
    return reduced_data, n_sensors


def load_data(config, data_path, balance_data=True):

    x = read_file('dataset', data_path, file_format='npy')
    y = read_file('labels', data_path, file_format='npy')
    try:
        idx = read_file('obsID', data_path, file_format='npy')
    except:
        idx = np.arange(x.shape[0])
    idx = idx.reshape(-1, 1)

    # sensor selection (config.sel_sensors)
    if config.sel_sensors != config.avail_sensors:
        x, config.n_sensors = select_sensors(
            x, config.avail_sensors, config.sel_sensors, config.orig_seq_lngth)

    # sort provided data and set up well-balanced datasets
    x, y, idx = frwk.select_data(x, y, config, idx=idx, balance_data=balance_data)

    if config.mode != 'ev':
        if 'testsets' in data_path:
            test_size = config.nSamplesTestSize
        else:
            test_size = config.test_size
        # split neg data into train and test data
        # x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        #     x, y, idx, test_size=test_size,
        #     shuffle=True, random_state=config.random)

        x_inlier = x[y==0]
        y_inlier = y[y==0]
        idx_inlier = idx[y==0]

        x_outlier = x[y != 0]
        y_outlier = y[y != 0]
        idx_outlier = idx[y != 0]

        x_trainInlier, x_testInlier, y_trainInlier, y_testInlier, idx_trainInlier, idx_testInlier = \
            train_test_split(x_inlier, y_inlier, idx_inlier, test_size=test_size, shuffle=True, random_state=config.random)

        x_trainOutlier, x_testOutlier, y_trainOutlier, y_testOutlier, idx_trainOutlier, idx_testOutlier = \
            train_test_split(x_outlier, y_outlier, idx_outlier, test_size=test_size, shuffle=True,
                             random_state=config.random)

        if config.nSamplesTrainingInlier > 0:
            _, x_trainInlier, _, y_trainInlier, _, idx_trainInlier = \
                train_test_split(x_trainInlier, y_trainInlier, idx_trainInlier, test_size=config.nSamplesTrainingInlier, shuffle=True,
                                 random_state=config.random)
            _, x_trainOutlier, _, y_trainOutlier, _, idx_trainOutlier = \
                train_test_split(x_trainOutlier, y_trainOutlier, idx_trainOutlier, test_size=config.nSamplesTrainingInlier,
                                 shuffle=True,
                                 random_state=config.random)
        x_train = x_trainInlier
        y_train = y_trainInlier
        idx_train = idx_trainInlier
        # add outlier to training data, because they are needed in validation data
        x_train = np.append(x_train, x_trainOutlier, axis=0)
        y_train = np.append(y_train, y_trainOutlier, axis=0)
        idx_train = np.append(idx_train, idx_trainOutlier, axis=0)


        print('Size x_trainInlier: %d' % x_trainInlier.shape[0])
        print('Size x_trainOutlier: %d' % x_trainOutlier.shape[0])
        print('Size x_testInlier: %d' % x_testInlier.shape[0])
        print('Size x_testOutlier: %d' % x_testOutlier.shape[0])

        x_test = np.append(x_testInlier, x_testOutlier, axis=0)
        y_test = np.append(y_testInlier, y_testOutlier, axis=0)
        idx_test = np.append(idx_testInlier, idx_testOutlier, axis=0)

        x_train, y_train, idx_train = shuffle(x_train, y_train, idx_train, random_state=config.random)
        x_test, y_test, idx_test = shuffle(x_test, y_test, idx_test, random_state=config.random)

    else:
        x_train = x_test = y_train = y_test = idx_train = idx_test = []

    return x, x_train, x_test, y, y_train, y_test, config, idx_train, idx_test, idx
