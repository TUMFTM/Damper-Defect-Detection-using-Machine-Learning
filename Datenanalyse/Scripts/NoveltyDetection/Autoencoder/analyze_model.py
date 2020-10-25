#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: framework.py -----------------------------------------------------------------------------
# Framework for data handling, model training and evaluation.
# -----------------------------------------------------------------------------
# This program was developed as a part of a Semester Thesis at the
# Technical University of Munich (Germany).
# It was later refined by Thomas Zehelein
#
# Programmer: Trumpp, Raphael F. and Thomas Zehelein
# -----------------------------------------------------------------------------
# Copyright 2019 Raphael Frederik Trumpp and Thomas Zehelein
# -----------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import sys, os
import numpy as np
from time import time
from joblib import Parallel, delayed, parallel_backend

from scipy import interp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn import metrics

import matplotlib.pyplot as plt
import NoveltyDetection.Autoencoder.scripts.inits as inits
import NoveltyDetection.Autoencoder.scripts.utilities.data_handling as datahandler
from tensorflow.keras.models import load_model

# custom packages
# import scripts.utilities.plot as cplt


def plot_vae_weights(model, config):
    plt.figure()
    for i in range(0, 7):
        tmp = np.mean(model.model_.trainable_weights[2],axis=0)
        test = np.abs(model.model_.trainable_weights[0][i * 32:(i + 1) * 32, :])
        plt.plot(np.linspace(0, 50, 32),
                 np.mean(np.abs(model.model_.trainable_weights[0][i * 32:(i + 1) * 32, :]), axis=1),
                 label=config.sel_sensors[i])
    plt.show()
    plt.legend()

if __name__ == '__main__':

    config = inits.init_parser()

    print('[INFO] EVALUATE MODEL ON DATA')

    # Load fitted/trained model and pipe
    config.model = datahandler.read_file('/model', config.model_path, 'csv')[0][0]

    print('[INFO] Load trained tf model')

    model = inits.init_model(config.model, **vars(config))

    # Load models with custom functions / layers
    if config.model == 'pvae':
        model.model_ = load_model(
            config.model_path + '/model.h5',
            custom_objects={'<lambda>': lambda x, rv_x: -rv_x.log_prob(x)})
    elif config.model == 'vae':
        model.model_ = load_model(
            config.model_path + '/model.h5',
            custom_objects={'_sampling': model._sampling})
    else:
        model.model_ = load_model(config.model_path + '/model.h5')

    plot_vae_weights(model, config)
