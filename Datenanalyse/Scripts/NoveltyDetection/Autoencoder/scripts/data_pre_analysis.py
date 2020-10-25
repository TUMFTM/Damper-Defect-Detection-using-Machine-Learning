#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: data_pre_analysis.py (main function)
# -----------------------------------------------------------------------------
# QUICK AND DIRTY evaluation of dataset
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

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# custom packages
import scripts.utilities.data_handling as datahandler
import scripts.framework_DND as framework
import scripts.pipeline as piper
import scripts.utilities.plot as cplt
import scripts.utilities.pre_analysis_tools as analysis
print('[INFO] packages imported')

config = framework.parser_init()
cplt.plot_init()

# data selection
config.pos_class = 100
config.sel_sensors = ['SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR', 'YAW_RATE']
config.preprocessing_type = 'fourier_v2'
config.scaling_type = 'quantile'

config.datapath = './data/DD2_raw_512_FlexRay/'
config.pos_class = 100
random_state = 42
print('[INFO] start data import')

#%% import (preprocessed data)
data = datahandler.read_file('dataset', config.datapath, file_format='npy')
labels = datahandler.read_file('labels', config.datapath, file_format='npy')
config.orig_seq_lngth, _, config.avail_sensors = datahandler.read_infotxt(
    config.datapath)

data_DD = np.array(datahandler.read_file(
    'dataset', './data/DD_raw_512/', file_format='csv'))
labels_DD = np.array(datahandler.read_file(
    'labels', './data/DD_raw_512/', file_format='csv'))

#data = np.append(data, data_DD, axis=0)
#labels = np.append(labels, labels_DD, axis=0)

# sensor selection (config.sel_sensors)
if config.sel_sensors != config.avail_sensors:
    data, sensors, n_channels = datahandler.select_channels(
        data, config.avail_sensors, config.orig_seq_lngth, config.sel_sensors)
else: n_channels = len(config.sel_sensors)

# make the unprocessed datasets
datasets = framework.Datasets(data, labels, config.pos_class)

print('--> data succesfully imported')

#%% make analysis
print('[INFO] start analysis')
print('--> analyse EXAMPLES')

# create train and test data (unprocessed)
datasets._data_neg_train, datasets._data_neg_test, datasets._labels_neg_train, datasets._labels_neg_test= \
                train_test_split(datasets._data_neg, datasets._labels_neg,
                                 test_size=0.1, shuffle=True, random_state=random_state)

data_before = datasets._data_neg_train

# make preprocessing pipeline with sklearn
pipe = Pipeline(steps=[('detrender', piper.Detrender(
    n_channels, config.orig_seq_lngth)),
                      ('fourierer', piper.Fourierer(
                          n_channels, config.orig_seq_lngth, nfft=128)),
                      ('scaler', piper.make_Scaler(
                          config.scaling_type))])

data_train, data_val, labels_val = framework.make_data_plain_run(
    datasets._data_neg_train, datasets._data_pos,
    test_size=0.1, random_state=random_state)

# analysis of dataset feature structure
data_all = pipe[0:2].transform(data)

data_neg = pipe[0:2].transform(datasets._data_neg)
data_pos = pipe[0:2].transform(datasets._data_pos)
data_pos_all = pipe[0:2].transform(datasets._data_pos_all)


analysis.analyse_dataset(data_neg, data_pos, config)

# No scaler --> only detrend + FFT
data_train = pipe[0:2].transform(data_train)

data_val = pipe[0:2].transform(data_val)
data_val_neg = data_val[:int(len(data_val)/2)]
data_val_pos = data_val[int(len(data_val)/2):]



# analysis before scaler is fitted

# _,_,_,_,_ = analysis.analyse_features(data_train, config,
#                                     data_name='data_train',
#                                     fitted_on='none')

# _,_,_,_,_ = analysis.analyse_features(data_val_neg, config,
#                                     data_name='data_val_neg',
#                                     fitted_on='none')

# _,_,_,_,_ = analysis.analyse_features(data_val_pos, config,
#                                     data_name='data_val_pos',
#                                     fitted_on='none')

#analysis.analyse_spectrum(data_train, data_val_neg, data_val_pos, config, 'None')


# fit scaler and analysis

# fitted on data_train
pipe[2].fit(data_train[0:1000])

data_train_fit_1 = pipe[2].transform(data_train)

data_val_fit_1 = pipe[2].transform(data_val)
data_val_neg_fit_1 = data_val_fit_1[:int(len(data_val_fit_1)/2)]
data_val_pos_fit_1 = data_val_fit_1[int(len(data_val_fit_1)/2):]

analysis.analyse_spectrum(
    data_train_fit_1, data_val_neg_fit_1, data_val_pos_fit_1, config, 'data_train_[0:1000]')
# analysis.analyse_spectrum_v2(data_val_neg_fit_1, data_val_pos_fit_1, config, '[data_train]')


# _,_,_,_,_ = analysis.analyse_features(data_train_fit_1, config,
#                                       data_name='data_train',
#                                       fitted_on='data_train')
# _,_,_,_,_ = analysis.analyse_features(data_val_neg_fit_1, config,
#                                       data_name='data_val_neg',
#                                       fitted_on='data_train')
# _,_,_,_,_ = analysis.analyse_features(data_val_pos_fit_1, config,
#                                       data_name='data_val_pos',
#                                       fitted_on='data_train')

# _,_,_,_,_ = analysis.analyse_examples(data_val_neg_fit_1, config,
#                                       data_name='data_val_neg',
#                                       fitted_on='data_train')
# _,_,_,_,_ = analysis.analyse_examples(data_val_pos_fit_1, config,
#                                       data_name='data_val_pos',
#                                       fitted_on='data_train')


# fitted on data_all
pipe[2].fit(data_all)

data_train_fit_2 = pipe[2].transform(data_train)

data_val_fit_2 = pipe[2].transform(data_val)
data_val_neg_fit_2 = data_val_fit_2[:int(len(data_val_fit_2)/2)]
data_val_pos_fit_2 = data_val_fit_2[int(len(data_val_fit_2)/2):]

analysis.analyse_spectrum(data_train_fit_2, data_val_neg_fit_2,
                          data_val_pos_fit_2, config, 'data_all')
#analysis.analyse_spectrum_v2(data_val_neg_fit_2, data_val_pos_fit_1, config, '[data_all]')


# _,_,_,_,_ = analysis.analyse_features(data_train_fit_2, config,
#                                       data_name='data_train',
#                                       fitted_on='data_all')
# _,_,_,_,_ = analysis.analyse_features(data_val_neg_fit_2, config,
#                                       data_name='data_val_neg',
#                                       fitted_on='data_all')
# _,_,_,_,_ = analysis.analyse_features(data_val_pos_fit_2, config,
#                                       data_name='data_val_pos',
#                                       fitted_on='data_all')

# _,_,_,_,_ = analysis.analyse_examples(data_val_neg_fit_2, config,
#                                       data_name='data_val_neg',
#                                       fitted_on='data_all')
# _,_,_,_,_ = analysis.analyse_examples(data_val_pos_fit_2, config,
#                                       data_name='data_val_pos',
#                                       fitted_on='data_all')

# mean feature value


