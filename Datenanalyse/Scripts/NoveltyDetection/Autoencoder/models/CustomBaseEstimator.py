#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: CustomBaseEstimator.py 
# -----------------------------------------------------------------------------
# Base class for estimators models
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

# make abstract class
from abc import ABC, abstractmethod
# import sklearn super classes
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import dump, load
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import plot_model
from tensorflow.summary import create_file_writer

# custom packages
from scripts.callbacks import RocCurve, BestModelCheckpoint, Blogger
from scripts.callbacks import ProgbarLogger, MahDistance


class CustomBaseEstimator(ABC, BaseEstimator, ClassifierMixin):
    # Following abstract methods have to be implemented by subclass
    # _build(self, x);...; return model
    # _train(self, x, y, x_val, y_val, callbacks, **kwargs);...; return history
    # nov_score(self, x);...; return nov_score
    
    def fit(self, x, y, x_val, y_val, callbacks, **kwargs):

        # Use private function _build() of selected model
        
        # derive info from inout data
        n_features = np.shape(x)[1]

        # file_writer = create_file_writer(kwargs['model_path'] + "/metrics")
        # file_writer = create_file_writer(kwargs['model_path'])
        # file_writer.set_as_default()

        self._build(n_features)
        
        # Print model structure
        # self.summary()

        # Train model
        self._train(x, y, x_val, y_val, callbacks, **kwargs)

        return self

    def fit_production(self, x, y, callbacks, **kwargs):

        # Train model without acces to positive data, hence roc auc cannot
        # be calculated as metric for EarlyStopping
        self._build(x)
        self._train(x, y, [], [], callbacks, **kwargs)

        return self

    @abstractmethod
    def _build(self, x):
        # private function
        # describes how model is built
        # return model
        pass

    def get_history(self):
        return self.history_.history

    def _set_callbacks(self, callbacks, **kwargs):

        cb_dict = {}

        for call in callbacks:
            
            if call == 'MahDistance':
                # Get code of intermediate layer
                cb_dict['MahDistance'] = MahDistance(
                    self, kwargs['x_train'], kwargs['x_val'])
                
            if call == 'RocCurve':
                # display roc values of each epoch, used for EarlyStopping!
                # give self (object) to RocAuc callback, that all methods of
                # CustomBaseEstimator class can be used
                cb_dict['RocCurve'] = RocCurve(
                    self, kwargs['x_val'], kwargs['y_val'], verbose=1)

            if call == 'EarlyStopping':
               cb_dict['EarlyStopping'] = EarlyStopping(
                   monitor='roc_auc', min_delta=0.0001, patience=20,
                   mode='max', verbose=0)

            if call == 'ModelCheckpoint':
                cb_dict['ModelCheckpoint'] = ModelCheckpoint(
                    kwargs['model_path'] + '/best_model_n' + \
                        str(kwargs['i_cv']) + '.h5', monitor='roc_auc',
                        mode='max', verbose=0, save_best_only=True)

            if call == 'BestModelCheckpoint':
                cb_dict['BestModelCheckpoint'] = BestModelCheckpoint()

            if call == 'Blogger':
                cb_dict['Blogger'] = Blogger()

            if call == 'CSVLogger':
                cb_dict['CSVLogger'] = CSVLogger(
                      kwargs['model_path'] + '/metrics_n' + \
                          str(kwargs['i_cv']) + '.log', separator=',',
                          append=False)

            if call == 'TensorBoard':
                 cb_dict['TensorBoard'] = TensorBoard(log_dir=kwargs['model_path'], profile_batch=100000000, histogram_freq=0)

            if call == 'ProgbarLogger':
                 cb_dict['ProgbarLogger'] = ProgbarLogger()


            if not call: # if calls is empty
                cb_dict = None

        return cb_dict

    def predict(self, x):
        return self.model_.predict(x)

    @abstractmethod
    def nov_score(self, x):
        # describes how novelty score is calculated
        # return nov_score
        pass

    def score(self, x, y):
        # standard metric for evaluation is roc auc value
        # get novelty score of fitted model
        nov_score = self.nov_score(x)
        roc_auc = roc_auc_score(y, nov_score)

        return roc_auc

    def summary(self):
        # print of all set hyperparamter values
        print('- selected hyperparameter of model:')
        print('   - selected loss function: ', self.loss)
        print('   - selected optimizer: ', self.opt)
        self.model_.summary()

    def save(self, log_path, name='/model.h5'):
        dump(self, log_path+name)

    def load(self, model_path):
        load(model_path)

    @abstractmethod
    def get_params_grid(self):
        # describes how novelty score is calculated
        # return nov_score
        pass

    def plot(self, path):
        pic = plot_model(self.model_, to_file=path+'/model.png', show_shapes=True,
                        show_layer_names=True)
        return pic

    # @abstractmethod
    # def encode(self, x):
    #     # return latent variables
    #     pass


