#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: callbacks.py 
# -----------------------------------------------------------------------------
# Callbacks for evaluation of training process
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

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
# from sklearn.base import clone

# custom packages
import scripts.framework as frwk

class RocCurve(Callback):
    # based on https://www.kaggle.com/tilii7/keras-averaging-runs-
    # gini-early-stopping
    def __init__(self, estimator, x_val, y_val, verbose=0):
        self.x = x_val
        self.y = y_val
        self.verbose = verbose
        self.estimator = estimator

        self.roc_auc_max = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs):

        # use estimator function (which is super to self.model) to predict
        # novelty score
        nov_score = self.estimator.nov_score(self.x)
        # get values for roc curve
        roc_auc, fprs, tprs, thresholds = frwk.calc_roc_curve(self.y,nov_score)

        # add to logs
        logs['roc_auc'], logs['thresholds'] = roc_auc, thresholds
        logs['fprs'], logs['tprs']= fprs, tprs

        if roc_auc > self.roc_auc_max: self.roc_auc_max = roc_auc
        logs['roc_auc_max'] = self.roc_auc_max

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class BestModelCheckpoint(Callback):

    def __init__(self):
        self.best_model_weights = []
        self.best_roc_auc = 0

    def on_epoch_end(self, epoch, logs):

        if self.best_roc_auc < logs['roc_auc']:

            #self.best_model = clone_model(self.model)
            self.best_model_weights = self.model.get_weights()
    
            # update best roc auc
            self.best_roc_auc = logs['roc_auc']

        return

class Blogger(Callback):
    def __init__(self):
        self.max_epochs = 0

    def on_train_begin(self, logs=None):
        self.max_epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs):

        print('Epoch %i/%i' %((epoch+1), self.max_epochs))
        if 'loss' in logs.keys():
            print(' - loss: %0.4f\t' %(logs['loss']), end="" )
        if 'roc_auc' in logs.keys():
            print(' - ROC AUC %.2f%%' %(logs['roc_auc']*100))

        return

class ProgbarLogger(Callback):

    def on_train_begin(self, logs=None):
        #print('Epoch %d/%d' % (epoch + 1, self.epochs))
        self.target = self.params['epochs']
        self.stateful_metrics = ['loss', 'roc_auc', 'roc_auc_max']
        self.roc_auc_max = 0
        self.progbar = Progbar(self.target, verbose=1,
                               stateful_metrics=self.stateful_metrics)
        self.seen = 0

    def on_epoch_begin(self, epoch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_epoch_end(self, epoch, logs=None):

        self.seen += 1
        logs = logs or {}

        for k in logs:
            if k in ['loss', 'roc_auc', 'roc_auc_max']:
                self.log_values.append((k, logs[k]))

        if self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_train_end(self, logs=None):
        # Necessary to end line
        print('')

        return


class MahDistance(Callback):
    
    def __init__(self, estimator, x_train, x_val):
        self.estimator = estimator
        self.x_train = x_train
        self.x_val = x_val
        
    def on_epoch_end(self, batch, logs={}):
        
        # Get code of middle layer of training data        
        code_train = self.estimator.get_code(self.x_train)
        code_val = self.estimator.get_code(self.x_val)
        x_train_p = self.estimator.predict(self.x_train)
        
        # Evaluate statistics of code
        # Set rowvar=False: rows contain observations
        self.estimator.code_cov = np.cov(code_train, rowvar=False) 
        self.estimator.code_mean = np.mean(code_train, axis=0)
        
        self.estimator.alpha = 1/np.std(frwk.calc_dm(
            code_val, self.estimator.code_mean, self.estimator.code_cov))
        self.estimator.beta = 1/np.std(frwk.calc_rmse(self.x_train, x_train_p))
        
        return


# class TensorBoard(Callback):
#
#     def __init__(self, estimator, model_path):
#         self.estimator = estimator
#         self.model_path = model_path
#         self.tensorboard_callback = None
#
#     def on_train_begin(self):
#         self.tensorboard_callback = TensorBoard(log_dir=self.estimator.model_path)
#
#     def on_train_end(self):
#         self.tensorboard_callback.set_model(self.estimator)
