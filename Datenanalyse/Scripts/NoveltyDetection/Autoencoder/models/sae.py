#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: sae.py 
# -----------------------------------------------------------------------------
# Class function for stacked autoencoder model
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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input #, LeakyReLU, ReLU
import tensorflow.keras.backend as K

import numpy as np
# custom packages
import scripts.framework as frwk
from models.CustomBaseEstimator import CustomBaseEstimator

# MODEL TYPE: AUTOENCODER
class Sae(CustomBaseEstimator):

    def __init__(self, epochs=200, batch_size = 64, loss='mse', opt='adam',
                 layer_config=[190,110,10,110,190], act='relu', nov_fun = 'l2',
                 **kwargs):

        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_config = layer_config
        self.loss = loss
        self.opt = opt
        self.act = act
        self.nov_fun = nov_fun

    def _build(self, n_features):
        
        self.n_features_ = n_features

        n_layers = len(self.layer_config)

        # input layer of model
        input_layer = layer = Input(shape=(n_features,), name='Input')

        # loop over desired number of layers
        for i in range(n_layers):
            # neurons of current layer
            units = self.layer_config[i]

            hidden = Dense(units=units, #activation=self.act,
                           name='Dense{}_{}'.format(i, units))
            # add to previous layer_config
            layer = hidden(layer)
            layer = tf.keras.activations.relu(layer)
            # layer = LeakyReLU(alpha=0.01)(layer)

        # output layer
        output_layer = Dense(units=n_features, name='Output')(layer)
        
        # compile model
        self.model_ = Model(input_layer, output_layer, name='Ae')
        self.model_.compile(loss=self.loss, optimizer=self.opt)

        return

    def _train(self, x, y, x_val, y_val, callbacks, **kwargs):

        # verify that x only contains neg class
        if np.count_nonzero(y==1)!=0:
            raise ValueError('Training on pos class not allowed')
            
        if self.nov_fun =='mah':
             callbacks.insert(0, 'MahDistance')

        # get list with objects of selected callbacks
        self.callbacks = self._set_callbacks(callbacks, x_train = x,
                                             x_val=x_val, y_val=y_val,
                                             max_epochs=self.epochs, **kwargs)

        # train with selected settings
        # use Blogger callback instead of verbose
        self.history_ = self.model_.fit(x, x,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_data=None,
                                        verbose=0,
                                        callbacks=list(self.callbacks.values()))
        
        # BestModelCheckpoint saves weights of model continually
        if 'BestModelCheckpoint' in list(self.callbacks.keys()):
            self.best_model_weights_ = \
                self.callbacks['BestModelCheckpoint'].best_model_weights

        return
    
    def get_code(self, x):
        
        code_layer = int((len(self.model_.layers)-1)/2)

        # make helper function to get output of intermediate layer, recommended
        # on keras homepage
        get_inter_layer_output = K.function(
            [self.model_.layers[0].input],
            [self.model_.layers[code_layer].output])
    
        layer_output = get_inter_layer_output([x])[0]
    
        return layer_output
        
    def nov_score(self, x):
        
        # novelty score is RMSE
        if self.nov_fun == 'mah':            
                        
            code = self.get_code(x)
            x_p = self.predict(x)
            
            # Get distance measures
            dm = frwk.calc_dm(code, self.code_mean, self.code_cov)
            l2 = frwk.calc_rmse(x, x_p)
                        
            nov_score = self.alpha*dm + self.beta*l2
            
        # custom novelty score to be implemented    
        elif self.nov_fun == 'cus':
            
            x_p = self.predict(x)
            
            n_sensors = 7
            n_seg = int(x.shape[1] / n_sensors)
            nov_score_sensor = np.zeros((x.shape[0], n_sensors))
            
            for i in range(n_sensors):
                 nov_score_sensor[:,i] = frwk.calc_rmse(
                     x[:,i*n_seg:(i+1)*n_seg],
                     x_p[:,i*n_seg:(i+1)*n_seg])
                             
            nov_score = np.sum(nov_score_sensor, axis=1)
                          
        elif self.nov_fun == 'l2':
            x_p = self.predict(x)
            nov_score = frwk.calc_rmse(x, x_p)
        return nov_score

    def get_params_grid(self):

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config (l_1 > l_2)
        layer_config = [
            [l_1, l_2, l_3, l_2, l_1] for l_1 in np.arange(30, 210, 20) \
            for l_2 in np.arange(50, 150, 20) for l_3 in np.arange(10, 100, 10) \
            if l_1 > l_2 > l_3]

        params_grid['layer_config'] = layer_config
        
        return params_grid
    




