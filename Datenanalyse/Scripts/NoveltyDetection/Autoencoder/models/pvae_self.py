#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: ocnn.py
# -----------------------------------------------------------------------------
# Class function for one-class neural network
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

from models.CustomBaseEstimator import CustomBaseEstimator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda #, LeakyReLU, ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

# custom packages
import scripts.framework as framework

class Pvae(CustomBaseEstimator):

    def __init__(self, epochs=200, batch_size = 64, loss='mse', opt='adam',
                 layer_config=[190, 70, 25, 70, 190], act='relu', **kwargs):

        # hyperparameter set of model
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_config = layer_config
        self.loss = loss
        self.opt = opt
        self.act = act

    def _build(self, n_features):
        
        h_1 = h_5 = self.layer_config[0]
        h_2 = h_4 = self.layer_config[1]
        h_3 = self.layer_config[2]


        # build layers of vae model
        encoder_in= Input(shape=(n_features,), name='Input')

        layers = Dense(units=h_1, activation='relu')(encoder_in)

        layers = Dense(units=h_2, activation='relu')(layers)

        z_mean = Dense(units=h_3, name='Z_mean')(layers)
        z_log_var = Dense(units=h_3, name='Z_log_var')(layers)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(
            self._sampling, output_shape=(25,), name='Z')([z_mean, z_log_var])

        #instantiate encoder model
        # self.model_encoder_ = Model(
        #     encoder_in, [z_mean, z_log_var, z], name='Encoder')
        model_encoder_ = Model(
            encoder_in, [z_mean, z_log_var, z], name='Encoder')

        # build decoder model
        decoder_in = Input(shape=(h_3,), name='Z_sampling')

        layers_2 = Dense(units=h_4, activation='relu')(decoder_in)

        layers_2 = Dense(units=h_5, activation='relu')(layers_2)

        decoder_out_mean = Dense(units=n_features, name='decoder_mean')(layers_2)
        decoder_out_log_var = Dense(units=n_features, name='decoder_var')(layers_2)

        # instantiate decoder model
        # self.model_decoder_ = Model(decoder_in, [decoder_out_mean, decoder_out_var], name='Decoder')
        model_decoder_ = Model(decoder_in, [decoder_out_mean, decoder_out_log_var], name='Decoder')

        # instantiate VAE model
        model_output = model_decoder_(model_encoder_(encoder_in)[2])
        self.model_ = Model(encoder_in, model_output, name='Vae')

        # make custom loss
        # reconstruction loss
        # rec_loss = mse(encoder_in, model_output)
        # rec_loss *= n_features
        rec_loss = self.log_prob(encoder_in, model_output[0], model_output[1])

        # Kullback L loss
        kl_loss = 1+z_log_var-K.square(z_mean)-K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(rec_loss + kl_loss)
        # vae_loss = K.mean(rec_loss)

        # add custom loss
        self.model_.add_loss(vae_loss)
        # Compile
        opt = optimizers.Adam()
        # self.model_.compile(optimizer=opt, fine_tune_batch_norm=False)
        self.model_.compile(optimizer=opt)

        self.test = 0

    def _sampling(self, args):
        # reparameterization trick
        # insteas of sampling from Q(z|X), sample epsilon = N(O,I)
        # z = z_mean + sqrt(var) * epsilon
        z_mean, z_log_var = args
    
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _train(self, x, y, x_val, y_val, callbacks, **kwargs):

        # verify that x only contains neg class
        if np.count_nonzero(y==1)!=0:
            raise ValueError('Training on pos class not allowed')

        # Get list with objects of selected callbacks
        callbacks = self._set_callbacks(callbacks, x_val=x_val, y_val=y_val,
                                        max_epochs=self.epochs, **kwargs)

        # Train with selected settings
        # Use Blogger-Callback instead of verbose
        self.history_ = self.model_.fit(
            x, epochs=self.epochs, batch_size=self.batch_size,
            shuffle=True, validation_data=None, verbose=0,
            callbacks=list(callbacks.values()))

        # BestModelCheckpoint saves weights of model continually
        if 'BestModelCheckpoint' in list(callbacks.keys()):
            self.best_model_weights_ = \
                callbacks['BestModelCheckpoint'].best_model_weights

        return

    def log_prob(self, x_in, x_out, var_out):
        if tf.is_tensor(x_out):
            var_out = K.exp(var_out)
            return K.sum(-K.log(
                1 / (K.sqrt(2 * np.pi * var_out)) * K.exp(-K.square(x_in - x_out) / (2 * var_out))),
                         axis=-1)
        else:
            return np.sum(
                -np.log(1 / (np.sqrt(2 * np.pi * np.exp(var_out))) * np.exp(-np.square(x_in - x_out) / (2 * np.exp(var_out)))), axis=-1)


    def nov_score(self, x):
        # Novelty score is root-mean-square-error
        x_p = self.predict(x)
        nov_score = self.log_prob(x, x_p[0], x_p[1])
        # nov_score = np.nan_to_num(framework.calc_rmse(x, x_p))
        return nov_score

    def get_params_grid(self):

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config
        layer_config = [
            [l_1, l_2, l_3, l_2, l_1] for l_1 in np.arange(10, 200, 20) \
            for l_2 in np.arange(50, 150, 20) for l_3 in np.arange(5, 30, 5) \
            if l_1 > l_2 > l_3]
            
        params_grid['layer_config'] = layer_config

        return params_grid






