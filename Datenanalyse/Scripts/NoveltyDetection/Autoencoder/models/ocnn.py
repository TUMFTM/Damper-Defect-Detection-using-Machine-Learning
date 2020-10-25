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
import numpy as np
# import tensorflow as tf
from models.CustomBaseEstimator import CustomBaseEstimator
from tensorflow.keras.layers import Input, Dense #, Lambda, LeakyReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

class rUpdateCall(Callback):
    def __init__(self, x_val_latent):
        self.x = x_val_latent
    def on_train_begin(self, logs={}):
        self.model.r = 1

    def on_batch_end(self, batch, logs={}):
        y = self.model.predict(self.x)
        self.model.r = np.quantile(y, 0.01)

        print(self.model.r, batch)

class Ocnn(CustomBaseEstimator):

    def __init__(self, epochs=200, batch_size = 256, loss='mse', opt='adam',
                 n_ae_hidden=60, n_ocnn_hidden=25, act='relu'):

        # hyperparameter set of model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_ae_hidden = n_ae_hidden
        self.n_ocnn_hidden = n_ocnn_hidden

        self.loss = loss
        self.opt = opt
        self.act = act

        self.nu = 0.01
        self.kvar = 0.0
        self.r = 1

    def _build(self, n_features):

        # Build autoencoder
        ae_input = Input(shape=(n_features, ), name='ae_input')
        ae_hidden = Dense(units=150, activation='relu', name='1242414')(ae_input)
        ae_hidden = Dense(units=self.n_ae_hidden, activation='relu', name='ae_hidden')(ae_input)
        ae_output = Dense(units=150, name='12412')(ae_hidden)
        ae_output = Dense(units=n_features, name='ae_output')(ae_output)

        # Hidden layer used as input for ocnn
        self.model_encoder_ = Model(ae_input, ae_hidden, name='ae_encoder')
        self.model_ae_ = Model(ae_input, ae_output, name='ae')

        #layer.trainable = False

        # self.Ae_decoder = Model(ae_hidden, ae_output, name='ae_decoder')
        # ae_output = self.Ae_decoder(self.Ae_encoder(ae_input))
        # self.Ae = Model(ae_input, self.outputs, nam='vae)

        # create oc nn
        ocnn_input = Input(shape=(self.n_ae_hidden,), name='ocnn_input')

        ocnn_layer = Dense(units=self.n_ocnn_hidden)

        ocnn_hidden= ocnn_layer(ocnn_input)
        #ocnn_act_1 = LeakyReLU(alpha=0.3)(ocnn_hidden)

        ocnn_output = Dense(units=1)
        output = ocnn_output(ocnn_hidden)



        # get layer weights (tensor objects)
        V = ocnn_layer.kernel
        w = ocnn_output.kernel

        self.VV = V
        self.ww = w
        # make ocnn model
        self.model_ocnn = Model(ocnn_input, output, name='ocnn')
        #make layer as loss function

        term_1 = 0.5*K.sum(K.square(w))
        term_2 = 0.5*K.sum(K.square(V))
        term_3 = 1/self.nu*K.mean(K.maximum(0.0, self.r-(output)), axis=-1)
        term_4 = -1*self.r

        loss_fn = term_1 +term_2 + term_3+ term_4

        self.model_ocnn.add_loss(loss_fn)


        self.model_encoder_.compile(optimizer=self.opt, loss='mse')
        self.model_ae_.compile(optimizer=self.opt, loss='mse')

        self.model_ocnn.compile(optimizer=self.opt)#, loss=self.custom_loss(w, V))


        return

    def custom_loss(self, w, V):
    # make a custom loss function by using function closure
    # (custom function for loss always needs to have arg(y_true, y_pred))
        def custom(y_true, y_pred):


            # define custom loss
            term_1 = 0.5*K.sum(K.square(w))
            term_2 = 0.5*K.sum(K.square(V))
            term_3 = 1/self.nu*K.mean(K.maximum(0.0, self.r-(y_pred)))
            term_4 = -1*self.r

            # update r
            # a = K.max(y_pred, axis=1)
            # self.r = tfp.stats.percentile(a, self.nu*100)
            # a = K.print_tensor(a,[a])

            return (term_1 + term_2 + term_3 + term_4)

        return custom




    def _train(self, x, y, x_val, y_val, callbacks, **kwargs):

        # Get list with objects of selected callbacks
        callbacks = self._set_callbacks(callbacks, x_val=x_val, y_val=y_val,
                                        max_epochs=self.epochs, **kwargs)

        self.model_ae_.fit(x, x,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=None,
                    verbose=2,
                    callbacks=None)

        x_train_latent = self.model_encoder_.predict(x)

        x_val_latent = self.model_encoder_.predict(x_val)


        # idee für iteratives training:
        # make loop with only one training step and then update r

        self.history_ = self.model_ocnn.fit(x_train_latent,
                      epochs=100,
                      batch_size=256,
                      shuffle=True,
                      validation_data=None,
                      verbose=2,
                      callbacks=[rUpdateCall(x_train_latent)])#list(callbacks.values()))



        return

    def nov_score(self, x):
        # Novelty score is root-mean-square-error
        x_latent = self.model_encoder_.predict(x)


        x_p = self.model_ocnn.predict(x_latent)
        #print(x_p)

        nov_score = x_p-self.r

        #a = np.array([1 if score > 0 else 0 for score in nov_score])

        return nov_score

    def get_params_grid(self):

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config
        layer_config =[[200,100,200],[30,10,30],[100,10,100],[300,200,300]]
        params_grid['layer_config'] = layer_config

        return params_grid






