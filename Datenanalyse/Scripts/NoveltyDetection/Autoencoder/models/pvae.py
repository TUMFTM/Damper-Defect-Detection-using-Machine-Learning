#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: pvae.py
# -----------------------------------------------------------------------------
# Class function for probabilistic variational autoencoder
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
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.layers as tfpl
import tensorflow_probability.python.distributions as tpd
from models.CustomBaseEstimator import CustomBaseEstimator

class Pvae(CustomBaseEstimator):

    def __init__(self, epochs=200, batch_size = 64, loss='mse', opt='adam',
                 layer_config=[340,100,35], act='relu', **kwargs):

        # hyperparameter set of model
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_config = layer_config
        self.loss = loss
        self.opt = opt
        self.act = act

    def _build(self, n_features):

        if len(self.layer_config) == 3:
            h_1 = self.layer_config[0]
            h_2 = self.layer_config[1]
            h_3 = self.layer_config[2]
            h_4 = self.layer_config[1]
            h_5 = self.layer_config[0]

            # prior p(z) as isotropic Gaussian
            self.prior_z = tpd.Independent(
                tpd.Normal(loc=tf.zeros(h_3), scale=1),
                reinterpreted_batch_ndims=1)

            encoder = tfk.Sequential([
                tfkl.Input(shape=(n_features,), name='input'),
                # tfkl.Dropout(rate=0.1),
                tfkl.Dense(units=h_1, activation=self.act, name='h1'),
                tfkl.Dense(units=h_2, activation=self.act, name='h2'),
                tfkl.Dense(tfpl.IndependentNormal.params_size(h_3), activation=None),
                tfpl.IndependentNormal(
                    h_3, activity_regularizer=
                    tfpl.KLDivergenceRegularizer(self.prior_z, weight=1.0)),
                ], name='encoder')

            decoder = tfk.Sequential([
                tfkl.Input(shape=encoder.output.shape, name='input'),
                tfkl.Dense(units=h_4, activation=self.act, name='h3'),
                tfkl.Dense(units=h_5, activation=self.act, name='h4'),
                tfkl.Dense(tfpl.IndependentNormal.params_size(n_features), activation=None),
                tfpl.IndependentNormal(n_features)
                ], name='decoder')

        else:
            h_1 = self.layer_config[0]
            # h_2 = self.layer_config[1]
            h_3 = self.layer_config[1]
            # h_4 = self.layer_config[1]
            h_5 = self.layer_config[0]

            # prior p(z) as isotropic Gaussian
            self.prior_z = tpd.Independent(
                tpd.Normal(loc=tf.zeros(h_3), scale=1),
                reinterpreted_batch_ndims=1)

            encoder = tfk.Sequential([
                tfkl.Input(shape=(n_features,), name='input'),
                # tfkl.Dropout(rate=0.1),
                tfkl.Dense(units=h_1, activation=self.act, name='h1'),
                # tfkl.Dense(units=h_2, activation=self.act, name='h2'),
                tfkl.Dense(tfpl.IndependentNormal.params_size(h_3), activation=None),
                tfpl.IndependentNormal(
                    h_3, activity_regularizer=
                    tfpl.KLDivergenceRegularizer(self.prior_z, weight=1.0)),
            ], name='encoder')

            decoder = tfk.Sequential([
                tfkl.Input(shape=encoder.output.shape, name='input'),
                # tfkl.Dense(units=h_4, activation=self.act, name='h3'),
                tfkl.Dense(units=h_5, activation=self.act, name='h4'),
                tfkl.Dense(tfpl.IndependentNormal.params_size(n_features), activation=None),
                tfpl.IndependentNormal(n_features)
            ], name='decoder')

        # self.prior_z = tpd.Independent(
        #     tpd.Normal(loc=tf.zeros(self.layer_config[0]), scale=1),
        #     reinterpreted_batch_ndims=1)
        #
        # encoder = tfk.Sequential([
        #     tfkl.Input(shape=(n_features,), name='input'),
        #     # tfkl.Dropout(rate=0.1),
        #     # tfkl.Dense(units=h_1, activation=self.act, name='h1'),
        #     # tfkl.Dense(units=h_2, activation=self.act, name='h2'),
        #     tfkl.Dense(tfpl.IndependentNormal.params_size(self.layer_config[-1]), activation=None),
        #     tfpl.IndependentNormal(
        #         self.layer_config[-1], activity_regularizer=
        #         tfpl.KLDivergenceRegularizer(self.prior_z, weight=1.0)),
        # ], name='encoder')
        #
        # decoder = tfk.Sequential([
        #     tfkl.Input(shape=encoder.output.shape, name='input'),
        #     # tfkl.Dense(units=h_4, activation=self.act, name='h3'),
        #     # tfkl.Dense(units=h_5, activation=self.act, name='h4'),
        #     tfkl.Dense(tfpl.IndependentNormal.params_size(n_features), activation=None),
        #     tfpl.IndependentNormal(n_features)
        # ], name='decoder')

        self.model_ = tfk.Model(inputs=encoder.inputs,
                                outputs=decoder(encoder.outputs[0]))

        # add loss function
        # loss function gets: x and output of network
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

        self.model_.compile(optimizer=self.opt,
                            loss=neg_log_likelihood)

    def _train(self, x, y, x_val, y_val, callbacks, **kwargs):

        # verify that x only contains neg class
        if np.count_nonzero(y==1)!=0:
            raise ValueError('Training on pos class not allowed')

        # Get list with objects of selected callbacks
        self.callbacks = self._set_callbacks(callbacks, x_val=x_val, y_val=y_val,
                                             max_epochs=self.epochs, **kwargs)

        # Train with selected settings
        # Use Blogger-Callback instead of verbose
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

    def nov_score(self, x):
        # Novelty score is the reconstruction log probability
        p_z = self.model_(x)
        nov_score = np.nan_to_num(-p_z.log_prob(x))
        # nov_score = np.nan_to_num(-p_z.prob(x))

        return nov_score

    def get_params_grid(self):

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config
        # layer_config = [
        #     [l_1, l_2, l_3, l_2, l_1] for l_1 in np.arange(10, 300, 20) \
        #     for l_2 in np.arange(50, 150, 20) for l_3 in np.arange(5, 40, 5) \
        #     if l_1 > l_2 > l_3]
        layer_config = [
            [l_1, l_3, l_1] for l_1 in np.arange(10, 300, 20) \
            for l_3 in np.arange(5, 50, 5) \
            if l_1 > l_3]

        params_grid['layer_config'] = layer_config

        return params_grid

    # def encode(self, x):
