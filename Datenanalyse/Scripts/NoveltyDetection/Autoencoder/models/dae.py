
# =============================================================================
# This model is build that it can be used as an estimator class of sklearn
# Inspired by http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
#
#
# =============================================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input # , LeakyReLU, ReLU
import tensorflow as tf
import numpy as np
from numpy.random import normal

# custom packages
import scripts.framework as frwk
from models.CustomBaseEstimator import CustomBaseEstimator

# MODEL TYPE: AUTOENCODER
class Dae(CustomBaseEstimator):

    def __init__(self, epochs=200, batch_size = 64, loss='mse', opt='adam',
                 layer_config=[190, 110, 10, 110, 190], noise_scale=0.01, act='relu', **kwargs):

        # hyperparameter set of model
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_config = layer_config
        self.noise_scale = noise_scale
        self.loss = loss
        self.opt = opt
        self.act = act

    def _build(self, n_features):

        #n_layers = shape(self.layer_config)[0]
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
            layer = tf.keras.activations.relu(layer)#layer = ReLU()(layer)
            #layer = LeakyReLU(alpha=0.3)(layer)

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

        # Get list with objects of selected callbacks
        callbacks = self._set_callbacks(callbacks, x_val=x_val, y_val=y_val,
                                        max_epochs=self.epochs, **kwargs)

        # add gaussian noise to tranings data
        x_noisy = x + normal(loc=0.0, scale=self.noise_scale, size=x.shape)

        # Train with selected settings
        # Use Blogger-Callback instead of verbose
        self.history_ = self.model_.fit(
            x_noisy, x, epochs=self.epochs, batch_size=self.batch_size,
            shuffle=True, validation_data=None, verbose=0,
            callbacks=list(callbacks.values()))

        # BestModelCheckpoint saves weights of model continually
        if 'BestModelCheckpoint' in list(callbacks.keys()):
            self.best_model_weights_ = \
                callbacks['BestModelCheckpoint'].best_model_weights

        return

    def nov_score(self, x):
        # Novelty score is root-mean-square-error
        x_p = self.predict(x)
        nov_score = frwk.calc_rmse(x, x_p)
        return nov_score

    def get_params_grid(self):

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config
        # layer_config = [[l_1, l_2, l_3, l_2, l_1] for l_1 in arange(10, 300, 10) \
        #                 for l_2 in arange(10, 300, 10) for l_3 in arange(10, 300, 10) \
        #                 if l_1 > l_2 > l_3]
        noise_scale_grid =[[n] for n in np.arange(0.1, 3.0, 0.2)]
        noise_scale_grid.append([0.01])
        params_grid['noise_scale'] = noise_scale_grid

        return params_grid




