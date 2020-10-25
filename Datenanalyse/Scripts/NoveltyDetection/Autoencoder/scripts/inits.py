#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: inits.py
# -----------------------------------------------------------------------------
# Init commands for general settings of model training and evaluation
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
import argparse
import datetime

from sklearn.pipeline import Pipeline
from cycler import cycler
from matplotlib.pyplot import rcParams, cm
import numpy as np

# import estimator models
from models.sae import Sae
from models.dae import Dae
from models.vae import Vae
from models.pvae import Pvae
from models.pvae_self import Pvae as Pvae_self

# import transformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, Normalizer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scripts.transformer import Detrend, Fourierer


def init_model(model_name, **params):
    # create model
    if model_name == 'sae': model = Sae(**params)
    elif model_name == 'dae': model = Dae(**params)
    elif model_name == 'vae': model = Vae(**params)
    elif model_name == 'pvae': model = Pvae(**params)
    elif model_name == 'pvae_self': model = Pvae_self(**params)
    else: raise NameError('model type is unknown')

    return model

def init_pipe(config):

    # list for in config.prep_options selected transformers
    transformers = []

    for step in config.prep_options:
        if step == 'detrend':
            transformers.append(
                ['detrend', Detrend(n_channels=config.n_sensors,
                                    seq_length=config.orig_seq_lngth)])
        if step == 'fourier':
            transformers.append(
                ['fourier', Fourierer(config.n_sensors, config.orig_seq_lngth,
                                      nfft=config.nfft)])
        if step == 'minmax_01':
            transformers.append(
                ['minmax_01', MinMaxScaler(feature_range=(0, 1))])

        if step == 'minmax_11':
            transformers.append(
                ['minmax_11', MinMaxScaler(feature_range=(-10, 10))])
            
        if step == 'standard':
            transformers.append(
                ['standard', RobustScaler()])

        if step == 'robust':
            transformers.append(
                ['robust', StandardScaler()])

        if step == 'quantile':
            transformers.append(
                ['quantile', QuantileTransformer(
                    # fit to gaussian distribution
                    output_distribution ='normal', random_state=42)])

        if step == 'power':
            transformers.append(
                ['power', PowerTransformer(method='box-cox')])

        if step == 'normalizer':
            transformers.append(
                ['normalizer', Normalizer(norm='l2')])

    # make pipeline with selected transformers
    pipe = Pipeline(transformers)

    return pipe

def init_plot():
    # change rcParams for custom look - show all settings: plt.rcParams.keys()
    style_c = cycler('linestyle', ['-', '--', '-.',':','-', '--', '-.',':'])
    color_c = cycler('color', cm.Greys((np.linspace(1, 0, 10)[1:-1])))
    rcParams['axes.prop_cycle']=(style_c+color_c) 

    rcParams['lines.linewidth'] = 2

    rcParams['axes.grid'] = True
    rcParams['grid.color'] = 'k'
    rcParams['grid.linestyle'] = ':'
    rcParams['grid.linewidth'] = 0.7

    rcParams['figure.autolayout'] = True # tight_layout for all plots
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300

    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.family'] = "sans-serif"
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['mathtext.default'] = 'regular'

    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 1


def init_parser():
    # init of argument parser
    
    # Choices of implemented models
    model_chc = ['sae', 'dae', 'vae', 'pvae', 'pvae_self']
    mode_chc = ['op', 'cv', 'pr', 'ev']
    # Choices for preprocessing
    prep_chc = ['detrend', 'fourier', 'minmax_01', 'minmax_11', 'robust',
                'quantile', 'power', 'normalizer', 'standard']

    data_run = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%M_%S")

    # argument of the argument parser
    parser = CustomArgumentParser(fromfile_prefix_chars='@')
    # Tweak to add info about run to parser
    data_run = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%M_%S")
    parser.add_argument("--date_run", default=data_run)

    # Path to data and model
    parser.add_argument('--data_folder_path', type=str, default='./../../../Datensatz/DataForCNN/data')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument('--dataset_mass_name', type=str, default='/testsets/DD_mass_raw_512_FlexRay_THP')
    parser.add_argument('--dataset_tire_name', type=str, default='/testsets/DD_ReiterEngineering_raw_512_FlexRay')
    parser.add_argument('--load_mass_data', type=int, default=0)
    parser.add_argument('--load_tire_data', type=int, default=0)
    parser.add_argument('--combine_training_data', type=int, default=0)
    parser.add_argument('--combine_training_dataset', type=str, default='')
    parser.add_argument('--combine_testing_data', type=int, default=0)
    parser.add_argument('--balance_training_datasets', type=int, default=0)
    parser.add_argument('--balance_testing_datasets', type=int, default=0)
    parser.add_argument('--nSamplesTestSize', type=int, default=250)
    parser.add_argument('--nSamplesTrainingInlier', type=int, default=0)

    # data parameter ('-' indicates an optional argument)
    parser.add_argument("--dataset_name", type=str, default='./')
    parser.add_argument("--mode", type=str, choices=mode_chc, default='cv')
    parser.add_argument('--n_opti_iter', type=int, default=0)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--pos_class", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="% of dataset used as held out test data")

    parser.add_argument("--prep_options", choices=prep_chc, nargs="*",
                        type=str)
    parser.add_argument('--nfft', type=int, default=128,
                        choices=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--scaling_type', type=str, default='power')
    parser.add_argument("--sel_sensors", nargs='*', type=str)
    parser.add_argument("--random", type=int, default=42)
    parser.add_argument('--analyse_data', action='store_true',
                        help='analyse the structure of the data')

    # Logging parameters
    parser.add_argument('--save_log', action='store_true',
                        help='redirects print commands to log.txt in logspath')
    parser.add_argument('--base_log_path', type=str, default='./../../../ND_AE/logs/')
    parser.add_argument('--save_plot', action='store_true',
                        help='save all plots')
    parser.add_argument('--save_model', action='store_true',
                        help='save the trained model')
    parser.add_argument('--save_scores', action='store_true',
                        help='save novelty scores')

    # model hyperparameters
    parser.add_argument("--model", type=str, default='ae', choices=model_chc,
                        help='model type to train/evaluate during run')
    parser.add_argument("--epochs", type=int, default=50,
                        help='number of epochs, 1 epoch includes \
                        all training samples')
    parser.add_argument("-bs", "--batch_size", type=int, default=128,
                        choices=[8, 16, 32, 64, 128, 256, 512],
                        help="number of training samples per mini-batch")
    parser.add_argument("--optimizer", type=str, default='adam',
                        choices=['adam', 'adaGrad'],
                        help="number of training samples per mini-batch")
    parser.add_argument("--layer_config", type=int, nargs='+',
                        help="number of neurons in layer")

    return parser.parse_args(['@config.args'])

class CustomArgumentParser(argparse.ArgumentParser):
    # used to improve original argparser
    # allows comments by using '#' and reads space separated args
    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg
        return line.split()


if __name__ == '__main__':
    pass
