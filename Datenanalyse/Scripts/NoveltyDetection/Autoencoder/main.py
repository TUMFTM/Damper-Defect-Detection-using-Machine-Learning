#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: main.py (main function)
# -----------------------------------------------------------------------------
# Main function for training and evalaution of DL-based models used for ND.
# -----------------------------------------------------------------------------
# This programm was developed as a part of a Semester Thesis at the
# Technical University of Munich (Germany).
# It was further adapted as part of the research project on
# predictive maintenance of automotive chassis systems.
#
# Programmer: Trumpp, Raphael F. and Zehelein, Thomas
# Email: raphael.trumpp@tum.de, thomas.zehelein@tum.de
# -----------------------------------------------------------------------------
# Copyright 2020 Raphael Frederik Trumpp and Thomas Zehelein
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

# import packages
from time import time; start_time = time()
import warnings; warnings.filterwarnings("ignore")
import os, sys; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split, ParameterSampler
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import TensorBoard
from functools import partial
import concurrent.futures

# current path
sys.path.insert(0, os.path.abspath(os.path.join('../')))
# import custom packages
import NoveltyDetection.Autoencoder.scripts.utilities.data_handling as datahandler
import NoveltyDetection.Autoencoder.scripts.framework as frwk
import NoveltyDetection.Autoencoder.scripts.inits as inits
import NoveltyDetection.Autoencoder.scripts.utilities.plot as cplt
import NoveltyDetection.Autoencoder.analyze_model as analyze_model



def main():

    # set up a parser called config
    config = inits.init_parser()

    # add paths to config
    config.data_path = config.data_folder_path + '/' + config.dataset_name + '/'
    if config.save_log:
        config.log_path = config.base_log_path + config.date_run + '_' + config.model
        os.makedirs(config.log_path)
        # change stdout to custom logger that saves log to log.txt
        sys.stdout = frwk.Logger(config)
    if config.save_plot:
        config.plot_path = config.base_log_path + config.date_run + '_' + config.model + '/plots'
        os.makedirs(config.plot_path)
    if config.save_model and config.mode == 'pr':
        config.model_path = config.base_log_path + config.date_run + '_' + config.model + '/model'
        os.makedirs(config.model_path)
        print(config.model_path)

    print('-----------------------------------------------------------------')
    print('***************     START PROGRAM: MAIN.PY    *******************')
    print('-----------------------------------------------------------------')
    print('[INFO] Importing packages - done! Duration: %.2fs' %(time()-start_time))

    # set style for plot_path
    inits.init_plot()

    # data import
    start_time_data_import = time()
    print('[INFO] Importing data', end='')

    # read dataset info from info.txt and add them to parser dynamically
    config.orig_seq_lngth, config.n_sensors, config.avail_sensors = \
        datahandler.read_infotxt(config.data_path)

    # use datahandler to imported data .npy file
    # note: importing .npy files is way faster than .csv files

    print('\nNormal data:')
    if config.mode == 'ev':
        balance_data = False
    else:
        balance_data = True

    x, x_train, x_test, y, y_train, y_test, config, idx_train, idx_test, idx = datahandler.load_data(data_path=config.data_path, config=config, balance_data=balance_data)

    x_train_rob = dict()
    y_train_rob = dict()
    idx_train_rob = dict()
    x_test_rob = dict()
    y_test_rob = dict()
    idx_test_rob = dict()
    if config.load_mass_data:
        print('\nMass data:')
        _, x_train_rob['mass'], x_test_rob['mass'], _, y_train_rob['mass'], y_test_rob['mass'], _, idx_train_rob['mass'], idx_test_rob['mass'], _ = \
            datahandler.load_data(data_path=config.data_folder_path+config.dataset_mass_name, config=config)
    if config.load_tire_data:
        print('\nTire data:')
        _, x_train_rob['tire'], x_test_rob['tire'], _, y_train_rob['tire'], y_test_rob['tire'], _, idx_train_rob['tire'], idx_test_rob['tire'], _ = \
            datahandler.load_data(data_path=config.data_folder_path+config.dataset_tire_name, config=config)

    # reduce initial dataset to have a fair influence of mass and tire data
    if config.load_mass_data or config.load_tire_data:
        if config.combine_training_data:
            if config.balance_training_datasets:
                minimumTrainObsPerDataset = 99999
                minimumTrainObsPerDataset = 99999
                for key in x_train_rob.keys():
                    if x_train_rob[key].shape[0] < minimumTrainObsPerDataset:
                        minimumTrainObsPerDataset = x_train_rob[key].shape[0]
                # limit initial datasets to balanced amount of observations
                x_train = x_train[:minimumTrainObsPerDataset, :]
                y_train = y_train[:minimumTrainObsPerDataset]
                idx_train = idx_train[:minimumTrainObsPerDataset]
                # add balanced amount of observations robustness datasets to train and test dataset
                for key in x_train_rob.keys():
                    if key in config.combine_training_dataset:
                        x_train = np.append(x_train, x_train_rob[key][:minimumTrainObsPerDataset, :], axis=0)
                        y_train = np.append(y_train, y_train_rob[key][:minimumTrainObsPerDataset], axis=0)
                        idx_train = np.append(idx_train, idx_train_rob[key][:minimumTrainObsPerDataset], axis=0)
            else:
                # add complete robustness datasets to train dataset
                for key in x_train_rob.keys():
                    if key in config.combine_training_dataset:
                        x_train = np.append(x_train, x_train_rob[key], axis=0)
                        y_train = np.append(y_train, y_train_rob[key], axis=0)
                        idx_train = np.append(idx_train, idx_train_rob[key], axis=0)

        if config.balance_testing_datasets:
            minimumTestObsPerDataset = 99999
            for key in x_test_rob.keys():
                if x_test_rob[key].shape[0] < minimumTestObsPerDataset:
                    minimumTestObsPerDataset = x_test_rob[key].shape[0]
            # limit initial datasets to balanced amount of observations
            x_test = x_test[:minimumTestObsPerDataset, :]
            y_test = y_test[:minimumTestObsPerDataset]
            idx_test = idx_test[:minimumTestObsPerDataset]

            for key in x_test_rob.keys():
                x_test_rob[key] = x_test_rob[key][:minimumTestObsPerDataset, :]
                y_test_rob[key] = y_test_rob[key][:minimumTestObsPerDataset]
                idx_test_rob[key] = idx_test_rob[key][:minimumTestObsPerDataset]

    # add scaler data to traning dataset
    x_scal = [] # datahandler.read_file('dataset_scal', config.data_path, file_format='npy')

    print(' - done! Duration: %.2fs' %(time()-start_time_data_import))

    if config.mode != 'ev':

        start_time_data_selection = time()
        print('[INFO] Selecting data and build pipe', end='')

        # init pipeline process
        pipe = inits.init_pipe(config)

        # Init model with standard params
        model = inits.init_model(config.model, **vars(config))
        model.set_params()

        print(' - done! Duration: %.2fs' %(time()-start_time_data_selection))

        print('Number of training observations: %d' % x_train.shape[0])
        print('Number of testing observations: %d' % x_test.shape[0])

    else:
        print('Number of observations in dataset: %d' % x.shape[0])

    # OPTIMIZATION OF HYPERPARAMETERS
    if config.mode == 'op':

        print('[INFO] OPTIMIZE HYPERPARAMETERS')

        # get parameter space to be optimized using random search
        # params_grid = model.get_params_grid()

        # define hyperparameter search space
        params_grid = {}
        # make list for layer config
        layer_config = [
            [l_1, l_2, l_3, l_2, l_1] for l_1 in np.arange(200, 400, 20) \
            for l_2 in np.arange(80, 250, 20) for l_3 in np.arange(10, 50, 5) \
            if l_1 > l_2 > l_3]
        # layer_config = [
        #     [l_1, l_3, l_1] for l_1 in np.arange(10, 300, 20) \
        #     for l_3 in np.arange(5, 50, 5) \
        #     if l_1 > l_3]
        params_grid['layer_config'] = layer_config

        params_space = list(ParameterSampler(params_grid, config.n_opti_iter, 42))
        n_param_space = len(params_space)
        score = []

        # transform data using pipe for preprocessing
        print(' - Preprocessing steps: %s' % (config.prep_options))
        pipe.fit(x_train)
        x_train = pipe.transform(x_train)

        # cv data splits as list, val data is homogen of neg/pos class
        x_train_cv , x_val_cv, y_train_cv, y_val_cv, idx_train_cv, idx_val_cv = frwk.make_data_cv(
            x_train, y_train, idx_train, config)

        start_time_par_loop = time()

        print('[INFO] Start parallel loop for fast data transformation')
        print(' - Preprocessing steps:%s' %(config.prep_options))


        print(' - done! Duration: %.2fs' %(time()-start_time_par_loop))

        score = [None]*n_param_space

        func = partial(frwk.single_optimization_run, params_space=params_space, model=model, config=config,
                       x_train_cv=x_train_cv, y_train_cv=y_train_cv, idx_train_cv=idx_train_cv,
                       x_val_cv=x_val_cv, y_val_cv=y_val_cv, idx_val_cv=idx_val_cv)

        # use parallel computation
        with concurrent.futures.ProcessPoolExecutor() as executor:  # ThreadPoolExecutor ProcessPoolExecutor
            for scores_single_opt, i_iter in executor.map(func, range(0, n_param_space)):
                score[i_iter] = scores_single_opt

        # loop for optimization
        # for i_iter, params in enumerate(params_space):

            # start_time_opti = time()
            # print('[INFO] Start iteration no %i/%i' %(i_iter, n_param_space))
            #
            # # set model parameter of this optimization iteration
            # model.set_params(**params)
            # print(' - Model parameters: %s' %(str(model.get_params())))
            #
            # time_fit = time()
            # print('[INFO] Start cv runs')
            #
            # # train model in cross validation loop
            # metrics_hist, _ =  frwk.fit_cv(
            #     model, x_train_cv, y_train_cv, x_val_cv, y_val_cv,
            #     ['RocCurve', 'EarlyStopping'], config)
            #
            # print('[INFO] All cv runs done! Duration: %.2fs' %(time()-time_fit))
            #
            # metrics = frwk.eval_metrics_cv(metrics_hist)
            #
            # # append score to optimization history
            # score.append(metrics['roc_auc'])
            #
            # print('[INFO] Iteration no %i - done! Duration %.2fs'
            #       %(i_iter, time()-start_time_opti))
            # print('[INFO] Best ROC AUC score so far: %.2f%%'
            #       %(max(score)*100))

        print('[INFO] Best ROC AUC score so far: %.2f%%' %(max(score)*100))

        # evaluate hyperparameter settings
        opti_sort_idx = np.argsort(score)[::-1] # high to low order
        params_sort = [params_space[index] for index in opti_sort_idx]
        roc_auc_sort = [score[index] for index in opti_sort_idx]
        # best parameters and belonging roc auc value
        params_best = params_sort[0]
        roc_auc_best = roc_auc_sort[0]

        # save parameters and their results to .csv-file
        with open(config.log_path+'/params_space.txt', 'w+') as f:
            f.write('Params of model: %s' %(str(model.get_params())) + '\n')
            f.write('Ranking:\n')
            for i in range(len(params_sort)):
                f.write('%i.\tROC AUC %0.2f\t' %((i+1), roc_auc_sort[i]*100))
                f.write('parameter space: %s' %(str(params_sort[i])) + '\n')

        print('[INFO] Optimization finished - duration: %.2f' %(time()-start_time))
        print(' - Max ROC AUC: %.2f%%' %(roc_auc_best*100))
        print(' - Hyperparameter setting: ', params_best)
        print('[END]')

    # CROSS VALIDATION RUN
    elif config.mode == 'cv':

        print('[INFO] CROSS-VALIDATION RUN')

        # transform data using pipe for preprocessing
        print(' - Preprocessing steps: %s' % (config.prep_options))
        pipe.fit(x_train)
        x_train = pipe.transform(x_train)

        # Cv data splits as list, val data is homogen of neg/pos class
        x_train_cv , x_val_cv, y_train_cv, y_val_cv, idx_train_cv, idx_val_cv = frwk.make_data_cv(
            x_train, y_train, idx_train, config)

        start_time_transform= time()

        print(' - done! Duration: %.2fs' %(time()-start_time_transform))

        time_fit = time()
        print('[INFO] Start cv runs')

        # fit model in cross validation loop (build + train each run)
        metrics_hist, trained_model = frwk.fit_cv(
            model, x_train_cv, y_train_cv, idx_train_cv, x_val_cv, y_val_cv, idx_val_cv,
            ['RocCurve'], config) #, 'EarlyStopping' , 'ProgbarLogger'

        # get different metrics derived from complete metrics history
        metrics = frwk.eval_metrics_cv(metrics_hist)

        print('[INFO] CROSS-VALIDATION RUN - done! Duration: %.2fs' \
              %(time()-time_fit))

        print('[INFO] Evaluate and plot metrics')
        print(' - Mean ROC AUC: %.2f%% \xb1%.2f%% (1 std. dev)'
              %(metrics['roc_auc']*100, metrics['roc_auc_std']*100))
        print(' - Max ROC AUC: %.2f%%' %(max(metrics['roc_auc_all']*100)))
        print(' - ROC AUC of all runs: ', end='')
        [print('%.2f%%' %(i*100), end='  ') for i in metrics['roc_auc_all']]
        print('')

        # plot different metrics
        # loss history
        cplt.plot_loss_hist(metrics['loss_hist'], metrics['loss_hist_std'], config)
        # ROC AUC history
        cplt.plot_roc_hist(
            metrics['roc_auc_hist'], metrics['roc_auc_hist_std'], config)
        # ROC curve
        cplt.plot_roc_curve(
            metrics['roc_auc'], metrics['tprs'], metrics['fprs'],
            metrics['roc_auc_std'], metrics['tprs_std'], config)

        if config.save_log:
            with open(config.log_path + '/config.args', 'w+') as file:
                file.write(str(config))

        print('[END] Duration: %.2fs' \
              %(time()-start_time))

        ## test data
        # now, evaluate performance on test data
        print('[INFO] Now, evaluate performance on test data')
        if isinstance(trained_model, list):
            for cntCV, single_trained_model in enumerate(trained_model):
                frwk.test_classifier(x_test, y_test, idx_test, config, single_trained_model, 'test', cntCV, pipe=pipe)
        else:
            frwk.test_classifier(x_test, y_test, idx_test, config, trained_model, 'test', 0, pipe=pipe)

        if config.load_mass_data or config.load_tire_data:
            for key in x_test_rob.keys():
                # now, evaluate performance on test data
                print('[INFO] evaluate performance on %s data' % key)
                if isinstance(trained_model, list):
                    for cntCV, single_trained_model in enumerate(trained_model):
                        frwk.test_classifier(x_test_rob[key], y_test_rob[key], idx_test_rob[key], config, single_trained_model, key, cntCV, pipe=pipe)
                else:
                    frwk.test_classifier(x_test_rob[key], y_test_rob[key], idx_test_rob[key], config, trained_model, key, 0, pipe=pipe)

    # PLAIN RUN
    elif config.mode == 'pr':

        print('[INFO] PLAIN RUN')

        # data splits, val data is homogen of neg/pos class
        x_train , x_val, y_train, y_val, idx_train, idx_val = frwk.make_data_pr(
            x_train, y_train, idx_train, config)

        start_time_transform = time()

        print('[INFO] Data Transformation')
        print(' - Preprocessing steps:%s' %(config.prep_options))

        # transform data using pipe for preprocessing
        x_train, x_val, pipe = frwk.transform_data_pr(pipe, x_train,
                                                x_val, x_scal=x_scal)

        print(' - duration: %.2fs' %(time()-start_time_transform))

        # analyse transformed data
        if config.analyse_data:
            frwk.analyse_data(x_train, x_val[y_val==0], x_val[y_val==1],
                                   'datmoda_train', config)

        time_fit = time()
        print('[INFO] Start plain run')

        # fit model in plain run (build + train)
        model.fit(x_train, y_train, x_val, y_val,
                  ['RocCurve', 'BestModelCheckpoint', 'Blogger', 'TensorBoard'],
                   #'ProgbarLogger'],
                  model_path=config.model_path, i_cv=0)

        # get metrics of each run with complete metrics history
        metrics_hist = model.get_history()
        metrics = frwk.eval_metrics_pr(metrics_hist)

        print('[INFO] Plain run - done! Duration: %.2fs' %(time()-time_fit))

        print('[INFO] Evaluate and plot metrics')
        print(' - ROC AUC (val data): %.2f%%' %(metrics['roc_auc']*100))

        # loss history
        cplt.plot_loss_hist(metrics['loss_hist'], 0, config)
        # ROC AUC history
        cplt.plot_roc_hist(metrics['roc_auc_hist'], 0, config)
        # ROC curve
        cplt.plot_roc_curve(metrics['roc_auc'], metrics['tprs'],
                            metrics['fprs'], 0, 0, config)

        # best weights are saved during training
        model.model_.set_weights(model.best_model_weights_)

        analyze_model.plot_vae_weights(model, config)

        if config.save_log:
            with open(config.log_path + '/config.csv', 'w+') as file:
                file.write(str(config))

        if config.save_model:
            print('[INFO] Save model and pipeline')
            # save only keras model
            model.model_.save(config.model_path+'/model.h5')

            # save model name for later init of model
            with open(config.model_path + '/model.csv', 'w+') as file:
                file.write(str(config.model))

            # save pipe using joblib
            dump(pipe, config.model_path+'/pipe.h5')
            print(' - saved successfully')

        # model.plot(config.log_path)

        ## evaluate validiation data
        # now, evaluate performance on test data
        print('[INFO] Now, evaluate performance on validiation data')
        if isinstance(model, list):
            for cntCV, single_trained_model in enumerate(model):
                frwk.test_classifier(x_val, y_val, idx_val, config, single_trained_model, 'val', cntCV, pipe=None)
        else:
            frwk.test_classifier(x_val, y_val, idx_val, config, model, 'val', 0, pipe=None)

        ## test data
        # now, evaluate performance on test data
        print('[INFO] Now, evaluate performance on test data')
        if isinstance(model, list):
            for cntCV, single_trained_model in enumerate(model):
                frwk.test_classifier(x_test, y_test, idx_test, config, single_trained_model, 'test', cntCV, pipe=pipe)
        else:
            frwk.test_classifier(x_test, y_test, idx_test, config, model, 'test', 0, pipe=pipe)

        if config.load_mass_data or config.load_tire_data:
            for key in x_test_rob.keys():
                # now, evaluate performance on test data
                print('[INFO] evaluate performance on %s data' % key)
                if isinstance(model, list):
                    for cntCV, single_trained_model in enumerate(model):
                        frwk.test_classifier(x_test_rob[key], y_test_rob[key], idx_test_rob[key], config,
                                             single_trained_model, key, cntCV, pipe=pipe)
                else:
                    frwk.test_classifier(x_test_rob[key], y_test_rob[key], idx_test_rob[key], config, model,
                                         key, 0, pipe=pipe)

        print('[END]')


    elif config.mode == 'ev':

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

        print('[INFO] Load fitted sklearn pipeline')

        pipe = load(config.model_path + '/pipe.h5')
        # Use pipe to transform the data to evaluated
        # print('[INFO] Transform evaluation dataset')
        # x_ = pipe.transform(x)
        #
        # print('[INFO] Use model to detect novelties')
        #
        # # Use the roc auc metric to evaluate performance on data
        # roc_auc_eval = model.score(x_, y)
        # print('[INFO] Evaluation')
        # print(' - ROC AUC : %.2f%%' %(roc_auc_eval*100))

        print('[INFO] Now, evaluate performance on test data')
        if isinstance(model, list):
            for cntCV, single_trained_model in enumerate(model):
                frwk.test_classifier(x, y, idx, config, single_trained_model, 'test', cntCV, pipe=pipe)
        else:
            frwk.test_classifier(x, y, idx, config, model, 'test', 0, pipe=pipe)

    else: raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()