#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: framework.py -----------------------------------------------------------------------------
# Framework for data handling, model training and evaluation.
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

import sys, os
import numpy as np
from time import time
from joblib import Parallel, delayed, parallel_backend

from scipy import interp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn import metrics

import matplotlib.pyplot as plt

# custom packages
import scripts.utilities.plot as cplt

def current_path():
    # get current PYTHONPATH
    value = os.getcwd()
    print(value)
    return value

class Logger(object):

    # change stdout that terminal output is printed and also saved in log file
    def __init__(self, config):
        self.stdout_old = sys.stdout
        self.terminal = sys.stdout
        self.log = open(os.path.join(config.log_path, 'log.txt'), 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        # needed for python 3 compatibility, handles the flush command by
        # doing nothing
        pass


def select_data(x, y, config, idx=None, balance_data=True):

    y = np.reshape(y, -1)

    if idx is None:
        idx = np.arange(x.shape[0])

    # get neg class
    x_neg = x[y==0]
    idx_neg = idx[y==0]

    # select pos class and shuffle examples
    if config.pos_class == 111:
        x_pos, idx_pos = shuffle(x[y != 0], idx[y != 0], random_state=config.random)
    else:
        x_pos, idx_pos = shuffle(x[y == config.pos_class], idx[y == config.pos_class], random_state=config.random)

    # if config.pos_class == 111:
    #     x_pos, idx_pos = shuffle(x[y!=0], idx[y!=0], random_state=config.random)
    # elif config.pos_class == 100:
    #     x_pos = shuffle(x[y==100], random_state=config.random)
    # elif config.pos_class == 101:
    #     x_pos = shuffle(x[y==101], random_state=config.random)
    # elif config.pos_class == 104:
    #     x_pos = shuffle(x[y==104], random_state=config.random)
    # else:
    #     raise ValueError('Possible pos classes are: 111, 100, 101 and 104')

    if balance_data:
        if len(x_neg) < len(x_pos):
            x_pos = x_pos[:len(x_neg)]
            idx_pos = idx_pos[:len(x_neg)]
        elif len(x_neg) > len(x_pos):
            x_neg = x_neg[:len(x_pos)]
            idx_neg = idx_neg[:len(x_pos)]

    # make labels: no defect is neg class, defect is pos class
    y_neg = np.zeros(len(x_neg))
    y_pos = np.ones(len(x_pos))

    # build a single (homogen) dataset
    x = np.append(x_neg, x_pos, axis=0)
    y = np.append(y_neg, y_pos, axis=0)
    idx = np.append(idx_neg, idx_pos, axis=0)

    return x, y, idx


def make_data_pr(x, y, idx, config):

    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x, y, idx, test_size=config.test_size, shuffle=True,
        random_state=config.random)

    # trainings data consist only of neg class
    x_train = x_train[y_train==0]
    idx_train = idx_train[y_train == 0]
    y_train = y_train[y_train==0]

    return x_train, x_val, y_train, y_val, idx_train, idx_val


def make_data_cv(x, y, idx, config):

    idx_cv = StratifiedKFold(n_splits=config.cv_splits,
                             shuffle=True, random_state=config.random)

    x_train_cv, x_val_cv, y_train_cv, y_val_cv, idx_train_cv, idx_val_cv = [], [], [], [], [], []

    for idx_train_split, idx_val_split in idx_cv.split(x, y):

        x_train = x[idx_train_split]
        y_train = y[idx_train_split]
        idx_train = idx[idx_train_split]

        # trainings data of each run in one list
        x_train_cv.append(x_train[y_train==0])
        y_train_cv.append(y_train[y_train==0])
        idx_train_cv.append(idx_train[y_train==0])

        # validation data of each run in one list
        x_val_cv.append(x[idx_val_split])
        y_val_cv.append(y[idx_val_split])
        idx_val_cv.append(idx[idx_val_split])

    return x_train_cv, x_val_cv, y_train_cv, y_val_cv, idx_train_cv, idx_val_cv

def transform_data_pr(pipe, x_train, x_val, x_scal=[]):

    if x_scal != []: x_scal = np.append(x_train, x_scal, axis=0)
    else: x_scal = x_train

    # Fit transformer to trainings data (+ specific scaling data)
    pipe.fit(x_scal)

    # Transform with fitted params
    x_train = pipe.transform(x_train)
    x_val = pipe.transform(x_val)

    return x_train, x_val, pipe

def transform_data_cv(pipe, x_train_cv, x_val_cv, cv_splits, x_scal=[]):

    pipe_cv = []

    # use joblib parallel loop for fast calculation
    try:
        # set backend to 'pickle' which is faster than 'cloudpickle'
        with parallel_backend('multiprocessing'):
            para_res = Parallel(n_jobs=-1, verbose=0) \
                (delayed(transform_data_pr)
                (pipe, x_train_cv[i_cv], x_val_cv[i_cv], x_scal=x_scal)
                for i_cv in range(cv_splits))
    except:
        # if 'multiprocessing' backend not available, use standard backend
        with parallel_backend('loky'):
            para_res = Parallel(n_jobs=-1, verbose=0) \
                (delayed(transform_data_pr)
                (pipe, x_train_cv[i_cv], x_val_cv[i_cv], x_scal=x_scal)
                for i_cv in range(cv_splits))

    # unzip result / using zip() more canonical, but produces tuples
    x_train_cv = [i[0] for i in para_res]
    x_val_cv = [i[1] for i in para_res]
    pipe_cv = [i[2] for i in para_res]

    return x_train_cv, x_val_cv, pipe_cv

def fit_pr(model, x_train, x_val, y_train, y_val):
    pass

def fit_cv(model, x_train_cv, y_train_cv, idx_train_cv, x_val_cv, y_val_cv, idx_val_cv,
           callbacks, config):

    metrics_hist = []
    trained_model = []

    for i_cv in range(config.cv_splits):

        start_time_cv = time()

        print('[INFO] Start cv run no. %i' %(i_cv))

        # get data of current cv run
        x_train, x_val = x_train_cv[i_cv], x_val_cv[i_cv]
        y_train, y_val = y_train_cv[i_cv], y_val_cv[i_cv]
        idx_train, idx_val = idx_train_cv[i_cv], idx_val_cv[i_cv]
    
        # analyse transformed data
        if config.analyse_data:
            analyse_data(x_train, x_val[y_val==0], x_val[y_val==1],
                         'data_train', config)
    
        # fit model in current cv run (build + train)
        model.fit(x_train, y_train, x_val, y_val, callbacks)

        trained_model.append(model)

        # get metrics of each run with complete metrics history
        metrics_hist.append(model.get_history())

        ## evaluate validiation data
        # now, evaluate performance on valdiation data
        print('[INFO] Now, evaluate performance on validiation data')
        test_classifier(x_val, y_val, idx_val, config, model, 'val', i_cv, pipe=None)

        print('[INFO] Cv run no. %i - done! Duration: %.2fs' \
              %(i_cv, (time()-start_time_cv)))

    return metrics_hist, trained_model


def analyse_data(train, val_neg, val_pos, fitted_on, config):

    data =[train, val_neg, val_pos]
    name_list = ['train', 'val_neg', 'val_pos']

    data_mean, data_std = [], []

    for i in range(len(data)):
        data_mean.append(np.mean(data[i], axis=0))
        data_std.append(np.std(data[i], axis=0))

    cplt.plot_analyse_data(name_list, data_mean, data_std, fitted_on, config)

def eval_metrics_pr(metrics_all):

    metrics = {}

    metrics['loss_hist'] = metrics_all['loss']
    metrics['roc_auc_hist'] = metrics_all['roc_auc']

    # Get max roc auc
    metrics['roc_auc'] = np.max(metrics_all['roc_auc'])
    roc_auc_idx = np.argmax(metrics_all['roc_auc'])

    metrics['tprs'] = metrics_all['tprs'][roc_auc_idx]
    metrics['fprs'] = metrics_all['fprs'][roc_auc_idx]

    return metrics

def eval_metrics_cv(metrics):
    
    # dict for all mean metrics
    mean_metrics = {}

    # loss evaluation
    
    loss_all = flatten_list_dic(metrics, 'loss')
    # different length of list entries due to early stopping
    length_red = min([np.shape(loss)[0] for loss in loss_all])
    loss_all = [i[:length_red] for i in loss_all]
    mean_metrics['loss_hist'] = np.mean(loss_all, axis=0)
    mean_metrics['loss_hist_std'] = np.std(loss_all, axis=0)

    # ROC AUC evaluation

    roc_auc_hist_all = flatten_list_dic(metrics, 'roc_auc')
    # evaluate hist of runs (diff len of list due to EarlyStopping)
    length_red = min([np.shape(i)[0] for i in roc_auc_hist_all])
    roc_auc_hist_red = [i[:length_red] for i in roc_auc_hist_all]

    mean_metrics['roc_auc_hist'] = np.mean(roc_auc_hist_red, axis=0)
    mean_metrics['roc_auc_hist_std'] = np.std(roc_auc_hist_red, axis=0)

    # evaluate best ROC AUC of each run
    roc_auc_max_all = [np.max(i) for i in roc_auc_hist_all]
    roc_auc_max_idx = [np.argmax(i) for i in roc_auc_hist_all]
    mean_metrics['roc_auc'] = np.mean(roc_auc_max_all, axis=0)
    mean_metrics['roc_auc_std'] =np.std(roc_auc_max_all, axis=0)
    mean_metrics['roc_auc_all'] = np.array(roc_auc_max_all)

    # TPRS evaluation of best roc auc of each run
    tprs_hist_all = flatten_list_dic(metrics, 'tprs')
    tprs_all = [tprs[roc_auc_max_idx[i]] for i, tprs in enumerate(tprs_hist_all)]
    mean_metrics['tprs'] = np.mean(tprs_all, axis=0)
    mean_metrics['tprs_std'] = np.std(tprs_all, axis=0)

    # FPRS evaluation
    fprs_hist_all = flatten_list_dic(metrics, 'fprs')
    fprs_all = [fprs[roc_auc_max_idx[i]] for i, fprs in enumerate(fprs_hist_all)]
    mean_metrics['fprs'] = np.mean(fprs_all, axis=0)

    return mean_metrics

def calc_roc_curve(y, y_score):

    # get roc (auc)
    fprs, tprs, thresholds = metrics.roc_curve(y, np.nan_to_num(y_score), pos_label=1)
    roc_auc = metrics.auc(fprs, tprs)

    # interpolate (100 points)
    fprs_int = np.linspace(0, 1, 100)
    tprs_int = interp(fprs_int, fprs, tprs)
    tprs_int[0] = 0.0
    thresholds_int = interp(fprs_int, fprs, thresholds)

    return roc_auc, fprs_int, tprs_int, thresholds_int

def calc_rmse(a, b, axis=1):
    return np.sqrt(np.mean(np.square(a - b), axis=axis))

def calc_dm(x, x_mean, x_cov):
    # calculate mahalanobis distance
    dm = np.sqrt(np.diag(np.matmul(np.matmul((x-x_mean), np.linalg.inv(x_cov)), 
                                   (x-x_mean).transpose())))
    return dm

def flatten_list_dic(list_dic, topic):

    flat_dic = []
    for dic in list_dic:
        for key in dic:
            if key == topic:
                flat_dic.append(dic[key])
    return flat_dic


def test_classifier(x_test, y_test, idx_test, config, model, nameTestset, cntCV, pipe=None):
    # transform test data
    if pipe is not None:
        x_test_pr = pipe.transform(x_test)
    else:
        x_test_pr = x_test
    y_test_pr = y_test
    idx_test_pr = idx_test

    # get novelty score of predicted data to calculate ROC metric
    nov_score = model.nov_score(x_test_pr)
    roc_auc, fprs_t, tprs_t, thresholds_t = calc_roc_curve(
        y_test_pr, nov_score)

    print('- ROC AUC (%s data, fold %d): %.2f%%' % (nameTestset, cntCV, roc_auc * 100))

    if config.save_scores:
        # save prediction scores and labels for final test procedure
        if not os.path.isdir(config.log_path + '/result/'+nameTestset+'/cv'+str(cntCV)):
            os.makedirs(config.log_path + '/result/'+nameTestset+'/cv'+str(cntCV))
        np.savetxt(config.log_path + '/result/'+nameTestset+'/cv'+str(cntCV)+'/PredictionScores.csv', nov_score, fmt='%.4f', delimiter=",")
        np.savetxt(config.log_path + '/result/'+nameTestset+'/cv'+str(cntCV)+'/TrueLabels.csv', y_test_pr, fmt='%d', delimiter=",")
        np.savetxt(config.log_path + '/result/'+nameTestset+'/cv'+str(cntCV)+'/Index.csv', idx_test_pr, fmt='%d', delimiter=",")

    # ROC curve
    if config.mode is 'pr':
        cplt.plot_roc_curve(roc_auc, tprs_t, fprs_t, 0, 0, config)
        cplt.plot_nov_score_density(nov_score, y_test_pr, config)
        cplt.plot_reconstructed_data(x_test_pr, model.predict(x_test_pr), seq_length=config.nfft/2, n_sensors=config.n_sensors, i_example=0, config=config)

    return nov_score, roc_auc


def single_optimization_run(i_iter, params_space, model, config, x_train_cv, y_train_cv, idx_train_cv, x_val_cv, y_val_cv, idx_val_cv,):
    # start_time_opti = time()
    print('[INFO] Start iteration no %i' % (i_iter))

    # set model parameter of this optimization iteration
    params = params_space[i_iter]
    model.set_params(**params)


    # time_fit = time()
    # print('[INFO] Start cv runs', file=printfile)

    # train model in cross validation loop
    metrics_hist, _ = fit_cv(
        model, x_train_cv, y_train_cv, idx_train_cv, x_val_cv, y_val_cv, idx_val_cv,
        ['RocCurve', 'EarlyStopping'], config)

    # print('[INFO] All cv runs done! Duration: %.2fs' % (time() - time_fit))

    metrics = eval_metrics_cv(metrics_hist)

    print('Model parameters: %s, AUC: %.2f' % (str(model.get_params()), metrics['roc_auc']))

    # append score to optimization history
    # score.append(metrics['roc_auc'])

    # print('[INFO] Iteration no %i - done! Duration %.2fs'
    #       % (i_iter, time() - start_time_opti))
    # print('[INFO] Best ROC AUC score so far: %.2f%%'
    #       % (max(score) * 100))

    return metrics['roc_auc'], i_iter


if __name__ == '__main__':
    pass
