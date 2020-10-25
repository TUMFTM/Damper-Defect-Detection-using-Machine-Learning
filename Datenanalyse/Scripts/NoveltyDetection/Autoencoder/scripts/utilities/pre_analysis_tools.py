#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: pre_analysis_tools.py
# -----------------------------------------------------------------------------
# Tools for analysis of data before training
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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import tikzplotlib as tikz

class Data_analysis(object):
    def __init__(self, data, name):
        self._data = data
        self._name = name

def save_plot(config, title):
    analysis_path = config.datapath + 'analysis'
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)

    plt.savefig(analysis_path+'/'+title+'.pdf', bbox_inches='tight')
    #tikz.save(analysis_path+'/'+title+'.tikz', encoding='utf8')

def analyse_examples(data, config, data_name, fitted_on):
    # reduce each feature to characteristic values
    mean_exam_all = np.mean(data, axis=1)
    mean_exam = np.mean(mean_exam_all)

    std_exam_all = np.std(data, axis=1)
    std_exam = np.mean(std_exam_all)

    max_exam_all = np.max(data, axis=1)
    max_exam = np.mean(max_exam_all)

    min_exam_all = np.min(data, axis=1)
    min_exam = np.mean(min_exam_all)

    range_q90_exam_all = np.quantile(data, 0.95, axis=1)-np.quantile(data, 0.05, axis=1)
    range_q90_exam = np.mean(range_q90_exam_all)

     # make plot with hist of value range
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), dpi=300, sharey='row')
    bins = 30

    # subplot 0
    ax[0,0].hist(mean_exam_all, bins=bins, label='Calculated for each example')
    ax[0,0].axvline(x=mean_exam, label='Mean over all examples', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[0,0].set(xlabel='Mean', ylabel='Quantity', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[0,0].legend()

    # subplot 1
    ax[0,1].hist(std_exam_all, bins=bins, label='Calculated for each example')
    ax[0,1].axvline(x=std_exam, label='Mean over all examples', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[0,1].set(xlabel='Standard deviation', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[0,1].legend()

    # subplot 2
    ax[1,0].hist(min_exam_all, bins=bins, label='Min for each example', color='#CCCCC6')
    ax[1,0].hist(max_exam_all, bins=bins, label='Max for each example')
    ax[1,0].axvline(x=min_exam, label='Mean min over all examples', color='#808080', ls='--')
    ax[1,0].axvline(x=max_exam, label='Mean max over all examples', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[1,0].set(xlabel='Values', ylabel='Quantity', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[1,0].legend()

    # subplot 3
    ax[1,1].hist(range_q90_exam_all, bins=bins, label='Calculated for each example')
    ax[1,1].axvline(x=range_q90_exam, label='Mean over all examples', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[1,1].set(xlabel='Quantile-90 range', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[1,1].legend()

    save_plot(config, title='analysis_example_%s_fitted_on_%s'%(data_name, fitted_on))

    return mean_exam, std_exam, max_exam, min_exam, range_q90_exam

def analyse_features(data, config, data_name, fitted_on):
    # reduce each feature to characteristic values
    mean_feat_all = np.mean(data, axis=0)
    mean_feat = np.mean(mean_feat_all)

    std_feat_all = np.std(data, axis=0)
    std_feat = np.mean(std_feat_all)

    max_feat_all = np.max(data, axis=0)
    max_feat = np.mean(max_feat_all)

    min_feat_all = np.min(data, axis=0)
    min_feat = np.mean(min_feat_all)

    range_q90_feat_all = np.quantile(data, 0.95, axis=0)-np.quantile(data, 0.05, axis=0)
    range_q90_feat = np.mean(range_q90_feat_all)

     # make plot with hist of value range
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), dpi=300, sharey='row')
    bins = 30

    # subplot 0
    ax[0,0].hist(mean_feat_all, bins=bins, label='Calculated for each feature')
    ax[0,0].axvline(x=mean_feat, label='Mean over all features', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[0,0].set(xlabel='Mean', ylabel='Quantity', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[0,0].legend()

    # subplot 1
    ax[0,1].hist(std_feat_all, bins=bins, label='Calculated for each feature')
    ax[0,1].axvline(x=std_feat, label='Mean over all features', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[0,1].set(xlabel='Standard deviation', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[0,1].legend()

    # subplot 2
    ax[1,0].hist(min_feat_all, bins=bins, label='Min for each feature', color='#CCCCC6')
    ax[1,0].hist(max_feat_all, bins=bins, label='Max for each feature')
    ax[1,0].axvline(x=min_feat, label='Mean min over all features', color='#808080', ls='--')
    ax[1,0].axvline(x=max_feat, label='Mean max over all features', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[1,0].set(xlabel='Values', ylabel='Quantity', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[1,0].legend()

    # subplot 3
    ax[1,1].hist(range_q90_feat_all, bins=bins, label='Calculated for each feature')
    ax[1,1].axvline(x=range_q90_feat, label='Mean over all features', color='#808080')
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))
    ax[1,1].set(xlabel='Quantile-90 range', title='Analysis of [%s]\n(scaler fitted on [%s])'%(data_name, fitted_on))
    ax[1,1].legend()

    save_plot(config, title='analysis_feature_%s_fitted_on_%s'%(data_name, fitted_on))

    return mean_feat, std_feat, max_feat, min_feat, range_q90_feat

def plot_data_hist(data_normal, data_1, data_2, config, data_name, fitted_on):

    data = [data_normal, data_1, data_2]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6,12), dpi=300, sharey='row')

    bins = 30

    for i, ax in enumerate(axs.flat):
        ax.hist(data[i], bins=bins, label='Fitted on '+fitted_on[i])
        ax.set(xlabel='Feature Value', ylabel='Quantity', title='Analysis of [%s]'%(data_name))
        ax.legend()

    save_plot(config, title='analysis_feature_hist_%s'%(data_name))

def analyse_spectrum(train, val_neg, val_pos, config, fitted_on):

    train_mean = np.mean(train, axis=0)
    val_neg_mean = np.mean(val_neg, axis=0)
    val_pos_mean = np.mean(val_pos, axis=0)

    train_std = np.std(train, axis=0)
    val_neg_std = np.std(val_neg, axis=0)
    val_pos_std = np.std(val_pos, axis=0)


    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=300, sharey='row')

    axs[0].plot(train_mean, color='k', label='Mean')
    axs[1].plot(val_neg_mean, ls='--', color='g', label='Mean')
    axs[2].plot(val_pos_mean, ls=':', color='b', label='Mean')

    axs[0].fill_between(np.arange(len(train_mean)),
                    train_mean+train_std,
                    train_mean-train_std,
                    color='k', alpha=0.3,
                    label='$\pm$ std dev')

    axs[1].fill_between(np.arange(len(val_neg_mean)),
                    val_neg_mean+val_neg_std,
                    val_neg_mean-val_neg_std,
                    color='g', alpha=0.3,
                    label='$\pm$ std dev')


    axs[2].fill_between(np.arange(len(val_pos_mean)),
                    val_pos_mean+val_pos_std,
                    val_pos_mean-val_pos_std,
                    color='b', alpha=0.3,
                    label='$\pm$ std dev')


    axs[0].set(xlabel='Feature', ylabel='Value', ylim=[-3,3], title='Analysis FFT-spectrum of [data_train]\n(scaler fitted on [%s])'%(fitted_on))
    axs[1].set(xlabel='Feature', ylabel='Value', title='Analysis FFT-spectrum of [data_val_neg]\n(scaler fitted on [%s])'%(fitted_on))
    axs[2].set(xlabel='Feature', title='Analysis FFT-spectrum of [data_val_pos]\n(scaler fitted on [%s])'%(fitted_on))
    axs[0].legend()
    axs[1].legend()
    axs[1].legend()

    save_plot(config, title='analysis_spectrum_fitted on_%s'%(fitted_on))

def analyse_spectrum_v2(val_neg, val_pos, config, fitted_on):

    val_neg_mean = np.mean(val_neg, axis=0)
    val_pos_mean = np.mean(val_pos, axis=0)

    val_neg_std = np.std(val_neg, axis=0)
    val_pos_std = np.std(val_pos, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=300, sharey='row')
    ax.plot(val_neg_mean, color='k', label='Mean of [data_val_neg]')
    ax.plot(val_pos_mean, ls='--', color='b', label='Mean of [data_val_pos]')

    ax.fill_between(np.arange(len(val_neg_mean)),
                    val_neg_mean+val_neg_std,
                    val_neg_mean-val_neg_std,
                    color='k', alpha=0.3,
                    label='Std dev [data_val_neg]')
    ax.fill_between(np.arange(len(val_pos_mean)),
                    val_pos_mean+val_pos_std,
                    val_pos_mean-val_pos_std,
                    color='b', alpha=0.3,
                    label='Std dev [data_val_pos]')

    ax.set(xlabel='Feature', ylabel='Value', title='Analysis of FFT-spectrum\n(scaler fitted on [%s])'%(fitted_on))


def analyse_dataset(data_neg, data_pos, config):

    data_neg_mean = np.mean(data_neg, axis=0)
    data_pos_mean = np.mean(data_pos, axis=0)
    
    data_neg_std = np.std(data_neg, axis=0)
    data_pos_std = np.std(data_pos, axis=0)

    data_neg_max = np.max(data_neg, axis=0)
    data_pos_max = np.max(data_pos, axis=0)

    data_neg_min = np.min(data_neg, axis=0)
    data_pos_min = np.min(data_pos, axis=0)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), dpi=300, sharey='row')

    axs[0].plot(data_neg_mean, color='k', label='Mean of [data_neg]')
    axs[1].plot(data_pos_mean, color='k', label='Mean of [data_pos]')

    axs[0].fill_between(np.arange(len(data_neg_mean)),
                        data_neg_mean+data_neg_std,
                        data_neg_mean-data_neg_std,
                        color='k', alpha=0.3,
                        label='$\pm$ std dev')

    axs[0].fill_between(np.arange(len(data_neg_mean)),
                        data_neg_max,
                        data_neg_min,
                        color='k', alpha=0.1,
                        label='Min max')

    axs[1].fill_between(np.arange(len(data_pos_mean)),
                        data_pos_mean+data_pos_std,
                        data_pos_mean-data_pos_std,
                        color='k', alpha=0.3,
                        label='$\pm$ std dev')

    axs[1].fill_between(np.arange(len(data_pos_mean)),
                         data_pos_max,
                         data_pos_min,
                         color='k', alpha=0.1,
                         label='Min max')

    axs[0].set(ylabel='Amplitude', xlabel='Feature', title='Analysis FFT-spectrum of data_neg\n(scaler fitted on [None]')
    axs[1].set(xlabel='Feature', title='Analysis FFT-spectrum of data_pos\n(scaler fitted on [None]')
    axs[0].legend()
    axs[1].legend()

    save_plot(config, title='analysis_dataset')



    












