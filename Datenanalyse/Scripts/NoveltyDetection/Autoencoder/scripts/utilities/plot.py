#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: plot.py
# -----------------------------------------------------------------------------
# Utilities for data plotting
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
import matplotlib.pyplot as plt
from matplotlib import ticker
import tikzplotlib as tikz
from sklearn import metrics
# from matplotlib.ticker import PercentFormatter

###  define colors
# main
TUMmain = '#0065BD'
TUMwhite = '#ffffff'
TUMblack = '#000000'
# secondary
TUMgray1 = '#333333'
TUMgray2 = '#808080'
TUMgray3 = '#CCCCC6'
TUMblue1 = '#005293'
TUMblue2 = '#003359'
# accent
TUMaccent1 = '#64A0C8'
TUMaccent2 = '#98C6EA'
TUMaccent3 = '#DAD7CB'
TUMaccent4 = '#A2AD00'
TUMaccent5 = '#E37222'

def save_plot(config, title):
     plt.savefig(config.plot_path+'/'+title+'.pdf', bbox_inches='tight')
     tikz.save(config.plot_path+'/'+title+'.tikz', encoding='utf8')

def trim_axs(axs, n):
    """little helper to make the axs list to have correct length"""
    axs = axs.flat
    for ax in axs[n:]:
        ax.remove()
    return axs[:n]

def plot_loss_hist(loss, loss_std, config):
    # create loss curve

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), dpi=300)

    if config.mode == 'cv':
        loss_upper = loss+loss_std
        loss_lower = loss-loss_std
        ax.plot(loss, ls='-', color=TUMgray1, label='Mean loss')
        ax.fill_between(np.arange(0,len(loss)),loss_lower, loss_upper, color=TUMgray3,
        label=r'$\pm$ 1 std. dev.')
    else:
        ax.plot(loss, ls='-', color=TUMgray1, label='Loss')

    ax.set(xlabel='Epochs', ylabel='Loss', title='Loss curve')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    try:
        if config.save_plot == True: save_plot(config, title='plot_loss_curve')
    except: print()
    
def plot_roc_hist(roc_auc, roc_auc_std, config):
    # create loss curve

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), dpi=300)

    if config.mode == 'cv':
        roc_auc_upper = roc_auc+roc_auc_std
        roc_auc_lower = roc_auc-roc_auc_std
        ax.plot(roc_auc, ls='-', color=TUMgray1, label='Mean loss')
        ax.fill_between(np.arange(0,len(roc_auc)),roc_auc_lower, roc_auc_upper, color=TUMgray3,
        label=r'$\pm$ 1 std. dev.')
    else:
        ax.plot(roc_auc, ls='-', color=TUMgray1, label='Loss')

    ax.set(xlabel='Epochs', ylabel='ROC AUC', title='ROC AUC curve')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    
    try:
        if config.save_plot == True: save_plot(config, title='plot_roc_auc_hist_curve')
    except: print()

def plot_roc_curve(roc_auc, tprs, fprs, std_roc_auc, std_tprs, config):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), dpi=300)

    # chance line
    ax.plot([0,1], [0,1], lw=2, ls='--', color=TUMgray2, label='Chance')

    # for cv run
    if config.mode == 'cv':
        ax.plot(fprs, tprs, lw=2, alpha=1, color=TUMgray1, ls='-',
                    label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc, std_roc_auc))

        tprs_upper = np.minimum(tprs+std_tprs, 1)
        tprs_lower = np.maximum(tprs-std_tprs, 0)
        ax.fill_between(fprs, tprs_lower, tprs_upper, color=TUMgray3,
                            label=r'$\pm$ 1 std. dev.')
    else:
        ax.plot(fprs, tprs, lw=2, alpha=1, color=TUMgray1 , ls='-',
                    label='ROC (AUC = %0.2f)' % (roc_auc))

    ax.set(xlim=[-0.05, 1.05], xlabel='False positive rate',
               ylim=[-0.05, 1.05], ylabel='True positive rate',
               title='ROC curve')

    ax.legend()
    
    try:
        if config.save_plot == True: save_plot(config, title='plot_roc_auc_curve')
    except: print()

def plot_nov_score_density(score, labels, config):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), dpi=300)

    bins_neg = int((np.max(score[labels==0])-np.min(score[labels==0]))/0.02)
    bins_pos = int((np.max(score[labels==1])-np.min(score[labels==0]))/0.02)

    ax.hist(score[labels==0], density=True, bins=bins_neg, lw=2, ls='-', color=TUMgray1,
                         histtype='step', label='Negative class')

    ax.hist(score[labels==1], density=True, bins=bins_pos, lw=2, ls='--', color=TUMgray2,
                         histtype='step', label='Positive class')

    ax.set(xlabel='Novely score', ylabel='Fraction', title='Novelty score of predicted validation data')
    #ax_rmse_density.yaxis.set_major_formatter(PercentFormatter(xmax=len(labels[labels==1])))
    ax.legend()

    if config.save_plot == True: save_plot(config, title='plot_nov_score_density')


def plot_reconstructed_data(orig_data, pred_data, seq_length, n_sensors, i_example, config):

    seq_length = int(seq_length)
    ncols=int(n_sensors/2+0.5)

    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(12,6),
                        sharex='col', sharey='row')

    axs = trim_axs(axs, n_sensors)

    for i_sen, ax in enumerate(axs):
        ax.plot(orig_data[i_example, i_sen*seq_length:(i_sen+1)*seq_length],
                label='Original', lw=2, color='#333333')
        ax.plot(pred_data[i_example, i_sen*seq_length:(i_sen+1)*seq_length],
                label='Reconstructed', lw=2, color='#808080')
        # only first plot with (shared) amplitude
        if i_sen == 0:
            ax.set(xlabel='Data points', ylabel='Amplitude', title='Sensor '+str(config.sel_sensors[i_sen]))
        else: ax.set(xlabel='Data points', title='Sensor '+str(config.sel_sensors[i_sen]))

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1-1/ncols, 0., 1/ncols, 0.4))
    try:
        if config.save_plot == True: save_plot(config, title='plot_reconstructed_data['+str(i_example)+']')
    except: print()
    
def plot_confusion_matrix(labels, pred_labels, normalize, config):
    # plot confusion matrix
    cm = metrics.confusion_matrix(labels, pred_labels)
    classes = [1, 0]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    ax.matshow(cm, interpolation=None, cmap=plt.cm.Greys)

    ax.set(xticklabels=['']+classes, yticklabels=['']+classes,
           ylabel='True label', xlabel='Predicted label')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    ax.grid(b=False)
    
    try:
        if config.save_plot == True: save_plot(config, title='plot_confusion_matrix')
    except: print()
    
def plot_analyse_data(name_list, data_mean, data_std, fitted_on, config):

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=300, sharey='row')

    for i, ax in enumerate(axs.flat):

        ax.plot(data_mean[i], label='Mean')
        ax.fill_between(np.arange(len(data_mean[i])),
                            data_mean[i]+data_std[i],
                            data_mean[i]-data_std[i],
                            alpha=0.3, label='$\pm$ std dev')

        ax.set(xlabel='Feature', ylabel='Value', title='Analysis FFT-spectrum of [%s]\n(scaler fitted on [%s])' %(name_list[i], fitted_on))
        ax.legend()
        
    try:
        if config.save_plot == True: save_plot(config, title='analysis_spectrum_fitted on_%s' %(fitted_on))
    except: print('COULD NOT SAVE PLOT')