#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: transformer.py
# -----------------------------------------------------------------------------
# Class objects for data preprocessing of input data used in pipeline.
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

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import detrend
import numpy as np

class Detrend(BaseEstimator, TransformerMixin):
    # detrend function for pipline
    def __init__(self, n_channels, seq_length):
        self.n_channels = n_channels
        self.seq_length = seq_length

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        breakpoints=[self.seq_length * i for i in range(self.n_channels)]
        data_detrended = np.zeros(data.shape)
        for i_sample in range(np.shape(data)[0]):
            data_detrended[i_sample] = detrend(
                data[i_sample], bp=breakpoints, type='linear')
        return data_detrended

class Fourierer(BaseEstimator, TransformerMixin):

    def __init__(self, n_channels, seq_length, nfft):
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.nfft = nfft

    def fit(self, X, y=None):
        return  self

    def transform(self, data):

        fft_length = int(self.nfft/2)
        # initialize fft_data matrix
        fft_data = np.zeros((data.shape[0], fft_length*self.n_channels))
        # calc fft for each channel
        for i_ch in range(self.n_channels):
            fft_data[:, i_ch*fft_length:(i_ch+1)*fft_length] = \
            self._fourier_transform(
                data=data[:, i_ch*self.seq_length:(i_ch+1)*self.seq_length], 
                signal_len=self.seq_length, nfft=self.nfft)
        return fft_data

    def _fourier_transform(self, data, signal_len, nfft=64, \
                           fs=0, window='hamming'):

        # overlap between segments for FFT calculation
        overlap=0.5

        # if samples are not an integer multiple of nfft then do zero padding
        if signal_len % nfft != 0:
           signal_len_zeros = signal_len + nfft - (signal_len % nfft)
           data = np.append(data, np.zeros(
               [data.shape[0], nfft-(signal_len % nfft)]), axis=1)
           signal_len=signal_len_zeros
           
        n_segs = signal_len / nfft
        
        # overlap is always 0.5
        n_overlapping_seg = (n_segs-1)*2+1

        window = np.hamming(nfft)

        for i_segs in range(int(n_overlapping_seg)):
            data_win = window * \
                data[:,int(i_segs*nfft*overlap):int((i_segs+2)*nfft*overlap)]

            if i_segs==0: # first iteration
                spectra = \
                    np.abs(np.fft.fft(data_win)/len(window))[:,0:int(nfft/2)]
            else:
                spectra = spectra + \
                    np.abs(np.fft.fft(data_win)/len(window))[:,0:int(nfft/2)]

        return spectra/n_overlapping_seg


if __name__ == '__main__':
    pass
