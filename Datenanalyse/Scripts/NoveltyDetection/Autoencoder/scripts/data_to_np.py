#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Description: data_to_np.py
# -----------------------------------------------------------------------------
# Convert .csv-files to .np-files for faster data import. Specific data
# examples can be deleted.
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

import sys
import os.path
from numpy import array, save, delete
# inser path to find scripts
sys.path.insert(0, os.path.abspath(os.path.join('../')))
# custom package
from scripts.utilities.data_handling import read_file

# data name and path
# data_name = 'DD2_raw_512_FlexRay'
# data_path = '../data/' + data_name + '/'

data_name = 'DD_mass_raw_512_FlexRay_TZ'
data_path = './../../../../Datensatz/DataForCNN/data/testsets/' + data_name + '/'

# import data
x = read_file('_dataset', data_path, file_format='csv')
y = read_file('_labels', data_path, file_format='csv')

# convert data to numpy array and save them for fast data import
x = array(x)
y = array(y)

# delete specific data?
# idxs = [11080, 1643]
# for idx in idxs:
#     x = delete(x, idx, axis=0)
#     y = delete(y, idx, axis=0)

# get shape of data
print(x.shape)
print(y.shape)

# save arrays as np-arrays
save(data_path + 'dataset', x, allow_pickle=True, fix_imports=True)
save(data_path + 'labels', y, allow_pickle=True, fix_imports=True)

x_test = read_file('dataset', data_path, file_format='npy')

print(x==x_test)