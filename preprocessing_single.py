#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess OPM data from a single run.  

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np

from utils import get_mne_data

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
# %%
data_dir = r'C:\Files\Coding\Python\Neuro\data\Gesture\Rock Paper Scissors'
# data_dir = r'C:\Users\user\Desktop\MasterThesis\data_nottingham'
# data_dir = r'D:\PhD\data\2023-06-21_nottingham'
day = '20230623'
# acq_time = '095228'  # Noise
# acq_time = '100008'  # Noise
# acq_time = '100245'  # Noise
# acq_time = '102814'  # Run 1
# acq_time = '104104'  # Run 2
# acq_time = '105342'  # Run 3 (has some data from Run 4 after 681.499 s)
# acq_time = '110808'  # Run 4
# acq_time = '112029'  # Rest

acq_time_list = ['102814', '104104', '105342', '110808']
run_idx = 0

raw, events, event_id = get_mne_data(data_dir, day, acq_time_list[run_idx])
# Crop the end of Run 3 as it includes the beginning of Run 4
if run_idx == 2:
    raw.crop(tmax=681.499)
    idx = np.where(events[:, 2] == 255)[0][1] + 1
    events = events[:idx, :]

raw.info['bads'] = [
    'LN[X]',
    'LN[Y]',
    'LN[Z]',
    'FZ[X]',
    'K9[X]',
    'KF[Y]',
    'MU[Z]',
    'LB[X]',
    'LB[Y]',
    'MV[X]',
    'MV[Y]',
    'MV[Z]',
    'HF[Y]',
    'HJ[X]',
]

# %%
# Plotting for sanity check
# raw.compute_psd().plot(picks='data', exclude='bads')
raw.plot(events=events, event_id=event_id, scalings='auto')

# Bad channels (Run 1): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z]
# Bad channels (Run 2): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z]
# Bad channels (Run 3): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]
# Bad channels (Run 4): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]
