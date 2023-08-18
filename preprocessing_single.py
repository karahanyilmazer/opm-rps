#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess OPM data from a single run.  

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.base.EEG import EEG
from src.preprocessing.FeatureExtractor import FeatureExtractor
from src.utils.DataLoader import DataLoader

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
# %%
loader = DataLoader()

data_dir = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\Rock Paper Scissors'
day = '20230623'  # or '20230622'
acq_time_dict = {
    'noise_1': '095228',
    'noise_2': '100008',
    'noise_3': '100245',
    'run_1': '102814',
    'run_2': '104104',
    'run_3': '105342',  # has some data from run_4 after 681.499 s
    'run_4': '110808',
    'rest': '112029',
}

acq_time_key = 'run_1'

raw, events, event_id = loader.get_meg_data(data_dir, day, acq_time_dict[acq_time_key])
# Crop the end of Run 3 as it includes the beginning of Run 4
if acq_time_key == 'run_3':
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

x_axis_channels = [ch for ch in raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

raws = dict()
raws['x'] = raw.copy().pick(x_axis_channels)
raws['y'] = raw.copy().pick(y_axis_channels)
raws['z'] = raw.copy().pick(z_axis_channels)

del raw
# %%
# Plotting for sanity check
# raw.compute_psd().plot(picks='data', exclude='bads')
raw.plot(events=events, event_id=event_id, scalings='auto')

# Bad channels (Run 1): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z]
# Bad channels (Run 2): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z]
# Bad channels (Run 3): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]
# Bad channels (Run 4): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]

# %%
# Define events of interest
event_id_subset = {'cue_1': 1, 'cue_2': 2, 'cue_3': 4}

# Create epochs
epochs = mne.Epochs(raw, events, event_id_subset, tmin=-0.5, tmax=2.1, baseline=None)
