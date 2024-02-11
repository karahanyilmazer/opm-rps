#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess OPM data from a single run.  

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')
os.environ[
    'SUBJECTS_DIR'
] = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation'

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.base.EEG import EEG
from src.preprocessing.FeatureExtractor import FeatureExtractor

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

run = 'run_1'
config['run'] = run

meg = EEG(
    config,
    bp_filt=(config['fmin'], config['fmax']),
    epoching=(config['tmin'], config['tmax']),
    cropping=config['cropping'][run],
    events=config['events'],
    apply_notch=config['apply_notch'],
    extra_notch_freqs=config['notch_freqs'][run],
    notch_params={'notch_widths': config['notch_widths'][run]},
    apply_hfc=config['apply_hfc'],
    bad_chs=config['bad_channels'],
    bad_epochs=config['bad_epochs'][run],
    logger_name='meg_analysis',
)

meg.plot_epochs(['KB[X]', 'KB[Y]', 'KB[Z]', 'LG[X]'])

# %%
x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

raws = dict()
raws['x'] = meg.raw.copy().pick(x_axis_channels)
raws['y'] = meg.raw.copy().pick(y_axis_channels)
raws['z'] = meg.raw.copy().pick(z_axis_channels)

raws['x'].compute_psd().plot(picks='data')
raws['y'].compute_psd().plot(picks='data')
raws['z'].compute_psd().plot(picks='data')

raws['x'].compute_psd().plot(picks='data', average=True)
raws['y'].compute_psd().plot(picks='data', average=True)
raws['z'].compute_psd().plot(picks='data', average=True)
# %%
# Plotting for sanity check
# raws['x'].plot(events=events, event_id=event_id, scalings='auto')
raws['x'].compute_psd().plot(picks='data', average=True)
# %%
# Define events of interest
event_id_subset = {'cue_1': 1, 'cue_2': 2, 'cue_3': 4}
tmin, tmax = -0.5, 2.1

# Create epochs
epochs = dict()
epochs['x'] = mne.Epochs(
    raws['x'], meg.event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
)
epochs['y'] = mne.Epochs(
    raws['y'], meg.event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
)
epochs['z'] = mne.Epochs(
    raws['z'], meg.event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
)

# %%
epochs['x'].plot(scalings='auto', butterfly=True)

# %%
_, axs = plt.subplots(3, 1)
epochs_spectrum = dict()
epochs_spectrum['x'] = epochs['x'].compute_psd()
epochs_spectrum['x']['cue_1'].plot(picks='data', exclude='bads', axes=axs[0])
epochs_spectrum['x']['cue_2'].plot(picks='data', exclude='bads', axes=axs[1])
epochs_spectrum['x']['cue_3'].plot(picks='data', exclude='bads', axes=axs[2])

# %%
data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25, method="empirical")
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method="empirical")
data_cov.plot(epochs.info)

# %%
fe = FeatureExtractor(epochs['x'])
X_var = fe.get_var_feat(epochs['x'].get_data(), epochs['x'].ch_names)
X_bp_x = fe.get_bp_feat(epochs['x'].get_data(), epochs['x'].ch_names)
fe = FeatureExtractor(epochs['y'])
X_bp_y = fe.get_bp_feat(epochs['y'].get_data(), epochs['y'].ch_names)
fe = FeatureExtractor(epochs['z'])
X_bp_z = fe.get_bp_feat(epochs['z'].get_data(), epochs['z'].ch_names)

y = epochs['x'].events[:, 2]

# %%
# X = X_bp_x
X = np.concatenate([X_bp_x, X_bp_y, X_bp_z], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv_train_scores = []
cv_test_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ss = StandardScaler()
svm = SVC(gamma='auto')

pipe = make_pipeline(ss, svm)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    pipe.fit(X_train, y_train)
    cv_train_scores.append(np.round(pipe.score(X_train, y_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_test, y_test), 3))

print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))

# %%
