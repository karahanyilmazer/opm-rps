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
from scipy.stats import zscore
from sklearn.decomposition import PCA
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
data_dict = {
    'data_dir': r'C:\Files\Coding\Python\Neuro\data',
    'paradigm': 'Gesture',
    'dataset': 'Nottingham Gesture',
    'device': 'OPM',
    'subject': 11766,
    'session': 20230623,  # or 20230622
    'run': 'run_3',
}

cropping = (0, None)
if data_dict['run'] == 'run_3':
    # Crop the end of Run 3 as it includes the beginning of Run 4
    cropping = (0, 681.449)

fmin, fmax = None, 400
tmin, tmax = -0.5, 2.1
events = ('cue_1', 'cue_2', 'cue_3')
# events = None
combined_bads = [
    'FR[Z]',
    'FZ[X]',
    'HF[Y]',
    'HJ[X]',
    'HJ[Y]',
    'K9[X]',
    'KC[Y]',
    'KC[Z]',
    'KE[Y]',
    'KF[Y]',
    'LB[X]',
    'LB[Y]',
    'LF[X]',
    'LN[X]',
    'LN[Y]',
    'LN[Z]',
    'LP[X]',
    'LP[Y]',
    'MV[X]',
    'MV[Y]',
    'MV[Z]',
    'MU[Z]',
]

meg = EEG(
    data_dict,
    bp_filt=(fmin, fmax),
    epoching=(tmin, tmax),
    cropping=cropping,
    events=events,
    apply_notch=True,
    bad_chs=combined_bads,
    logger_name='meg_analysis',
)

# %%
# First look:
# Bad channels (Run 1): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z]
# Bad channels (Run 2): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z]
# Bad channels (Run 3): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]
# Bad channels (Run 4): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]

# Second look:
# Bad channels (Run 1): LN[X, Y, Z], FZ[X], K9[X], KF[Y], MU[Z], FR[Z]
# Sus channels (Run 1): FR[X, Y], LB[X, Y], HJ[X], HF[Y], KD[Y], KB[Z]

# Bad channels (Run 2): LN[X, Y, Z], MV[X, Y, Z], LB[X, Y], FZ[X], K9[X], HF[Y], KC[Y], KF[Y], MU[Z]
# Sus channels (Run 2): G0[X], HJ[X], LF[X], K9[Y], KC[Z]

# Bad channels (Run 3): LN[X, Y, Z], K9[X], FZ[X], MV[X, Y, Z], LB[X, Y], LP[X, Y], HJ[X, Y], KC[Y, Z], KE[Y], KF[Y], MU[Z]
# Sus channels (Run 3): HO[Z], LF[X]

# Bad channels (Run 4): LN[X, Y, Z], K9[X], FZ[X], LB[X], HJ[X], LP[X], KF[Y], MV[Y, Z], KC[Y, Z], KE[Y], HF[Y], LB[Y], MU[Z]
# Sus channels (Run 4): G0[X], K9[Z], LM[Z], LP[Z]

x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

raws = dict()
raws['x'] = meg.raw.copy().pick(x_axis_channels)
raws['y'] = meg.raw.copy().pick(y_axis_channels)
raws['z'] = meg.raw.copy().pick(z_axis_channels)
# %%
# Plotting for sanity check
# raws['x'].plot(events=events, event_id=event_id, scalings='auto')
raws['x'].compute_psd().plot(picks='data', exclude='bads', average=True)
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
