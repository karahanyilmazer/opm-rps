#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess OPM data from multiple runs.  

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import pickle
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')
os.environ[
    'SUBJECTS_DIR'
] = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation'

import matplotlib.pyplot as plt
import mne
import numpy as np
from src.preprocessing.FeatureExtractor import FeatureExtractor
from src.utils.MNELoader import MNELoader

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
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

raws = []
event_arrs = []

for run in ['run_1', 'run_2', 'run_3', 'run_4']:
    data_dict = {
        'data_dir': r'C:\Files\Coding\Python\Neuro\data',
        'paradigm': 'Gesture',
        'dataset': 'Nottingham Gesture',
        'device': 'OPM',
        'subject': 11766,
        'session': 20230623,  # or 20230622
        'run': run,
    }

    # MNE objects
    mne_loader = MNELoader(data_dict)
    raw, event_arr, event_id, device = mne_loader.get_objects()

    if data_dict['run'] == 'run_3':
        # Crop the end of Run 3 as it includes the beginning of Run 4
        cropping = (0, 681.449)
        raw.crop(*cropping)
        # Convert the cropping times to indices
        low, high = raw.time_as_index(cropping)
        # Create a Boolean mask for the relevant part of the array
        mask = np.logical_and(event_arr[:, 0] >= low, event_arr[:, 0] <= high)
        # Crop the events array
        event_arr = event_arr[mask]

    raws.append(raw)
    event_arrs.append(event_arr)

# Concatenate all raw and event objects
raw, event_arr = mne.concatenate_raws(raws, events_list=event_arrs)
del raws, event_arrs

# %%
raw_filt = raw.copy()

notch_freqs = np.arange(50, 600, 50)
notch_freqs = np.insert(notch_freqs, 5, 227)
notch_freqs = np.insert(notch_freqs, 6, 277)
notch_freqs = np.insert(notch_freqs, 13, 554)
raw.notch_filter(notch_freqs)

fmin, fmax = None, 400
raw.filter(fmin, fmax)

raw.info['bads'].extend(combined_bads)

# %%
tmin, tmax = -0.5, 2.1
events = ('cue_1', 'cue_2', 'cue_3')

x_axis_channels = [ch for ch in raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

raws = dict()
raws['x'] = raw.copy().pick(x_axis_channels)
raws['y'] = raw.copy().pick(y_axis_channels)
raws['z'] = raw.copy().pick(z_axis_channels)
# %%
# Plotting for sanity check
# raws['x'].plot(events=events, event_id=event_id, scalings='auto')
raws['x'].compute_psd().plot(picks='data', exclude='bads', average=False)
# %%
# Define events of interest
event_id_subset = {'cue_1': 1, 'cue_2': 2, 'cue_3': 4}
tmin, tmax = -0.5, 2.1

# Create epochs
epochs = dict()
epochs['x'] = mne.Epochs(
    raws['x'], event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
)
epochs['y'] = mne.Epochs(
    raws['y'], event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
)
epochs['z'] = mne.Epochs(
    raws['z'], event_arr, event_id_subset, tmin=tmin, tmax=tmax, baseline=None
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
X_var_x = fe.get_var_feat(epochs['x'].get_data(), epochs['x'].ch_names)
X_bp_x = fe.get_bp_feat(epochs['x'].get_data(), epochs['x'].ch_names)
X_welch_x = fe.get_welch_feat(epochs['x'].get_data(), epochs['x'].ch_names)

fe = FeatureExtractor(epochs['y'])
X_var_y = fe.get_var_feat(epochs['y'].get_data(), epochs['y'].ch_names)
X_bp_y = fe.get_bp_feat(epochs['y'].get_data(), epochs['y'].ch_names)
X_welch_y = fe.get_welch_feat(epochs['y'].get_data(), epochs['y'].ch_names)

fe = FeatureExtractor(epochs['z'])
X_var_z = fe.get_var_feat(epochs['z'].get_data(), epochs['z'].ch_names)
X_bp_z = fe.get_bp_feat(epochs['z'].get_data(), epochs['z'].ch_names)
X_welch_z = fe.get_welch_feat(epochs['z'].get_data(), epochs['z'].ch_names)

y = epochs['x'].events[:, 2]

# %%
file = os.path.join('pickles', 'feature_matrices', 'y' + '.pkl')
# Open a file to dump the data
with open(file, 'wb') as pkl_file:
    # Dump the list to the pickle file
    pickle.dump(y, pkl_file)

# %%
