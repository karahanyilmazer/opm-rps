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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ[
    'SUBJECTS_DIR'
] = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation'

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from src.base.EEG import EEG
from src.preprocessing.FeatureExtractor import FeatureExtractor
from src.utils.MNELoader import MNELoader
from tqdm import tqdm

from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
with open('preprocessing_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

raws = []
event_arrs = []

for run in tqdm(['run_2', 'run_3', 'run_4']):
    config['run'] = run

    # MNE objects
    mne_loader = MNELoader(config)
    raw, event_arr, event_id, device = mne_loader.get_objects()

    if run == 'run_3':
        # Crop the end of Run 3 as it includes the beginning of Run 4
        raw.crop(*config['cropping'][run])
        # Convert the cropping times to indices
        low, high = raw.time_as_index(config['cropping'][run])
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
mne_objects = raw, event_arr, event_id, device
meg = EEG(
    None,
    bp_filt=(config['fmin'], config['fmax']),
    epoching=(config['tmin'], config['tmax']),
    cropping=config['cropping'][run],
    events=config['events'],
    apply_notch=config['apply_notch'],
    extra_notch_freqs=config['notch_freqs'][run],
    notch_params={'notch_widths': config['notch_widths'][run]},
    apply_hfc=config['apply_hfc'],
    bad_chs=config['bad_channels'],
    mne_objects=mne_objects,
    logger_name='meg_analysis',
)
del raw, event_arr, event_id, device

# %%
# Get lists of channels for each axis
x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

# Create epochs
epochs = dict()
epochs['x'] = meg.epochs.copy().pick(x_axis_channels)
epochs['y'] = meg.epochs.copy().pick(y_axis_channels)
epochs['z'] = meg.epochs.copy().pick(z_axis_channels)

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
file_list = [
    'X_var_x',
    'X_var_y',
    'X_var_z',
    'X_bp_x',
    'X_bp_y',
    'X_bp_z',
    'X_welch_x',
    'X_welch_y',
    'X_welch_z',
    'y',
]
var_list = [
    X_var_x,
    X_var_y,
    X_var_z,
    X_bp_x,
    X_bp_y,
    X_bp_z,
    X_welch_x,
    X_welch_y,
    X_welch_z,
    y,
]
for file, var in zip(file_list, var_list):
    file = os.path.join('pickles', 'feature_matrices', file + '.pkl')
    # Open a file to dump the data
    with open(file, 'wb') as pkl_file:
        # Dump the list to the pickle file
        pickle.dump(var, pkl_file)

# %%
