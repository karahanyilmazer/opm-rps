#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract the time-frequency spectrograms for each run.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys
from pickle import dump

from utils import get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
sys.path.insert(0, os.path.join(base_dir, 'data'))
os.environ['SUBJECTS_DIR'] = os.path.join(
    base_dir, 'data', 'Gesture', 'Nottingham', 'MRI', 'Segmentation'
)

import matplotlib.pyplot as plt
import numpy as np
from src.base.EEG import EEG
from tqdm import tqdm
from yaml import safe_load

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = safe_load(file)

run = 'run_4'
config['run'] = run

# Read in the data
meg = EEG(
    config,
    apply_bandpass=False,
    apply_notch=False,
    apply_hfc=True,
    detrend=config['detrend'],
    epoching=(config['tmin'], config['tmax']),
    cropping=config['cropping'][run],
    events=config['events'],
    bad_chs=config['bad_channels'],
    bad_epochs=config['bad_epochs'][run],
    logger_name='meg_analysis',
)

# %%
# Get a list of channels for each axis
x_chs = [ch for ch in meg.raw.ch_names if '[X]' in ch]
y_chs = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
z_chs = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

# raw_x = meg.raw.copy().pick(x_chs)
# raw_y = meg.raw.copy().pick(y_chs)
# raw_z = meg.raw.copy().pick(z_chs)

epochs_x = meg.epochs.copy().pick(x_chs)
epochs_y = meg.epochs.copy().pick(y_chs)
epochs_z = meg.epochs.copy().pick(z_chs)
del meg

# %%
# Choose the axis
config['axis'] = 'Z'

# Choose the epochs object
if config['axis'] == 'X':
    epochs = epochs_x
elif config['axis'] == 'Y':
    epochs = epochs_y
elif config['axis'] == 'Z':
    epochs = epochs_z

# Get the time vector
time = epochs.times
srate = epochs.info['sfreq']

# Get the data for different conditions
data_roc = epochs['roc'].get_data(copy=True)
data_pap = epochs['pap'].get_data(copy=True)
data_sci = epochs['sci'].get_data(copy=True)

# Get the channel(s) of interest
chs = [f'KB[{config["axis"]}]']  # One channel
# chs = [ch for ch in config['motor_channels'] if [config["axis"]] in ch]  # Motor chans

# Get the channel indices
ch_idx = [epochs.ch_names.index(ch) for ch in chs]
ch_idx = ch_idx[0] if len(ch_idx) == 1 else ch_idx  # If there's only one channel

# Get the trials data for the channel of interest
data_roc = np.squeeze(data_roc[:, ch_idx, :]).T
data_pap = np.squeeze(data_pap[:, ch_idx, :]).T
data_sci = np.squeeze(data_sci[:, ch_idx, :]).T

# Get the shape of the data (samples x trials)
roc_shape = data_roc.shape
pap_shape = data_pap.shape
sci_shape = data_sci.shape

# Get the ERPs
erp_roc = np.mean(data_roc, axis=1)
erp_pap = np.mean(data_pap, axis=1)
erp_sci = np.mean(data_sci, axis=1)

# Get the non-phase locked data
npl_roc = data_roc - erp_roc[:, np.newaxis]
npl_pap = data_pap - erp_pap[:, np.newaxis]
npl_sci = data_sci - erp_sci[:, np.newaxis]

# %%
# Reshape the data to be 1D --> form a "super-trial"
all_roc = data_roc.reshape(-1, order='F')
all_pap = data_pap.reshape(-1, order='F')
all_sci = data_sci.reshape(-1, order='F')
all_roc_npl = npl_roc.reshape(-1, order='F')
all_pap_npl = npl_pap.reshape(-1, order='F')
all_sci_npl = npl_sci.reshape(-1, order='F')

# Wavelet parameters
min_freq = 2  # in Hz
max_freq = 50  # in Hz
n_freqs = 40  # in count
freqs = np.linspace(min_freq, max_freq, n_freqs)

# Variable number of cycles: 4 at low frequencies, 10 at high frequencies
range_cycles = [4, 10]
s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), n_freqs)
s /= 2 * np.pi * freqs

# Create the wavelet time vector
wave_time = np.arange(0, 2 + 1 / srate, 1 / srate)
wave_time = np.r_[-wave_time[::-1], wave_time[1:]]

# Convolution parameters
n_kern = len(wave_time)
n_conv_roc = np.prod(roc_shape) + n_kern - 1
n_conv_pap = np.prod(pap_shape) + n_kern - 1
n_conv_sci = np.prod(sci_shape) + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# FFT of the data
rocX = np.fft.fft(all_roc, n_conv_roc)
papX = np.fft.fft(all_pap, n_conv_pap)
sciX = np.fft.fft(all_sci, n_conv_sci)
roc_nplX = np.fft.fft(all_roc_npl, n_conv_roc)
pap_nplX = np.fft.fft(all_pap_npl, n_conv_pap)
sci_nplX = np.fft.fft(all_sci_npl, n_conv_sci)

# Initialize the TF matrix
# 1st dim: total/NPL
# 2nd dim: frequencies
# 3rd dim: time
# 4th dim: trials
tf_roc = np.zeros((2, n_freqs, len(time), roc_shape[1]))
tf_pap = np.zeros((2, n_freqs, len(time), pap_shape[1]))
tf_sci = np.zeros((2, n_freqs, len(time), sci_shape[1]))

for fi in tqdm(range(len(freqs))):
    # Create wavelet
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -(wave_time**2) / (2 * s[fi] ** 2)
    )

    # Iterate over the conditions
    for i, (tf, dataX, nplX, data_shape, n_conv) in enumerate(
        zip(
            [tf_roc, tf_pap, tf_sci],
            [rocX, papX, sciX],
            [roc_nplX, pap_nplX, sci_nplX],
            [roc_shape, pap_shape, sci_shape],
            [n_conv_roc, n_conv_pap, n_conv_sci],
        )
    ):
        # FFT of the wavelet
        cmwX = np.fft.fft(cmw, n_conv)
        cmwX = cmwX / cmwX[np.argmax(np.abs(cmwX))]

        # Run convolution, trim edges, and reshape to 2D (time X trials)
        as_total = np.fft.ifft(dataX * cmwX)
        as_total = as_total[half_wave:-half_wave]
        as_total = as_total.reshape(data_shape, order='F')

        as_npl = np.fft.ifft(nplX * cmwX)
        as_npl = as_npl[half_wave:-half_wave]
        as_npl = as_npl.reshape(data_shape, order='F')

        # Average power and phase over trials and put in matrix
        tf[0, fi, :, :] = np.abs(as_total) ** 2
        tf[1, fi, :, :] = np.abs(as_npl) ** 2

# %%
# Get the baseline window
base_win = config['baseline_window']
base_idx = [np.argmin(np.abs(time - base)) for base in base_win]

# Initialize dB normalized TF matrices
tf_db_roc = tf_roc.copy()
tf_db_pap = tf_pap.copy()
tf_db_sci = tf_sci.copy()

# dB normalization of the TF matrices
for i in range(2):  # Total/NPL
    for j in range(roc_shape[1]):  # Trials
        tf_db_roc[i, :, :, j] = 10 * np.log10(
            tf_db_roc[i, :, :, j]
            / np.mean(tf_db_roc[i, :, base_idx[0] : base_idx[1], j], axis=1)[:, None]
        )

for i in range(2):  # Total/NPL
    for j in range(pap_shape[1]):  # Trials
        tf_db_pap[i, :, :, j] = 10 * np.log10(
            tf_db_pap[i, :, :, j]
            / np.mean(tf_db_pap[i, :, base_idx[0] : base_idx[1], j], axis=1)[:, None]
        )

for i in range(2):  # Total/NPL
    for j in range(sci_shape[1]):  # Trials
        tf_db_sci[i, :, :, j] = 10 * np.log10(
            tf_db_sci[i, :, :, j]
            / np.mean(tf_db_sci[i, :, base_idx[0] : base_idx[1], j], axis=1)[:, None]
        )

# %%
file_names = [
    f'tf_roc-{config["run"]}-{config["axis"]}',
    f'tf_pap-{config["run"]}-{config["axis"]}',
    f'tf_sci-{config["run"]}-{config["axis"]}',
    f'tf_db_roc-{config["run"]}-{config["axis"]}',
    f'tf_db_pap-{config["run"]}-{config["axis"]}',
    f'tf_db_sci-{config["run"]}-{config["axis"]}',
]

var_names = [
    tf_roc,
    tf_pap,
    tf_sci,
    tf_db_roc,
    tf_db_pap,
    tf_db_sci,
]

# Save the TF matrices to pickle files
for file, var in zip(file_names, var_names):
    file = os.path.join('pickles', 'tf_matrices', 'single', file + '.pkl')
    # Open a file to dump the data
    with open(file, 'wb') as pkl_file:
        # Dump the list to the pickle file
        dump(var, pkl_file)

# %%
