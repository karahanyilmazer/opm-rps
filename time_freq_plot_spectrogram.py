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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ[
    'SUBJECTS_DIR'
] = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation'

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from src.base.EEG import EEG
from tqdm import tqdm

from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

run = 'run_2'
config['run'] = run

# Read in the data
meg = EEG(
    config,
    apply_bandpass=False,
    apply_notch=False,
    apply_hfc=True,
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

raw_x = meg.raw.copy().pick(x_chs)

epochs_x = meg.epochs.copy().pick(x_chs)
epochs_y = meg.epochs.copy().pick(y_chs)
epochs_z = meg.epochs.copy().pick(z_chs)
del meg
# %%
# Get the time vector
time = epochs_x.times
srate = epochs_x.info['sfreq']

# Get the data for different conditions
data_roc = epochs_x['roc'].get_data()
data_pap = epochs_x['pap'].get_data()
data_sci = epochs_x['sci'].get_data()

# Get the channel(s) of interest
chs = ['KB[X]']  # One channel
# chs = [ch for ch in config['motor_channels'] if '[X]' in ch]  # Motor channels

# Get the channel indices
ch_idx = [epochs_x.ch_names.index(ch) for ch in chs]

# Get the trials data for the channel of interest
data_roc = np.squeeze(data_roc[:, ch_idx, :]).T
data_pap = np.squeeze(data_pap[:, ch_idx, :]).T
data_sci = np.squeeze(data_sci[:, ch_idx, :]).T

# Get the ERPs
erp_roc = np.mean(data_roc, axis=1)
erp_pap = np.mean(data_pap, axis=1)
erp_sci = np.mean(data_sci, axis=1)

# Plot the ERP
plt.plot(time, erp_roc, label='Rock')
plt.plot(time, erp_pap, label='Paper')
plt.plot(time, erp_sci, label='Scissors')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('ERPs Over the Motor Cortex')
plt.legend()
plt.show()

# %%
# Reshape the data to be 1D --> form a "super-trial"
all_roc = data_roc.reshape(-1, order='F')
all_pap = data_pap.reshape(-1, order='F')
all_sci = data_sci.reshape(-1, order='F')

# Wavelet parameters
min_freq = 1  # in Hz
max_freq = 40  # in Hz
n_freqs = 40  # in count

freqs = np.linspace(min_freq, max_freq, n_freqs)

# Variable number of cycles: 4 at low frequencies, 13 at high frequencies
range_cycles = [4, 13]
n_cycles = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), n_freqs)

wave_time = np.arange(0, 2 * srate + 1) / srate
wave_time = wave_time - np.mean(wave_time)

# Convolution parameters
n_data = len(all_roc)
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# FFT of the data
rocX = np.fft.fft(all_roc, n_conv)
papX = np.fft.fft(all_pap, n_conv)
sciX = np.fft.fft(all_sci, n_conv)

# Initialize the TF matrix
tf = np.zeros((3, n_freqs, len(time)))

for cyci in tqdm(range(len(n_cycles))):
    for fi in range(n_freqs):
        # Create wavelet
        s = n_cycles[cyci] / (2 * np.pi * freqs[fi])
        cmw = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
            -(time**2) / (2 * s**2)
        )

        # FFT of the wavelet
        cmwX = np.fft.fft(cmw, n_conv)
        cmwX = cmwX / cmwX[np.argmax(np.abs(cmwX))]

        # Iterate over the conditions
        for i, (dataX, data) in enumerate(
            zip([rocX, papX, sciX], [data_roc, data_pap, data_sci])
        ):
            # Run convolution, trim edges, and reshape to 2D (time X trials)
            comp_sig = np.fft.ifft(dataX * cmwX)
            comp_sig = comp_sig[half_wave:-half_wave]
            comp_sig = comp_sig.reshape(data.shape, order='F')

            # Average power over trials and put in matrix
            tf[i, fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

# %%
# Get the baseline window
base_win = [-0.45, -0.25]
base_idx = [np.argmin(np.abs(time - base)) for base in base_win]

# Initialize dB normalized TF matrix
tf_db = np.zeros_like(tf)

# dB normalization of the TF matrix
for i in range(3):
    tf_db[i, :, :] = 10 * np.log10(
        tf[i, :, :] / np.mean(tf[i, :, base_idx[0] : base_idx[1]], axis=1)[:, None]
    )

# Plot the TF matrices
fig, axs = plt.subplots(1, 3)
c = axs[0].contourf(time, freqs, tf_db[0, :, :], 40, cmap=cmap)
axs[0].set_title('Rock')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')

c = axs[1].contourf(time, freqs, tf_db[1, :, :], 40, cmap=cmap)
axs[1].set_title('Paper')
axs[1].set_xlabel('Time (s)')

c = axs[2].contourf(time, freqs, tf_db[2, :, :], 40, cmap=cmap)
axs[2].set_title('Scissors')
axs[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()

# %%
