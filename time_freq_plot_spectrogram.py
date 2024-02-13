#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the time-frequency spectrogram of the EEG data for different conditions.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

from utils import get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
sys.path.insert(0, os.path.join(base_dir, 'data'))
os.environ['SUBJECTS_DIR'] = os.path.join(
    base_dir, 'data', 'Gesture', 'Nottingham', 'MRI', 'Segmentation'
)

import matplotlib.pyplot as plt
import mne
import numpy as np
from src.base.EEG import EEG
from tqdm import tqdm
from yaml import safe_load

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = safe_load(file)

run = 'run_2'
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
# Get the time vector
time = epochs_x.times
srate = epochs_x.info['sfreq']

# Get the data for different conditions
data_roc = epochs_x['roc'].get_data(copy=True)
data_pap = epochs_x['pap'].get_data(copy=True)
data_sci = epochs_x['sci'].get_data(copy=True)

# Get the channel(s) of interest
chs = ['KB[X]']  # One channel
# chs = [ch for ch in config['motor_channels'] if '[X]' in ch]  # Motor channels

# Get the channel indices
ch_idx = [epochs_x.ch_names.index(ch) for ch in chs]
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
n_data = len(all_roc)
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# FFT of the data
rocX = np.fft.fft(all_roc, n_conv)
papX = np.fft.fft(all_pap, n_conv)
sciX = np.fft.fft(all_sci, n_conv)
roc_nplX = np.fft.fft(all_roc_npl, n_conv)
pap_nplX = np.fft.fft(all_pap_npl, n_conv)
sci_nplX = np.fft.fft(all_sci_npl, n_conv)

# Initialize the TF matrix
# 1st dim: total/NPL
# 2nd dim: power/ITPC
# 3rd dim: conditions
# 4th dim: frequencies
# 5th dim: time
tf = np.zeros((2, 2, 3, n_freqs, len(time)))

for fi in tqdm(range(len(freqs))):
    # Create wavelet
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -(wave_time**2) / (2 * s[fi] ** 2)
    )

    # FFT of the wavelet
    cmwX = np.fft.fft(cmw, n_conv)
    cmwX = cmwX / cmwX[np.argmax(np.abs(cmwX))]

    # Iterate over the conditions
    for i, (dataX, nplX, data_shape) in enumerate(
        zip(
            [rocX, papX, sciX],
            [roc_nplX, pap_nplX, sci_nplX],
            [roc_shape, pap_shape, sci_shape],
        )
    ):
        # Run convolution, trim edges, and reshape to 2D (time X trials)
        as_total = np.fft.ifft(dataX * cmwX)
        as_total = as_total[half_wave:-half_wave]
        as_total = as_total.reshape(data_shape, order='F')

        as_npl = np.fft.ifft(nplX * cmwX)
        as_npl = as_npl[half_wave:-half_wave]
        as_npl = as_npl.reshape(data_shape, order='F')

        # Average power and phase over trials and put in matrix
        tf[0, 0, i, fi, :] = np.mean(np.abs(as_total) ** 2, axis=1)
        tf[0, 1, i, fi, :] = np.abs(np.mean(np.exp(1j * np.angle(as_total)), axis=1))

        tf[1, 0, i, fi, :] = np.mean(np.abs(as_npl) ** 2, axis=1)
        tf[1, 1, i, fi, :] = np.abs(np.mean(np.exp(1j * np.angle(as_npl)), axis=1))


# %%
# Plot the TF matrices for each condition
def plot_tf_matrices(cond, tf_slice, time, freqs, cmap, dB=False):
    fig, axs = plt.subplots(2, 3)

    title_suffix = 'in dB' if dB else ''
    fig_suffix = '-dB' if dB else ''
    fig.suptitle(f'Time-Frequency Analysis {title_suffix} ({cond})')

    titles = [
        'Total Power',
        'Non-Phase-Locked Power',
        'Phase-Locked Power',
        'Total ITPC',
        'Non-Phase-Locked ITPC',
        'Phase-Locked ITPC',
    ]

    for i, ax in enumerate(axs.flat):
        row, col = divmod(i, 3)
        # For Phase-Locked plots, subtract the non-phase-locked from the total
        if col == 2:
            data = tf_slice[0, row, :, :] - tf_slice[1, row, :, :]
        else:
            data = tf_slice[col, row, :, :]

        # Plot the TF matrix
        ax.contourf(time, freqs, data, 40, cmap=cmap)

        # Plot customization
        ax.set_title(titles[i])
        if row == 1:
            ax.set_xlabel('Time (s)')
        if col == 0:
            ax.set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig(
        os.path.join('img', f'tf_{cond.lower()[:3]}-{config["run"]}{fig_suffix}.png')
    )
    plt.show()


plot_tf_matrices('Rock', tf[:, :, 0, :, :], time, freqs, cmap)
plot_tf_matrices('Paper', tf[:, :, 1, :, :], time, freqs, cmap)
plot_tf_matrices('Scissors', tf[:, :, 2, :, :], time, freqs, cmap)

# %%
# Get the baseline window
base_win = [-0.45, -0.25]
base_idx = [np.argmin(np.abs(time - base)) for base in base_win]

# Initialize dB normalized TF matrix
tf_db = tf.copy()

# dB normalization of the TF matrix
for i in tqdm(range(3)):  # Conditions
    for j in range(2):  # Total/NPL
        tf_db[j, 0, i, :, :] = 10 * np.log10(
            tf_db[j, 0, i, :, :]
            / np.mean(tf_db[j, 0, i, :, base_idx[0] : base_idx[1]], axis=1)[:, None]
        )

plot_tf_matrices('Rock', tf_db[:, :, 0, :, :], time, freqs, cmap, dB=True)
plot_tf_matrices('Paper', tf_db[:, :, 1, :, :], time, freqs, cmap, dB=True)
plot_tf_matrices('Scissors', tf_db[:, :, 2, :, :], time, freqs, cmap, dB=True)

# %%
