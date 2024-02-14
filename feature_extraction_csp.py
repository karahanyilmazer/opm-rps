#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract CSP features from all runs.

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
import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import filtfilt, firls, hilbert
from src.base.EEG import EEG
from tqdm import tqdm
from yaml import safe_load

# %%
# Concatenate the epochs
# epochs_x = mne.read_epochs(os.path.join('pickles/epochs/all_runs_x-epo.fif.gz'))
# epochs_y = mne.read_epochs(os.path.join('pickles/epochs/all_runs_y-epo.fif.gz'))
epochs_z = mne.read_epochs(os.path.join('pickles/epochs/all_runs_z-epo.fif.gz'))

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    bad_epochs_idx = safe_load(file)['bad_epoch_idx']

with open('analysis_parameters.yaml', 'r') as file:
    config = safe_load(file)

# Choose the axis
config['axis'] = 'Z'

# Choose the epochs object
if config['axis'] == 'X':
    epochs = epochs_x
elif config['axis'] == 'Y':
    epochs = epochs_y
elif config['axis'] == 'Z':
    epochs = epochs_z

# Narrowband filtering via FIR

# Filter parameters
time = epochs.times
srate = epochs.info['sfreq']
nyquist = srate / 2
f_range = [config['fmin'], config['fmax']]  # [1, 30]
trans_w = 0.2
order = np.round(3 * srate / f_range[0]).astype(int)
order = order + 1 if order % 2 == 0 else order  # Convert to an even number (for firls)

gains = [0, 0, 1, 1, 0, 0]
freqs = (
    np.array(
        [
            0,
            f_range[0] - f_range[0] * trans_w,
            *f_range,
            f_range[1] + f_range[1] * trans_w,
            nyquist,
        ]
    )
    / nyquist
)

# Filter kernel
filt_kern = firls(order, freqs, gains)

# Compute the power spectrum of the filter kernel
hz = np.linspace(0, srate / 2, len(filt_kern) // 2 + 1)
filt_pow = np.abs(np.fft.fft(filt_kern))  # ** 2
filt_pow = filt_pow[: len(hz)]

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, :])

# Plot the filter kernel
ax1.plot(filt_kern)
ax1.set_title('Filter Kernel')
ax1.set_xlabel('Time Points')
ax1.set_ylabel('Amplitude')

# Plot amplitude spectrum of the filter kernel
ax2.plot(hz, filt_pow, 'ks-', markerfacecolor='w', label='Actual')
ax2.plot(freqs * nyquist, gains, 'ro-', markerfacecolor='w', label='Ideal')
ax2.set_xlim(0, f_range[1] * 1.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Filter Gain')
ax2.legend()

# Plot filter gain in dB
ax3.axvline(f_range[0], color='grey', linestyle=':')
ax3.axvline(f_range[1], color='grey', linestyle=':')
ax3.plot(hz, 10 * np.log10(filt_pow), 'ks-', markerfacecolor='w')
ax3.set_xlim(0, f_range[1] * 1.5)
ax3.set_ylim(-30, 2)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Filter Gain (dB)')

ax4.set_title('Frequency Response')
ax4.axis('off')

plt.tight_layout()
plt.show()

# %%
# Apply the filter to the epochs


# %%
# Apply the filter to random noise

# Generate random noise as "signal"
signal = np.random.randn(int(srate) * 4)

# Apply the filter kernel to the signal
filt_sig = filtfilt(filt_kern, 1, signal)

fig = plt.figure()
gs = GridSpec(1, 4, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 3])

# Plot time series
ax1.plot(signal, 'r', label='Original')
ax1.plot(filt_sig, 'k', label='Filtered')
ax1.set_xlim(1, len(signal))
ax1.set_title('Time Domain')
ax1.set_xlabel('Time (a.u.)')
ax1.set_ylabel('Amplitude (a.u.)')
ax1.legend()

# Plot power spectrum
hz = np.linspace(0, srate, len(signal))
ax2.plot(hz, np.abs(np.fft.fft(signal)), 'r')
ax2.plot(hz, np.abs(np.fft.fft(filt_sig)), 'k')
ax2.set_xlim(0, f_range[1] * 2)
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %%
