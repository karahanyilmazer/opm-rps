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
from mne.decoding import CSP
from scipy.signal import filtfilt, firls, hilbert
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.base.EEG import EEG
from tqdm import tqdm
from yaml import safe_load

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = safe_load(file)
    bad_epochs_idx = set(
        config['bad_epoch_idx']['X']
        + config['bad_epoch_idx']['Y']
        + config['bad_epoch_idx']['Z']
    )  # Store for later

# Concatenate the raw
raw_list = []
raw_x_list = []
raw_y_list = []
raw_z_list = []
events_list = []

# Aggregate the data from all runs
for run in ['run_1', 'run_2', 'run_3', 'run_4']:
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
        logger_name='meg_analysis',
    )

    raw_list.append(meg.raw)

    # Get a list of channels for each axis
    # x_chs = [ch for ch in meg.raw.ch_names if '[X]' in ch]
    # y_chs = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
    # z_chs = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

    # raw_x_list.append(meg.raw.copy().pick(x_chs))
    # raw_y_list.append(meg.raw.copy().pick(y_chs))
    # raw_z_list.append(meg.raw.copy().pick(z_chs))

    events_list.append(meg.event_arr)

event_id = meg.event_id
del meg

# Concatenate the raw
raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
# raw_x, events = mne.concatenate_raws(raw_x_list, events_list=events_list)
# raw_y, _ = mne.concatenate_raws(raw_y_list, events_list=events_list)
# raw_z, _ = mne.concatenate_raws(raw_z_list, events_list=events_list)

# %%
# Choose the axis
config['axis'] = 'Z'

# Choose the epochs object
if config['axis'] == 'X':
    raw = raw_x
elif config['axis'] == 'Y':
    raw = raw_y
elif config['axis'] == 'Z':
    raw = raw_z

# %%
with open('analysis_parameters.yaml', 'r') as file:
    config = safe_load(file)

# Narrowband filtering via FIR

# Filter parameters
time = raw.times
srate = raw.info['sfreq']
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
# Apply the filter to the raw data
signal = raw.get_data()
filt_sig = filtfilt(filt_kern, 1, signal)
signal_plot = signal[0, :]
filt_plot = filt_sig[0, :]

# %%
# Plot time series
fig = plt.figure()
plt.plot(signal_plot, 'r', label='Original')
plt.plot(filt_plot, 'k', label='Filtered')
plt.xlim(1, len(signal_plot))
plt.title('Time Domain')
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude (a.u.)')
plt.legend()
plt.show()

# %%
# Plot power spectrum
hz = np.linspace(0, srate, len(signal_plot))

plt.figure()
plt.plot(hz, np.abs(np.fft.fft(signal_plot)), 'r', label='Original')
plt.plot(hz, np.abs(np.fft.fft(filt_plot)), 'k', label='Filtered')
plt.xlim(0, f_range[1] * 1.5)
plt.ylim(0, 1.1 * np.max(np.abs(np.fft.fft(filt_plot))))
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.show()

# %%
# Apply the filter to each channel in the Raw data
raw.load_data()  # Ensure the Raw data is preloaded
raw._data = filt_sig

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=config['tmin'],
    tmax=config['tmax'],
    detrend=config['detrend'],
)
epochs.info.normalize_proj()
epochs.drop(list(bad_epochs_idx))
epochs.load_data()

X = epochs[['roc', 'sci']].crop(tmin=0.5, tmax=1.5).get_data(copy=True)
y = epochs[['roc', 'sci']].events[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)

# %%‚
cv_train_scores = []
cv_test_scores = []

skf = KFold(n_splits=3, shuffle=True, random_state=50)

ss = StandardScaler()
svm = SVC(gamma='auto')
lda = LDA()
csp = CSP(n_components=4, reg=1e-4, log=True, norm_trace=False)

pipe = Pipeline([('CSP', csp), ('SVM', svm)])
# pipe = Pipeline([('CSP', csp), ('LDA', lda)])

for train_index, test_index in skf.split(X_train, y_train):
    X_cv_train = X_train[train_index]
    y_cv_train = y_train[train_index]
    X_cv_test = X_train[test_index]
    y_cv_test = y_train[test_index]

    pipe.fit(X_train, y_train)
    cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))

# Use scikit-learn Pipeline with cross_val_score function
# scores = cross_val_score(pipe, X_train, y_train, cv=skf, n_jobs=-1)

# Printing the results
# class_balance = np.mean(y == y[0])
# class_balance = max(class_balance, 1.0 - class_balance)
# print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")


# %%
