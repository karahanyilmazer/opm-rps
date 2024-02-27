#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the event-related potentials of aggregated data for each condition.

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
from mne import read_epochs
from mne.viz import plot_sensors
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable
from yaml import safe_load

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = safe_load(file)

# Concatenate the epochs
epochs_x = read_epochs(
    os.path.join('data', 'pickles', 'epochs', 'all_runs_x-epo.fif.gz')
)
epochs_y = read_epochs(
    os.path.join('data', 'pickles', 'epochs', 'all_runs_y-epo.fif.gz')
)
epochs_z = read_epochs(
    os.path.join('data', 'pickles', 'epochs', 'all_runs_z-epo.fif.gz')
)

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 5))
for ax, axis in zip(axs, ['X', 'Y', 'Z']):
    # Choose the axis
    config['axis'] = axis

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

    # Downsampled time vector
    resolution = 0.007
    times_to_save = np.arange(-0.5, 2.1 + resolution, resolution)
    t_idx = np.argmin(np.abs(time[:, np.newaxis] - times_to_save), axis=0)

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

    # Get the ERPs
    erp_roc = np.mean(data_roc, axis=1)
    erp_pap = np.mean(data_pap, axis=1)
    erp_sci = np.mean(data_sci, axis=1)

    # Plot the ERPs
    ax.plot(time, erp_roc * 1e15, label='Rock')
    ax.plot(time, erp_pap * 1e15, label='Paper')
    ax.plot(time, erp_sci * 1e15, label='Scissors')

    ax.set_ylim(-375, 450)
    ax.set_xlabel('Time (s)')
    if axis == 'X':
        ax.set_ylabel('Amplitude (fT)')
    else:
        ax.set_yticklabels([])
    if axis == 'Y':
        ax.set_title(f'ERPs Over Different Axes')

    ax.legend(loc='lower left', fontsize=10)

    axins = inset_locator.inset_axes(ax, width='30%', height='30%', loc='upper right')

    plot_sensors(
        epochs.copy().pick(f'KB[{config["axis"]}]').info,
        axes=axins,
        show_names=True,
        pointsize=5,
        linewidth=0.1,
    )

plt.tight_layout()
plt.savefig(os.path.join('img', 'presentation', 'erp', f'erp-all_runs.png'))
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 5))
for ax, axis in zip(axs, ['X', 'Y', 'Z']):
    # Choose the axis
    config['axis'] = axis

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

    # Downsampled time vector
    resolution = 0.007
    times_to_save = np.arange(-0.5, 2.1 + resolution, resolution)
    t_idx = np.argmin(np.abs(time[:, np.newaxis] - times_to_save), axis=0)

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

    # Get the ERPs
    erp_roc = np.mean(data_roc, axis=1)
    erp_pap = np.mean(data_pap, axis=1)
    erp_sci = np.mean(data_sci, axis=1)

    # Plot the ERPs
    ax.plot(time, erp_roc * 1e15, label='Rock')
    ax.plot(time, erp_pap * 1e15, label='Paper')
    ax.plot(time, erp_sci * 1e15, label='Scissors')
    ax.axvspan(0.1, 0.9, alpha=0.5, color='gray')

    ax.set_ylim(-375, 450)
    ax.set_xlabel('Time (s)')
    if axis == 'X':
        ax.set_ylabel('Amplitude (fT)')
    else:
        ax.set_yticklabels([])
    if axis == 'Y':
        ax.set_title(f'ERPs Over Different Axes')

    ax.legend(loc='lower left', fontsize=10)

    axins = inset_locator.inset_axes(ax, width='30%', height='30%', loc='upper right')

    plot_sensors(
        epochs.copy().pick(f'KB[{config["axis"]}]').info,
        axes=axins,
        show_names=True,
        pointsize=5,
        linewidth=0.1,
    )

plt.tight_layout()
plt.savefig(os.path.join('img', 'presentation', 'erp', f'erp-all_runs-interval.png'))
plt.show()
