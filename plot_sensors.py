#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot sensor positions and bad channels.

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
from yaml import safe_load

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = safe_load(file)

# Concatenate the raw
raw_list = []
raw_x_list = []
raw_y_list = []
raw_z_list = []
events_list = []

# Aggregate the data from all runs
# for run in ['run_1', 'run_2', 'run_3', 'run_4']:
for run in ['run_1']:
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

    meg.raw.drop_channels([ch for ch in meg.raw.ch_names if 'Trigger' in ch])
    raw_list.append(meg.raw)

    # Get a list of channels for each axis
    x_chs = [ch for ch in meg.raw.ch_names if '[X]' in ch]
    y_chs = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
    z_chs = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

    raw_x_list.append(meg.raw.copy().pick(x_chs))
    raw_y_list.append(meg.raw.copy().pick(y_chs))
    raw_z_list.append(meg.raw.copy().pick(z_chs))

    events_list.append(meg.event_arr)

event_id = meg.event_id
del meg

# Concatenate the raw
raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
# raw.drop_channels([ch for ch in raw.ch_names if 'Trigger' in ch] + raw.info['bads'])
raw_x, events = mne.concatenate_raws(raw_x_list, events_list=events_list)
raw_y, _ = mne.concatenate_raws(raw_y_list, events_list=events_list)
raw_z, _ = mne.concatenate_raws(raw_z_list, events_list=events_list)

# %%
# Plot all the sensors with bad channels marked
ratio_dropped_x = np.round(len(raw_x.info['bads']) / len(raw_x.ch_names) * 100, 1)
ratio_dropped_y = np.round(len(raw_y.info['bads']) / len(raw_y.ch_names) * 100, 1)
ratio_dropped_z = np.round(len(raw_z.info['bads']) / len(raw_z.ch_names) * 100, 1)

fig, axs = plt.subplots(1, 3)
raw_x.plot_sensors(axes=axs[0])
raw_y.plot_sensors(axes=axs[1])
raw_z.plot_sensors(axes=axs[2])

axs[0].set_title(f'X-Axis\n({ratio_dropped_x}% Dropped)')
axs[1].set_title(f'Y-Axis\n({ratio_dropped_y}% Dropped)')
axs[2].set_title(f'Z-Axis\n({ratio_dropped_z}% Dropped)')

plt.tight_layout()
plt.savefig(os.path.join('img', 'sensors', 'sensors_bad.png'))
plt.show()
# %%
# Plot all the sensors
raw_x.info['bads'] = []
raw_y.info['bads'] = []
raw_z.info['bads'] = []

fig, axs = plt.subplots(1, 3)
raw_x.plot_sensors(axes=axs[0])
raw_y.plot_sensors(axes=axs[1])
raw_z.plot_sensors(axes=axs[2])

axs[0].set_title(f'X-Axis\n')
axs[1].set_title(f'Y-Axis\n')
axs[2].set_title(f'Z-Axis\n')

plt.tight_layout()
plt.savefig(os.path.join('img', 'sensors', 'sensors_all.png'))
plt.show()
