#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot PSD plots before and after preprocessing for single runs.

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
import yaml
from src.base.EEG import EEG

from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
# Load the YAML file
with open('preprocessing_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

for run in ['run_1', 'run_2', 'run_3', 'run_4']:
    config['run'] = run

    meg = EEG(
        config,
        epoching=(config['tmin'], config['tmax']),
        cropping=config['cropping'][run],
        events=config['events'],
        apply_notch=False,
        extra_notch_freqs=config['notch_freqs'][run],
        notch_params={'notch_widths': config['notch_widths'][run]},
        apply_bandpass=False,
        bad_epochs=config['bad_epochs'][run],
        logger_name='meg_analysis',
    )

    x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
    y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
    z_axis_channels = [
        ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch
    ]

    # Unprocessed PSD plots
    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(x_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, X-Axis, All Channels')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_x-all_channels.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(y_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Y-Axis, All Channels')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_y-all_channels.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(z_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Z-Axis, All Channels')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_z-all_channels.png')
    )

    # Bad channels dropped PSD plots
    meg.raw.drop_channels(
        [ch for ch in config['bad_channels'] if ch in meg.raw.ch_names]
    )

    x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
    y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
    z_axis_channels = [
        ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch
    ]

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(x_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, X-Axis, Bad Channels Dropped')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_x-bads_dropped.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(y_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Y-Axis, Bad Channels Dropped')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_y-bads_dropped.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(z_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Z-Axis, Bad Channels Dropped')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_z-bads_dropped.png')
    )

    # HFC filtered PSD plots
    projs = mne.preprocessing.compute_proj_hfc(meg.raw.info, order=2)
    meg.raw.add_proj(projs).apply_proj()

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(x_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, X-Axis, HFC Applied')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_x-hfc_applied.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(y_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Y-Axis, HFC Applied')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_y-hfc_applied.png')
    )

    fig, axs = plt.subplots(figsize=(12, 6))
    meg.raw.copy().pick(z_axis_channels).compute_psd().plot(axes=axs, show=False)
    axs.set_title(f'Run {run[-1]}, Z-Axis, HFC Applied')
    plt.savefig(
        os.path.join('img', 'presentation', 'psd', f'raw-{run[-1]}_z-hfc_applied.png')
    )

    plt.close('all')

# %%
