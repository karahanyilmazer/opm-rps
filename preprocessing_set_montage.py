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
os.environ[
    'FREESURFER_HOME'
] = r'\\wsl.localhost\Ubuntu-22.04\usr\local\freesurfer\7.4.1'

import matplotlib.pyplot as plt
import mne
from src.base.EEG import EEG

from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
data_dict = {
    'data_dir': r'C:\Files\Coding\Python\Neuro\data',
    'paradigm': 'Gesture',
    'dataset': 'Nottingham Gesture',
    'device': 'OPM',
    'subject': 11766,
    'session': 20230623,  # or 20230622
    'run': 'run_1',
}

cropping = (0, None)
if data_dict['run'] == 'run_3':
    # Crop the end of Run 3 as it includes the beginning of Run 4
    cropping = (0, 681.449)

fmin, fmax = None, None
tmin, tmax = -0.5, 2.1
# events = ('cue_1', 'cue_2', 'cue_3')
events = None

meg = EEG(
    data_dict,
    bp_filt=(fmin, fmax),
    epoching=(tmin, tmax),
    cropping=cropping,
    events=events,
    apply_notch=False,
    logger_name='meg_analysis',
)

# %%
fiducials = mne.io.read_fiducials(
    r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation\11766\bem\11766-fiducials.fif'
)

# Get a list of indices for the MEG channels
positions = mne.pick_types(meg.raw.info, meg=True, exclude='bads')

# Get channel names
ch_names = [meg.raw.info['ch_names'][i] for i in positions]

# Get the 3D positions of the channels
chs = [meg.raw.info['chs'][i] for i in positions]
ch_locs = [ch['loc'][:3] for ch in chs]

# Create dictionary with channel names as keys and locations as values
ch_dict = dict(zip(ch_names, ch_locs))

dig_montage = mne.channels.make_dig_montage(
    ch_pos=ch_dict,
    nasion=fiducials[0][1]['r'],
    lpa=fiducials[0][0]['r'],
    rpa=fiducials[0][2]['r'],
    coord_frame='meg',
)

meg.raw.set_montage(dig_montage)

# %%
plot_kwargs = dict(
    subject='11766',
    surfaces='head-dense',
    dig='fiducials',
    eeg=[],
    meg='sensors',
    show_axes=True,
    coord_frame='meg',
)
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

coreg = mne.coreg.Coregistration(meg.raw.info, '11766', fiducials=fiducials)
fig = mne.viz.plot_alignment(meg.raw.info, trans=coreg.trans, **plot_kwargs)

# %%
x_axis_channels = [ch for ch in meg.raw.ch_names if '[X]' in ch]
y_axis_channels = [ch for ch in meg.raw.ch_names if '[Y]' in ch]
z_axis_channels = [ch for ch in meg.raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

raws = dict()
raws['x'] = meg.raw.copy().pick(x_axis_channels)
raws['y'] = meg.raw.copy().pick(y_axis_channels)
raws['z'] = meg.raw.copy().pick(z_axis_channels)
# %%
raws['z'].info["bads"] = []
raw_check = raws['z'].copy()
(
    auto_noisy_chs,
    auto_flat_chs,
    auto_scores,
) = mne.preprocessing.find_bad_channels_maxwell(
    raw_check,
    h_freq=None,
    int_order=6,
    return_scores=True,
    verbose=True,
)
print(auto_noisy_chs)  # we should find them!
print(auto_flat_chs)  # none for this dataset
# %%
raw_sss = mne.preprocessing.maxwell_filter(raws['z'], int_order=6, verbose=True)
