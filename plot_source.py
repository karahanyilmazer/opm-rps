#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize source space objects.

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

# %%
# Read in the source space objects
src = mne.read_source_spaces('11766_misaligned-oct6-src.fif')
bem_model = mne.read_bem_surfaces('11766-misaligned-ico5-bem.fif')
fwd = mne.read_forward_solution('11766-misaligned-ico5-fwd.fif')
inv = mne.minimum_norm.read_inverse_operator('11766-misaligned-ico5-meg-inv.fif')
stc = mne.read_source_estimate('11766-misaligned-ico5')
stc.subject = '11766'

# %%
method = "dSPM"
# Plot the source estimates
fig, ax = plt.subplots()
ax.plot(stc.times, stc.data[::100, :].T)
ax.set_title('Source Time Courses')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'{method} value')
plt.show()

# %%
vertno_max, time_max = stc.get_peak(hemi="lh")

surfer_kwargs = dict(
    hemi="lh",
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=time_max,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=10,
)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi="lh",
    color="blue",
    scale_factor=0.6,
    alpha=0.5,
)
brain.add_text(
    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
)

# The documentation website's movie is generated with:
# brain.save_movie(..., tmin=0.05, tmax=0.15, interpolation='linear',
#                  time_dilation=20, framerate=10, time_viewer=True)

# %%
