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
method = "sLORETA"
# Plot the source estimates
fig, ax = plt.subplots()
ax.plot(stc.times, stc.data[::100, :].T, alpha=0.1, color="k")
ax.plot(stc.times, stc.data.mean(0), color="r")
ax.set_title('Source Time Courses')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'{method} Value')
plt.show()

# %%
# Plot the source activity
vertno_max, time_max = stc.get_peak(hemi="lh")

surfer_kwargs = dict(
    hemi="lh",
    clim=dict(kind="value", lims=[2, 2.6, 6]),
    views="lateral",
    initial_time=time_max,
    time_unit="s",
    size=(1200, 800),
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

brain.save_movie(
    filename='source_activity.mp4',
    tmin=-0.05,
    tmax=0.4,
    interpolation='linear',
    time_dilation=20,
    framerate=10,
    time_viewer=True,
)
# %%
report = mne.Report(title="Source estimate example")
report.add_stc(
    stc='11766-misaligned-ico5',
    subject="11766",
    title="Source estimate",
    n_time_points=200,  # few for speed
    stc_plot_kwargs=dict(
        hemi="lh",
        clim=dict(kind="value", lims=[2, 2.6, 6]),
        views="lateral",
        initial_time=time_max,
        time_unit="s",
        smoothing_steps=10,
    ),
)
report.save("report_inverse_sol.html", overwrite=True)

# %%

# Use MNE for source activity

# Read in cleaned epochs
epochs = mne.read_epochs('all_runs-filt_7_30-epo.fif.gz')
epochs.load_data()
epochs.drop_channels([ch for ch in epochs.ch_names if 'Trigger' in ch])

# Get a list of channels for each axis
x_chs = [ch for ch in epochs.ch_names if '[X]' in ch]
y_chs = [ch for ch in epochs.ch_names if '[Y]' in ch]
z_chs = [ch for ch in epochs.ch_names if '[Z]' in ch and 'Trigger' not in ch]

evoked = epochs.average()

method = "MNE"
snr = 3.0
lambda2 = 1.0 / snr**2
stc = mne.minimum_norm.apply_inverse(
    evoked, inv, lambda2, method=method, pick_ori=None, verbose=True
)

vertno_max, time_max = stc.get_peak(hemi="lh")

surfer_kwargs = dict(
    hemi="lh",
    # clim=dict(kind="value", lims=[2, 2.6, 6]),
    clim=dict(kind="percent", lims=[99, 99.9, 99.99]),
    views="lateral",
    initial_time=time_max,
    time_unit="s",
    size=(1200, 800),
    smoothing_steps=10,
    background="w",
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

# %%
