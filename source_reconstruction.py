#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute and save source space objects.

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

import mne

# %%
# Read in cleaned epochs
epochs = mne.read_epochs('all_runs-filt_7_30-epo.fif.gz')
epochs.load_data()
epochs.drop_channels([ch for ch in epochs.ch_names if 'Trigger' in ch])

# Get a list of channels for each axis
x_chs = [ch for ch in epochs.ch_names if '[X]' in ch]
y_chs = [ch for ch in epochs.ch_names if '[Y]' in ch]
z_chs = [ch for ch in epochs.ch_names if '[Z]' in ch and 'Trigger' not in ch]

evoked = epochs.average()

# %%
# Plot the annotated left hemisphere
Brain = mne.viz.get_brain_class()
brain = Brain('11766', hemi='lh', surf='pial', size=(800, 600))
brain.add_annotation('aparc.a2009s', borders=False)
# %%
# Plotting arguments
plot_bem_kwargs = dict(
    subject='11766',
    brain_surfaces='white',
    orientation='coronal',
    slices=[50, 100, 150, 200],
)

# Plot the BEM surfaces
mne.viz.plot_bem(**plot_bem_kwargs)

# %%
# Set up bilateral hemisphere surface-based source space with subsampling
src = mne.setup_source_space('11766', add_dist='patch', spacing='oct6', n_jobs=-1)
mne.write_source_spaces('11766_misaligned-oct6-src.fif', src, overwrite=True)
print(src)

# %%
# Plot the sources on the cortical slices
mne.viz.plot_bem(src=src, **plot_bem_kwargs)

# %%
# Plot the sources on the brain
fig = mne.viz.plot_alignment(
    subject='11766',
    surfaces="white",
    coord_frame="mri",
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)
# %%
# Define layer conductivities
conductivity = (0.3,)  # Single layer --> can be used for MEG but not for EEG
# conductivity = (0.3, 0.006, 0.3)  # Three layers

# Compute the BEM model and solution
model = mne.make_bem_model(subject='11766', ico=5, conductivity=conductivity)
bem = mne.make_bem_solution(model)
mne.write_bem_solution('11766-misaligned-ico5-bem.fif', bem, overwrite=True)

# %%
# Compute the forward operator, i.e., the gain or leadfield matrix
fwd = mne.make_forward_solution(
    epochs.info,
    trans=mne.transforms.Transform('mri', 'head'),
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    n_jobs=-1,
    verbose=True,
)
print(fwd)

print(f"Before: {src}")
print(f'After:  {fwd["src"]}')

leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

mne.write_forward_solution('11766-misaligned-ico5-fwd.fif', fwd, overwrite=True)

# %%
# Extract the forward operator corresponding to the
# source space with cortical orientation constraint
# fwd_fixed = mne.convert_forward_solution(
#     fwd, surf_ori=True, force_fixed=True, use_cps=True
# )
# leadfield = fwd_fixed["sol"]["data"]
# print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
# %%
# Compute the noise covariance matrix
cov = mne.read_cov('11766-misaligned-ico5-cov.fif')
# cov = mne.compute_covariance(epochs, method='auto')
# cov.save('11766-misaligned-ico5-cov.fif')

# %%
# Compute the inverse operator
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2)
mne.minimum_norm.write_inverse_operator('11766-misaligned-ico5-meg-inv.fif', inv)

# %%
# Compute the source estimates
method = 'sLORETA'
# snr = 3.0
# lambda2 = 1.0 / snr**2
stc, residual = mne.minimum_norm.apply_inverse(
    evoked,
    inv,
    # lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)
stc.save('11766-misaligned-ico5', overwrite=True)

# %%
# Perform group analysis
# morph = mne.compute_source_morph(stc, subject_from='11766', subject_to='fsaverage')
# stc_fsaverage = morph.apply(stc)

# %%
# Plot the source activity
vertno_max, time_max = stc.get_peak(hemi="lh")

surfer_kwargs = dict(
    hemi="lh",
    clim=dict(kind="value", lims=[2, 2.5, 4]),
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
    hemi="lh",
    tmin=-0.05,
    tmax=0.4,
    interpolation='linear',
    time_dilation=20,
    framerate=10,
    time_viewer=True,
)


# %%
# Create a report
report = mne.Report(title="Source Space")
report.add_bem(
    subject="11766",
    title="MRI & BEM",
    # decim=40,
    width=512,
    n_jobs=-1,
)
report.add_trans(
    trans=mne.transforms.Transform('mri', 'head'),
    info=epochs.info,
    subject="11766",
    alpha=1.0,
    title="Coregistration",
)
report.save("report_source.html", overwrite=True)
