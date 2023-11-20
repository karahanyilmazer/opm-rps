#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computing the forward model from OPM data.  

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')
os.environ[
    'SUBJECTS_DIR'
] = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\MRI\Segmentation'

import matplotlib.pyplot as plt
import mne
from src.base.EEG import EEG

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
# %%
data_dict = {
    'data_dir': r'C:\Files\Coding\Python\Neuro\data',
    'paradigm': 'Gesture',
    'dataset': 'Nottingham Gesture',
    'device': 'OPM',
    'subject': '11766',
    'session': 20230623,  # or 20230622
    'run': 'run_4',
}

# First look:
# Bad channels (Run 1): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z]
# Bad channels (Run 2): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z]
# Bad channels (Run 3): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]
# Bad channels (Run 4): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z], LB[X], LB[Y], MV[X], MV[Y], MV[Z], HF[Y], HJ[X]

# Second look:
# Bad channels (Run 1): LN[X, Y, Z], FZ[X], K9[X], KF[Y], MU[Z], FR[Z]
# Sus channels (Run 1): FR[X, Y], LB[X, Y], HJ[X], HF[Y], KD[Y], KB[Z]

# Bad channels (Run 2): LN[X, Y, Z], MV[X, Y, Z], LB[X, Y], FZ[X], K9[X], HF[Y], KC[Y], KF[Y], MU[Z]
# Sus channels (Run 2): G0[X], HJ[X], LF[X], K9[Y], KC[Z]

# Bad channels (Run 3): LN[X, Y, Z], K9[X], FZ[X], MV[X, Y, Z], LB[X, Y], LP[X, Y], HJ[X, Y], KC[Y, Z], KE[Y], KF[Y], MU[Z]
# Sus channels (Run 3): HO[Z], LF[X]

# Bad channels (Run 4): LN[X, Y, Z], K9[X], FZ[X], LB[X], HJ[X], LP[X], KF[Y], MV[Y, Z], KC[Y, Z], KE[Y], HF[Y], LB[Y], MU[Z]
# Sus channels (Run 4): G0[X], K9[Z], LM[Z], LP[Z]

combined_bads = [
    'FR[Z]',
    'FZ[X]',
    'G0[X]',
    'HF[Y]',
    'HJ[X]',
    'HJ[Y]',
    'K9[X]',
    'KC[Y]',
    'KC[Z]',
    'KE[Y]',
    'KF[Y]',
    'LB[X]',
    'LB[Y]',
    'LF[X]',
    'LN[X]',
    'LN[Y]',
    'LN[Z]',
    'LP[X]',
    'LP[Y]',
    'LP[Z]',
    'MV[X]',
    'MV[Y]',
    'MV[Z]',
    'MU[Z]',
]

cropping = (0, None)
if data_dict['run'] == 'run_3':
    # Crop the end of Run 3 as it includes the beginning of Run 4
    cropping = (0, 680)

fmin, fmax = 1, 400
tmin, tmax = -0.5, 2.1
events = ('roc', 'pap', 'sci')

notch_freqs = {
    'run_1': (227, 277.15),
    'run_2': (227, 277.15),  # 182
    'run_3': (227, 277.15),
    'run_4': (227, 277.15),
}

meg = EEG(
    data_dict,
    bp_filt=(fmin, fmax),
    extra_notch_freqs=notch_freqs[data_dict['run']],
    notch_params={'notch_widths': 2},
    epoching=(tmin, tmax),
    cropping=cropping,
    events=events,
    apply_notch=True,
    bad_chs=combined_bads,
    logger_name='meg_analysis',
)
meg.raw.compute_psd().plot()

# %%
# Plot the MRI <-> MEG alignment
mne.viz.plot_alignment(
    meg.raw.info,
    subject=data_dict['subject'],
    trans=mne.transforms.Transform('head', 'mri'),
    surfaces=['white', 'inner_skull', 'outer_skull', 'outer_skin'],
    coord_frame='meg',
    show_axes=True,
)

# %%
# Plot BEM surfaces
plot_bem_kwargs = dict(
    subject=data_dict['subject'],
    brain_surfaces='white',
    orientation='coronal',
    slices=[25, 50, 75, 100, 125, 150, 175, 200],
)

mne.viz.plot_bem(**plot_bem_kwargs)

# %%
# Compute the surface-based source space
src = mne.setup_source_space(data_dict['subject'], spacing="oct6", add_dist="patch")
print(src)

# %%
# Plot the sources
mne.viz.plot_bem(src=src, **plot_bem_kwargs)

# %%
# Compute the volume-based source space
surface = os.path.join(
    os.environ['SUBJECTS_DIR'], data_dict['subject'], 'bem', 'inner_skull.surf'
)
vol_src = mne.setup_volume_source_space(
    data_dict['subject'],
    surface=surface,
    add_interpolator=False,
)  # Just for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

# %%
fig = mne.viz.plot_alignment(
    subject=data_dict['subject'],
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
# Compute one-layer BEM
conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=data_dict['subject'],
    ico=4,
    conductivity=conductivity,
)
bem = mne.make_bem_solution(model)

# %%
# Compute the forward model
fwd = mne.make_forward_solution(
    meg.raw.info,
    trans=mne.transforms.Transform('head', 'mri'),
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
)
print(fwd)
print(f"Before: {src}")
print(f'After:  {fwd["src"]}')

# %%
# Explore the content of the forward model
leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

# Extract the forward operator corresponding to the source space
fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

# %%
# Save the forward model
fname = os.path.join('data', 'forward_model', f'meg_{data_dict["run"]}-fwd.fif.gz')
mne.write_forward_solution(fname, fwd)

# %%
