#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Localize sources from OPM data.

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
from mne.minimum_norm import (
    apply_inverse,
    make_inverse_operator,
    write_inverse_operator,
)
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

# %%
# Compute the covariance matrices
data_cov = mne.compute_covariance(meg.epochs, tmin=0.01, tmax=0.25, method='empirical')
noise_cov = mne.compute_covariance(
    meg.epochs, tmin=tmin, tmax=0, method=['shrunk', 'empirical']
)
# data_cov.plot(meg.epochs.info)
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, meg.raw.info)

# %%
evoked = meg.epochs.average().pick('meg')
evoked.plot(time_unit='s')
# evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag')

# %%
evoked.plot_white(noise_cov, time_unit='s')

# %%
fname = os.path.join('data', 'forward_model', f'meg_{data_dict["run"]}-fwd.fif.gz')
fwd = mne.read_forward_solution(fname)

# %%
inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)
del fwd

fname = os.path.join('data', 'inverse_model', f'meg_{data_dict["run"]}-inv.fif.gz')
# write_inverse_operator(fname, inv)
# %%
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2
stc, residual = apply_inverse(
    evoked,
    inv,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# %%
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel="time (ms)", ylabel="%s value" % method)

# %%
# evoked.plot()
residual.plot()

# %%
vertno_max, time_max = stc.get_peak(hemi="lh")

brain = stc.plot(
    hemi="split",
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=0,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=10,
)


# brain.add_foci(
#     vertno_max,
#     coords_as_verts=True,
#     hemi="lh",
#     color="blue",
#     scale_factor=0.6,
#     alpha=0.5,
# )

brain.add_text(
    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
)

# %%
brain = stc.plot(subject='11766', initial_time=0)

# %%
