#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for convenience of plotting multiple recordings.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
import os
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib._pylab_helpers import Gcf
from src.utils.DataLoader import DataLoader

# High-DPI monitor settings
if 'qApp' not in vars():
    from matplotlib.backends.qt_compat import QtWidgets

    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
# %%
loader = DataLoader()

data_dir = r'C:\Files\Coding\Python\Neuro\data\Gesture\Nottingham\Rock Paper Scissors'
day = '20230623'  # or '20230622'
acq_time_dict = {
    'noise_1': '095228',
    'noise_2': '100008',
    'noise_3': '100245',
    'run_1': '102814',
    'run_2': '104104',
    'run_3': '105342',  # has some data from run_4 after 681.499 s
    'run_4': '110808',
    'rest': '112029',
}

for acq_time_key in ['run_1', 'run_2', 'run_3', 'run_4']:
    raw, events, event_id = loader.get_meg_data(
        data_dir, day, acq_time_dict[acq_time_key]
    )
    # Crop the end of Run 3 as it includes the beginning of Run 4
    if acq_time_key == 'run_3':
        raw.crop(tmax=681.499)
        idx = np.where(events[:, 2] == 255)[0][1] + 1
        events = events[:idx, :]

    combined_bads = [
        'FR[Z]',
        'FZ[X]',
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
        'LN[X]',
        'LN[Y]',
        'LN[Z]',
        'LP[X]',
        'LP[Y]',
        'MV[X]',
        'MV[Y]',
        'MV[Z]',
        'MU[Z]',
    ]

    raw.info['bads'] = combined_bads

    x_axis_channels = [ch for ch in raw.ch_names if '[X]' in ch]
    y_axis_channels = [ch for ch in raw.ch_names if '[Y]' in ch]
    z_axis_channels = [ch for ch in raw.ch_names if '[Z]' in ch and 'Trigger' not in ch]

    raws = dict()
    raws['x'] = raw.copy().pick(x_axis_channels)
    raws['y'] = raw.copy().pick(y_axis_channels)
    raws['z'] = raw.copy().pick(z_axis_channels)
    del raw

    for axis in ['x', 'y', 'z']:
        mne_fig = raws[axis].compute_psd().plot(picks='data', exclude='bads')
        mne_fig._localaxes[0].set_title(
            f'MEG (Run {acq_time_key[-1]}, All Bads Dropped, {axis.upper()}-axis)'
        )
        manager = Gcf.get_fig_manager(mne_fig.number)
        manager.window.showMaximized()
        f_name = f'psd-{acq_time_key}_all_dropped_{axis}.png'
        plt.pause(0.1)
        plt.savefig(os.path.join('figures', f_name))

# %%
# Bad channels (Run 1): LN[X, Y, Z], FZ[X], K9[X], KF[Y], MU[Z], FR[Z]
# Sus channels (Run 1): FR[X, Y], LB[X, Y], HJ[X], HF[Y], KD[Y], KB[Z]

# Bad channels (Run 2): LN[X, Y, Z], MV[X, Y, Z], LB[X, Y], FZ[X], K9[X], HF[Y], KC[Y], KF[Y], MU[Z]
# Sus channels (Run 2): G0[X], HJ[X], LF[X], K9[Y], KC[Z]

# Bad channels (Run 3): LN[X, Y, Z], K9[X], FZ[X], MV[X, Y, Z], LB[X, Y], LP[X, Y], HJ[X, Y], KC[Y, Z], KE[Y], KF[Y], MU[Z]
# Sus channels (Run 3): HO[Z]

# Bad channels (Run 4): LN[X, Y, Z], K9[X], FZ[X], LB[X], HJ[X], LP[X], KF[Y], MV[Y, Z], KC[Y, Z], KE[Y], HF[Y], LB[Y], MU[Z]
# Sus channels (Run 4): G0[X], K9[Z], LM[Z], LP[Z]
