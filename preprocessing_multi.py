#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess OPM data from multiple runs.  

@author: Karahan Yilmazer
"""

# %% Import packages
# !%matplotlib qt
import mne
import numpy as np

from utils import get_mne_data

# %%
data_dir = r'C:\Files\Coding\Python\Neuro\data\Gesture\Rock Paper Scissors'
# data_dir = r'C:\Users\user\Desktop\MasterThesis\data_nottingham'
# data_dir = r'D:\PhD\data\2023-06-21_nottingham'
day = '20230623'
# acq_time = '095228'  # Noise
# acq_time = '100008'  # Noise
# acq_time = '100245'  # Noise
# acq_time = '102814'  # Run 1
# acq_time = '104104'  # Run 2
# acq_time = '105342'  # Run 3 (has some data from Run 4 after 681.499 s)
# acq_time = '110808'  # Run 4
# acq_time = '112029'  # Rest

acq_time_list = ['102814', '104104', '105342', '110808']
raws_list = []
events_list = []

# Iterate over all the runs
for acq_time in acq_time_list:
    raw, events, event_id = get_mne_data(data_dir, day, acq_time)
    # Crop the end of Run 3 as it includes the beginning of Run 4
    if acq_time == '105342':
        raw.crop(tmax=681.499)
        idx = np.where(events[:, 2] == 255)[0][1] + 1
        events = events[:idx, :]
    raws_list.append(raw)
    events_list.append(events)

# %%
# Plotting for sanity check
raws_list[0].compute_psd().plot()
# raws_list[0].plot(events=events_list[0], event_id=event_id, scalings='auto')

# Bad channels (Run 1): LN[X], LN[Y], LN[Z], FZ[X], K9[X], KF[Y], MU[Z]

# %%
# Concatenate all raw and event objects
raw, events = mne.concatenate_raws(raws_list, events_list=events_list)

# %%