# -*- coding: utf-8 -*-
"""
Load OPM Data, convert for use in MNE

@author: Ryan Hill, Molly Rea, Martin Iniguez
"""

# %% Import packages
import mne

from utils import get_mne_data

# %%
data_dir = r"C:\Files\Coding\Python\Neuro\data\Gesture\Rock Paper Scissors"
# data_dir = r"C:\Users\user\Desktop\MasterThesis\data_nottingham"
# data_dir = r'D:\PhD\data\2023-06-21_nottingham'
day = "20230623"
# acq_time = "095228"  # Noise
# acq_time = "100008"  # Noise
# acq_time = "100245"  # Noise
# acq_time = "102814"  # Run 1
# acq_time = "104104"  # Run 2
# acq_time = "105342"  # Run 3 (might have some data from Run 4)
# acq_time = "110808"  # Run 4
# acq_time = "112029"  # Rest

acq_time_list = ["102814", "104104", "105342", "110808"]
raw_list = []
events_list = []

for acq_time in acq_time_list:
    raw, events, event_id = get_mne_data(data_dir, day, acq_time)
    raw_list.append(raw)
    events_list.append(events)


# %%
raw, events = mne.concatenate_raws(raw_list, events_list=events_list)

# %%
