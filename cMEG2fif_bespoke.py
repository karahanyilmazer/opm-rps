# -*- coding: utf-8 -*-
"""
Load OPM Data, convert for use in MNE

@author: Ryan Hill, Molly Rea, Martin Iniguez
"""

# %% Import packages
import json
import os
from typing import Tuple

import mne
import numpy as np
import pandas as pd
from mne.io.constants import FIFF

from utils import (
    calc_pos,
    conv_square_window,
    create_chans_dict,
    find_matching_indices,
    get_channels_and_data,
    read_old_cMEG,
)


def get_data_mne(
    data_dir: str, day="20230622", acq_time="155445"
) -> Tuple[mne.io.array.array.RawArray, np.ndarray, dict]:
    """Get data from a cMEG file and convert it to a MNE raw object.

    :param subjects_dir: The path to the data directory.
    :type subjects_dir: str
    :param day: The day of the scan.
    :type day: str
    :param acq_time: The time of the scan.
    :type acq_time: str
    :return: The MNE raw object, the events and dictionary of event ids.
    :rtype: Tuple[mne.io.RawArray, np.ndarray, dict]
    """

    # configure subjects directory
    # data_dir = r'C:\Users\user\Desktop\MasterThesis\data_nottingham'
    # subject = "11766"

    # Data filename and path
    file_path = os.path.join(
        data_dir,
        day,
        "Bespoke scans",
        day + "_" + acq_time + "_cMEG_Data",
        day + "_" + acq_time + "_meg.cMEG",
    )
    file_path = file_path.replace("\\", "/")
    file_path_split = os.path.split(file_path)
    fpath = file_path_split[0] + "/"
    fname = file_path_split[1]

    # Load Data
    # Requires a single cMEG file, doesn't concatenate runs yet
    print("Loading File")

    # Load data
    data_input = read_old_cMEG(fpath + fname)
    data_raw = data_input[1:, :]  # Remove first row, including time stamps
    del data_input

    fname_pre = fname.split("_meg.cMEG")[0]
    f = open(fpath + fname_pre + "_meg.json")
    # TODO: check if the names make sense between channels and HelmConfig
    tsv_file = {
        "channels": pd.read_csv(fpath + fname_pre + "_channels.tsv", sep="\t"),
        "HelmConfig": pd.read_csv(fpath + fname_pre + "_HelmConfig.tsv", sep="\t"),
        "SensorTransform": pd.read_csv(
            fpath + fname_pre + "_SensorTransform.tsv", header=None, sep="\t"
        ),
        "JSON": json.load(f),
    }
    f.close()
    samp_freq = tsv_file["JSON"]["SamplingFrequency"]

    # Sensor indexes and locations
    names = tsv_file["channels"]["name"]
    sensors = tsv_file["HelmConfig"]["Sensor"]
    loc_idx = find_matching_indices(names, sensors)
    chans = create_chans_dict(tsv_file, loc_idx)
    # TODO: make it so it is a DataFrame instead of a dict of lists.

    # Sensor information
    print("Sorting Sensor Information")
    try:
        ch_scale = pd.Series.tolist(tsv_file["channels"]["nT/V"])
    except KeyError:
        tsv_file["channels"].rename(columns={"nT0x2FV": "nT/V"}, inplace=True)
        ch_scale = pd.Series.tolist(
            tsv_file["channels"]["nT/V"]
        )  # Scale factor from V to nT
    ch_names, ch_types, data = get_channels_and_data(data_raw, tsv_file, ch_scale)
    sfreq = samp_freq

    # Create MNE info object
    print("Creating MNE Info")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info["line_freq"] = tsv_file["JSON"]["PowerLineFrequency"]

    # Sort sensor locations
    print("Sensor Location Information")
    nmeg = 0
    nstim = 0
    nref = 0
    chs = list()

    for ii in range(tsv_file["channels"].shape[0]):
        # Create channel information
        ch = dict(
            scanno=ii + 1,
            range=1.0,
            cal=1.0,
            loc=np.full(12, np.nan),
            unit_mul=FIFF.FIFF_UNITM_NONE,
            ch_name=tsv_file["channels"]["name"][ii].replace(" ", ""),
            coil_type=FIFF.FIFFV_COIL_NONE,
        )

        pos = chans["Locations"][ii]
        ori = chans["Orientations"][ii]

        # Calculate sensor position
        if sum(np.isnan(pos)) == 0:
            ch["loc"] = calc_pos(pos, ori)

        # Update channel depending on type
        # TODO: check if we can do ch_types instead of this weird replacement
        if chans["Channel_Type"][ii].replace(" ", "") == "TRIG":  # its a trigger!
            nstim += 1
            info["chs"][ii].update(
                logno=nstim,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_STIM_CH,
                unit=FIFF.FIFF_UNIT_V,
                cal=1,
            )

        elif chans["Channel_Type"][ii].replace(" ", "") == "MISC":  # its a BNC channel
            nref += 1
            info["chs"][ii].update(
                logno=nstim,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_STIM_CH,
                unit=FIFF.FIFF_UNIT_V,
                cal=1,
            )

        elif sum(np.isnan(pos)) == 3:  # its a sensor with no location info
            nref += 1
            info["chs"][ii].update(
                logno=nref,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_REF_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                cal=1e-9 / tsv_file["channels"]["nT/V"][ii],
            )

        else:  # its a sensor!
            nmeg += 1
            info["chs"][ii].update(
                logno=nmeg,
                coord_frame=FIFF.FIFFV_COORD_DEVICE,
                kind=FIFF.FIFFV_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                loc=ch["loc"],
                cal=1e-9 / tsv_file["channels"]["nT/V"][ii],
            )
            # TODO: check if we are not multiplying twice by the factor.

        chs.append(ch)

    # Might need some transform for the sensor positions (from MNE to head reference
    # frame) Quaternion 4x4 matrix. For now for us we see its identity.
    info["dev_head_t"] = mne.transforms.Transform(
        "meg", "head", pd.DataFrame(tsv_file["SensorTransform"]).to_numpy()
    )

    # Create MNE raw object
    print("Create raw object")
    raw = mne.io.RawArray(data, info)

    # Set bad channels defined in channels.tsv. Do not know if its something we need to
    # do ourselves.
    idx = tsv_file["channels"].status.str.strip() == "Bad"
    bad_ch = tsv_file["channels"].name[idx.values]
    raw.info["bads"] = bad_ch.str.replace(" ", "").to_list()

    # Create events
    stm_misc_chans = mne.pick_types(info, stim=True, misc=True)
    data_stim = data[stm_misc_chans]
    trig_data_sum = (np.sum(data_stim, axis=0) >= 1) * 1.0
    on_inds = np.where(np.diff(trig_data_sum, prepend=0) == 1)[0]

    # Convolute to fix when triggers are not happening exactly at same sample time
    data_stim_conv = conv_square_window(data=data_stim, window_size=5)

    event_values = []
    for on_ind in on_inds:
        event_values.append(
            np.sum((data_stim_conv[:, on_ind] > 0.5) * 2 ** np.arange(0, 8))
        )

    events = np.array(
        [(on_ind, 0, value) for on_ind, value in zip(on_inds, event_values)]
    )

    # Define the event_id dictionary with swapped keys and values
    event_id = {
        "cue_1": 1,
        "cue_2": 2,
        "cue_3": 3,
        "cue_4": 4,
        "cue_5": 5,
        "end_trial": 7,
        "press_1": 8,
        "press_2": 16,
        "press_3": 32,
        "press_4": 64,
        "press_5": 128,
        "experiment_marker": 255,
    }

    # #%% Digitisation and montage

    # print("Digitisation")
    # ch_pos = dict()
    # for ii in range(tsv_file["channels"].shape[0]):
    #     pos1 = chans["Locations"][ii]
    #     if sum(np.isnan(pos1)) == 0:
    #         ch_pos[chans["Channel_Name"][ii].replace(" ", "")] = pos1

    # It is a system of 3D points. We need to convert it to a montage. Can be used for
    # Source Analysis and ICA maybe? We might leave it out for now?
    # mtg = mne.channels.make_dig_montage(ch_pos=ch_pos)
    # raw.set_montage(mtg)  # TODO: problems setting the montage

    return raw, events, event_id


# %%
