import json
import os
import re
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
from mne.io.constants import FIFF


def read_old_cMEG(filename: str) -> np.ndarray:
    """
    Reads old cMEG files from the OPM system. The file is read in binary format
    and the header is used to determine the dimensions of the array. The
    dimensions are then used to reshape the array.

    :param filename: The filename of the file to be read.
    :type filename: str
    :return: The data from the file. We assume it will have the shape (Nch, N_samples).
    :rtype: np.ndarray
    """
    size = os.path.getsize(filename)  # Find its byte size
    array_conv = np.array([2**32, 2 ^ 16, 2 ^ 8, 1])  # header convertion table
    arrays = []
    with open(filename, "rb") as fid:
        while fid.tell() < size:
            # Read the header of the array which gives its dimensions
            Nch = np.fromfile(fid, ">u1", sep="", count=4)
            N_samples = np.fromfile(fid, ">u1", sep="", count=4)
            # Multiply by convertion array
            dims = np.array([np.dot(array_conv, Nch), np.dot(array_conv, N_samples)])
            # Read the array and shape it to dimensions given by header
            array = np.fromfile(fid, ">f8", sep="", count=dims.prod())
            arrays.append(array.reshape(dims))

        data = np.concatenate(arrays, axis=1)
        data = data

    return data


def find_matching_indices(
    names: pd.core.series.Series, sensors: pd.core.series.Series
) -> np.ndarray:
    """
    Find matching indices between listen of possible sensors and actual channel names.

    :param names: A list of actual channel names.
    :type names: pd.core.series.Series
    :param sensors: A list of total possible sensors.
    :type sensors: pd.core.series.Series
    :return: An array of indices for the matching names.
    :rtype: np.ndarray

    Example:
        >>> names = ["sensor1", "sensor2", "sensor3"]
        >>> sensors = ["sensor2", "sensor4", "sensor1"]
        >>> loc_idx = find_matching_indices(names, sensors)
        >>> print(loc_idx)
        [2, 0, nan]
    """
    loc_idx = np.full(np.size(names), np.nan)
    count2 = 0
    for n in pd.Series.tolist(names):
        match_idx = np.full(np.size(pd.Series.tolist(sensors)), 0)
        count = 0
        for nn in pd.Series.tolist(sensors):
            if re.sub("[\W_]+", "", n) == re.sub("[\W_]+", "", nn):
                match_idx[count] = 1
            count = count + 1
        if np.array(np.where(match_idx == 1)).size > 0:
            loc_idx[count2] = np.array(np.where(match_idx == 1))
        count2 = count2 + 1
    return loc_idx


def create_chans_dict(tsv_file: dict, loc_idx: np.ndarray) -> dict:
    """
    Create the 'chans' dictionary with channel information.

    :param tsv_file: Dictionary containing channel information.
    :type tsv_file: dict
    :param loc_idx: Array of indices for the matching names.
    :type loc_idx: np.ndarray
    :return: Dictionary containing channel information.
    :rtype: dict
    """
    chans = {
        "Channel_Name": pd.Series.tolist(tsv_file["channels"]["name"]),
        "Channel_Type": pd.Series.tolist(tsv_file["channels"]["type"]),
        "Locations": np.zeros(
            (np.size(pd.Series.tolist(tsv_file["channels"]["name"])), 3)
        ),
        "Orientations": np.zeros(
            (np.size(pd.Series.tolist(tsv_file["channels"]["name"])), 3)
        ),
        "Loc_Name": [None] * np.size(pd.Series.tolist(tsv_file["channels"]["name"])),
    }
    for n in range(np.size(loc_idx)):
        if not np.isnan(loc_idx[n]):
            chans["Locations"][n][0] = float(tsv_file["HelmConfig"]["Px"][loc_idx[n]])
            chans["Locations"][n][1] = float(tsv_file["HelmConfig"]["Py"][loc_idx[n]])
            chans["Locations"][n][2] = float(tsv_file["HelmConfig"]["Pz"][loc_idx[n]])
            chans["Orientations"][n][0] = float(
                tsv_file["HelmConfig"]["Ox"][loc_idx[n]]
            )
            chans["Orientations"][n][1] = float(
                tsv_file["HelmConfig"]["Oy"][loc_idx[n]]
            )
            chans["Orientations"][n][2] = float(
                tsv_file["HelmConfig"]["Oz"][loc_idx[n]]
            )
            chans["Loc_Name"][n] = tsv_file["HelmConfig"]["Name"][loc_idx[n]]
        else:
            chans["Locations"][n][0] = np.nan
            chans["Locations"][n][1] = np.nan
            chans["Locations"][n][2] = np.nan
            chans["Orientations"][n][0] = np.nan
            chans["Orientations"][n][1] = np.nan
            chans["Orientations"][n][2] = np.nan

    return chans


def get_channel_names(tsv_file: dict) -> List[str]:
    """
    Get the channel names from the tsv_file.

    :param tsv_file: Dictionary containing the tsv file data.
    :type tsv_file: dict
    :return: List of channel names.
    :rtype: List[str]
    """
    ch_names1 = pd.Series.tolist(tsv_file["channels"]["name"])
    ch_names = [n.replace(" ", "") for n in ch_names1]
    return ch_names


def get_channels_and_data(
    data_raw: np.ndarray, tsv_file: dict, ch_scale: List[float]
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Get the channel names, types and data from the tsv_file.

    :param data_raw: The raw data from the file.
    :type data_raw: np.ndarray
    :param tsv_file: Dictionary containing the tsv file data.
    :type tsv_file: dict
    :param ch_scale: List of channel scales.
    :type ch_scale: List[float]
    :return: Tuple containing the channel names, types and processed data.
    :rtype: Tuple[List[str], List[str], np.ndarray]
    """
    ch_types = []
    data = np.empty(data_raw.shape)
    ch_names = get_channel_names(tsv_file)
    # We got through the type of channels.
    for count, n in enumerate(pd.Series.tolist(tsv_file["channels"]["type"])):
        # Change MEGMAG to mag because that way it is MNE compatible.
        if n.replace(" ", "") == "MEGMAG":
            ch_types.append("mag")
            data[count, :] = (
                1e-9 * data_raw[count, :] / ch_scale[count]
            )  # convert mag channels to T (nT to T)
        elif n.replace(" ", "") == "TRIG":
            ch_types.append("stim")
            data[count, :] = data_raw[count, :]  # Trigger channels stay as Volts
        elif n.replace(" ", "") == "MISC":
            ch_types.append("stim")
            data[count, :] = data_raw[count, :]  # BNC channels stay as Volts

    return ch_names, ch_types, data


def _calc_tangent(RDip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the tangent vectors for a given dipole.

    :param RDip: The dipole location.
    :type RDip: np.ndarray
    :return: The tangent vectors.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    x = RDip[0]
    y = RDip[1]
    z = RDip[2]
    r = np.sqrt(x * x + y * y + z * z)
    tanu = np.zeros(3)
    tanv = np.zeros(3)
    if x == 0 and y == 0:
        tanu[0] = 1.0
        tanu[1] = 0
        tanu[2] = 0
        tanv[0] = 0
        tanv[1] = 1.0
        tanv[2] = 0
    else:
        rzxy = -(r - z) * x * y
        x2y2 = 1 / (x * x + y * y)

        tanu[0] = (z * x * x + r * y * y) * x2y2 / r
        tanu[1] = rzxy * x2y2 / r
        tanu[2] = -x / r

        tanv[0] = rzxy * x2y2 / r
        tanv[1] = (z * y * y + r * x * x) * x2y2 / r
        tanv[2] = -y / r

    return tanu, tanv


def calc_pos(pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """Create the position information for a given sensor.

    :param pos: The position of the sensor.
    :type pos: np.ndarray
    :param ori: The orientation of the sensor.
    :type ori: np.ndarray
    :return: The position information. 12 coordinates per sensor.
        r0: location (3 coordinates)
        ex: orientation vector (plane x) (3 coordinates)
        ey: orientation vector (plane y) (3 coordinates)
        ez: orientation vector (plane z) (3 coordinates)
    :rtype: np.ndarray
    """
    r0 = pos.copy()
    ez = ori.copy()
    ez = ez / np.linalg.norm(ez)
    ex, ey = _calc_tangent(ez)
    loc = np.concatenate([r0, ex, ey, ez])
    return loc


def conv_square_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convolve the data with a square window.

    :param data: The data to be convolved.
    :type data: np.ndarray
    :param window_size: The size of the window.
    :type window_size: int
    :return: The convolved data.
    :rtype: np.ndarray
    """
    data_stim_conv = []
    for channel_idx in range(data.shape[0]):
        window = np.ones(window_size)
        window = window / sum(window)

        data_stim_conv.append(np.convolve(data[channel_idx, :], window, mode="same"))

    return np.array(data_stim_conv)


def get_mne_data(
    data_dir: str, day="20230623", acq_time="155445", paradigm="gesture"
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
    if paradigm == "gesture":
        event_id = {
            "cue_1": 1,
            "cue_2": 2,
            "cue_3": 4,
            "end_trial": 7,
            "experiment_marker": 255,
        }
    elif paradigm == "finger":
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
