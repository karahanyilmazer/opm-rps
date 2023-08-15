import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List


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
