"""Read the Nottingham OPM recordings (``.cMEG`` + sidecar files) into MNE.

Each acquisition consists of a binary ``<stem>_meg.cMEG`` file and the sidecars
``<stem>_meg.json``, ``<stem>_channels.tsv``, ``<stem>_HelmConfig.tsv`` and
``<stem>_SensorTransform.tsv``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd
from mne.io.constants import FIFF

if TYPE_CHECKING:
    from .config import Config

#: Trigger codes of the rock-paper-scissors paradigm
EVENT_ID = {'roc': 1, 'pap': 2, 'sci': 4, 'end_trial': 7, 'experiment_marker': 255}

#: Display names of the conditions
CONDITION_NAMES = {'roc': 'Rock', 'pap': 'Paper', 'sci': 'Scissors'}


def read_run(cfg: Config, run: str) -> tuple[mne.io.RawArray, np.ndarray | None]:
    """Read one acquisition of the configured session, e.g. ``'run_1'``."""
    fname = cfg.paths.cmeg_file(cfg.session, cfg.acquisitions[run])
    return read_raw_cmeg(fname)


def read_raw_cmeg(fname: str | Path) -> tuple[mne.io.RawArray, np.ndarray | None]:
    """Convert a ``.cMEG`` recording to an MNE Raw object.

    Returns
    -------
    raw
        Magnetometer channels in T and trigger channels in V. Channels marked
        "Bad" in the channels.tsv sidecar are listed in ``raw.info['bads']``.
    events
        MNE events decoded from the trigger channels (codes in :data:`EVENT_ID`),
        or None if the recording contains no triggers.
    """
    fname = Path(fname)
    stem = fname.with_name(fname.name.removesuffix('_meg.cMEG'))
    meta = json.loads(Path(f'{stem}_meg.json').read_text())
    channels = pd.read_csv(f'{stem}_channels.tsv', sep='\t')
    helmet = pd.read_csv(f'{stem}_HelmConfig.tsv', sep='\t')
    sensor_transform = pd.read_csv(
        f'{stem}_SensorTransform.tsv', header=None, sep='\t'
    ).to_numpy()
    # Some exports escape the slash in the name of the scale-factor column
    channels = channels.rename(columns={'nT0x2FV': 'nT/V'})

    ch_names = channels['name'].str.replace(' ', '').tolist()
    kinds = channels['type'].str.replace(' ', '').tolist()
    scales = channels['nT/V'].to_numpy(dtype=float)
    positions, orientations = _sensor_geometry(channels['name'], helmet)

    raw_data = read_cmeg_data(fname)[1:]  # the first row holds the time stamps
    data = np.empty(raw_data.shape)
    ch_types = []
    for idx, kind in enumerate(kinds):
        if kind == 'MEGMAG':
            ch_types.append('mag')
            data[idx] = 1e-9 * raw_data[idx] / scales[idx]  # nT -> T
        elif kind in ('TRIG', 'MISC'):  # trigger and BNC channels stay in V
            ch_types.append('stim')
            data[idx] = raw_data[idx]
        else:
            raise ValueError(f'Unknown channel type {kind!r} of {ch_names[idx]}')

    info = mne.create_info(ch_names, sfreq=meta['SamplingFrequency'], ch_types=ch_types)
    info['line_freq'] = meta['PowerLineFrequency']
    counters = dict.fromkeys(('stim', 'ref', 'meg'), 0)
    for ch, kind, pos, ori, scale in zip(
        info['chs'], kinds, positions, orientations, scales, strict=True
    ):
        if kind in ('TRIG', 'MISC'):
            counters['stim'] += 1
            ch.update(
                logno=counters['stim'],
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_STIM_CH,
                unit=FIFF.FIFF_UNIT_V,
                cal=1.0,
            )
        elif np.isnan(pos).all():  # magnetometer without a position: reference
            counters['ref'] += 1
            ch.update(
                logno=counters['ref'],
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_REF_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                cal=1e-9 / scale,
            )
        else:
            counters['meg'] += 1
            has_position = not np.isnan(pos).any()
            ch.update(
                logno=counters['meg'],
                coord_frame=FIFF.FIFFV_COORD_DEVICE,
                kind=FIFF.FIFFV_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                loc=_sensor_loc(pos, ori) if has_position else np.full(12, np.nan),
                cal=1e-9 / scale,
            )
    # Sensor (device) to head transform; identity unless coregistered
    info['dev_head_t'] = mne.transforms.Transform('meg', 'head', sensor_transform)

    raw = mne.io.RawArray(data, info)
    is_bad = channels['status'].str.strip() == 'Bad'
    raw.info['bads'] = [ch for ch, bad in zip(ch_names, is_bad, strict=True) if bad]

    stim_idx = [idx for idx, ch_type in enumerate(ch_types) if ch_type == 'stim']
    events = find_trigger_events(data[stim_idx])
    return raw, events


def read_cmeg_data(fname: str | Path) -> np.ndarray:
    """Read the binary ``.cMEG`` file into an array of shape (n_channels, n_samples).

    The file is a sequence of chunks, each a header of two big-endian uint32
    (number of channels and samples) followed by the samples as big-endian float64.
    The first channel holds the time stamps.
    """
    fname = Path(fname)
    size = fname.stat().st_size
    chunks = []
    with open(fname, 'rb') as fid:
        while fid.tell() < size:
            n_channels, n_samples = (int(n) for n in np.fromfile(fid, '>u4', count=2))
            chunk = np.fromfile(fid, '>f8', count=n_channels * n_samples)
            chunks.append(chunk.reshape(n_channels, n_samples))
    return np.concatenate(chunks, axis=1).astype(np.float64)


def find_trigger_events(stim_data: np.ndarray) -> np.ndarray | None:
    """Decode events from trigger channels that each carry one bit of the code.

    An event starts whenever any trigger channel rises above 1 V. Its code is read
    after smoothing the channels with a 5-sample boxcar, which tolerates bits that
    switch a few samples apart.
    """
    active = (stim_data.sum(axis=0) >= 1).astype(float)
    onsets = np.flatnonzero(np.diff(active, prepend=0) == 1)
    if len(onsets) == 0:
        return None

    boxcar = np.ones(5) / 5
    smoothed = np.array([np.convolve(ch, boxcar, mode='same') for ch in stim_data])
    bits = 2 ** np.arange(len(stim_data))
    codes = [np.sum((smoothed[:, onset] > 0.5) * bits) for onset in onsets]
    return np.array(
        [(onset, 0, code) for onset, code in zip(onsets, codes, strict=True)]
    )


def _sensor_geometry(
    names: pd.Series, helmet: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Look up position and orientation of each channel in the helmet config.

    Names are matched ignoring spaces and punctuation; channels without a match get
    NaN.
    """

    def normalize(name: str) -> str:
        return re.sub(r'[\W_]+', '', name)

    row_of = {}
    for row, sensor in enumerate(helmet['Sensor']):
        row_of.setdefault(normalize(sensor), row)
    helmet_pos = helmet[['Px', 'Py', 'Pz']].to_numpy(float)
    helmet_ori = helmet[['Ox', 'Oy', 'Oz']].to_numpy(float)

    positions = np.full((len(names), 3), np.nan)
    orientations = np.full((len(names), 3), np.nan)
    for idx, name in enumerate(names):
        row = row_of.get(normalize(name))
        if row is not None:
            positions[idx] = helmet_pos[row]
            orientations[idx] = helmet_ori[row]
    return positions, orientations


def _sensor_loc(pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """MNE channel location: position followed by the coil's x, y and z unit vectors.

    The coil's z axis is the sensitive axis (the sensor orientation); x and y are
    tangential to it.
    """
    ez = ori / np.linalg.norm(ori)
    ex, ey = _tangent_vectors(ez)
    return np.concatenate([pos, ex, ey, ez])


def _tangent_vectors(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two vectors orthogonal to ``direction`` and to each other."""
    x, y, z = direction
    r = np.sqrt(x * x + y * y + z * z)
    if x == 0 and y == 0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    rzxy = -(r - z) * x * y
    x2y2 = 1 / (x * x + y * y)
    tanu = np.array([(z * x * x + r * y * y) * x2y2 / r, rzxy * x2y2 / r, -x / r])
    tanv = np.array([rzxy * x2y2 / r, (z * y * y + r * x * x) * x2y2 / r, -y / r])
    return tanu, tanv
