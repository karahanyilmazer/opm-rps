"""Preprocessing steps shared by the analysis scripts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import mne
import numpy as np
from scipy.signal import filtfilt, firls

from .config import Config
from .io import read_run

#: 50 Hz line noise and its harmonics below the Nyquist frequency of the recordings
LINE_NOISE_FREQS = np.arange(50, 600, 50)


def load_run(
    cfg: Config, run: str, *, hfc: bool = True
) -> tuple[mne.io.RawArray, np.ndarray | None]:
    """Read a run, mark the bad channels, apply HFC and crop it as configured.

    Only the events of the configured conditions are returned (None for recordings
    without triggers).
    """
    raw, events = read_run(cfg, run)
    mark_bad_channels(raw, cfg.bad_channels)
    if hfc:
        apply_hfc(raw, order=cfg.hfc_order)
    if run in cfg.crop:
        events = crop(raw, events, *cfg.crop[run])
    if events is not None:
        events = events[np.isin(events[:, 2], list(cfg.event_id.values()))]
    return raw, events


def load_epochs(
    cfg: Config, runs: str | Sequence[str], *, drop_bad_trials: bool = True
) -> mne.BaseEpochs:
    """HFC-corrected, linearly detrended epochs of one or more runs.

    Each run is epoched separately and its rejected trials (``bad_epochs`` in the
    config) are dropped before the runs are concatenated.
    """
    runs = [runs] if isinstance(runs, str) else list(runs)
    tmin, tmax = cfg.epoch_window
    epochs_list = []
    for run in runs:
        raw, events = load_run(cfg, run)
        epochs = mne.Epochs(
            raw,
            events,
            cfg.event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            detrend=cfg.detrend,
            preload=True,
        )
        if drop_bad_trials:
            drop_trials(epochs, cfg.bad_epochs.get(run, []))
        epochs_list.append(epochs)

    if len(epochs_list) == 1:
        return epochs_list[0]
    return mne.concatenate_epochs(epochs_list)


def mark_bad_channels(raw: mne.io.BaseRaw, names: Iterable[str]) -> None:
    """Add channels to ``raw.info['bads']``, skipping names the recording lacks."""
    names = list(names)
    present = [ch for ch in names if ch in raw.ch_names]
    if missing := sorted(set(names) - set(present)):
        warnings.warn(f'Bad channels not in the recording: {missing}', stacklevel=2)
    raw.info['bads'] = list(dict.fromkeys([*raw.info['bads'], *present]))


def apply_hfc(raw: mne.io.BaseRaw, order: int = 2) -> None:
    """Apply homogeneous field correction, fitted on the good channels."""
    projs = mne.preprocessing.compute_proj_hfc(raw.info, order=order)
    raw.add_proj(projs).apply_proj()


def notch_filter(raw: mne.io.BaseRaw, extra_freqs: Iterable[float] = ()) -> None:
    """Remove line noise, its harmonics and additional sensor-specific peaks."""
    raw.notch_filter(np.append(LINE_NOISE_FREQS, list(extra_freqs)))


def crop(
    raw: mne.io.BaseRaw, events: np.ndarray | None, tmin: float, tmax: float | None
) -> np.ndarray | None:
    """Crop a run in place and return the events that remain inside it."""
    raw.crop(tmin, tmax)
    if events is None:
        return None
    inside = (events[:, 0] >= raw.first_samp) & (events[:, 0] <= raw.last_samp)
    return events[inside]


def drop_trials(epochs: mne.BaseEpochs, samples: Iterable[int]) -> None:
    """Drop the epochs whose events occur at the given sample indices."""
    idx = np.flatnonzero(np.isin(epochs.events[:, 0], list(samples)))
    epochs.drop(idx)


def trigger_channels(ch_names: Iterable[str]) -> list[str]:
    """Names of the trigger channels."""
    return [ch for ch in ch_names if 'Trigger' in ch]


def axis_channels(ch_names: Iterable[str], axis: str) -> list[str]:
    """Names of the channels measuring along ``axis`` (``'X'``, ``'Y'`` or ``'Z'``)."""
    return [ch for ch in ch_names if f'[{axis.upper()}]' in ch and 'Trigger' not in ch]


def firls_bandpass(
    sfreq: float, l_freq: float, h_freq: float, transition: float = 0.2
) -> np.ndarray:
    """Least-squares linear-phase FIR band-pass kernel.

    The kernel spans three cycles of the low cutoff, and the transition bands are
    ``transition`` times the respective cutoff frequency wide.
    """
    nyquist = sfreq / 2
    order = np.round(3 * sfreq / l_freq).astype(int)
    order = order + 1 if order % 2 == 0 else order  # firls needs an odd length
    band_edges = np.array(
        [
            0,
            l_freq - l_freq * transition,
            l_freq,
            h_freq,
            h_freq + h_freq * transition,
            nyquist,
        ]
    )
    return firls(order, band_edges / nyquist, [0, 0, 1, 1, 0, 0])


def apply_fir(raw: mne.io.BaseRaw, kernel: np.ndarray) -> None:
    """Zero-phase filter all channels of a preloaded Raw in place."""
    raw.apply_function(
        lambda data: filtfilt(kernel, 1, data), picks='all', channel_wise=False
    )
