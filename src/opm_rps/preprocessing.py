"""Preprocessing steps shared by the analysis scripts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import mne
import numpy as np
from scipy.signal import filtfilt, firls

from .config import Config
from .io import CONDITION_NAMES, EVENT_ID, read_run

#: 50 Hz line noise and its harmonics below the Nyquist frequency of the recordings
LINE_NOISE_FREQS = np.arange(50, 600, 50)

#: Trigger codes of the cues, one per trial
CUE_CODES = [EVENT_ID[condition] for condition in CONDITION_NAMES]

# Events of the runs read so far (after cropping), keyed by recording and crop
# window, so that bad_trials can count the cue events of every run cheaply
_events_cache: dict[tuple, np.ndarray | None] = {}


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
    _events_cache[_events_key(cfg, run)] = events
    if events is not None:
        events = events[np.isin(events[:, 2], list(cfg.event_id.values()))]
    return raw, events


def concatenate_runs(
    cfg: Config, runs: Sequence[str], *, hfc: bool = True
) -> tuple[mne.io.BaseRaw, np.ndarray, np.ndarray]:
    """Load runs with :func:`load_run` and concatenate them.

    Returns the concatenated recording, its events and the cue samples of the
    rejected trials (:func:`bad_trials`), shifted like the events.
    """
    raws, events_list = zip(*(load_run(cfg, run, hfc=hfc) for run in runs), strict=True)
    rejected = np.concatenate(
        [
            np.isin(events[:, 0], bad_trials(cfg, run))
            for run, events in zip(runs, events_list, strict=True)
        ]
    )
    raw, events = mne.concatenate_raws(list(raws), events_list=list(events_list))
    return raw, events, events[rejected, 0]


def load_epochs(
    cfg: Config, runs: str | Sequence[str], *, drop_bad_trials: bool = True
) -> mne.BaseEpochs:
    """HFC-corrected, linearly detrended epochs of one or more runs.

    Each run is epoched separately and its rejected trials (:func:`bad_trials`) are
    dropped before the runs are concatenated.
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
        epochs_list.append(epochs)
    if drop_bad_trials:
        # Once all runs are loaded, so that bad_trials does not read them again
        for run, epochs in zip(runs, epochs_list, strict=True):
            drop_trials(epochs, bad_trials(cfg, run))

    if len(epochs_list) == 1:
        return epochs_list[0]
    return mne.concatenate_epochs(epochs_list)


def bad_trials(cfg: Config, run: str) -> np.ndarray:
    """Cue samples of the rejected trials of a run.

    Every analysis rejects the union of the two inspections in the config:
    ``bad_epochs`` lists cue samples per run; ``bad_epoch_indices`` lists, per sensor
    axis, positions into the cue events of all runs in ``cfg.runs``, concatenated in
    that order after cropping. Cue events of every condition count, whatever
    ``cfg.events`` selects. The samples are those of the events of :func:`load_run`,
    so the rejection does not depend on the epoch window.

    Raises
    ------
    ValueError
        If a position lies beyond the last cue event, i.e. the config does not match
        the recordings.

    Warns
    -----
    UserWarning
        If a sample in ``bad_epochs`` matches no cue event of the run.
    """
    cues = _cue_samples(cfg, run)
    samples = cfg.bad_epochs.get(run) or []
    if unmatched := sorted(set(samples).difference(cues)):
        warnings.warn(
            f'bad_epochs of {run} match no cue event: {unmatched}', stacklevel=2
        )
    bad = np.isin(cues, samples)

    counts = [len(_cue_samples(cfg, other)) for other in cfg.runs]
    indices = sorted(set().union(*cfg.bad_epoch_indices.values()))
    if invalid := [idx for idx in indices if not 0 <= idx < sum(counts)]:
        raise ValueError(
            f'bad_epoch_indices {invalid} lie outside the {sum(counts)} cue events '
            f'of {cfg.runs}: the config does not match the recordings'
        )
    if run in cfg.runs:
        start = sum(counts[: cfg.runs.index(run)])
        in_run = [idx - start for idx in indices if start <= idx < start + len(cues)]
        bad[np.array(in_run, dtype=int)] = True
    return cues[bad]


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
    """Zero-phase filter all channels of a preloaded Raw in place.

    Channel by channel: filtering all channels at once needs several copies of the
    data, and gives the same result.
    """
    raw.apply_function(
        lambda data: filtfilt(kernel, 1, data), picks='all', channel_wise=True
    )


def _cue_samples(cfg: Config, run: str) -> np.ndarray:
    """Samples of the cue events of a run after cropping, reading it if needed."""
    key = _events_key(cfg, run)
    if key not in _events_cache:
        load_run(cfg, run, hfc=False)
    events = _events_cache[key]
    if events is None:
        return np.empty(0, dtype=int)
    return events[np.isin(events[:, 2], CUE_CODES), 0]


def _events_key(cfg: Config, run: str) -> tuple:
    fname = cfg.paths.cmeg_file(cfg.session, cfg.acquisitions[run])
    return fname, tuple(cfg.crop.get(run, ()))
