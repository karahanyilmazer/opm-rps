"""Time-frequency decomposition of single-channel trials with complex Morlet wavelets.

Each trial is convolved separately with each wavelet in the frequency domain (one
batched FFT over the trials per frequency), so the trials do not mix. Near the edges
of a trial the wavelets reach past its data, which distorts the decomposition there,
most of all at the low frequencies, whose wavelets are the widest. The trials are
therefore epoched with a buffer of recorded data on both sides of the analysis
window (:func:`wavelet_buffer`, :func:`load_buffered_epochs`), and their
decomposition is cropped back to the window before the baseline normalisation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

import mne
import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from .config import Config
from .preprocessing import load_epochs

#: Half-length (s) of the wavelets' time axis
WAVELET_DURATION = 2.0


def morlet_freqs(
    fmin: float, fmax: float, n_freqs: int, cycles: tuple[float, float] = (4, 10)
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly spaced frequencies and log-spaced numbers of cycles."""
    freqs = np.linspace(fmin, fmax, n_freqs)
    n_cycles = np.logspace(np.log10(cycles[0]), np.log10(cycles[1]), n_freqs)
    return freqs, n_cycles


def wavelet_buffer(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    n_widths: float = 3,
    wavelet_duration: float = WAVELET_DURATION,
) -> float:
    """Data (s) needed on both sides of an analysis window to avoid edge effects.

    ``n_widths`` Gaussian widths of the widest wavelet, at most the wavelets'
    half-length. Beyond three widths a wavelet's envelope has decayed to 1.1 % of its
    peak and holds 0.13 % of its weight on each side, so cropping the decomposition of
    buffered trials to the window gives nearly the same values as decomposing the
    continuous recording (exactly the same with a buffer of ``wavelet_duration``).
    """
    widths = n_cycles / (2 * np.pi * freqs)
    return min(n_widths * np.max(widths), wavelet_duration)


def load_buffered_epochs(
    cfg: Config, runs: str | Sequence[str], buffer: float
) -> tuple[mne.BaseEpochs, slice]:
    """Epochs extended by ``buffer`` seconds on both sides of ``cfg.epoch_window``.

    See :func:`~opm_rps.preprocessing.load_epochs`. Trials whose buffered epoch does
    not fit into the (cropped) recording are dropped, and the detrending is fitted to
    the buffered epochs.

    Returns
    -------
    epochs
        The buffered epochs.
    window
        Slice of their time axis that covers ``cfg.epoch_window``.
    """
    tmin, tmax = cfg.epoch_window
    buffered = dataclasses.replace(cfg, epoch_window=(tmin - buffer, tmax + buffer))
    epochs = load_epochs(buffered, runs)
    start, stop = nearest_time_indices(epochs.times, cfg.epoch_window)
    return epochs, slice(start, stop + 1)


def wavelet_convolution(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    wavelet_duration: float = WAVELET_DURATION,
) -> Iterator[np.ndarray]:
    """Convolve each trial separately with one Morlet wavelet per frequency.

    The trials are zero-padded, so their edges are distorted unless they are buffers
    that are cropped off afterwards (see :func:`load_buffered_epochs`). The wavelets
    are scaled such that their spectra peak at 1.

    Parameters
    ----------
    data
        Trials of a single channel, shape (n_trials, n_times).
    sfreq
        Sampling frequency (Hz).
    freqs, n_cycles
        Wavelet frequencies (Hz) and their numbers of cycles.
    wavelet_duration
        Half-length (s) of the wavelet's time axis.

    Yields
    ------
    analytic
        Complex analytic signal of shape (n_trials, n_times), one per frequency.
    """
    n_times = data.shape[-1]
    wave_time = np.arange(0, wavelet_duration + 1 / sfreq, 1 / sfreq)
    wave_time = np.r_[-wave_time[::-1], wave_time[1:]]
    half_wave = (wave_time.size - 1) // 2
    n_fft = next_fast_len(n_times + wave_time.size - 1)
    data_fft = fft(data, n_fft, axis=-1)  # every trial on its own

    widths = n_cycles / (2 * np.pi * freqs)  # Gaussian width (s) of each wavelet
    for freq, width in zip(freqs, widths, strict=True):
        envelope = np.exp(-(wave_time**2) / (2 * width**2))
        # The spectrum peaks at the wavelet frequency, at the sum of the envelope
        wavelet = np.exp(2j * np.pi * freq * wave_time) * envelope / envelope.sum()
        analytic = ifft(data_fft * fft(wavelet, n_fft), axis=-1)
        yield analytic[..., half_wave : half_wave + n_times]


def single_trial_power(
    data: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: np.ndarray
) -> np.ndarray:
    """Power of every trial, shape (n_trials, n_freqs, n_times)."""
    power = np.empty((data.shape[0], len(freqs), data.shape[1]))
    for fi, analytic in enumerate(wavelet_convolution(data, sfreq, freqs, n_cycles)):
        power[:, fi] = np.abs(analytic) ** 2
    return power


def average_power_itpc(
    data: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-averaged power and inter-trial phase coherence (ITPC).

    Returns two arrays of shape (n_freqs, n_times).
    """
    power = np.empty((len(freqs), data.shape[1]))
    itpc = np.empty((len(freqs), data.shape[1]))
    for fi, analytic in enumerate(wavelet_convolution(data, sfreq, freqs, n_cycles)):
        power[fi] = np.mean(np.abs(analytic) ** 2, axis=0)
        itpc[fi] = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    return power, itpc


def non_phase_locked(data: np.ndarray) -> np.ndarray:
    """Remove the evoked response (trial average) from each trial."""
    return data - data.mean(axis=0)


def db_normalize(
    power: np.ndarray, times: np.ndarray, baseline: tuple[float, float]
) -> np.ndarray:
    """Convert power to dB relative to its mean in the baseline window.

    The baseline is taken separately for every frequency (and trial) along the last
    (time) axis.
    """
    start, stop = nearest_time_indices(times, baseline)
    baseline_power = power[..., start:stop].mean(axis=-1, keepdims=True)
    return 10 * np.log10(power / baseline_power)


def nearest_time_indices(times: np.ndarray, targets) -> np.ndarray:
    """Indices of the samples closest to each target time."""
    targets = np.atleast_1d(targets)
    return np.argmin(np.abs(times[:, np.newaxis] - targets), axis=0)
