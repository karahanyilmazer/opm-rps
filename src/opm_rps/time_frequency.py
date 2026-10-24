"""Time-frequency decomposition of single-channel trials with complex Morlet wavelets.

The trials of one condition are concatenated into a single "super-trial" that is
convolved with each wavelet in the frequency domain (one FFT per frequency instead
of one per trial). The convolution therefore also runs across trial boundaries,
exactly as in the original analysis.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def morlet_freqs(
    fmin: float, fmax: float, n_freqs: int, cycles: tuple[float, float] = (4, 10)
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly spaced frequencies and log-spaced numbers of cycles."""
    freqs = np.linspace(fmin, fmax, n_freqs)
    n_cycles = np.logspace(np.log10(cycles[0]), np.log10(cycles[1]), n_freqs)
    return freqs, n_cycles


def wavelet_convolution(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    wavelet_duration: float = 2.0,
) -> Iterator[np.ndarray]:
    """Convolve trials with one Morlet wavelet per frequency.

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
    n_trials, n_times = data.shape
    super_trial = data.reshape(-1)  # trial after trial

    wave_time = np.arange(0, wavelet_duration + 1 / sfreq, 1 / sfreq)
    wave_time = np.r_[-wave_time[::-1], wave_time[1:]]
    n_conv = super_trial.size + wave_time.size - 1
    half_wave = (wave_time.size - 1) // 2
    data_fft = np.fft.fft(super_trial, n_conv)

    widths = n_cycles / (2 * np.pi * freqs)  # Gaussian width (s) of each wavelet
    for freq, width in zip(freqs, widths, strict=True):
        wavelet = np.exp(2j * np.pi * freq * wave_time) * np.exp(
            -(wave_time**2) / (2 * width**2)
        )
        wavelet_fft = np.fft.fft(wavelet, n_conv)
        wavelet_fft = wavelet_fft / wavelet_fft[np.argmax(np.abs(wavelet_fft))]

        analytic = np.fft.ifft(data_fft * wavelet_fft)[half_wave:-half_wave]
        yield analytic.reshape(n_trials, n_times)


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
