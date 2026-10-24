import dataclasses

import numpy as np
import pytest

from opm_rps import time_frequency as tf
from opm_rps.preprocessing import load_epochs, load_run

SFREQ = 250.0


def sine(freq, times, phase=0.0):
    return np.sin(2 * np.pi * freq * times + phase)


def analytic_signals(data, sfreq, freqs, n_cycles):
    """Analytic signals of all wavelets, shape (n_freqs, n_trials, n_times)."""
    return np.stack(list(tf.wavelet_convolution(data, sfreq, freqs, n_cycles)))


def weight_beyond(lag, sfreq, freqs, n_cycles):
    """Largest fraction of a wavelet envelope's weight at lags beyond ``lag``."""
    half = tf.WAVELET_DURATION * sfreq
    lags = np.arange(-half, half + 1) / sfreq
    widths = n_cycles / (2 * np.pi * freqs)
    envelopes = np.exp(-(lags**2) / (2 * widths[:, np.newaxis] ** 2))
    beyond = envelopes[:, np.abs(lags) > lag].sum(axis=1)
    return np.max(beyond / envelopes.sum(axis=1))


def test_power_peaks_at_the_oscillation_frequency():
    times = np.arange(-1, 2, 1 / SFREQ)
    data = np.tile(sine(10, times), (5, 1))
    freqs, n_cycles = tf.morlet_freqs(4, 20, 17)
    power = tf.single_trial_power(data, SFREQ, freqs, n_cycles)
    assert power.shape == (5, len(freqs), len(times))
    middle = len(times) // 2
    assert freqs[power[:, :, middle].mean(axis=0).argmax()] == pytest.approx(10)


def test_wavelet_spectra_peak_at_one():
    times = np.arange(0, 6, 1 / SFREQ)
    freqs, n_cycles = tf.morlet_freqs(2, 40, 5)
    for fi, freq in enumerate(freqs):
        data = sine(freq, times)[np.newaxis]
        analytic = analytic_signals(data, SFREQ, freqs[[fi]], n_cycles[[fi]])
        # A unit sine is the sum of two complex exponentials of amplitude 1/2
        assert np.abs(analytic[0, 0, len(times) // 2]) == pytest.approx(0.5, rel=1e-9)


def test_trials_are_convolved_separately():
    data = np.random.default_rng(4).normal(size=(5, 400))
    freqs, n_cycles = tf.morlet_freqs(2, 30, 8)
    power = tf.single_trial_power(data, SFREQ, freqs, n_cycles)
    for trial in range(len(data)):
        alone = tf.single_trial_power(data[[trial]], SFREQ, freqs, n_cycles)
        np.testing.assert_allclose(alone[0], power[trial])


def test_wavelet_buffer():
    freqs, n_cycles = tf.morlet_freqs(2, 50, 40)
    widest = n_cycles[0] / (2 * np.pi * freqs[0])
    assert tf.wavelet_buffer(freqs, n_cycles) == pytest.approx(3 * widest)
    freqs, n_cycles = tf.morlet_freqs(0.5, 40, 60)
    assert tf.wavelet_buffer(freqs, n_cycles) == tf.WAVELET_DURATION


@pytest.mark.parametrize('n_widths', [3, np.inf])
def test_cropped_buffered_trials_match_the_continuous_signal(n_widths):
    rng = np.random.default_rng(5)
    times = np.arange(0, 60, 1 / SFREQ)
    signal = rng.normal(size=times.size) + 5 * sine(2.5, times) + 3 * sine(11, times)
    freqs, n_cycles = tf.morlet_freqs(2, 30, 8)
    continuous = analytic_signals(signal[np.newaxis], SFREQ, freqs, n_cycles)[:, 0]

    buffer = tf.wavelet_buffer(freqs, n_cycles, n_widths=n_widths)
    n_buffer = int(np.ceil(buffer * SFREQ))
    window = np.arange(round(-0.5 * SFREQ), round(2.1 * SFREQ) + 1)
    epoch = np.arange(window[0] - n_buffer, window[-1] + n_buffer + 1)
    cues = np.arange(5 * SFREQ, 55 * SFREQ, 3.6 * SFREQ).astype(int)[:, np.newaxis]
    trials = signal[cues + epoch]
    cropped = analytic_signals(trials, SFREQ, freqs, n_cycles)
    cropped = cropped[..., n_buffer : n_buffer + window.size]

    # The cropped trials only lack the data at lags beyond the buffer, so they
    # deviate by at most the wavelet weight there times the largest amplitude
    # (nothing, apart from rounding errors, with the full wavelet half-length)
    tolerance = weight_beyond(buffer, SFREQ, freqs, n_cycles) * np.abs(signal).max()
    np.testing.assert_allclose(
        cropped, continuous[:, cues + window], rtol=0, atol=tolerance + 1e-12
    )


def test_buffered_epochs_match_the_continuous_recording(cfg):
    cfg = dataclasses.replace(cfg, detrend=None)  # epochs = segments of the recording
    epochs, window = tf.load_buffered_epochs(cfg, 'run_1', tf.WAVELET_DURATION)
    unbuffered = load_epochs(cfg, 'run_1')
    np.testing.assert_array_equal(epochs.times[window], unbuffered.times)
    np.testing.assert_array_equal(epochs.events, unbuffered.events)
    np.testing.assert_array_equal(epochs.get_data()[..., window], unbuffered.get_data())

    raw, _ = load_run(cfg, 'run_1')
    channel, sfreq = cfg.roi_channel('X'), raw.info['sfreq']
    freqs, n_cycles = tf.morlet_freqs(2, 50, 10)
    recording = raw.get_data(picks=[channel])
    continuous = analytic_signals(recording, sfreq, freqs, n_cycles)[:, 0]
    offsets = np.round(unbuffered.times * sfreq).astype(int)
    samples = epochs.events[:, [0]] - raw.first_samp + offsets
    trials = epochs.get_data(picks=[channel])[:, 0]
    cropped = analytic_signals(trials, sfreq, freqs, n_cycles)[..., window]
    np.testing.assert_allclose(
        cropped, continuous[:, samples], rtol=0, atol=1e-9 * np.abs(continuous).max()
    )


def test_itpc_of_phase_locked_and_random_trials():
    rng = np.random.default_rng(0)
    times = np.arange(0, 2, 1 / SFREQ)
    locked = np.tile(sine(10, times), (20, 1))
    random = sine(10, times[np.newaxis], rng.uniform(0, 2 * np.pi, (20, 1)))
    freqs, n_cycles = tf.morlet_freqs(10, 10, 1)
    _, itpc_locked = tf.average_power_itpc(locked, SFREQ, freqs, n_cycles)
    _, itpc_random = tf.average_power_itpc(random, SFREQ, freqs, n_cycles)
    middle = len(times) // 2
    assert itpc_locked[0, middle] > 0.99
    assert itpc_random[0, middle] < 0.5


def test_average_power_is_the_mean_single_trial_power():
    data = np.random.default_rng(1).normal(size=(6, 300))
    freqs, n_cycles = tf.morlet_freqs(5, 30, 6)
    single = tf.single_trial_power(data, SFREQ, freqs, n_cycles)
    average, _ = tf.average_power_itpc(data, SFREQ, freqs, n_cycles)
    np.testing.assert_allclose(single.mean(axis=0), average)


def test_non_phase_locked_removes_the_evoked_response():
    data = np.random.default_rng(2).normal(size=(10, 50)) + 3.0
    np.testing.assert_allclose(tf.non_phase_locked(data).mean(axis=0), 0, atol=1e-12)


def test_db_normalize():
    times = np.linspace(-0.5, 1, 151)
    power = np.random.default_rng(3).uniform(1, 2, (3, len(times)))
    db = tf.db_normalize(power, times, (-0.45, -0.15))
    start, stop = tf.nearest_time_indices(times, (-0.45, -0.15))
    np.testing.assert_allclose((10 ** (db[:, start:stop] / 10)).mean(axis=-1), 1)
