import numpy as np
import pytest

from opm_rps import time_frequency as tf

SFREQ = 250.0


def sine(freq, times, phase=0.0):
    return np.sin(2 * np.pi * freq * times + phase)


def test_power_peaks_at_the_oscillation_frequency():
    times = np.arange(-1, 2, 1 / SFREQ)
    data = np.tile(sine(10, times), (5, 1))
    freqs, n_cycles = tf.morlet_freqs(4, 20, 17)
    power = tf.single_trial_power(data, SFREQ, freqs, n_cycles)
    assert power.shape == (5, len(freqs), len(times))
    middle = len(times) // 2
    assert freqs[power[:, :, middle].mean(axis=0).argmax()] == pytest.approx(10)


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
