import mne
import numpy as np
import pytest

from opm_rps import features

SFREQ = 1200.0


def make_epochs(freqs=(50, 150, None), bads=(), n_times=int(SFREQ)):
    """Four epochs with one channel per frequency: a sine plus noise (None: noise)."""
    rng = np.random.default_rng(0)
    times = np.arange(n_times) / SFREQ
    data = 0.1 * rng.normal(size=(4, len(freqs), len(times)))
    for ch, freq in enumerate(freqs):
        if freq is not None:
            data[:, ch] += np.sin(2 * np.pi * freq * times)
    names = [f'{chr(ord("A") + ch)}[X]' for ch in range(len(freqs))]
    epochs = mne.EpochsArray(data, mne.create_info(names, SFREQ, 'mag'), verbose=False)
    epochs.info['bads'] = list(bads)
    return epochs


def test_variance_excludes_bad_channels():
    epochs = make_epochs(bads=['B[X]'])
    expected = np.var(epochs.get_data(picks=['A[X]', 'C[X]']), axis=-1)
    np.testing.assert_allclose(features.variance(epochs), expected)


def test_welch_psd_resolution():
    _, freqs = features.welch_psd(make_epochs(n_times=3121))
    np.testing.assert_allclose(np.diff(freqs), 1.0)
    _, freqs = features.welch_psd(make_epochs(), resolution=2.0)
    np.testing.assert_allclose(np.diff(freqs), 2.0)


def test_band_power_is_ordered_band_by_band():
    epochs = make_epochs()
    power = features.band_power(epochs)
    assert power.shape == (len(epochs), len(features.FREQ_BANDS) * 3)

    bands = list(features.FREQ_BANDS)
    per_band = power.reshape(len(epochs), len(bands), 3)  # epochs x bands x channels
    assert np.all(per_band[:, :, 0].argmax(axis=1) == bands.index('low_gamma'))
    assert np.all(per_band[:, :, 1].argmax(axis=1) == bands.index('mid_gamma'))


def test_band_power_resolves_the_low_frequency_bands():
    epochs = make_epochs(freqs=(2, 6, 10), n_times=3121)  # 2.6 s, like the epochs
    per_band = features.band_power(epochs).reshape(4, len(features.FREQ_BANDS), 3)
    assert np.all(per_band > 0)
    bands = list(features.FREQ_BANDS)
    for ch, band in enumerate(['delta', 'theta', 'alpha']):
        assert np.all(per_band[:, :, ch].argmax(axis=1) == bands.index(band))


def test_relative_band_powers_add_up_to_the_power_below_300_hz():
    epochs = make_epochs(freqs=(10, 50, 250))
    epochs.filter(None, 290, verbose=False)
    per_band = features.band_power(epochs).reshape(4, len(features.FREQ_BANDS), 3)
    np.testing.assert_allclose(per_band.sum(axis=1), 1, atol=0.01)


def test_band_power_rejects_bands_narrower_than_the_resolution():
    with pytest.raises(ValueError, match='narrow band'):
        features.band_power(make_epochs(), bands={'narrow': (10, 10.5)})


def test_filter_bank_bands():
    thesis = features.filter_bank_bands()
    assert len(thesis) == 17
    assert thesis[0] == (4.0, 8.0) and thesis[-1] == (36.0, 40.0)

    # The bands of extract_filter_bank.py, spanning the decoding band
    decoding = features.filter_bank_bands(7, 30)
    assert len(decoding) == 10
    assert decoding[0] == (7.0, 11.0) and decoding[-1] == (25.0, 29.0)

    wide = features.filter_bank_bands(bandwidth=12.0)
    assert len(wide) == 13
    assert wide[-1] == (28.0, 40.0)


def test_welch_power_is_the_signal_variance():
    epochs = make_epochs(freqs=(None, None), bads=['B[X]'], n_times=3121)
    power = features.welch_power(epochs)
    assert power.shape == (4, 1)
    np.testing.assert_allclose(power, features.variance(epochs), rtol=0.1)
