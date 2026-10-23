import mne
import numpy as np

from opm_rps import features

SFREQ = 1200.0


def make_epochs(bads=()):
    """Four epochs of three channels: 50 Hz, 150 Hz and noise."""
    rng = np.random.default_rng(0)
    times = np.arange(int(SFREQ)) / SFREQ
    data = 0.1 * rng.normal(size=(4, 3, len(times)))
    data[:, 0] += np.sin(2 * np.pi * 50 * times)
    data[:, 1] += np.sin(2 * np.pi * 150 * times)
    info = mne.create_info(['A[X]', 'B[X]', 'C[X]'], SFREQ, 'mag')
    epochs = mne.EpochsArray(data, info, verbose=False)
    epochs.info['bads'] = list(bads)
    return epochs


def test_variance_excludes_bad_channels():
    epochs = make_epochs(bads=['B[X]'])
    expected = np.var(epochs.get_data(picks=['A[X]', 'C[X]']), axis=-1)
    np.testing.assert_allclose(features.variance(epochs), expected)


def test_band_power_is_ordered_band_by_band():
    epochs = make_epochs()
    power = features.band_power(epochs)
    assert power.shape == (len(epochs), len(features.FREQ_BANDS) * 3)

    bands = list(features.FREQ_BANDS)
    per_band = power.reshape(len(epochs), len(bands), 3)  # epochs x bands x channels
    assert np.all(per_band[:, :, 0].argmax(axis=1) == bands.index('low_gamma'))
    assert np.all(per_band[:, :, 1].argmax(axis=1) == bands.index('mid_gamma'))


def test_bands_with_a_single_bin_have_zero_power():
    per_band = features.band_power(make_epochs()).reshape(
        4, len(features.FREQ_BANDS), 3
    )
    # sfreq / n_fft = 4.7 Hz: delta, theta and alpha contain one frequency bin
    assert np.all(per_band[:, :3] == 0)


def test_welch_power():
    assert features.welch_power(make_epochs(bads=['C[X]'])).shape == (4, 2)
