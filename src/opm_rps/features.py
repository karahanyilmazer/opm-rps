"""Hand-crafted per-channel features of epochs, computed on the good channels."""

from __future__ import annotations

import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.integrate import simpson

#: Frequency bands (Hz) of the band-power features
FREQ_BANDS = {
    'delta': (0, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 100),
    'mid_gamma': (100, 200),
    'high_gamma': (200, 300),
}


def variance(epochs: mne.BaseEpochs) -> np.ndarray:
    """Signal variance per trial and channel, shape (n_epochs, n_channels)."""
    return np.var(_good_channel_data(epochs), axis=-1)


def welch_power(epochs: mne.BaseEpochs, n_fft: int = 256) -> np.ndarray:
    """Total Welch power per trial and channel, shape (n_epochs, n_channels).

    The power spectral density is integrated over all frequencies with Simpson's
    rule (unit frequency spacing, as in the original analysis).
    """
    psd, _ = psd_array_welch(
        _good_channel_data(epochs), epochs.info['sfreq'], n_fft=n_fft
    )
    return simpson(psd, axis=-1)


def band_power(
    epochs: mne.BaseEpochs,
    bands: dict[str, tuple[float, float]] = FREQ_BANDS,
    relative: bool = True,
    n_fft: int = 256,
) -> np.ndarray:
    """Power per frequency band, shape (n_epochs, n_bands * n_channels).

    Columns are ordered band by band: all channels of the first band, then all
    channels of the second band, and so on. With ``relative=True`` the band power is
    divided by the total power.

    Note that the Welch frequency resolution is ``sfreq / n_fft``; bands that contain
    a single frequency bin have zero power under Simpson's rule.
    """
    psd, freqs = psd_array_welch(
        _good_channel_data(epochs), epochs.info['sfreq'], n_fft=n_fft
    )
    total = simpson(psd, axis=-1)
    powers = []
    for f_low, f_high in bands.values():
        in_band = (freqs >= f_low) & (freqs <= f_high)
        power = simpson(psd[..., in_band], axis=-1)
        powers.append(power / total if relative else power)
    return np.concatenate(powers, axis=1)


def _good_channel_data(epochs: mne.BaseEpochs) -> np.ndarray:
    good = [ch for ch in epochs.ch_names if ch not in epochs.info['bads']]
    return epochs.get_data(picks=good)
