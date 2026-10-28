"""Hand-crafted per-channel features of epochs, computed on the good channels, and the
frequency bands of the filter bank CSP."""

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


def welch_psd(
    epochs: mne.BaseEpochs, resolution: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of the good channels and its frequencies.

    The spectrum is averaged over Hamming-windowed segments of ``1 / resolution``
    seconds that overlap by half, giving a frequency resolution of ``resolution`` Hz.
    """
    sfreq = epochs.info['sfreq']
    n_fft = round(sfreq / resolution)
    return psd_array_welch(
        _good_channel_data(epochs), sfreq, n_fft=n_fft, n_overlap=n_fft // 2
    )


def welch_power(epochs: mne.BaseEpochs, resolution: float = 1.0) -> np.ndarray:
    """Total power per trial and channel, shape (n_epochs, n_channels).

    The Welch power spectral density is integrated over all frequencies with
    Simpson's rule.
    """
    psd, freqs = welch_psd(epochs, resolution)
    return simpson(psd, x=freqs, axis=-1)


def band_power(
    epochs: mne.BaseEpochs,
    bands: dict[str, tuple[float, float]] = FREQ_BANDS,
    relative: bool = True,
    resolution: float = 1.0,
) -> np.ndarray:
    """Power per frequency band, shape (n_epochs, n_bands * n_channels).

    Columns are ordered band by band: all channels of the first band, then all
    channels of the second band, and so on. With ``relative=True`` the band power is
    divided by the total power.
    """
    psd, freqs = welch_psd(epochs, resolution)
    total = simpson(psd, x=freqs, axis=-1)
    powers = []
    for name, (f_low, f_high) in bands.items():
        in_band = (freqs >= f_low) & (freqs <= f_high)
        if in_band.sum() < 2:
            raise ValueError(
                f'The {name} band ({f_low}-{f_high} Hz) contains fewer than two '
                f'frequency bins at a resolution of {resolution} Hz.'
            )
        power = simpson(psd[..., in_band], x=freqs[in_band], axis=-1)
        powers.append(power / total if relative else power)
    return np.concatenate(powers, axis=1)


def filter_bank_bands(
    f_min: float = 4.0, f_max: float = 40.0, bandwidth: float = 4.0, step: float = 2.0
) -> list[tuple[float, float]]:
    """Overlapping frequency bands (Hz) between ``f_min`` and ``f_max``.

    The defaults give the 17 bands of the thesis's filter bank CSP: 4-8 Hz, 6-10 Hz,
    ..., 36-40 Hz.
    """
    lows = np.arange(f_min, f_max - bandwidth + step / 2, step)
    return [(float(low), float(low + bandwidth)) for low in lows]


def _good_channel_data(epochs: mne.BaseEpochs) -> np.ndarray:
    good = [ch for ch in epochs.ch_names if ch not in epochs.info['bads']]
    return epochs.get_data(picks=good)
