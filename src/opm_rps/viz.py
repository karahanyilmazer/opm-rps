"""Plotting style and figures shared by several scripts."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.resources import files

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .config import AXES
from .preprocessing import axis_channels


def set_style(notebook: bool = True, grid: bool = False) -> None:
    """Use the SciencePlots style (without LaTeX)."""
    import scienceplots  # noqa: F401  (registers the 'science' styles)

    styles = ['science', 'no-latex']
    if grid:
        styles.append('grid')
    if notebook:
        styles.append('notebook')
    plt.style.use(styles)
    if sys.platform == 'win32':  # high-DPI monitor
        plt.rcParams['figure.dpi'] = 260


def parula_cmap() -> LinearSegmentedColormap:
    """MATLAB's parula colormap."""
    colors = yaml.safe_load((files('opm_rps') / 'parula.yaml').read_text())['cmap']
    return LinearSegmentedColormap.from_list('parula', colors)


def plot_tf_grid(
    power: np.ndarray,
    power_npl: np.ndarray,
    itpc: np.ndarray,
    itpc_npl: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    title: str,
    cmap=None,
    levels: Sequence[np.ndarray] | None = None,
    power_unit: str = '',
) -> Figure:
    """Total, non-phase-locked and phase-locked power (top) and ITPC (bottom).

    Phase-locked = total minus non-phase-locked. ``levels`` optionally fixes the
    contour levels of the six panels (row by row); otherwise 40 levels are chosen
    automatically.
    """
    panels = [
        (power, 'Total Power'),
        (power_npl, 'Non-Phase-Locked Power'),
        (power - power_npl, 'Phase-Locked Power'),
        (itpc, 'Total ITPC'),
        (itpc_npl, 'Non-Phase-Locked ITPC'),
        (itpc - itpc_npl, 'Phase-Locked ITPC'),
    ]
    levels = levels if levels is not None else [40] * len(panels)

    fig, axs = plt.subplots(2, 3, figsize=(21, 14))
    for ax, (data, panel_title), panel_levels in zip(
        axs.flat, panels, levels, strict=True
    ):
        contours = ax.contourf(times, freqs, data, levels=panel_levels, cmap=cmap)
        is_power_panel = ax is axs[0, -1]
        fig.colorbar(
            contours, ax=ax, format='%.1f', label=power_unit if is_power_panel else ''
        )
        ax.set_title(panel_title)
    for ax in axs[:, 0]:
        ax.set_ylabel('Frequency (Hz)')
    for ax in axs[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.subplots_adjust(left=0.05, right=0.95)
    return fig


def plot_fir_response(
    kernel: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    transition: float = 0.2,
) -> Figure:
    """Kernel and amplitude response of a band-pass FIR filter."""
    nyquist = sfreq / 2
    freqs = np.linspace(0, nyquist, len(kernel) // 2 + 1)
    response = np.abs(np.fft.fft(kernel))[: len(freqs)]
    ideal_freqs = [
        0,
        l_freq - l_freq * transition,
        l_freq,
        h_freq,
        h_freq + h_freq * transition,
        nyquist,
    ]
    ideal_gains = [0, 0, 1, 1, 0, 0]

    fig = plt.figure(figsize=(10, 6))
    ax_kernel = fig.add_subplot(2, 1, 1)
    ax_gain = fig.add_subplot(2, 2, 3)
    ax_db = fig.add_subplot(2, 2, 4)

    ax_kernel.plot(kernel)
    ax_kernel.set(title='Filter Kernel', xlabel='Time Points', ylabel='Amplitude')

    ax_gain.plot(freqs, response, 'ks-', markerfacecolor='w', label='Actual')
    ax_gain.plot(ideal_freqs, ideal_gains, 'ro-', markerfacecolor='w', label='Ideal')
    ax_gain.set(
        title='Frequency Response',
        xlim=(0, h_freq * 1.5),
        xlabel='Frequency (Hz)',
        ylabel='Filter Gain',
    )
    ax_gain.legend()

    for cutoff in (l_freq, h_freq):
        ax_db.axvline(cutoff, color='grey', linestyle=':')
    ax_db.plot(freqs, 10 * np.log10(response), 'ks-', markerfacecolor='w')
    ax_db.set(
        title='Frequency Response (dB)',
        xlim=(0, h_freq * 1.5),
        ylim=(-30, 2),
        xlabel='Frequency (Hz)',
        ylabel='Filter Gain (dB)',
    )
    fig.tight_layout()
    return fig


def plot_topomaps_per_axis(
    data: np.ndarray, info: mne.Info, tmin: float, title: str, **topomap_kwargs
) -> None:
    """Topomaps of channel x time data, one figure per sensor axis.

    The X, Y and Z channels of a triaxial OPM share one position, so they cannot be
    shown in the same topomap.
    """
    for axis in AXES:
        idx = [
            info['ch_names'].index(ch) for ch in axis_channels(info['ch_names'], axis)
        ]
        if not idx:
            continue
        evoked = mne.EvokedArray(data[idx], mne.pick_info(info, idx), tmin=tmin)
        fig = evoked.plot_topomap(show=False, **topomap_kwargs)
        fig.suptitle(f'{title} ({axis}-axis)')
        plt.show()
