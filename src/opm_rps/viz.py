"""Plotting style and figures shared by several scripts."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.resources import files

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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
        # extend: values beyond fixed levels get the end colours instead of none
        contours = ax.contourf(
            times, freqs, data, levels=panel_levels, cmap=cmap, extend='both'
        )
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


def plot_test_accuracies(
    accuracies: pd.DataFrame, chance: pd.Series, threshold: pd.Series
) -> Figure:
    """Test accuracies of several pipelines against chance.

    Parameters
    ----------
    accuracies
        Test accuracies between 0 and 1, one row per dataset (e.g. a pair of
        gestures or a run) and one column per pipeline. Missing values are left out.
    chance
        Accuracy of always predicting the most frequent class of each row
        (:func:`opm_rps.classification.chance_level`), drawn as a dashed bar.
    threshold
        Lowest accuracy above chance of each row
        (:func:`opm_rps.classification.significance_threshold`), a black bar.
    """
    x = np.arange(len(accuracies))
    n_pipelines = accuracies.shape[1]
    offsets = (np.arange(n_pipelines) - (n_pipelines - 1) / 2) * 0.16

    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(x), 5), layout='constrained')
    for offset, column, marker in zip(offsets, accuracies, 'os^Dv', strict=False):
        values = 100 * accuracies[column].to_numpy(dtype=float)
        ax.plot(x + offset, values, ls='none', marker=marker, zorder=3, label=column)
    for values, style, label in (
        (chance, {'color': 'gray', 'ls': '--'}, 'Chance'),
        (threshold, {'color': 'k'}, 'Above chance from here ($p < 0.05$)'),
    ):
        heights = 100 * values.reindex(accuracies.index).to_numpy(dtype=float)
        ax.hlines(heights, x - 0.42, x + 0.42, zorder=2, label=label, **style)

    # Leave room for points at 0 % and for thresholds of 100 %
    ax.set(xlim=(-0.6, len(x) - 0.4), ylim=(-3, 103), ylabel='Test accuracy (%)')
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_xticks(x, accuracies.index)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.tick_params(axis='x', which='major', top=False)
    ax.grid(axis='y')
    fig.legend(loc='outside lower center', ncols=n_pipelines + 2)
    return fig


def plot_cross_accuracies(
    scores: dict[str, pd.DataFrame], chance: float = 0.5
) -> Figure:
    """Cross-evaluation tables as heat maps, one panel per table.

    Parameters
    ----------
    scores
        Tables of :func:`opm_rps.classification.cross_evaluate`, accuracies with the
        training data as rows and the test data as columns, under the title of their
        panel. The diagonal, the regular test accuracies, is outlined.
    chance
        Accuracy at the centre of the colour map, e.g. 1/3 for three classes.
    """
    norm = TwoSlopeNorm(vmin=0, vcenter=100 * chance, vmax=100)
    fig, axs = plt.subplots(
        1,
        len(scores),
        squeeze=False,
        figsize=(1 + 5 * len(scores), 4.5),
        layout='constrained',
    )
    for ax, (title, table) in zip(axs[0], scores.items(), strict=True):
        values = 100 * table.to_numpy(dtype=float)
        image = ax.imshow(values, cmap='RdBu', norm=norm)
        for (i, j), value in np.ndenumerate(values):
            if not np.isnan(value):
                color = 'white' if abs(norm(value) - 0.5) > 0.3 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color=color)
        for k in range(min(values.shape)):
            ax.add_patch(
                Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec='k', lw=1.5)
            )
        ax.set_xticks(range(values.shape[1]), table.columns)
        ax.set_yticks(range(values.shape[0]), table.index)
        ax.tick_params(which='both', length=0)
        ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
        ax.set(title=title, xlabel='Tested on')
    axs[0, 0].set_ylabel('Trained on')
    fig.colorbar(image, ax=axs[0].tolist(), label='Test accuracy (%)', aspect=30)
    return fig


def plot_lasso_coefficients(
    coefs: dict[str, np.ndarray], bands: Sequence[tuple[float, float]] | None = None
) -> Figure:
    """Absolute LASSO weights of the filter bank CSP features, one panel per model.

    Parameters
    ----------
    coefs
        LASSO weights of each model (e.g. of each class of a one-vs-rest
        classifier), with the features ordered band by band (see
        :class:`opm_rps.csp.FilterBankCSP`).
    bands
        Bands of the filter bank. The bars of each band are then placed at its
        centre frequency, otherwise at the index of their feature.
    """
    fig, axs = plt.subplots(
        1,
        len(coefs),
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * len(coefs), 4),
        layout='constrained',
    )
    for ax, (title, coef) in zip(axs[0], coefs.items(), strict=True):
        weights = np.abs(np.asarray(coef, dtype=float))
        if bands is None:
            ax.bar(np.arange(len(weights)), weights, width=0.8)
        else:
            # Each band contributes the same number of features, side by side
            per_band = len(weights) // len(bands)
            centres = np.array([(low + high) / 2 for low, high in bands])
            spacing = np.diff(centres).min() if len(centres) > 1 else 1.0
            width = 0.8 * spacing / per_band
            offsets = (np.arange(per_band) - (per_band - 1) / 2) * width
            positions = (centres[:, np.newaxis] + offsets[np.newaxis, :]).ravel()
            ax.bar(positions, weights, width=width)
        ax.set_title(title)
        ax.tick_params(axis='x', which='minor', bottom=False, top=False)
        ax.tick_params(axis='x', which='major', top=False)
        ax.set_xlabel('Feature index' if bands is None else 'Band centre (Hz)')
    axs[0, 0].set_ylabel('Absolute LASSO weight')
    return fig
