"""Plot the trial-averaged time-frequency decomposition of the ROI sensor.

For each condition and sensor axis: total, non-phase-locked (evoked response
removed) and phase-locked (total minus non-phase-locked) power in dB relative to the
pre-cue baseline, and the corresponding inter-trial phase coherence (ITPC).

The trials are epoched with a buffer of recorded data on both sides of
``epoch_window`` (three Gaussian widths of the 1 Hz wavelet, 1.9 s), so that the
wavelets do not reach past the data within the window; trials whose buffered epoch
does not fit into the recording are dropped. The decomposition is cropped back to
``epoch_window`` before the baseline normalisation.

Output: derivatives/figures/tf/
"""

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from opm_rps.config import AXES, load_config
from opm_rps.io import CONDITION_NAMES
from opm_rps.time_frequency import (
    average_power_itpc,
    db_normalize,
    load_buffered_epochs,
    morlet_freqs,
    non_phase_locked,
    wavelet_buffer,
)
from opm_rps.viz import parula_cmap, plot_tf_grid, set_style

set_style()
cfg = load_config()

RUNS = cfg.runs  # e.g. ['run_1'] for a single run
FREQS, N_CYCLES = morlet_freqs(fmin=1, fmax=40, n_freqs=60, cycles=(4, 10))
BUFFER = wavelet_buffer(FREQS, N_CYCLES)  # s of data on both sides of the epochs
# Fixed contour levels (dB power, ITPC) of the six panels; None for automatic levels
LEVELS = [
    np.linspace(-8.0, 4.2, 100),
    np.linspace(-8.0, 4.2, 100),
    np.linspace(-0.8, 4.2, 100),
    np.linspace(-0.1, 1, 100),
    np.linspace(-0.1, 1, 100),
    np.linspace(-0.1, 1, 100),
]

# %% Buffered trials of the ROI sensor, collected run by run to save memory
trials = defaultdict(list)
for run in RUNS:
    epochs, window = load_buffered_epochs(cfg, run, BUFFER)
    for axis in AXES:
        channel = cfg.roi_channel(axis)
        for condition in cfg.events:
            data = epochs[condition].get_data(picks=[channel])[:, 0]
            trials[axis, condition].append(data)
times, sfreq = epochs.times[window], epochs.info['sfreq']
runs_label = 'all_runs' if len(RUNS) > 1 else RUNS[0]
cmap = parula_cmap()

# %%
for axis in AXES:
    channel = cfg.roi_channel(axis)
    for condition in cfg.events:
        data = np.concatenate(trials[axis, condition])
        power, itpc = average_power_itpc(data, sfreq, FREQS, N_CYCLES)
        power_npl, itpc_npl = average_power_itpc(
            non_phase_locked(data), sfreq, FREQS, N_CYCLES
        )
        fig = plot_tf_grid(
            db_normalize(power[:, window], times, cfg.tf_baseline),
            db_normalize(power_npl[:, window], times, cfg.tf_baseline),
            itpc[:, window],
            itpc_npl[:, window],
            times,
            FREQS,
            title=f'Time-Frequency Analysis ({CONDITION_NAMES[condition]}, {channel})',
            cmap=cmap,
            levels=LEVELS,
            power_unit='dB',
        )
        fig.savefig(
            cfg.paths.figure('tf', f'tf_dB-{condition}-{runs_label}-{axis}.png')
        )
        plt.show()
