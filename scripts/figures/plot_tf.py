"""Plot the trial-averaged time-frequency decomposition of the ROI sensor.

For each condition and sensor axis: total, non-phase-locked (evoked response
removed) and phase-locked (total minus non-phase-locked) power in dB relative to the
pre-cue baseline, and the corresponding inter-trial phase coherence (ITPC).

Output: derivatives/figures/tf/
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from opm_rps.config import AXES, load_config
from opm_rps.io import CONDITION_NAMES
from opm_rps.preprocessing import load_epochs
from opm_rps.time_frequency import (
    average_power_itpc,
    db_normalize,
    morlet_freqs,
    non_phase_locked,
)
from opm_rps.viz import parula_cmap, plot_tf_grid, set_style

set_style()
cfg = load_config()

RUNS = cfg.runs  # e.g. ['run_1'] for a single run
FREQS, N_CYCLES = morlet_freqs(fmin=1, fmax=40, n_freqs=60, cycles=(4, 10))
# Fixed contour levels (dB power, ITPC) of the six panels; None for automatic levels
LEVELS = [
    np.linspace(-8.0, 4.2, 100),
    np.linspace(-8.0, 4.2, 100),
    np.linspace(-0.8, 4.2, 100),
    np.linspace(-0.1, 1, 100),
    np.linspace(-0.1, 1, 100),
    np.linspace(-0.1, 1, 100),
]

# %%
epochs = load_epochs(cfg, RUNS)
runs_label = 'all_runs' if len(RUNS) > 1 else RUNS[0]
times, sfreq = epochs.times, epochs.info['sfreq']
cmap = parula_cmap()

for axis in AXES:
    channel = cfg.roi_channel(axis)
    for condition in cfg.events:
        data = epochs[condition].get_data(picks=[channel])[:, 0]
        power, itpc = average_power_itpc(data, sfreq, FREQS, N_CYCLES)
        power_npl, itpc_npl = average_power_itpc(
            non_phase_locked(data), sfreq, FREQS, N_CYCLES
        )
        fig = plot_tf_grid(
            db_normalize(power, times, cfg.tf_baseline),
            db_normalize(power_npl, times, cfg.tf_baseline),
            itpc,
            itpc_npl,
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
