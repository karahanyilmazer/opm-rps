"""Extract single-trial time-frequency power of the ROI sensor for decoding.

For every run and sensor axis, the trials of each condition are convolved with
complex Morlet wavelets. Power is converted to dB relative to the pre-cue baseline
(``tf_baseline``) and downsampled in time.

Output: derivatives/tf/tf-<run>-<axis>.npz with power_db (trials x freqs x times),
the labels (trigger codes), freqs and times.
"""

# %%
import numpy as np

from opm_rps.config import AXES, load_config
from opm_rps.preprocessing import load_epochs
from opm_rps.time_frequency import (
    db_normalize,
    morlet_freqs,
    nearest_time_indices,
    single_trial_power,
)

cfg = load_config()
FREQS, N_CYCLES = morlet_freqs(fmin=2, fmax=50, n_freqs=40, cycles=(4, 10))
TIME_STEP = 0.007  # s, time resolution of the saved power

# %%
for run in cfg.runs:
    epochs = load_epochs(cfg, run)
    times = epochs.times
    tmin, tmax = cfg.epoch_window
    keep = nearest_time_indices(times, np.arange(tmin, tmax + TIME_STEP, TIME_STEP))

    for axis in AXES:
        channel = cfg.roi_channel(axis)
        power_db, labels = [], []
        for condition, code in cfg.event_id.items():
            data = epochs[condition].get_data(picks=[channel])[:, 0]
            power = single_trial_power(data, epochs.info['sfreq'], FREQS, N_CYCLES)
            power_db.append(db_normalize(power, times, cfg.tf_baseline)[..., keep])
            labels.append(np.full(len(data), code))

        np.savez(
            cfg.paths.tf(run, axis),
            power_db=np.concatenate(power_db),
            labels=np.concatenate(labels),
            freqs=FREQS,
            times=times[keep],
        )
