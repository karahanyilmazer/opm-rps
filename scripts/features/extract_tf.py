"""Extract single-trial time-frequency power of the ROI sensor for decoding.

For every run and sensor axis, each trial is convolved with complex Morlet wavelets.
The trials are epoched with a buffer of recorded data on both sides of
``epoch_window`` (three Gaussian widths of the 2 Hz wavelet, 0.95 s), so that the
wavelets do not reach past the data within the window; trials whose buffered epoch
does not fit into the recording are dropped. The power is cropped back to
``epoch_window``, converted to dB relative to the pre-cue baseline (``tf_baseline``)
and downsampled in time.

Output: derivatives/tf/tf-<run>-<axis>.npz with power_db (trials x freqs x times),
the labels (trigger codes), freqs and times.
"""

# %%
import numpy as np

from opm_rps.config import AXES, load_config
from opm_rps.time_frequency import (
    db_normalize,
    load_buffered_epochs,
    morlet_freqs,
    nearest_time_indices,
    single_trial_power,
    wavelet_buffer,
)

cfg = load_config()
FREQS, N_CYCLES = morlet_freqs(fmin=2, fmax=50, n_freqs=40, cycles=(4, 10))
BUFFER = wavelet_buffer(FREQS, N_CYCLES)  # s of data on both sides of the epochs
TIME_STEP = 0.007  # s, time resolution of the saved power

# %%
for run in cfg.runs:
    epochs, window = load_buffered_epochs(cfg, run, BUFFER)
    times = epochs.times[window]
    tmin, tmax = cfg.epoch_window
    keep = nearest_time_indices(times, np.arange(tmin, tmax + TIME_STEP, TIME_STEP))

    for axis in AXES:
        channel = cfg.roi_channel(axis)
        power_db, labels = [], []
        for condition, code in cfg.event_id.items():
            data = epochs[condition].get_data(picks=[channel])[:, 0]
            power = single_trial_power(data, epochs.info['sfreq'], FREQS, N_CYCLES)
            power = power[..., window]
            power_db.append(db_normalize(power, times, cfg.tf_baseline)[..., keep])
            labels.append(np.full(len(data), code))

        np.savez(
            cfg.paths.tf(run, axis),
            power_db=np.concatenate(power_db),
            labels=np.concatenate(labels),
            freqs=FREQS,
            times=times[keep],
        )
