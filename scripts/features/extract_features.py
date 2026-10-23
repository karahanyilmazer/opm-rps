"""Extract hand-crafted features from the broadband epochs of runs 2-4.

The runs are concatenated, HFC-corrected, notch-filtered (line noise harmonics and
``notch_freqs``) and band-passed (``broadband``), and the rejected trials
(``bad_epochs`` and ``bad_epoch_indices``, see ``opm_rps.preprocessing.bad_trials``)
are dropped. For the good channels of each sensor axis, three feature sets are
computed per trial: signal variance, relative power in 7 frequency bands, and total
Welch power.

Note: the Welch spectra use n_fft=256, i.e. a resolution of sfreq/256 Hz (~4.7 Hz
at 1200 Hz). The delta, theta and alpha bands then contain a single frequency bin
and their band power is zero.

Output: derivatives/features/features.npz with the arrays var_<axis>, bp_<axis>
and welch_<axis> (trials x features, axis in x/y/z) and the labels y (trigger codes).
"""

# %%
import mne
import numpy as np

from opm_rps import features
from opm_rps.config import AXES, load_config
from opm_rps.preprocessing import (
    apply_hfc,
    axis_channels,
    concatenate_runs,
    drop_trials,
    notch_filter,
)

cfg = load_config()
RUNS = ['run_2', 'run_3', 'run_4']  # the original feature matrices exclude run 1

# %% Preprocess the concatenated runs
raw, events, bad_samples = concatenate_runs(cfg, RUNS, hfc=False)
apply_hfc(raw, order=cfg.hfc_order)
notch_filter(raw, cfg.notch_freqs)
raw.filter(*cfg.broadband)

tmin, tmax = cfg.epoch_window
epochs = mne.Epochs(
    raw, events, cfg.event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True
)
drop_trials(epochs, bad_samples)

# %% Compute and save the feature matrices
matrices = {'y': epochs.events[:, 2]}
for axis in AXES:
    axis_epochs = epochs.copy().pick(axis_channels(epochs.ch_names, axis))
    matrices[f'var_{axis.lower()}'] = features.variance(axis_epochs)
    matrices[f'bp_{axis.lower()}'] = features.band_power(axis_epochs)
    matrices[f'welch_{axis.lower()}'] = features.welch_power(axis_epochs)
np.savez(cfg.paths.features, **matrices)
