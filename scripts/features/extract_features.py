"""Extract hand-crafted features from the broadband epochs of runs 2-4.

The runs are concatenated, HFC-corrected, notch-filtered (line noise harmonics and
``notch_freqs``) and band-passed (``broadband``). For the good channels of each
sensor axis, three feature sets are computed per trial: signal variance, relative
power in 7 frequency bands, and total Welch power.

The band power and the total power come from Welch spectra with a resolution of
1 Hz (1-s Hamming windows overlapping by half).

Output: derivatives/features/features.npz with the arrays var_<axis>, bp_<axis>
and welch_<axis> (trials x features, axis in x/y/z) and the labels y (trigger codes).
"""

# %%
import mne
import numpy as np

from opm_rps import features
from opm_rps.config import AXES, load_config
from opm_rps.preprocessing import apply_hfc, axis_channels, load_run, notch_filter

cfg = load_config()
RUNS = ['run_2', 'run_3', 'run_4']  # the original feature matrices exclude run 1

# %% Preprocess the concatenated runs
raws, events_list = zip(*(load_run(cfg, run, hfc=False) for run in RUNS), strict=True)
raw, events = mne.concatenate_raws(list(raws), events_list=list(events_list))
apply_hfc(raw, order=cfg.hfc_order)
notch_filter(raw, cfg.notch_freqs)
raw.filter(*cfg.broadband)

tmin, tmax = cfg.epoch_window
epochs = mne.Epochs(
    raw, events, cfg.event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True
)

# %% Compute and save the feature matrices
matrices = {'y': epochs.events[:, 2]}
for axis in AXES:
    axis_epochs = epochs.copy().pick(axis_channels(epochs.ch_names, axis))
    matrices[f'var_{axis.lower()}'] = features.variance(axis_epochs)
    matrices[f'bp_{axis.lower()}'] = features.band_power(axis_epochs)
    matrices[f'welch_{axis.lower()}'] = features.welch_power(axis_epochs)
np.savez(cfg.paths.features, **matrices)
