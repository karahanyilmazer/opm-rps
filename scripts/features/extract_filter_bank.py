"""Band-pass runs 1-4 into the decoding band and the bands of the filter bank CSP.

Each HFC-corrected run is reduced to the good magnetometers (the channels of the
cleaned epochs) and band-passed into the decoding band (``decoding_band``) and into
each 4 Hz band of the filter bank (2 Hz apart, spanning the decoding band), with
MNE's zero-phase FIR filter and 2 Hz wide transition bands. The continuous data is
filtered before it is epoched, so the filters have no edge effects within the short
epochs (``epoch_window``). The epochs are decimated to 150 Hz, five times the highest
cutoff, and the rejected trials (``bad_epochs`` and ``bad_epoch_indices``, see
``opm_rps.preprocessing.bad_trials``) are dropped.

Output: derivatives/features/filter_bank.npz with the arrays broadband (trials x
channels x times), filter_bank (trials x bands x channels x times), bands (low and
high cutoff of each band in Hz), times (s), y (trigger codes), run (run of each
trial) and ch_names.
"""

# %%
import numpy as np

from opm_rps.config import load_config
from opm_rps.features import filter_bank_bands
from opm_rps.preprocessing import filter_bank_epochs

cfg = load_config()

BANDS = filter_bank_bands(*cfg.decoding_band)  # 7-11 Hz, 9-13 Hz, ..., 25-29 Hz
DECIM = 8  # 1200 Hz -> 150 Hz

# %% Band-pass, epoch and save
broadband, *sub_bands = filter_bank_epochs(
    cfg, cfg.runs, [cfg.decoding_band, *BANDS], decim=DECIM
)
np.savez(
    cfg.paths.filter_bank,
    broadband=broadband.get_data(copy=False),
    filter_bank=np.stack([epochs.get_data(copy=False) for epochs in sub_bands], axis=1),
    bands=np.array(BANDS),
    times=broadband.times,
    y=broadband.events[:, 2],
    run=broadband.metadata['run'].to_numpy(dtype=str),
    ch_names=np.array(broadband.ch_names),
)
