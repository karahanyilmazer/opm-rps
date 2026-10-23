"""Build the band-passed epochs of all runs used for decoding and source analysis.

The HFC-corrected runs are concatenated, reduced to the good MEG channels and
band-passed with a zero-phase least-squares FIR filter (``decoding_band``). The
epochs are linearly detrended and baseline-corrected (start of the epoch to the
cue); the rejected trials (``bad_epochs`` and ``bad_epoch_indices``, see
``opm_rps.preprocessing.bad_trials``) are dropped.

Output: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz
"""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np

from opm_rps.config import load_config
from opm_rps.preprocessing import (
    apply_fir,
    concatenate_runs,
    drop_trials,
    firls_bandpass,
    trigger_channels,
)
from opm_rps.viz import plot_fir_response, set_style

set_style()
cfg = load_config()

# %% Concatenate the HFC-corrected runs
raw, events, bad_samples = concatenate_runs(cfg, cfg.runs)
raw.drop_channels(trigger_channels(raw.ch_names) + raw.info['bads'])

# %% Design the band-pass filter
sfreq = raw.info['sfreq']
l_freq, h_freq = cfg.decoding_band
kernel = firls_bandpass(sfreq, l_freq, h_freq)
fig = plot_fir_response(kernel, sfreq, l_freq, h_freq)
fig.savefig(cfg.paths.figure('preprocessing', 'fir_kernel.png'))
plt.show()

# %% Filter and compare the first channel before and after filtering
before = raw.get_data(picks=[0])[0]
apply_fir(raw, kernel)
after = raw.get_data(picks=[0])[0]

fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
ax_time.plot(before, 'r', label='Original')
ax_time.plot(after, 'k', label='Filtered')
ax_time.set(title='Time Domain', xlabel='Sample', ylabel='Amplitude (T)')
ax_time.legend()

freqs = np.linspace(0, sfreq, len(before))
spectrum_after = np.abs(np.fft.fft(after))
ax_freq.plot(freqs, np.abs(np.fft.fft(before)), 'r', label='Original')
ax_freq.plot(freqs, spectrum_after, 'k', label='Filtered')
ax_freq.set(
    title='Frequency Domain',
    xlabel='Frequency (Hz)',
    xlim=(0, h_freq * 1.5),
    ylim=(0, 1.1 * spectrum_after.max()),
)
ax_freq.legend()
plt.tight_layout()
plt.show()

# %% Epoch, drop the rejected trials and save
tmin, tmax = cfg.epoch_window
epochs = mne.Epochs(
    raw,
    events,
    cfg.event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(None, 0),
    detrend=cfg.detrend,
)
drop_trials(epochs, bad_samples)
epochs.save(cfg.paths.clean_epochs(l_freq, h_freq), fmt='double', overwrite=True)
