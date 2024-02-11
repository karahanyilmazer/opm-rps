# %%
import matplotlib.pyplot as plt
import mne
import numpy as np

# %%
opm_data_folder = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    opm_data_folder
    / "sub-002"
    / "ses-001"
    / "meg"
    / "sub-002_ses-001_task-aef_run-001_meg.bin"
)
# For now we are going to assume the device and head coordinate frames are
# identical (even though this is incorrect), so we pass verbose='error' for now
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()  # crop for speed
# %%
picks = mne.pick_types(raw.info, meg=True)

amp_scale = 1e12  # T->pT
stop = len(raw.times) - 300
step = 300
data_ds, time_ds = raw[picks[::5], :stop]
data_ds, time_ds = data_ds[:, ::step] * amp_scale, time_ds[::step]

fig, ax = plt.subplots(constrained_layout=True)
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-500, 500), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="No preprocessing", **set_kwargs)
# %%
# set flux channels to bad
bad_picks = mne.pick_channels_regexp(raw.ch_names, regexp="Flux.")
raw.info["bads"].extend([raw.ch_names[ii] for ii in bad_picks])
raw.info["bads"].extend(["G2-17-TAN"])

# compute the PSD for later using 1 Hz resolution
psd_kwargs = dict(fmax=20, n_fft=int(round(raw.info["sfreq"])))
psd_pre = raw.compute_psd(**psd_kwargs)

# filter and regress
raw.filter(None, 5, picks="ref_meg")
regress = mne.preprocessing.EOGRegression(picks, picks_artifact="ref_meg")
regress.fit(raw)
regress.apply(raw, copy=False)

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=":")
ax.set(title="After reference regression", **set_kwargs)
plt.show()

# compute the psd of the regressed data
psd_post_reg = raw.compute_psd(**psd_kwargs)
# %%
# include gradients by setting order to 2, set to 1 for homgenous components
projs = mne.preprocessing.compute_proj_hfc(raw.info, order=2)
raw.add_proj(projs).apply_proj(verbose="error")

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=":")
ax.set(title="After HFC", **set_kwargs)
plt.show()

# compute the psd of the regressed data
psd_post_hfc = raw.compute_psd(**psd_kwargs)

# %%
# notch
raw.notch_filter(np.arange(50, 251, 50), notch_widths=4)
# bandpass
raw.filter(2, 40, picks="meg")
# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale
fig, ax = plt.subplots(constrained_layout=True)
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-500, 500), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="After regression, HFC and filtering", **set_kwargs)
plt.show()

# %%
events = mne.find_events(raw, min_duration=0.1)
epochs = mne.Epochs(
    raw, events, tmin=-0.1, tmax=0.4, baseline=(-0.1, 0.0), verbose="error"
)
evoked = epochs.average()
evoked.plot()

# %%
