"""Visualise the sLORETA estimate of the evoked response and compare it with an MNE
(minimum-norm) estimate.

Inputs: the outputs of source_space/source_reconstruction.py
"""

# %%
import matplotlib.pyplot as plt
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator

from opm_rps.config import load_config
from opm_rps.viz import set_style

set_style()
cfg = load_config()
subjects_dir = cfg.paths.subjects_dir

inv = read_inverse_operator(cfg.source_file('inv'))
stc = mne.read_source_estimate(cfg.source_file('stc'), subject=cfg.subject)

# %% Source time courses (every 100th vertex) and their mean
fig, ax = plt.subplots()
ax.plot(stc.times, stc.data[::100, :].T, alpha=0.1, color='k')
ax.plot(stc.times, stc.data.mean(0), color='r')
ax.set(title='Source Time Courses', xlabel='Time (s)', ylabel='sLORETA Value')
plt.show()

# %% Activity at the peak of the left hemisphere and a movie of its time course
vertno_max, time_max = stc.get_peak(hemi='lh')
surfer_kwargs = dict(
    hemi='lh',
    # 99.8th and 99.96th percentile and close to the maximum of the sLORETA values
    clim=dict(kind='value', lims=[1.85, 2.4, 5.2]),
    views='lateral',
    initial_time=time_max,
    time_unit='s',
    smoothing_steps=10,
    subjects_dir=subjects_dir,
)
brain = stc.plot(size=(1200, 800), **surfer_kwargs)
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi='lh',
    color='blue',
    scale_factor=0.6,
    alpha=0.5,
)
brain.save_movie(
    filename=cfg.paths.figure('source', 'source_activity.mp4'),
    tmin=-0.05,
    tmax=0.4,
    interpolation='linear',
    time_dilation=20,
    framerate=10,
    time_viewer=True,
)

# %% Report
report = mne.Report(title='Source estimate')
report.add_stc(
    stc=stc,
    subject=cfg.subject,
    subjects_dir=subjects_dir,
    title='Source estimate',
    n_time_points=200,
    stc_plot_kwargs={k: v for k, v in surfer_kwargs.items() if k != 'subjects_dir'},
)
report.save(cfg.paths.report('source_estimate.html'), overwrite=True)

# %% MNE (minimum-norm) estimate of the evoked response for comparison
evoked = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band)).average()
snr = 3.0
stc_mne = apply_inverse(evoked, inv, lambda2=1.0 / snr**2, method='MNE', pick_ori=None)
vertno_max, time_max = stc_mne.get_peak(hemi='lh')
brain = stc_mne.plot(
    hemi='lh',
    clim=dict(kind='percent', lims=[99, 99.9, 99.99]),
    views='lateral',
    initial_time=time_max,
    time_unit='s',
    size=(1200, 800),
    smoothing_steps=10,
    background='w',
    subjects_dir=subjects_dir,
)
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi='lh',
    color='blue',
    scale_factor=0.6,
    alpha=0.5,
)
