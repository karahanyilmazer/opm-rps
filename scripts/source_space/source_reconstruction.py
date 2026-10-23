"""Compute the source space, BEM, forward and inverse operator, and the sLORETA
estimate of the evoked response.

The MEG-MRI coregistration has not been done: the identity transform is used, so
the source estimates are not anatomically aligned ("misaligned" in the file names).

Inputs: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz (preprocessing/clean_epochs.py)
        and the FreeSurfer segmentation of the subject
Outputs: derivatives/source/, derivatives/reports/source_space.html
"""

# %%
import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator

from opm_rps.config import load_config

cfg = load_config()
subject = cfg.subject
subjects_dir = cfg.paths.subjects_dir
TRANS = mne.transforms.Transform('head', 'mri')  # identity: not coregistered

epochs = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band))
evoked = epochs.average()

# %% Anatomy: cortical parcellation and BEM surfaces
Brain = mne.viz.get_brain_class()
brain = Brain(
    subject, hemi='lh', surf='pial', size=(800, 600), subjects_dir=subjects_dir
)
brain.add_annotation('aparc.a2009s', borders=False)

plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces='white',
    orientation='coronal',
    slices=[50, 100, 150, 200],
)
mne.viz.plot_bem(**plot_bem_kwargs)

# %% Surface source space of both hemispheres
src = mne.setup_source_space(
    subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir, n_jobs=-1
)
mne.write_source_spaces(cfg.source_file('src'), src, overwrite=True)
print(src)

mne.viz.plot_bem(src=src, **plot_bem_kwargs)
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces='white',
    coord_frame='mri',
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)

# %% BEM: a single layer (inner skull) is sufficient for MEG
model = mne.make_bem_model(
    subject=subject, ico=5, conductivity=(0.3,), subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)
mne.write_bem_solution(cfg.source_file('bem'), bem, overwrite=True)

# %% Forward model (leadfield)
fwd = mne.make_forward_solution(
    epochs.info, trans=TRANS, src=src, bem=bem, meg=True, eeg=False, n_jobs=-1
)
mne.write_forward_solution(cfg.source_file('fwd'), fwd, overwrite=True)
n_sensors, n_dipoles = fwd['sol']['data'].shape
print(f'Leadfield: {n_sensors} sensors x {n_dipoles} dipoles')

# %% Noise covariance from the pre-cue baseline (start of the epoch to the cue)
# The HFC projectors reduce the rank of the data by their number (8 for order 2)
cov = mne.compute_covariance(epochs, tmax=0, method='auto', rank='info')
cov.save(cfg.source_file('cov'), overwrite=True)
rank = mne.compute_rank(cov, info=epochs.info)['mag']
n_channels, n_projs = len(epochs.ch_names), len(epochs.info['projs'])
print(f'Noise covariance rank: {rank} ({n_channels} channels - {n_projs} projectors)')
mne.viz.plot_cov(cov, epochs.info)

# %% Inverse operator and sLORETA estimate of the evoked response
inv = make_inverse_operator(epochs.info, fwd, cov, loose=0.2, rank='info')
mne.minimum_norm.write_inverse_operator(cfg.source_file('inv'), inv, overwrite=True)
stc, residual = apply_inverse(
    evoked, inv, method='sLORETA', pick_ori=None, return_residual=True
)
stc.save(cfg.source_file('stc'), overwrite=True)

# %% Report of the MRI, BEM and (identity) coregistration
report = mne.Report(title='Source Space')
report.add_bem(
    subject=subject, subjects_dir=subjects_dir, title='MRI & BEM', width=512, n_jobs=-1
)
report.add_trans(
    trans=TRANS,
    info=epochs.info,
    subject=subject,
    subjects_dir=subjects_dir,
    alpha=1.0,
    title='Coregistration',
)
report.save(cfg.paths.report('source_space.html'), overwrite=True)
