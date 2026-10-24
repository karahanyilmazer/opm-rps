"""MEG-MRI coregistration: compute the head -> MRI transform of the forward model and
check where it puts the sensors relative to the scalp.

``coregistration`` in config.yaml selects the method (see opm_rps.coregistration).
The distances from the sensors to the scalp are compared between all methods: with a
correct transform, every sensor lies outside the head, a few millimetres to a few
centimetres from the scalp.

Inputs: derivatives/epochs/all_runs-filt_7_30-epo.fif (preprocessing/clean_epochs.py)
        with the sensor positions, the SensorTransform sidecars of the runs and the
        FreeSurfer segmentation of the subject
Outputs: derivatives/source/<subject>-trans.fif (not for ``identity``),
         derivatives/figures/source/, derivatives/reports/coregistration.html
"""

# %%
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from opm_rps.config import load_config
from opm_rps.coregistration import (
    METHODS,
    head_to_mri,
    read_sensor_transforms,
    scalp_distances,
    scanner_to_surface_ras,
)
from opm_rps.viz import set_style

set_style()
cfg = load_config()
subject = cfg.subject
subjects_dir = cfg.paths.subjects_dir
info = mne.io.read_info(cfg.paths.clean_epochs(*cfg.decoding_band))

# %% Device -> head transform: the SensorTransform sidecar of each run
for run, transform in read_sensor_transforms(cfg).items():
    print(f'SensorTransform of {run}:\n{transform}')
    if not np.allclose(transform, info['dev_head_t']['trans']):
        warnings.warn(
            f'The SensorTransform of {run} differs from the device -> head '
            'transform of the epochs, which is that of the first run',
            stacklevel=1,
        )

# %% Distances from the sensors to the scalp with each method
# identity and scanner_ras differ by the c_ras offset of the MRI
print(f'Scanner RAS -> surface RAS:\n{scanner_to_surface_ras(subject, subjects_dir)}')
distances = {
    method: 1000
    * scalp_distances(
        info, head_to_mri(method, subject, subjects_dir), subject, subjects_dir
    )
    for method in METHODS
}
summary = pd.DataFrame(
    {method: pd.Series(dist).describe() for method, dist in distances.items()}
).T.drop(columns='count')
summary['inside'] = [np.sum(dist < 0) for dist in distances.values()]
print('Sensor-to-scalp distance (mm) and number of channels inside the head:')
print(summary.round(1).to_string())

fig, ax = plt.subplots()
bins = np.histogram_bin_edges(np.concatenate(list(distances.values())), bins=30)
for method, dist in distances.items():
    ax.hist(dist, bins=bins, histtype='step', label=method)
ax.axvline(0, color='k', linestyle='--')
ax.set(title='Sensor-to-Scalp Distance', xlabel='Distance (mm)', ylabel='Channels')
ax.legend()
fig_dist_file = cfg.paths.figure('source', 'sensor_scalp_distance.png')
fig.savefig(fig_dist_file)
plt.show()

# %% Head -> MRI transform of the configured method
trans = head_to_mri(cfg.coregistration, subject, subjects_dir)
print(f'Head -> MRI transform ({cfg.coregistration}):\n{trans}')
if cfg.coregistration != 'identity':  # source_reconstruction.py reads it
    mne.write_trans(cfg.source_file('trans'), trans, overwrite=True)

fig = mne.viz.plot_alignment(
    info,
    trans=trans,
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces=dict(head=0.5, white=1.0),
    meg='sensors',
    coord_frame='mri',
    show_axes=True,
)

# %% Report
report = mne.Report(title='Coregistration')
report.add_trans(
    trans=trans,
    info=info,
    subject=subject,
    subjects_dir=subjects_dir,
    alpha=0.5,
    title=f'Coregistration: {cfg.coregistration}',
)
report.add_image(fig_dist_file, title='Sensor-to-scalp distance')
report.add_html(
    summary.round(1).to_html(), title='Sensor-to-scalp distance (mm) per method'
)
report.save(cfg.paths.report('coregistration.html'), overwrite=True)
