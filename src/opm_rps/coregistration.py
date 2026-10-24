"""MEG-MRI coregistration: where the sensors are relative to the subject's MRI.

The forward model places the sensors with two transforms, device -> head and
head -> MRI. The device frame is the helmet CAD frame of the HelmConfig sidecar. The
SensorTransform sidecar, which becomes ``info['dev_head_t']``, maps it to the frame of
a coregistration saved at acquisition (a 3D scan of the head or the anatomical MRI)
and is the identity if none was saved. The MRI frame is FreeSurfer's surface RAS
frame, in which the surfaces of the segmentation are defined.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd
from mne.transforms import Transform, apply_trans, combine_transforms
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .config import Config

#: Coregistration methods (``coregistration`` in config.yaml)
METHODS = ('identity', 'scanner_ras')


def head_to_mri(method: str, subject: str, subjects_dir: str | Path) -> Transform:
    """Head -> MRI transform of a coregistration method.

    ``'identity'``
        No coregistration: the head frame is taken as the MRI frame.
    ``'scanner_ras'``
        The head frame is the scanner RAS frame of the MRI, i.e. the world frame of
        the image the segmentation was made from: the helmet was designed on this
        MRI, or the SensorTransform maps the sensors into it.
    """
    if method == 'identity':
        return Transform('head', 'mri')
    if method == 'scanner_ras':
        ras_to_mri = scanner_to_surface_ras(subject, subjects_dir)
        return Transform('head', 'mri', ras_to_mri['trans'])
    raise ValueError(f'Unknown coregistration {method!r}, expected one of {METHODS}')


def read_sensor_transforms(cfg: Config) -> dict[str, np.ndarray]:
    """SensorTransform sidecar (4 x 4) of each run, the device -> head transform."""
    transforms = {}
    for run in cfg.runs:
        fname = cfg.paths.cmeg_file(cfg.session, cfg.acquisitions[run])
        stem = fname.name.removesuffix('_meg.cMEG')
        sidecar = fname.with_name(f'{stem}_SensorTransform.tsv')
        transforms[run] = pd.read_csv(sidecar, header=None, sep='\t').to_numpy()
    return transforms


def scanner_to_surface_ras(subject: str, subjects_dir: str | Path) -> Transform:
    """Transform from the scanner RAS to the surface RAS frame of the subject's MRI.

    Computed from the header of ``mri/T1.mgz`` as in MNE's OPM tutorial; for a
    FreeSurfer volume this is a translation by minus its ``c_ras``. Needs nibabel
    (``viz3d`` extra).
    """
    import nibabel as nib

    header = nib.load(Path(subjects_dir) / subject / 'mri' / 'T1.mgz').header
    ras_to_mri = header.get_vox2ras_tkr() @ np.linalg.inv(header.get_vox2ras())
    ras_to_mri[:3, 3] /= 1000  # mm -> m
    return Transform('ras', 'mri', ras_to_mri)


def sensor_positions(info: mne.Info, trans: Transform) -> np.ndarray:
    """Positions (m) of the MEG channels in the MRI frame.

    ``trans`` is the head -> MRI transform; channels without a position are NaN.
    """
    picks = mne.pick_types(info, meg=True, ref_meg=False, exclude=())
    device = np.array([info['chs'][pick]['loc'][:3] for pick in picks])
    dev_head_t = info['dev_head_t'] or Transform('meg', 'head')
    return apply_trans(combine_transforms(dev_head_t, trans, 'meg', 'mri'), device)


def scalp_distances(
    info: mne.Info, trans: Transform, subject: str, subjects_dir: str | Path
) -> np.ndarray:
    """Distance (m) from each MEG channel to the scalp, negative inside the head.

    Measured along the normal of the nearest vertex of the subject's scalp surface.
    ``trans`` is the head -> MRI transform.
    """
    head = _read_scalp(subject, subjects_dir)
    positions = sensor_positions(info, trans)
    located = np.isfinite(positions).all(axis=1)
    _, nearest = cKDTree(head['rr']).query(positions[located])
    offset = positions[located] - head['rr'][nearest]
    distances = np.full(len(positions), np.nan)
    distances[located] = np.sum(offset * head['nn'][nearest], axis=1)
    return distances


def _read_scalp(subject: str, subjects_dir: str | Path) -> dict:
    """Scalp surface of the subject (MRI frame, m) with outward vertex normals.

    The first of ``bem/<subject>-head-dense.fif``, ``-head.fif`` or the BEM, as in
    :func:`mne.dig_mri_distances`, else the watershed surface ``bem/outer_skin.surf``,
    which :func:`mne.viz.plot_alignment` shows as the head.
    """
    try:
        return mne.get_head_surf(
            subject, ('head-dense', 'head', 'bem'), subjects_dir=subjects_dir
        )
    except OSError:
        fname = Path(subjects_dir) / subject / 'bem' / 'outer_skin.surf'
        rr, tris = mne.read_surface(fname)
    rr = rr / 1000  # mm -> m
    tri_nn = np.cross(rr[tris[:, 1]] - rr[tris[:, 0]], rr[tris[:, 2]] - rr[tris[:, 0]])
    nn = np.zeros_like(rr)
    for corner in tris.T:
        np.add.at(nn, corner, tri_nn)
    return dict(rr=rr, tris=tris, nn=nn / np.linalg.norm(nn, axis=1, keepdims=True))
