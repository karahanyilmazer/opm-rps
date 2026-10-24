import dataclasses

import mne
import numpy as np
import pandas as pd
import pytest
from mne.transforms import Transform, apply_trans, rotation, translation
from scipy.spatial import ConvexHull
from synthetic import SENSORS, sensor_geometry, write_recording

from opm_rps import coregistration, io

SUBJECT = 'sphere'
HEAD_RADIUS = 0.09  # m
C_RAS = np.array([4.0, -17.0, 25.0])  # scanner RAS of the volume centre (mm)


def sphere_mesh():
    """Scalp: triangulated sphere (m) with outward-facing triangles."""
    golden = np.pi * (3 - np.sqrt(5))
    k = np.arange(4000)
    z = 1 - 2 * (k + 0.5) / k.size
    rho = np.sqrt(1 - z**2)
    rr = HEAD_RADIUS * np.column_stack(
        [rho * np.cos(golden * k), rho * np.sin(golden * k), z]
    )
    tris = ConvexHull(rr).simplices
    normals = np.cross(rr[tris[:, 1]] - rr[tris[:, 0]], rr[tris[:, 2]] - rr[tris[:, 0]])
    inward = np.sum(normals * rr[tris].mean(axis=1), axis=1) < 0
    tris[inward] = tris[inward][:, ::-1]
    return rr, tris


@pytest.fixture(scope='module')
def subjects_dir(tmp_path_factory):
    """FreeSurfer subject with a spherical scalp centred on the surface RAS origin."""
    subjects_dir = tmp_path_factory.mktemp('subjects')
    (subjects_dir / SUBJECT / 'bem').mkdir(parents=True)
    fname = subjects_dir / SUBJECT / 'bem' / f'{SUBJECT}-head.fif'
    mne.write_head_bem(fname, *sphere_mesh())
    return subjects_dir


@pytest.fixture(scope='module')
def t1(subjects_dir):
    """T1.mgz of the subject whose scanner RAS frame is offset by ``C_RAS``."""
    nib = pytest.importorskip('nibabel')
    (subjects_dir / SUBJECT / 'mri').mkdir()

    # Conformed (LIA) volume with 1 mm voxels whose centre lies at C_RAS
    n = 16
    vox2ras = np.array(
        [[-1.0, 0, 0, n / 2], [0, 0, 1, -n / 2], [0, -1, 0, n / 2], [0, 0, 0, 1]]
    )
    vox2ras[:3, 3] += C_RAS
    image = nib.MGHImage(np.zeros((n, n, n), np.uint8), vox2ras)
    nib.save(image, subjects_dir / SUBJECT / 'mri' / 'T1.mgz')


def place_helmet(fname, helmet_to_scanner, bespoke):
    """Put the helmet of a synthetic recording into the scanner RAS frame of the MRI.

    A bespoke helmet (designed on the MRI) has its HelmConfig in scanner RAS and an
    identity SensorTransform; otherwise the SensorTransform maps the helmet there.
    """
    stem = fname.with_name(fname.name.removesuffix('_meg.cMEG'))
    sensor_transform = helmet_to_scanner
    if bespoke:
        helmet = pd.read_csv(f'{stem}_HelmConfig.tsv', sep='\t')
        pos, ori = ['Px', 'Py', 'Pz'], ['Ox', 'Oy', 'Oz']
        helmet[pos] = apply_trans(helmet_to_scanner, helmet[pos].to_numpy())
        helmet[ori] = apply_trans(helmet_to_scanner, helmet[ori].to_numpy(), move=False)
        helmet.to_csv(f'{stem}_HelmConfig.tsv', sep='\t', index=False)
        sensor_transform = np.eye(4)
    pd.DataFrame(sensor_transform).to_csv(
        f'{stem}_SensorTransform.tsv', sep='\t', header=False, index=False
    )


@pytest.mark.usefixtures('t1')
def test_scanner_to_surface_ras(subjects_dir):
    trans = coregistration.scanner_to_surface_ras(SUBJECT, subjects_dir)
    assert (trans['from'], trans['to']) == (
        mne.io.constants.FIFF.FIFFV_MNE_COORD_RAS,
        mne.io.constants.FIFF.FIFFV_COORD_MRI,
    )
    np.testing.assert_allclose(trans['trans'][:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(trans['trans'][:3, 3], -C_RAS / 1000)


@pytest.mark.usefixtures('t1')
@pytest.mark.parametrize('bespoke', [False, True])
def test_scanner_ras_recovers_known_transform(tmp_path, subjects_dir, bespoke):
    fname, _ = write_recording(tmp_path, 'rec', n_trials=2, radius=0.1)
    helmet_to_brain = rotation(x=0.1, y=-0.2, z=0.3)  # in surface RAS
    place_helmet(fname, translation(*C_RAS / 1000) @ helmet_to_brain, bespoke)
    raw, _ = io.read_raw_cmeg(fname)

    trans = coregistration.head_to_mri('scanner_ras', SUBJECT, subjects_dir)
    positions = coregistration.sensor_positions(raw.info, trans)
    helmet, _ = sensor_geometry(len(SENSORS), 0.1)
    expected = apply_trans(helmet_to_brain, helmet)
    np.testing.assert_allclose(positions[::3], expected, atol=1e-6)
    distances = coregistration.scalp_distances(raw.info, trans, SUBJECT, subjects_dir)
    np.testing.assert_allclose(distances, 0.1 - HEAD_RADIUS, atol=1e-3)

    # Taking the scanner RAS frame for the surface RAS frame misplaces the sensors
    trans = coregistration.head_to_mri('identity', SUBJECT, subjects_dir)
    offset = coregistration.sensor_positions(raw.info, trans)[::3] - expected
    np.testing.assert_allclose(
        offset, np.tile(C_RAS / 1000, (len(SENSORS), 1)), atol=1e-6
    )


def test_unknown_method(tmp_path):
    with pytest.raises(ValueError, match='scanner_ras'):
        coregistration.head_to_mri('scan', SUBJECT, tmp_path)


def test_scalp_distances(tmp_path, subjects_dir):
    fname, _ = write_recording(tmp_path, 'rec', n_trials=2, radius=0.1)
    raw, _ = io.read_raw_cmeg(fname)
    identity = Transform('head', 'mri')
    distances = coregistration.scalp_distances(
        raw.info, identity, SUBJECT, subjects_dir
    )
    assert distances.shape == (3 * len(SENSORS),)
    np.testing.assert_allclose(distances, 0.1 - HEAD_RADIUS, atol=1e-3)

    # Moving the sensors 5 cm down puts the top of the helmet inside the head
    down = Transform('head', 'mri', translation(z=-0.05))
    distances = coregistration.scalp_distances(raw.info, down, SUBJECT, subjects_dir)
    helmet, _ = sensor_geometry(len(SENSORS), 0.1)
    expected = np.linalg.norm(helmet - [0, 0, 0.05], axis=1) - HEAD_RADIUS
    np.testing.assert_allclose(distances[::3], expected, atol=1e-3)
    assert (distances < 0).any() and (distances > 0).any()


def test_scalp_from_watershed_surface(tmp_path):
    # Without a head surface FIF file, the watershed bem/outer_skin.surf is used
    pytest.importorskip('nibabel')  # to write the FreeSurfer surface
    rr, tris = sphere_mesh()
    (tmp_path / 'subjects' / 'ws' / 'bem').mkdir(parents=True)
    mne.write_surface(tmp_path / 'subjects/ws/bem/outer_skin.surf', 1000 * rr, tris)
    fname, _ = write_recording(tmp_path, 'rec', n_trials=2, radius=0.1)
    raw, _ = io.read_raw_cmeg(fname)
    identity = Transform('head', 'mri')
    distances = coregistration.scalp_distances(
        raw.info, identity, 'ws', tmp_path / 'subjects'
    )
    np.testing.assert_allclose(distances, 0.1 - HEAD_RADIUS, atol=1e-3)


def test_read_sensor_transforms(cfg):
    transforms = coregistration.read_sensor_transforms(cfg)
    assert list(transforms) == cfg.runs
    for transform in transforms.values():
        np.testing.assert_array_equal(transform, np.eye(4))


def test_source_file_names(cfg):
    assert cfg.source_file('cov').name == f'{cfg.subject}-baseline-cov.fif'
    assert cfg.source_file('trans').name == f'{cfg.subject}-trans.fif'

    misaligned = dataclasses.replace(cfg, coregistration='identity')
    for kind in ('fwd', 'inv', 'stc'):
        assert '-misaligned-' in misaligned.source_file(kind).name
    aligned = dataclasses.replace(cfg, coregistration='scanner_ras')
    assert aligned.source_file('fwd').name == f'{cfg.subject}-ico5-fwd.fif'
