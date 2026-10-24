import numpy as np
from synthetic import (
    N_TRIGGERS,
    SENSORS,
    axis_orientations,
    sensor_geometry,
    write_recording,
)

from opm_rps import io


def test_read_raw_cmeg(tmp_path):
    fname, true_events = write_recording(
        tmp_path, 'rec', n_trials=6, bad_in_tsv=('KB[Y]',)
    )
    raw, events = io.read_raw_cmeg(fname)

    types = raw.get_channel_types()
    assert types.count('mag') == 3 * len(SENSORS)
    assert types.count('stim') == N_TRIGGERS
    assert raw.ch_names[:3] == ['KB[X]', 'KB[Y]', 'KB[Z]']
    assert raw.info['bads'] == ['KB[Y]']
    np.testing.assert_array_equal(events, true_events)
    assert np.abs(raw.get_data(picks='mag')).max() < 1e-9  # tesla

    # Position and orientation of KB[Z] come from the helmet config
    positions, radial = sensor_geometry(len(SENSORS))
    loc = raw.info['chs'][2]['loc']
    np.testing.assert_allclose(loc[:3], positions[0])
    np.testing.assert_allclose(loc[9:], axis_orientations(radial)[0, 2])
    coil_frame = loc[3:].reshape(3, 3)
    np.testing.assert_allclose(coil_frame @ coil_frame.T, np.eye(3), atol=1e-12)


def test_helmet_config_with_metadata_line(tmp_path):
    fname, _ = write_recording(tmp_path, 'rec', n_trials=2)
    expected, _ = io.read_raw_cmeg(fname)
    # Like the real sidecars: a last line with the helmet name and creation date
    with open(tmp_path / 'rec_HelmConfig.tsv', 'a') as file:
        file.write('Helmet:\tdense\tCreated:\t22\t6\t2023\tdd/mm/yyyy\thh:mm\n')
    raw, _ = io.read_raw_cmeg(fname)
    for ch, ch_expected in zip(raw.info['chs'], expected.info['chs'], strict=True):
        np.testing.assert_array_equal(ch['loc'], ch_expected['loc'])


def test_reference_sensor_and_escaped_scale_column(tmp_path):
    fname, _ = write_recording(
        tmp_path, 'rec', n_trials=2, reference_sensor=True, escaped_scale_column=True
    )
    raw, _ = io.read_raw_cmeg(fname)
    assert raw.get_channel_types(picks=['RF[Z]']) == ['ref_meg']


def test_chunk_size_does_not_matter(tmp_path):
    small, _ = write_recording(tmp_path / 'small', 'rec', n_trials=2, chunk_size=100)
    large, _ = write_recording(tmp_path / 'large', 'rec', n_trials=2, chunk_size=1000)
    np.testing.assert_array_equal(io.read_cmeg_data(small), io.read_cmeg_data(large))


def test_trigger_bits_switching_a_few_samples_apart():
    stim = np.zeros((N_TRIGGERS, 200))
    stim[0, 50:80] = 5  # bit 1
    stim[2, 52:80] = 5  # bit 4, two samples late
    assert io.find_trigger_events(stim).tolist() == [[50, 0, 5]]


def test_no_triggers():
    assert io.find_trigger_events(np.zeros((N_TRIGGERS, 100))) is None
