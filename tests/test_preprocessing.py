import numpy as np
import pytest
from scipy.signal import freqz

from opm_rps import io, preprocessing


def test_load_run(cfg):
    raw, events = preprocessing.load_run(cfg, 'run_1')
    assert set(cfg.bad_channels) <= set(raw.info['bads'])
    assert 'LN[Z]' in raw.info['bads']  # marked bad in the channels.tsv
    assert raw.info['projs'] and all(proj['active'] for proj in raw.info['projs'])
    assert set(events[:, 2]) == set(cfg.event_id.values())


def test_crop_removes_trials_outside_the_run(cfg):
    _, events_uncropped = preprocessing.load_run(cfg, 'run_2')
    raw, events = preprocessing.load_run(cfg, 'run_3')
    assert raw.times[-1] == pytest.approx(cfg.crop['run_3'][1])
    assert len(events) == len(events_uncropped) - 1
    assert events[:, 0].max() <= raw.last_samp


def test_crop_recording_without_triggers(cfg):
    raw, _ = io.read_run(cfg, 'run_1')
    assert preprocessing.crop(raw, None, 0, 10) is None
    assert raw.times[-1] == pytest.approx(10)


def test_load_epochs_drops_the_bad_trials(cfg):
    _, events = preprocessing.load_run(cfg, 'run_1')
    epochs = preprocessing.load_epochs(cfg, 'run_1')
    assert len(epochs) == len(events) - len(cfg.bad_epochs['run_1'])
    assert not np.isin(epochs.events[:, 0], cfg.bad_epochs['run_1']).any()
    assert epochs.baseline is None
    assert (epochs.tmin, epochs.tmax) == pytest.approx(cfg.epoch_window)


def test_load_epochs_concatenates_runs(cfg):
    run_1 = preprocessing.load_epochs(cfg, 'run_1')
    run_2 = preprocessing.load_epochs(cfg, 'run_2')
    both = preprocessing.load_epochs(cfg, ['run_1', 'run_2'])
    assert len(both) == len(run_1) + len(run_2)


def test_mark_bad_channels_skips_unknown_names(cfg):
    raw, _ = io.read_run(cfg, 'run_1')
    with pytest.warns(UserWarning, match='NOPE'):
        preprocessing.mark_bad_channels(raw, ['KB[X]', 'NOPE[X]'])
    assert 'KB[X]' in raw.info['bads']
    assert 'NOPE[X]' not in raw.info['bads']


def test_channel_selection():
    names = ['KB[X]', 'KB[Y]', 'KB[Z]', 'Trigger1', 'Trigger[Z]']
    assert preprocessing.axis_channels(names, 'Z') == ['KB[Z]']
    assert preprocessing.axis_channels(names, 'x') == ['KB[X]']
    assert preprocessing.trigger_channels(names) == ['Trigger1', 'Trigger[Z]']


def test_firls_bandpass():
    sfreq = 1200
    kernel = preprocessing.firls_bandpass(sfreq, 7, 30)
    assert len(kernel) % 2 == 1

    freqs, response = freqz(kernel, worN=2**14, fs=sfreq)
    gain = np.abs(response) ** 2  # zero-phase filtering applies the kernel twice
    passband = (freqs >= 9) & (freqs <= 28)
    stopband = (freqs <= 3) | (freqs >= 36)
    assert np.all(np.abs(gain[passband] - 1) < 0.15)  # least-squares ripple
    assert np.all(gain[stopband] < 0.01)
