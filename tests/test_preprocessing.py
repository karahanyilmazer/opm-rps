import dataclasses
import re

import mne
import numpy as np
import pytest
from scipy.signal import freqz

from opm_rps import io, preprocessing


def manual_drops(epochs, events_by_run):
    """(run, cue sample) of the trials dropped by drop_trials, from the drop log."""
    trials = [
        (run, sample)
        for run, events in events_by_run.items()
        for sample in events[:, 0]
    ]
    return {
        trial
        for trial, log in zip(trials, epochs.drop_log, strict=True)
        if log == ('USER',)
    }


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
    bad = preprocessing.bad_trials(cfg, 'run_1')
    epochs = preprocessing.load_epochs(cfg, 'run_1')
    assert len(epochs) == len(events) - len(bad)
    assert not np.isin(epochs.events[:, 0], bad).any()
    assert epochs.baseline is None
    assert (epochs.tmin, epochs.tmax) == pytest.approx(cfg.epoch_window)


def test_bad_trials_is_the_union_of_both_inspections(cfg):
    # 12 trials per run, 11 in run 3 after the crop. bad_epochs: trials 1 and 5 of
    # each run. bad_epoch_indices: 0, 3 and 7 in run 1; 40 and 42 in run 4, which
    # starts after the 35 cue events of runs 1-3
    expected = {
        'run_1': [0, 1, 3, 5, 7],
        'run_2': [1, 5],
        'run_3': [1, 5],
        'run_4': [1, 5, 7],
    }
    for run, trials in expected.items():
        _, events = preprocessing.load_run(cfg, run)
        np.testing.assert_array_equal(
            preprocessing.bad_trials(cfg, run), events[trials, 0]
        )


def test_bad_trials_validates_the_config(cfg):
    n_cues = sum(len(preprocessing.load_run(cfg, run)[1]) for run in cfg.runs)
    _, events = preprocessing.load_run(cfg, 'run_4')
    last = dataclasses.replace(cfg, bad_epoch_indices={'X': [n_cues - 1]})
    assert events[-1, 0] in preprocessing.bad_trials(last, 'run_4')
    for index in (n_cues, -1):
        invalid = dataclasses.replace(cfg, bad_epoch_indices={'X': [index]})
        with pytest.raises(ValueError, match='does not match the recordings'):
            preprocessing.bad_trials(invalid, 'run_1')

    _, events = preprocessing.load_run(cfg, 'run_2')
    sample = int(events[1, 0])
    typo = dataclasses.replace(cfg, bad_epochs={'run_2': [sample, sample + 1]})
    with pytest.warns(UserWarning, match=re.escape(f'no cue event: [{sample + 1}]')):
        np.testing.assert_array_equal(preprocessing.bad_trials(typo, 'run_2'), [sample])


def test_every_analysis_rejects_the_same_trials(cfg):
    events = {run: preprocessing.load_run(cfg, run)[1] for run in cfg.runs}
    rejected = {
        (run, sample)
        for run in cfg.runs
        for sample in preprocessing.bad_trials(cfg, run)
    }

    per_run = set()
    for run in cfg.runs:
        epochs = preprocessing.load_epochs(cfg, run)
        per_run |= manual_drops(epochs, {run: events[run]})
    assert per_run == rejected

    # The epoching of clean_epochs.py (all runs, lazy) and extract_features.py (runs
    # 2-4 without HFC, preloaded)
    tmin, tmax = cfg.epoch_window
    for runs, hfc, preload in [
        (cfg.runs, True, False),
        (['run_2', 'run_3', 'run_4'], False, True),
    ]:
        raw, concatenated, bad = preprocessing.concatenate_runs(cfg, runs, hfc=hfc)
        epochs = mne.Epochs(
            raw, concatenated, cfg.event_id, tmin=tmin, tmax=tmax, preload=preload
        )
        preprocessing.drop_trials(epochs, bad)
        assert manual_drops(epochs, {run: events[run] for run in runs}) == {
            trial for trial in rejected if trial[0] in runs
        }


def test_rejection_does_not_depend_on_the_epoch_window(cfg):
    _, events = preprocessing.load_run(cfg, 'run_2')
    cues = events[:, 0]
    bad = preprocessing.bad_trials(cfg, 'run_2')  # trials 1 and 5
    kept = preprocessing.load_epochs(cfg, 'run_2').events[:, 0]
    tmin, tmax = cfg.epoch_window
    # The buffers of the wavelet analyses, and one so wide that MNE also drops the
    # first and the last trial, which shifts the positions of the later trials
    for buffer, edge_trials in [(0.95, []), (1.91, []), (7, [cues[0], cues[-1]])]:
        wide = dataclasses.replace(cfg, epoch_window=(tmin - buffer, tmax + buffer))
        epochs = preprocessing.load_epochs(wide, 'run_2')
        assert not np.isin(epochs.events[:, 0], bad).any()
        np.testing.assert_array_equal(
            epochs.events[:, 0], np.setdiff1d(kept, edge_trials)
        )


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
