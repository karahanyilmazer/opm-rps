"""Synthetic recordings in the format of the Nottingham OPM system, for tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SFREQ = 1200.0
CUE_CODES = (1, 2, 4)  # rock, paper, scissors
END_TRIAL = 7
EXPERIMENT_MARKER = 255
N_TRIGGERS = 8

#: Sensor codes: the ROI sensor KB, the configured motor sensors and a few others
SENSORS = (
    'KB FV LJ LQ LI I7 FW HI FR FZ G0 HF HJ K9 KC KD KE KF LB LF LN LM LP MV MU '
    'A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3'
).split()


def sensor_geometry(
    n_sensors: int, radius: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor positions on a hemisphere (m) and their radial unit vectors."""
    golden = np.pi * (3 - np.sqrt(5))
    k = np.arange(n_sensors)
    z = 0.15 + 0.85 * (k + 0.5) / n_sensors
    rho = np.sqrt(1 - z**2)
    radial = np.column_stack([rho * np.cos(golden * k), rho * np.sin(golden * k), z])
    return radius * radial, radial


def axis_orientations(radial: np.ndarray) -> np.ndarray:
    """Orthonormal (X, Y, Z) sensing directions per sensor, Z being radial."""
    helper = np.where(np.abs(radial[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    x_dir = np.cross(radial, helper)
    x_dir /= np.linalg.norm(x_dir, axis=1, keepdims=True)
    y_dir = np.cross(radial, x_dir)
    return np.stack([x_dir, y_dir, radial], axis=1)  # (n_sensors, 3 axes, 3)


def write_recording(
    directory: str | Path,
    stem: str,
    *,
    n_trials: int = 30,
    trial_spacing: int = 4400,
    sensors: tuple[str, ...] | list[str] = SENSORS,
    bad_in_tsv: tuple[str, ...] = (),
    reference_sensor: bool = False,
    chunk_size: int = 100,
    escaped_scale_column: bool = False,
    radius: float = 0.1,
    seed: int = 0,
) -> tuple[Path, np.ndarray]:
    """Write ``<stem>_meg.cMEG`` and its sidecar files into ``directory``.

    Returns the path of the .cMEG file and the true events (sample, 0, code).
    """
    rng = np.random.default_rng(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    positions, radial = sensor_geometry(len(sensors), radius)
    orientations = axis_orientations(radial)
    # Sensor gains are fixed across the runs of a session
    gains = 2.7 + 0.1 * np.random.default_rng(1234).random(3 * len(sensors))

    # Channel table: three axes per sensor, optional reference, triggers
    names, kinds, scales, chan_ori, chan_pos = [], [], [], [], []
    for code, pos, oris in zip(sensors, positions, orientations, strict=True):
        for axis, ori in zip('XYZ', oris, strict=True):
            names.append(f'{code} [{axis}]')
            kinds.append('MEGMAG')
            scales.append(gains[len(scales)])
            chan_ori.append(ori)
            chan_pos.append(pos)
    if reference_sensor:
        names.append('RF [Z]')  # not in the helmet config -> reference channel
        kinds.append('MEGMAG')
        scales.append(2.7)
        chan_ori.append(np.array([0.0, 0.0, 1.0]))
        chan_pos.append(np.array([0.0, 0.0, 0.3]))
    n_meg = len(names)
    names += [f'Trigger {i + 1}' for i in range(N_TRIGGERS)]
    kinds += ['TRIG'] * N_TRIGGERS
    scales += [1.0] * N_TRIGGERS
    status = [
        'Bad' if name.replace(' ', '') in bad_in_tsv else 'Good' for name in names
    ]

    # Events: experiment markers around cues in random order, each cue followed by
    # an end-of-trial marker
    first = 2 * trial_spacing
    cue_onsets = first + trial_spacing * np.arange(n_trials)
    cue_codes = rng.permutation(np.resize(CUE_CODES, n_trials))
    n_samples = cue_onsets[-1] + 2 * trial_spacing
    events = [(trial_spacing, 0, EXPERIMENT_MARKER)]
    for onset, code in zip(cue_onsets, cue_codes, strict=True):
        events += [(onset, 0, code), (onset + 3000, 0, END_TRIAL)]
    events.append((n_samples - trial_spacing, 0, EXPERIMENT_MARKER))
    events = np.array(events)

    # Magnetic field (T): homogeneous environmental field, line noise, sensor noise
    # and a condition-dependent 12 Hz response over the ROI sensor (first sensor)
    times = np.arange(n_samples) / SFREQ
    env_field = np.cumsum(rng.normal(0, 2e-13, (3, n_samples)), axis=1)
    env_field += 5e-12 * np.sin(2 * np.pi * 50 * times)
    field = np.array(chan_ori) @ env_field
    field += rng.normal(0, 3e-13, field.shape)
    distance = np.linalg.norm(np.array(chan_pos) - positions[0], axis=1)
    weight = np.exp(-((distance / 0.03) ** 2))
    burst_time = np.arange(int(SFREQ)) / SFREQ
    burst = np.sin(2 * np.pi * 12 * burst_time) * np.hanning(burst_time.size)
    for onset, code in zip(cue_onsets, cue_codes, strict=True):
        start = onset + int(0.5 * SFREQ)
        amplitude = {1: 1e-12, 2: 2e-12, 4: 3e-12}[code]
        field[:, start : start + burst.size] += amplitude * np.outer(weight, burst)
    meg = field * 1e9 * np.array(scales[:n_meg])[:, None]  # T -> V

    triggers = np.zeros((N_TRIGGERS, n_samples))
    for onset, _, code in events:
        for bit in range(N_TRIGGERS):
            if code >> bit & 1:
                triggers[bit, onset : onset + 60] = 5.0

    data = np.vstack([times, meg, triggers])  # first row: time stamps
    with open(directory / f'{stem}_meg.cMEG', 'wb') as fid:
        for start in range(0, n_samples, chunk_size):
            chunk = data[:, start : start + chunk_size]
            np.array(chunk.shape, dtype='>u4').tofile(fid)
            chunk.astype('>f8').tofile(fid)

    scale_column = 'nT0x2FV' if escaped_scale_column else 'nT/V'
    pd.DataFrame(
        {'name': names, 'type': kinds, scale_column: scales, 'status': status}
    ).to_csv(directory / f'{stem}_channels.tsv', sep='\t', index=False)
    n_located = 3 * len(sensors)  # the reference sensor has no helmet entry
    helmet_rows = [
        (name.replace(' [', '_').rstrip(']'), *pos, *ori)
        for name, pos, ori in zip(
            names[:n_located], chan_pos[:n_located], chan_ori[:n_located], strict=True
        )
    ]
    pd.DataFrame(
        helmet_rows, columns=['Sensor', 'Px', 'Py', 'Pz', 'Ox', 'Oy', 'Oz']
    ).assign(Name=lambda df: df['Sensor']).to_csv(
        directory / f'{stem}_HelmConfig.tsv', sep='\t', index=False
    )
    pd.DataFrame(np.eye(4)).to_csv(
        directory / f'{stem}_SensorTransform.tsv', sep='\t', header=False, index=False
    )
    (directory / f'{stem}_meg.json').write_text(
        json.dumps({'SamplingFrequency': SFREQ, 'PowerLineFrequency': 50})
    )
    return directory / f'{stem}_meg.cMEG', events


def write_dataset(root: str | Path, n_trials: int = 30, **recording_kwargs) -> Path:
    """Write a four-run session and a matching config file; return the config path.

    Run 3 is cropped, a few channels and trials are marked bad, mirroring the real
    configuration.
    """
    root = Path(root)
    params = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    session = params['session']
    raw_dir = root / 'data' / params['raw_subdir'] / session
    bad_samples = {}
    for seed, run in enumerate(params['runs']):
        acquisition = params['acquisitions'][run]
        stem = f'{session}_{acquisition}'
        _, events = write_recording(
            raw_dir / f'{stem}_cMEG_Data',
            stem,
            n_trials=n_trials,
            bad_in_tsv=('LN[Z]',),
            seed=seed,
            **recording_kwargs,
        )
        cues = events[np.isin(events[:, 2], CUE_CODES), 0]
        bad_samples[run] = [int(cues[1]), int(cues[5])]
        if run == 'run_3':  # crop away the last trial, like the real run 3
            crop_tmax = int(cues[-1] / SFREQ) - 1

    params.update(
        crop={'run_3': [0, crop_tmax]},
        bad_epochs=bad_samples,
        bad_epoch_indices={'X': [0, 3], 'Y': [3, 40], 'Z': [7]},
        bad_channels=['FR[Y]', 'FZ[X]', 'K9[X]', 'LN[X]', 'LN[Y]', 'MV[Z]'],
    )
    config_file = root / 'config.yaml'
    config_file.write_text(yaml.safe_dump(params, sort_keys=False))
    return config_file
