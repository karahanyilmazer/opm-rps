"""Project configuration: analysis parameters and file locations.

Analysis parameters live in ``config.yaml`` at the repository root. Machine-specific
paths are read from environment variables, which can be set in a ``.env`` file at the
repository root (see the README):

- ``OPM_RPS_DATA_DIR``: folder containing the raw cMEG data and the MRI segmentation
- ``OPM_RPS_DERIVATIVES_DIR`` (optional): where derived files are written,
  defaults to ``<repo>/derivatives``
- ``OPM_RPS_CONFIG`` (optional): alternative config file, defaults to
  ``<repo>/config.yaml``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .io import EVENT_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / 'config.yaml'

#: Sensor axes of the triaxial OPMs; channel names end in ``[X]``, ``[Y]`` or ``[Z]``
AXES = ('X', 'Y', 'Z')


@dataclass(frozen=True)
class Paths:
    """Locations of the raw data and of every derived file."""

    data_dir: Path | None
    raw_subdir: str
    mri_subdir: str
    derivatives: Path

    # --- Inputs ---------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Folder with one subfolder of cMEG recordings per session."""
        return self._data_dir() / self.raw_subdir

    @property
    def subjects_dir(self) -> Path:
        """FreeSurfer ``SUBJECTS_DIR`` with the MRI segmentation."""
        return self._data_dir() / self.mri_subdir

    def cmeg_file(self, session: str, acquisition: str) -> Path:
        """Path of the ``.cMEG`` file of one acquisition."""
        stem = f'{session}_{acquisition}'
        return self.raw_dir / session / f'{stem}_cMEG_Data' / f'{stem}_meg.cMEG'

    # --- Outputs --------------------------------------------------------------
    def clean_epochs(self, l_freq: float, h_freq: float) -> Path:
        """Band-passed epochs of all runs used for decoding and source analysis."""
        return self._output('epochs', f'all_runs-filt_{l_freq}_{h_freq}-epo.fif.gz')

    @property
    def features(self) -> Path:
        """Hand-crafted feature matrices (variance, band power, Welch)."""
        return self._output('features', 'features.npz')

    def tf(self, run: str, axis: str) -> Path:
        """Single-trial time-frequency power of the ROI sensor of one run and axis."""
        return self._output('tf', f'tf-{run}-{axis}.npz')

    @property
    def filter_bank(self) -> Path:
        """Epochs band-passed into the decoding band and the filter bank CSP bands."""
        return self._output('features', 'filter_bank.npz')

    def source(self, name: str) -> Path:
        """Source-space object (source space, BEM, forward, covariance, ...)."""
        return self._output('source', name)

    def figure(self, *parts: str) -> Path:
        """Figure file, e.g. ``paths.figure('tf', 'rock.png')``."""
        return self._output('figures', *parts)

    def report(self, name: str) -> Path:
        """MNE HTML report."""
        return self._output('reports', name)

    def _output(self, *parts: str) -> Path:
        path = self.derivatives.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _data_dir(self) -> Path:
        if self.data_dir is None:
            raise RuntimeError(
                'OPM_RPS_DATA_DIR is not set. Set it in a .env file at the repository '
                'root (see the README) to the folder that contains the raw data and '
                'the MRI segmentation.'
            )
        return self.data_dir


@dataclass(frozen=True)
class Config:
    """Analysis parameters, see ``config.yaml`` for their meaning."""

    subject: str
    session: str
    acquisitions: dict[str, str]
    runs: list[str]
    events: list[str]
    roi_sensor: str
    hfc_order: int
    detrend: int | None
    epoch_window: tuple[float, float]
    tf_baseline: tuple[float, float]
    crop: dict[str, tuple[float, float | None]]
    notch_freqs: list[float]
    broadband: tuple[float, float]
    decoding_band: tuple[float, float]
    bad_channels: list[str]
    bad_epochs: dict[str, list[int]]
    bad_epoch_indices: dict[str, list[int]]
    motor_channels: list[str]
    coregistration: str
    paths: Paths

    @property
    def event_id(self) -> dict[str, int]:
        """Trigger codes of the analysed conditions."""
        return {name: EVENT_ID[name] for name in self.events}

    def roi_channel(self, axis: str) -> str:
        """Name of the ROI sensor's channel along one axis, e.g. ``'KB[X]'``."""
        return f'{self.roi_sensor}[{axis}]'

    def source_file(self, kind: str) -> Path:
        """Source-space file: ``'trans'``, ``'src'``, ``'bem'``, ``'cov'``, ``'fwd'``,
        ``'inv'`` or ``'stc'``.

        Without a MEG-MRI coregistration (``coregistration: identity``), everything
        that depends on the sensor positions relative to the brain is labelled
        "misaligned".
        """
        aligned = '-misaligned' if self.coregistration == 'identity' else ''
        names = {
            'trans': f'{self.subject}-trans.fif',
            'src': f'{self.subject}-oct6-src.fif',
            'bem': f'{self.subject}-ico5-bem-sol.fif',
            'cov': f'{self.subject}-baseline-cov.fif',
            'fwd': f'{self.subject}{aligned}-ico5-fwd.fif',
            'inv': f'{self.subject}{aligned}-ico5-meg-inv.fif',
            'stc': f'{self.subject}{aligned}-ico5-sloreta',  # + -lh/-rh.stc
        }
        return self.paths.source(names[kind])


def load_config(path: str | Path | None = None) -> Config:
    """Read the config file and resolve the data paths from the environment."""
    load_dotenv(PROJECT_ROOT / '.env')
    path = path or os.getenv('OPM_RPS_CONFIG') or CONFIG_FILE
    with open(path) as file:
        params = yaml.safe_load(file)

    data_dir = os.getenv('OPM_RPS_DATA_DIR')
    derivatives = os.getenv('OPM_RPS_DERIVATIVES_DIR')
    paths = Paths(
        data_dir=Path(data_dir).expanduser() if data_dir else None,
        raw_subdir=params.pop('raw_subdir'),
        mri_subdir=params.pop('mri_subdir'),
        derivatives=(
            Path(derivatives).expanduser()
            if derivatives
            else PROJECT_ROOT / 'derivatives'
        ),
    )

    for key in ('epoch_window', 'tf_baseline', 'broadband', 'decoding_band'):
        params[key] = tuple(params[key])
    params['crop'] = {run: tuple(window) for run, window in params['crop'].items()}

    return Config(**params, paths=paths)
