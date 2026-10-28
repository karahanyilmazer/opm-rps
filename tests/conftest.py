import matplotlib
import numpy as np
import pytest
from synthetic import simulate_csp_data, write_dataset

from opm_rps.config import load_config

matplotlib.use('Agg')


@pytest.fixture(scope='session')
def cfg(tmp_path_factory):
    """Config of a synthetic four-run session."""
    root = tmp_path_factory.mktemp('dataset')
    config_file = write_dataset(root, n_trials=12)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('OPM_RPS_DATA_DIR', str(root / 'data'))
        mp.setenv('OPM_RPS_DERIVATIVES_DIR', str(root / 'derivatives'))
        yield load_config(config_file)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def csp_data(rng):
    """Two classes whose first and second hidden source differ in variance."""
    return simulate_csp_data(rng)
