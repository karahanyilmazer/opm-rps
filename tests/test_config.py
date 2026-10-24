import pytest

from opm_rps import config


def test_repository_config():
    cfg = config.load_config(config.CONFIG_FILE)
    assert cfg.runs == ['run_1', 'run_2', 'run_3', 'run_4']
    assert cfg.event_id == {'roc': 1, 'pap': 2, 'sci': 4}
    assert cfg.epoch_window == (-0.5, 2.1)
    assert cfg.crop == {'run_3': (0, 681)}
    assert set(cfg.bad_epochs) == set(cfg.runs)
    assert set(cfg.bad_epoch_indices) == set(config.AXES)
    assert cfg.roi_channel('X') == 'KB[X]'
    assert cfg.source_file('fwd').name == '11766-ico5-fwd.fif'


def test_missing_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.delenv('OPM_RPS_DATA_DIR', raising=False)
    monkeypatch.setenv('OPM_RPS_DERIVATIVES_DIR', str(tmp_path))
    cfg = config.load_config(config.CONFIG_FILE)
    with pytest.raises(RuntimeError, match='OPM_RPS_DATA_DIR'):
        _ = cfg.paths.raw_dir

    # Outputs do not need the data and their folders are created on demand
    figure = cfg.paths.figure('tf', 'rock.png')
    assert figure == tmp_path / 'figures' / 'tf' / 'rock.png'
    assert figure.parent.is_dir()


def test_raw_file_layout(cfg):
    fname = cfg.paths.cmeg_file(cfg.session, cfg.acquisitions['run_1'])
    assert fname.is_file()
    assert fname.name == f'{cfg.session}_{cfg.acquisitions["run_1"]}_meg.cMEG'
