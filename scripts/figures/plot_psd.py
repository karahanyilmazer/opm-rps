"""Plot the PSDs of each run and sensor axis at three stages: all channels, after
dropping the bad channels and after homogeneous field correction (HFC).

Output: derivatives/figures/psd/
"""

# %%
import matplotlib.pyplot as plt

from opm_rps.config import AXES, load_config
from opm_rps.io import read_run
from opm_rps.preprocessing import apply_hfc, axis_channels, crop
from opm_rps.viz import set_style

set_style()
cfg = load_config()


def plot_axis_psds(raw, run: str, stage: str, suffix: str) -> None:
    """One PSD figure per sensor axis."""
    spectrum = raw.compute_psd()  # good data channels of all axes, without copying raw
    for axis in AXES:
        fig, ax = plt.subplots(figsize=(12, 6))
        spectrum.copy().pick(axis_channels(spectrum.ch_names, axis)).plot(
            axes=ax, show=False
        )
        ax.set_title(f'Run {run[-1]}, {axis}-Axis, {stage}')
        fig.savefig(
            cfg.paths.figure('psd', f'raw-{run[-1]}_{axis.lower()}-{suffix}.png')
        )


# %%
for run in cfg.runs:
    raw, events = read_run(cfg, run)
    if run in cfg.crop:
        crop(raw, events, *cfg.crop[run])
    plot_axis_psds(raw, run, 'All Channels', 'all_channels')

    raw.drop_channels([ch for ch in cfg.bad_channels if ch in raw.ch_names])
    plot_axis_psds(raw, run, 'Bad Channels Dropped', 'bads_dropped')

    apply_hfc(raw, order=cfg.hfc_order)
    plot_axis_psds(raw, run, 'HFC Applied', 'hfc_applied')
    plt.close('all')
