"""Plot the sensor layout of each axis, with and without the bad channels marked.

Output: derivatives/figures/sensors/
"""

# %%
import matplotlib.pyplot as plt
from mne import pick_channels, pick_info
from mne.viz import plot_sensors

from opm_rps.config import AXES, load_config
from opm_rps.io import read_run
from opm_rps.preprocessing import axis_channels, mark_bad_channels
from opm_rps.viz import set_style

set_style()
cfg = load_config()

RUN = 'run_1'

raw, _ = read_run(cfg, RUN)
mark_bad_channels(raw, cfg.bad_channels)
# Only the channel info is needed, so the data is not copied
axis_infos = {
    axis: pick_info(
        raw.info, pick_channels(raw.ch_names, axis_channels(raw.ch_names, axis))
    )
    for axis in AXES
}
del raw

# %%
for show_bads, fname in ((True, 'sensors_bad.png'), (False, 'sensors_all.png')):
    fig, axs = plt.subplots(1, 3)
    for ax, axis in zip(axs, AXES, strict=True):
        info = axis_infos[axis].copy()
        title = f'{axis}-Axis\n'
        if show_bads:
            dropped = 100 * len(info['bads']) / len(info['ch_names'])
            title += f'({dropped:.1f}% Dropped)'
        else:
            info['bads'] = []
        plot_sensors(info, axes=ax, show=False)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(cfg.paths.figure('sensors', fname))
    plt.show()
