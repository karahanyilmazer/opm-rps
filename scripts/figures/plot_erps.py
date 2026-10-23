"""Plot the event-related fields of the ROI sensor along each axis (all runs).

Output: derivatives/figures/erp/
"""

# %%
import matplotlib.pyplot as plt
from mne.viz import plot_sensors
from mpl_toolkits.axes_grid1 import inset_locator

from opm_rps.config import AXES, load_config
from opm_rps.io import CONDITION_NAMES
from opm_rps.preprocessing import load_epochs
from opm_rps.viz import set_style

set_style()
cfg = load_config()

epochs = load_epochs(cfg, cfg.runs)


# %%
def plot_erps(highlight: tuple[float, float] | None = None) -> plt.Figure:
    """ERPs of the three axes side by side, optionally shading a time window."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    for ax, axis in zip(axs, AXES, strict=True):
        channel = cfg.roi_channel(axis)
        for condition in cfg.events:
            erp = epochs[condition].get_data(picks=[channel])[:, 0].mean(axis=0)
            ax.plot(epochs.times, erp * 1e15, label=CONDITION_NAMES[condition])
        if highlight is not None:
            ax.axvspan(*highlight, alpha=0.5, color='gray')

        ax.set_ylim(-375, 450)
        ax.set_xlabel('Time (s)')
        if axis == AXES[0]:
            ax.set_ylabel('Amplitude (fT)')
        else:
            ax.set_yticklabels([])
        if axis == AXES[1]:
            ax.set_title('ERPs Over Different Axes')
        ax.legend(loc='lower left', fontsize=10)

        # Position of the sensor
        inset = inset_locator.inset_axes(
            ax, width='30%', height='30%', loc='upper right'
        )
        plot_sensors(
            epochs.copy().pick(channel).info,
            axes=inset,
            show_names=True,
            pointsize=5,
            linewidth=0.1,
            show=False,
        )
    plt.tight_layout()
    return fig


fig = plot_erps()
fig.savefig(cfg.paths.figure('erp', 'erp-all_runs.png'))
plt.show()

fig = plot_erps(highlight=(0.1, 0.9))
fig.savefig(cfg.paths.figure('erp', 'erp-all_runs-interval.png'))
plt.show()
