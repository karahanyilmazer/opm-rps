"""Time-resolved decoding of the gesture in source space (left hemisphere).

The cleaned epochs are projected to the cortex with sLORETA (normal orientation) and
a logistic regression is trained at every time point.

Inputs: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz (preprocessing/clean_epochs.py)
        and the inverse operator (source_space/source_reconstruction.py)
"""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, get_coef
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opm_rps.config import load_config
from opm_rps.io import CONDITION_NAMES
from opm_rps.viz import set_style

set_style()
cfg = load_config()

DECIM = 7  # temporal decimation of the epochs
SNR = 3.0
CHANCE = 1 / len(cfg.events)

# %% Source estimates of all epochs
epochs = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band))
epochs.decimate(DECIM)
inv = read_inverse_operator(cfg.source_file('inv'))
vertices = inv['src'][0]['vertno']  # left hemisphere only
# One epoch at a time: the estimates of all epochs and both hemispheres would take
# twice the memory of X
stcs = apply_inverse_epochs(
    epochs,
    inv,
    lambda2=1.0 / SNR**2,
    method='sLORETA',
    pick_ori='normal',
    return_generator=True,
)
X = np.fromiter(
    (stc.lh_data for stc in stcs),
    dtype=(float, (len(vertices), len(epochs.times))),
    count=len(epochs),
)
y = epochs.events[:, 2]
times = epochs.times
tmin, tstep = times[0], 1 / epochs.info['sfreq']
del inv

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# A one-vs-rest logistic regression per time point
classifier = OneVsRestClassifier(LogisticRegression(C=1, solver='liblinear'))
time_decod = SlidingEstimator(
    make_pipeline(StandardScaler(), LinearModel(classifier)), scoring='accuracy'
)


def plot_scores(curves: dict, title: str, fname: str) -> None:
    fig, ax = plt.subplots()
    ax.axhline(CHANCE, color='k', linestyle='--', label='Chance')
    ax.axvline(0, color='k')
    for label, scores in curves.items():
        ax.plot(times, scores, label=label)
    ax.set(title=title, xlabel='Time (s)', ylabel='Accuracy')
    ax.legend(loc='lower right')
    fig.savefig(cfg.paths.figure('decoding', fname))
    plt.show()


# %% Cross-validated accuracy over time (training set)
scores = cross_val_multiscore(time_decod, X_train, y_train, cv=3)
plot_scores(
    {'Mean CV': scores.mean(0)},
    'Source Space Decoding (CV)',
    'decoding_spatio_temporal_source.png',
)

# %% Training and test accuracy over time
time_decod.fit(X_train, y_train)
plot_scores(
    {
        'Train': time_decod.score(X_train, y_train),
        'Test': time_decod.score(X_test, y_test),
    },
    'Source Space Decoding Accuracy Over Time',
    'decoding_accuracies_source.png',
)

# %% Spatial patterns of each class (model fitted on all trials)
time_decod.fit(X, y)
patterns = get_coef(time_decod, 'patterns_', inverse_transform=True)
for i, condition in enumerate(cfg.events):
    stc_patterns = mne.SourceEstimate(
        np.abs(patterns[:, i, :]),
        vertices=[vertices, np.array([], int)],  # no right hemisphere
        tmin=tmin,
        tstep=tstep,
        subject=cfg.subject,
    )
    brain = stc_patterns.plot(
        views=['lat'],
        transparent=True,
        initial_time=0.1,
        time_unit='s',
        subjects_dir=cfg.paths.subjects_dir,
        title=CONDITION_NAMES[condition],
    )
