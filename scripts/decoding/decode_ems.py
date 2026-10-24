"""Effect-matched spatial (EMS) filtering of two conditions and decoding of the
resulting single-trial surrogate time courses.

The surrogates shown in the figures come from a cross-validation of the EMS filters.
The decoding refits the filters within every fold of its own cross-validation, so
that the labels of its test trials never shape the surrogates it trains on.

Input: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz (preprocessing/clean_epochs.py)
"""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import EMS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from opm_rps.config import load_config
from opm_rps.decoding import cross_validate
from opm_rps.io import CONDITION_NAMES
from opm_rps.viz import plot_topomaps_per_axis, set_style

set_style()
cfg = load_config()

CONDITIONS = ['roc', 'sci']
PIPELINE = 'scaler_svm'  # key of PIPELINES below
PIPELINES = {
    'scaler_svm': Pipeline([('Scaler', StandardScaler()), ('SVM', SVC(gamma='auto'))]),
    'svm': Pipeline([('SVM', SVC(gamma='auto'))]),
    'lda': Pipeline([('LDA', LinearDiscriminantAnalysis())]),
    'scaler_lda': Pipeline(
        [('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())]
    ),
}

# %% Load the data
epochs = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band))[CONDITIONS]
X = epochs.get_data(copy=True)
y = epochs.events[:, -1]
n_epochs, n_channels, n_times = X.shape

# %% Cross-validated EMS surrogates and spatial filters
ems = EMS()
surrogates = np.zeros((n_epochs, n_times))
filters = []
for train, test in StratifiedKFold(n_splits=5).split(X, y):
    # Scale with the training data only (the original paper z-scores outside the CV)
    X_scaled = X / np.std(X[train])
    ems.fit(X_scaled[train], y[train])
    filters.append(ems.filters_)
    surrogates[test] = ems.transform(X_scaled[test])
filters = np.mean(filters, axis=0)

# %% Single-trial surrogates, sorted by condition
plt.figure()
plt.imshow(
    surrogates[y.argsort()],
    origin='lower',
    aspect='auto',
    extent=[epochs.times[0], epochs.times[-1], 1, len(surrogates)],
    cmap='RdBu_r',
)
plt.title('Single-Trial Surrogates')
plt.xlabel('Time (s)')
plt.ylabel('Trials (sorted by condition)')
plt.show()

# %% Average surrogate per condition
plt.figure()
for condition, code in epochs.event_id.items():
    plt.plot(
        epochs.times, surrogates[y == code].mean(0), label=CONDITION_NAMES[condition]
    )
plt.title('Average EMS Signal')
plt.xlabel('Time (s)')
plt.ylabel('a.u.')
plt.legend(loc='best')
plt.show()

# %% Spatial filters over time
plot_topomaps_per_axis(filters, epochs.info, epochs.tmin, 'EMS filters', scalings=1)

# %% Decode the conditions from the surrogates of EMS filters fitted in each fold
# Decoding the surrogates above would leak the labels of the test trials, which
# helped to fit the filters of the training trials (well above chance even for
# shuffled labels)
results = cross_validate(
    Pipeline([('EMS', EMS()), *PIPELINES[PIPELINE].steps]),
    X / np.std(X),
    y,
    cv=KFold(n_splits=3, shuffle=True, random_state=50),
)
