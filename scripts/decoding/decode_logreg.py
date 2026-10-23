"""Linear decoding of two conditions with logistic regression on all channels and time
points, and topomaps of the resulting spatial patterns and filters.

Input: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz (preprocessing/clean_epochs.py)
"""

# %%
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from opm_rps.config import load_config
from opm_rps.decoding import haufe_patterns
from opm_rps.viz import plot_topomaps_per_axis, set_style

set_style()
cfg = load_config()

CONDITIONS = ['roc', 'sci']
TIME_WINDOW = (0.5, 2.0)  # s after the cue

# %% Fit a logistic regression on the z-scored, vectorised epochs
epochs = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band))[CONDITIONS]
epochs.crop(*TIME_WINDOW)
X = epochs.get_data(copy=True).reshape(len(epochs), -1)
y = epochs.events[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(solver='liblinear').fit(X_scaled, y)
filters = clf.coef_[0]
patterns = haufe_patterns(X_scaled, filters)

# %% Spatial patterns and filters
for name, coef in (('patterns', patterns), ('filters', filters)):
    # Undo the z-scoring to make the coefficients interpretable, and reshape the
    # vectorised coefficients back to channels x time points
    coef = scaler.inverse_transform([coef])[0].reshape(len(epochs.ch_names), -1)
    plot_topomaps_per_axis(coef, epochs.info, epochs.tmin, f'MEG {name}')
