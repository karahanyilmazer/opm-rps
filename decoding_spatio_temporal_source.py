# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

from utils import get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
sys.path.insert(0, os.path.join(base_dir, 'data'))
os.environ['SUBJECTS_DIR'] = os.path.join(
    base_dir, 'data', 'Gesture', 'Nottingham', 'MRI', 'Segmentation'
)

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, get_coef
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
epochs = mne.read_epochs('all_runs-filt_7_30-epo.fif.gz')
epochs.drop_channels(
    [ch for ch in epochs.ch_names if 'Trigger' in ch] + epochs.info['bads']
)
epochs.decimate(7)
epochs.load_data()
time = epochs.times

# %%
# Compute inverse solution
snr = 3.0
inv = read_inverse_operator('11766-misaligned-ico5-meg-inv.fif')

stcs = apply_inverse_epochs(
    epochs,
    inv,
    lambda2=1.0 / snr**2,
    verbose=True,
    method="sLORETA",
    pick_ori="normal",
)
stc = stcs[0]
# %%
# Decoding in sensor space using a logistic regression

# Retrieve source space data into an array
X = np.array([stc.lh_data for stc in stcs])  # only keep left hemisphere
y = epochs.events[:, 2]

del epochs, stcs, snr, inv

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# %%
skf = KFold(n_splits=3, shuffle=True, random_state=26)

cv_train_scores = []
cv_test_scores = []

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    # SelectKBest(f_classif, k=),  # select features for speed
    LinearModel(LogisticRegression(C=1, solver="liblinear")),
)
time_decod = SlidingEstimator(clf, scoring="accuracy")

# %%

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(i, time.time())
    X_cv_train = X[train_index]
    X_cv_test = X[test_index]
    y_cv_train = y[train_index]
    y_cv_test = y[test_index]

    time_decod.fit(X_cv_train, y_cv_train)

    cv_train_scores.append(np.round(time_decod.score(X_cv_train, y_cv_train), 3))
    cv_test_scores.append(np.round(time_decod.score(X_cv_test, y_cv_test), 3))


# %%
# Run cross-validated decoding analyses:
scores = cross_val_multiscore(time_decod, X_train, y_train, cv=3, n_jobs=None)

# %%
# Plot average decoding scores of 5 splits
fig, ax = plt.subplots(1)
ax.axhline(0.33, color="k", linestyle="--", label="Chance")
ax.axvline(0, color="k")
ax.plot(time, scores.mean(0), label="Score")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join('img', 'decoding_spatio_temporal_source.png'))
plt.show()

# %%
time_decod.fit(X_train, y_train)
train_scores = time_decod.score(X_train, y_train)
test_scores = time_decod.score(X_test, y_test)

# %%

fig, ax = plt.subplots(1)
ax.axhline(0.33, color="k", linestyle="--", label="Chance")
ax.axvline(0, color="k")
ax.plot(time, train_scores, label='Train')
# ax.plot(time, scores.mean(0), label='Mean CV')
ax.plot(time, test_scores, label='Test')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accuracy")
ax.set_title('Source Space Decoding Accuracy Over Time')
# ax.set_ylim(0.05, None)
plt.legend(loc='lower right')
plt.savefig(os.path.join('img', 'decoding_accuracies_source.png'))
plt.show()

# %%
# To investigate weights, we need to retrieve the patterns of a fitted model

# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y)

# %%
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

# %%
# Condition 1
vertices = [stc.lh_vertno, np.array([], int)]  # empty array for right hemi
stc_feat = mne.SourceEstimate(
    np.abs(patterns[:, 0, :]),
    vertices=vertices,
    tmin=stc.tmin,
    tstep=stc.tstep,
    subject="11766",
)

brain = stc_feat.plot(
    views=["lat"],
    transparent=True,
    initial_time=0.1,
    time_unit="s",
)

# %%
# Condition 2
vertices = [stc.lh_vertno, np.array([], int)]  # empty array for right hemi
stc_feat = mne.SourceEstimate(
    np.abs(patterns[:, 1, :]),
    vertices=vertices,
    tmin=stc.tmin,
    tstep=stc.tstep,
    subject="11766",
)

brain = stc_feat.plot(
    views=["lat"],
    transparent=True,
    initial_time=0.1,
    time_unit="s",
)
# %%
# Condition 3
vertices = [stc.lh_vertno, np.array([], int)]  # empty array for right hemi
stc_feat = mne.SourceEstimate(
    np.abs(patterns[:, 2, :]),
    vertices=vertices,
    tmin=stc.tmin,
    tstep=stc.tstep,
    subject="11766",
)

brain = stc_feat.plot(
    views=["lat"],
    transparent=True,
    initial_time=0.1,
    time_unit="s",
)
# %%
