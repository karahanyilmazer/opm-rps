#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze epochs with effect-matched-spatial (EMS) filtering.

@author: Karahan Yilmazer
"""
# %% Import packages
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
from matplotlib.gridspec import GridSpec
from mne import EvokedArray
from mne.decoding import CSP, EMS, UnsupervisedSpatialFilter
from scipy.signal import filtfilt, firls, hilbert
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.base.EEG import EEG
from umap import UMAP
from yaml import safe_load

# %%
# Read in cleaned epochs
epochs = mne.read_epochs('all_runs-filt_7_30-epo.fif.gz')
epochs.load_data()
epochs.drop_channels([ch for ch in epochs.ch_names if 'Trigger' in ch])

# Get a list of channels for each axis
x_chs = [ch for ch in epochs.ch_names if '[X]' in ch]
y_chs = [ch for ch in epochs.ch_names if '[Y]' in ch]
z_chs = [ch for ch in epochs.ch_names if '[Z]' in ch and 'Trigger' not in ch]

# %%
with open('analysis_parameters.yaml', 'r') as file:
    config = safe_load(file)

events = ['roc', 'sci']

X = (
    epochs.copy()[events]
    # .pick(z_chs)
    .get_data(copy=True)
)
y = epochs.copy()[events].events[:, -1]
n_epochs, n_channels, n_times = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)

# Initialize EMS transformer
ems = EMS()

# Initialize the variables of interest
X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
filters = list()  # Spatial filters at each time point

for train, test in StratifiedKFold(n_splits=5).split(X, y):
    # In the original paper, the z-scoring is applied outside the CV.
    # However, we recommend to apply this preprocessing inside the CV.
    # Note that such scaling should be done separately for each channels if the
    # data contains multiple channel types.
    X_scaled = X / np.std(X[train])
    # X_scaled = np.std(X, axis=(1, 2), keepdims=True)

    # Fit and store the spatial filters
    ems.fit(X_scaled[train], y[train])

    # Store filters for future plotting
    filters.append(ems.filters_)

    # Generate the transformed data
    X_transform[test] = ems.transform(X_scaled[test])

# Average the spatial filters across folds
filters = np.mean(filters, axis=0)

# %%
# Plot individual trials
plt.figure()
plt.title("single trial surrogates")
plt.imshow(
    X_transform[y.argsort()],
    origin="lower",
    aspect="auto",
    extent=[epochs.times[0], epochs.times[-1], 1, len(X_transform)],
    cmap="RdBu_r",
)
plt.xlabel("Time (ms)")
plt.ylabel("Trials (reordered by condition)")
plt.show()

# %%
# Plot average response
plt.figure()
plt.title("Average EMS signal")
mappings = [(key, value) for key, value in epochs.event_id.items()]
for key, value in mappings:
    ems_ave = X_transform[y == value]
    plt.plot(epochs.times, ems_ave.mean(0), label=key)
plt.xlabel("Time (ms)")
plt.ylabel("a.u.")
plt.legend(loc="best")
plt.show()

# %%
# Visualize spatial filters across time
evoked = EvokedArray(filters, epochs.copy().pick(z_chs).info, tmin=epochs.tmin)
evoked.plot_topomap(scalings=1)

# %%
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=50)
skf = KFold(n_splits=3, shuffle=True, random_state=50)

ss = StandardScaler()
pca = UnsupervisedSpatialFilter(PCA(0.99), average=False)
umap = UnsupervisedSpatialFilter(UMAP(), average=False)
svm = SVC(gamma='auto')
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# pipe = Pipeline([('PCA', pca), ('CSP', csp), ('SVM', svm)])
# pipe = Pipeline([('UMAP', umap), ('CSP', csp), ('SVM', lda)])
# pipe = Pipeline([('PCA', pca), ('LDA', lda)])
# pipe = Pipeline([('CSP', csp), ('LDA', lda)])
pipe = Pipeline([('Scaler', ss), ('SVM', svm)])
# pipe = Pipeline([('SVM', svm)])
# pipe = Pipeline([('LDA', lda)])
# pipe = Pipeline([('Scaler', ss), ('LDA', lda)])

cv_train_scores = []
cv_test_scores = []

for train_index, test_index in skf.split(X_transform, y):
    X_cv_train = X_transform[train_index]
    y_cv_train = y[train_index]
    X_cv_test = X_transform[test_index]
    y_cv_test = y[test_index]

    pipe.fit(X_cv_train, y_cv_train)
    cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))


print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))

# %%
