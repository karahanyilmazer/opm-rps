#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract CSP features from all runs.

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
from mne.decoding import CSP, UnsupervisedSpatialFilter
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

X = (
    epochs.copy()
    .pick(z_chs)
    .crop(tmin=config['csp_tmin'], tmax=config['csp_tmax'])
    .get_data(copy=True)
)
y = epochs.events[:, -1]

# X = epochs[['roc', 'sci']].crop(tmin=0.5, tmax=1.5).get_data(copy=True)
# y = epochs[['roc', 'sci']].events[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)

# %%‚
cv_train_scores = []
cv_test_scores = []

# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=50)
skf = KFold(n_splits=3, shuffle=True, random_state=50)

ss = StandardScaler()
pca = UnsupervisedSpatialFilter(PCA(0.99), average=False)
umap = UnsupervisedSpatialFilter(UMAP(), average=False)
svm = SVC(gamma='auto')
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# pipe = Pipeline([('PCA', pca), ('CSP', csp), ('SVM', svm)])
pipe = Pipeline([('UMAP', umap), ('CSP', csp), ('SVM', lda)])
# pipe = Pipeline([('PCA', pca), ('LDA', lda)])
# pipe = Pipeline([('CSP', csp), ('LDA', lda)])

for train_index, test_index in skf.split(X_train, y_train):
    X_cv_train = X_train[train_index]
    y_cv_train = y_train[train_index]
    X_cv_test = X_train[test_index]
    y_cv_test = y_train[test_index]

    pipe.fit(X_cv_train, y_cv_train)
    cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

    dim_red = pipe.steps[0]
    dim_red_name = pipe.steps[0][0]
    dim_red_train = np.mean(dim_red.transform(X_cv_train), axis=2)
    dim_red_test = np.mean(dim_red.transform(X_cv_test), axis=2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    for label in np.unique(y_cv_train):
        idx = np.where(y_cv_train == label)
        axs[0].scatter(dim_red_train[idx, 0], dim_red_train[idx, 1], label=label)
        axs[0].set_title('Train Set')
        axs[0].set_xlabel(f'{dim_red_name} Component 1')
        axs[0].set_ylabel(f'{dim_red_name} Component 2')
        axs[0].legend()

        idx = np.where(y_cv_test == label)
        axs[1].scatter(dim_red_test[idx, 0], dim_red_test[idx, 1], label=label)
        axs[1].set_title('Test Set')
        axs[1].set_xlabel(f'{dim_red_name} Component 1')
        axs[1].set_ylabel(f'{dim_red_name} Component 2')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))


# %%
# Get the test set accuracy
pipe.score(X_test, y_test)

# %%
