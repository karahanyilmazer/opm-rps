#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze epochs with linear logistic regression.

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
from mne import EvokedArray, io
from mne.datasets import sample

# import a linear classifier from mne.decoding
from mne.decoding import (
    CSP,
    EMS,
    LinearModel,
    UnsupervisedSpatialFilter,
    Vectorizer,
    get_coef,
)
from scipy.signal import filtfilt, firls, hilbert
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
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

tmp_epochs = (
    epochs.copy()[events]
    # .pick(z_chs)
    .crop(tmin=0.5, tmax=2)
)

X = tmp_epochs.get_data(copy=True)
X = X.reshape(X.shape[0], -1)
y = tmp_epochs.events[:, -1]

clf = LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
scaler = StandardScaler()

# create a linear model with LogisticRegression
model = LinearModel(clf)

# fit the classifier on MEG data
model.fit(X, y)

# %%
# Extract and plot spatial filters and spatial patterns
for name, coef in (("patterns", model.patterns_), ("filters", model.filters_)):
    # We fitted the linear model onto Z-scored data. To make the filters
    # interpretable, we must reverse this normalization step
    coef = scaler.inverse_transform([coef])[0]

    # The data was vectorized to fit a single model across all time points and
    # all channels. We thus reshape it:
    coef = coef.reshape(len(tmp_epochs.ch_names), -1)

    # Plot
    evoked = EvokedArray(coef, tmp_epochs.info, tmin=epochs.tmin)
    fig = evoked.plot_topomap()
    fig.suptitle(f"MEG {name}")

# %%
