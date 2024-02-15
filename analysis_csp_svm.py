#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of OPM data using CSP + SVM.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')

import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.utils.DataLoader import DataLoader

# %%
data_loader = DataLoader(os.getcwd())
X_csp_mne_x = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_mne_x'))
X_csp_mne_y = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_mne_y'))
X_csp_mne_z = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_mne_z'))
X_csp_pyrie_x = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_pyrie_x'))
X_csp_pyrie_y = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_pyrie_y'))
X_csp_pyrie_z = data_loader.load_pkl(os.path.join('feature_matrices', 'X_csp_pyrie_z'))

y = data_loader.load_pkl(os.path.join('feature_matrices', 'y'))

# %%
X_csp_mne = np.concatenate([X_csp_mne_x, X_csp_mne_y, X_csp_mne_z], axis=1)
X_csp_pyrie = np.concatenate([X_csp_pyrie_x, X_csp_pyrie_y, X_csp_pyrie_z], axis=1)

# Choose the feature matrix
# X = X_csp_mne_x
# X = X_csp_mne_y
# X = X_csp_mne_z
X = X_csp_mne
# X = X_csp_pyrie

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
cv_train_scores = []
cv_test_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ss = StandardScaler()
svm = SVC(gamma='auto')

pipe = Pipeline([('Standard Scaler', ss), ('SVM', svm)])

for train_index, test_index in skf.split(X, y):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    pipe.fit(X_train, y_train)
    cv_train_scores.append(np.round(pipe.score(X_train, y_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_test, y_test), 3))

print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))


# %%
# Make predictions
y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

labels = {1: 'Rock', 2: 'Paper', 4: 'Scissors'}

# Calculate confusion matrix
cm_train = confusion_matrix(y_train, y_pred_train, labels=list(labels.keys()))
cm_test = confusion_matrix(y_test, y_pred_test, labels=list(labels.keys()))


# Plot confusion matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
heatmap(
    cm_train,
    annot=True,
    cmap="Blues",
    fmt='g',
    cbar=False,
    xticklabels=labels.values(),
    yticklabels=labels.values(),
    ax=axs[0],
)
heatmap(
    cm_test,
    annot=True,
    cmap="Blues",
    fmt='g',
    cbar=False,
    xticklabels=labels.values(),
    yticklabels=labels.values(),
    ax=axs[1],
)

axs[0].set_title('Train')
axs[1].set_title('CV')
axs[0].set_ylabel('True Labels', fontsize=13)
fig.suptitle('Predicted Labels', y=0.06, fontsize=13)

plt.show()
# %%
