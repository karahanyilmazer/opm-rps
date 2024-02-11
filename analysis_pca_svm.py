#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of OPM data using PCA + SVM.

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
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.utils.DataLoader import DataLoader

# %%
data_loader = DataLoader(os.getcwd())
X_bp_x = data_loader.load_pkl(os.path.join('feature_matrices', 'X_bp_x'))
X_bp_y = data_loader.load_pkl(os.path.join('feature_matrices', 'X_bp_y'))
X_bp_z = data_loader.load_pkl(os.path.join('feature_matrices', 'X_bp_z'))
X_var_x = data_loader.load_pkl(os.path.join('feature_matrices', 'X_var_x'))
X_var_y = data_loader.load_pkl(os.path.join('feature_matrices', 'X_var_y'))
X_var_z = data_loader.load_pkl(os.path.join('feature_matrices', 'X_var_z'))
X_welch_x = data_loader.load_pkl(os.path.join('feature_matrices', 'X_welch_x'))
X_welch_y = data_loader.load_pkl(os.path.join('feature_matrices', 'X_welch_y'))
X_welch_z = data_loader.load_pkl(os.path.join('feature_matrices', 'X_welch_z'))
y = data_loader.load_pkl(os.path.join('feature_matrices', 'y'))

# %%
X_bp = np.concatenate([X_bp_x, X_bp_y, X_bp_z], axis=1)
X_var = np.concatenate([X_var_x, X_var_y, X_var_z], axis=1)
X_welch = np.concatenate([X_welch_x, X_welch_y, X_welch_z], axis=1)

# Choose the feature matrix
# X = np.concatenate([X_bp, X_var, X_welch], axis=1)
X = X_bp
# X = X_var
# X = X_welch

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_norm = zscore(X_train, axis=1)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_norm)

# Plot the variance explained by each principal component
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].plot(pca.explained_variance_ratio_[:40])
axs[0].set_xlabel('Principal Component')
axs[0].set_ylabel('Variance Explained')
axs[0].set_xticks(np.arange(40))
axs[0].set_xticklabels(np.arange(1, 41))
axs[0].grid()

axs[1].plot(np.cumsum(pca.explained_variance_ratio_[:40]))
axs[1].set_xlabel('Principal Component')
axs[1].set_ylabel('Cumulative Variance Explained')
axs[1].set_xticks(np.arange(40))
axs[1].set_xticklabels(np.arange(1, 41))
axs[1].grid()

plt.show()

# %%
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
plt.show()

# %%
cv_train_scores = []
cv_test_scores = []
cv_train_means = []
cv_test_means = []

skf = StratifiedKFold(n_splits=5, shuffle=True)

ss = StandardScaler()
pca = PCA(n_components=2)
svm = SVC(gamma='auto')

pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVM', svm)])

for train_index, test_index in skf.split(X_train, y_train):
    X_cv_train = X_train[train_index]
    X_cv_test = X_train[test_index]
    y_cv_train = y_train[train_index]
    y_cv_test = y_train[test_index]

    pipe.fit(X_cv_train, y_cv_train)

    cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
    cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

    X_cv_train_pca = pipe.named_steps['Standard Scaler'].transform(X_cv_train)
    X_cv_train_pca = pipe.named_steps['PCA'].transform(X_cv_train_pca)
    X_cv_test_pca = pipe.named_steps['Standard Scaler'].transform(X_cv_test)
    X_cv_test_pca = pipe.named_steps['PCA'].transform(X_cv_test_pca)

    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(X_cv_train_pca[:, 0], X_cv_train_pca[:, 1], c=y_cv_train)
    axs[0].set_title('Training Set')
    axs[0].set_xlabel('PCA Component 1')
    axs[0].set_ylabel('PCA Component 2')

    axs[1].scatter(X_cv_test_pca[:, 0], X_cv_test_pca[:, 1], c=y_cv_test)
    axs[1].set_title('CV Set')
    axs[1].set_xlabel('PCA Component 1')
    axs[1].set_ylabel('PCA Component 2')
    plt.show()

cv_train_means.append(np.mean(cv_train_scores))
cv_test_means.append(np.mean(cv_test_scores))

cv_train_scores = []
cv_test_scores = []

# %%

print('TRAIN')
print(cv_train_scores)
print(np.mean(cv_train_scores))
print('TEST')
print(cv_test_scores)
print(np.mean(cv_test_scores))


# %%
