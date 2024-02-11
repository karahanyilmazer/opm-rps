#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of OPM data using LDA + SVM.

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.analysis.ConfusionMatrix import ConfusionMatrix
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
# X = X_csp

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ss = StandardScaler()

X_train_norm = ss.fit_transform(X_train)

lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
X_lda = lda.fit_transform(X_train_norm, y_train)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train)
plt.title('Training Set')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.show()

# %%
cv_train_scores = []
cv_test_scores = []
cv_train_means = []
cv_test_means = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_iterations = 20

for i in np.logspace(-4, 0, n_iterations):
    ss = StandardScaler()
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=i)
    svm = SVC(gamma='auto')

    pipe = Pipeline([('Standard Scaler', ss), ('LDA', lda), ('SVM', svm)])
    for train_index, test_index in skf.split(X_train, y_train):
        X_cv_train = X_train[train_index]
        X_cv_test = X_train[test_index]
        y_cv_train = y_train[train_index]
        y_cv_test = y_train[test_index]

        pipe.fit(X_cv_train, y_cv_train)
        cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
        cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

        X_cv_train_lda = pipe.named_steps['Standard Scaler'].transform(X_cv_train)
        X_cv_train_lda = pipe.named_steps['LDA'].transform(X_cv_train_lda)
        X_cv_test_lda = pipe.named_steps['Standard Scaler'].transform(X_cv_test)
        X_cv_test_lda = pipe.named_steps['LDA'].transform(X_cv_test_lda)

        _, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(X_cv_train_lda[:, 0], X_cv_train_lda[:, 1], c=y_cv_train)
        axs[0].set_title('Training Fold')
        axs[0].set_xlabel('LDA Component 1')
        axs[0].set_ylabel('LDA Component 2')

        axs[1].scatter(X_cv_test_lda[:, 0], X_cv_test_lda[:, 1], c=y_cv_test)
        axs[1].set_title('Test Fold')
        axs[1].set_xlabel('LDA Component 1')
        axs[1].set_ylabel('LDA Component 2')
        plt.show()

    cv_train_means.append(np.mean(cv_train_scores))
    cv_test_means.append(np.mean(cv_test_scores))

    cv_train_scores = []
    cv_test_scores = []

# %%
plt.plot(np.arange(1, n_iterations + 1), cv_train_means, label='Training ACC')
plt.plot(np.arange(1, n_iterations + 1), cv_test_means, label='CV ACC')
plt.xticks(np.arange(1, n_iterations + 1))
plt.xlabel('Iteration #')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# %%
# Make predictions
y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# Plot the confusion matrix
labels = {1: 'Rock', 2: 'Paper', 4: 'Scissors'}
cm = ConfusionMatrix(y_train, y_test, y_pred_train, y_pred_test, labels)
cm.plot()

# %%
