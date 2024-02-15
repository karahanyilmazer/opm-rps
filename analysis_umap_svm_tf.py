#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of TF matrices using UMAP + SVM.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%load_ext autoreload
# !%autoreload 2
import os
import sys
from pprint import pprint

from utils import get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
os.environ['WANDB_NOTEBOOK_NAME'] = 'analysis_umap_svm.py'

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.analysis.ConfusionMatrix import ConfusionMatrix
from src.utils.DataLoader import DataLoader
from umap import UMAP

# %%
# Load the TF matrices
data_loader = DataLoader(os.getcwd())
X_tf_db_1x = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_1-X'))
X_tf_db_2x = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_2-X'))
X_tf_db_3x = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_3-X'))
X_tf_db_4x = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_4-X'))
X_tf_db_1y = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_1-Y'))
X_tf_db_2y = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_2-Y'))
X_tf_db_3y = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_3-Y'))
X_tf_db_4y = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_4-Y'))
X_tf_db_1z = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_1-Z'))
X_tf_db_2z = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_2-Z'))
X_tf_db_3z = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_3-Z'))
X_tf_db_4z = data_loader.load_pkl(os.path.join('tf_matrices', 'tf_dB-run_run_4-Z'))

# %%
roc_x = np.stack(
    (
        X_tf_db_1x[0, 0, 0, :, :],
        X_tf_db_2x[0, 0, 0, :, :],
        X_tf_db_3x[0, 0, 0, :, :],
        X_tf_db_4x[0, 0, 0, :, :],
    )
).reshape(4, -1)
pap_x = np.stack(
    (
        X_tf_db_1x[0, 0, 1, :, :],
        X_tf_db_2x[0, 0, 1, :, :],
        X_tf_db_3x[0, 0, 1, :, :],
        X_tf_db_4x[0, 0, 1, :, :],
    )
).reshape(4, -1)
sci_x = np.stack(
    (
        X_tf_db_1x[0, 0, 2, :, :],
        X_tf_db_2x[0, 0, 2, :, :],
        X_tf_db_3x[0, 0, 2, :, :],
        X_tf_db_4x[0, 0, 2, :, :],
    )
).reshape(4, -1)
X_x = np.concatenate([roc_x, pap_x, sci_x], axis=0)

roc_y = np.stack(
    (
        X_tf_db_1y[0, 0, 0, :, :],
        X_tf_db_2y[0, 0, 0, :, :],
        X_tf_db_3y[0, 0, 0, :, :],
        X_tf_db_4y[0, 0, 0, :, :],
    )
).reshape(4, -1)
pap_y = np.stack(
    (
        X_tf_db_1y[0, 0, 1, :, :],
        X_tf_db_2y[0, 0, 1, :, :],
        X_tf_db_3y[0, 0, 1, :, :],
        X_tf_db_4y[0, 0, 1, :, :],
    )
).reshape(4, -1)
sci_y = np.stack(
    (
        X_tf_db_1y[0, 0, 2, :, :],
        X_tf_db_2y[0, 0, 2, :, :],
        X_tf_db_3y[0, 0, 2, :, :],
        X_tf_db_4y[0, 0, 2, :, :],
    )
).reshape(4, -1)
X_y = np.concatenate([roc_y, pap_y, sci_y], axis=0)

roc_z = np.stack(
    (
        X_tf_db_1z[0, 0, 0, :, :],
        X_tf_db_2z[0, 0, 0, :, :],
        X_tf_db_3z[0, 0, 0, :, :],
        X_tf_db_4z[0, 0, 0, :, :],
    )
).reshape(4, -1)
pap_z = np.stack(
    (
        X_tf_db_1z[0, 0, 1, :, :],
        X_tf_db_2z[0, 0, 1, :, :],
        X_tf_db_3z[0, 0, 1, :, :],
        X_tf_db_4z[0, 0, 1, :, :],
    )
).reshape(4, -1)
sci_z = np.stack(
    (
        X_tf_db_1z[0, 0, 2, :, :],
        X_tf_db_2z[0, 0, 2, :, :],
        X_tf_db_3z[0, 0, 2, :, :],
        X_tf_db_4z[0, 0, 2, :, :],
    )
).reshape(4, -1)
X_z = np.concatenate([roc_z, pap_z, sci_z], axis=0)

y = np.array([0] * 4 + [1] * 4 + [2] * 4)

# %%
# Choose the feature matrix
# X = np.concatenate([X_x, X_y, X_z], axis=1)
X = X_x
# X = X_y
# X = X_z

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
sweep_config = {
    'method': 'random',
    'name': 'UMAP+SVM Sweep 1',
    'metric': {'goal': 'maximize', 'name': 'cv_test_acc_mean'},
    'parameters': {
        'n_components': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 500,
        },
        'n_neighbors': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 400,
        },
        'min_dist': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.99,
        },
        'kernel': {'values': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
        'C': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,  # log10(0.001) = -3
            'max': 1000,  # log10(1000) = 3
        },
        'n_classes': {'value': 3},
        'dataset': {'value': 'OPM Nottingham'},
    },
}

pprint(sweep_config)
sweep_id = wandb.sweep(sweep=sweep_config, project='OPM Nottingham Classification')


# %%
def sweep_train(config=None):
    with wandb.init(config):
        config = wandb.config
        ss = StandardScaler()
        umap = UMAP(
            n_components=config['n_components'],
            n_neighbors=config['n_neighbors'],
            min_dist=config['min_dist'],
        )
        svm = SVC(kernel=config['kernel'], C=config['C'])

        pipe = Pipeline([('Scaler', ss), ('UMAP', umap), ('SVM', svm)])

        # Reset the scores
        cv_train_scores = []
        cv_test_scores = []

        # Define the CV folds
        skf = StratifiedKFold(n_splits=3, shuffle=True)

        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            # Get the CV folds
            X_cv_train = X_train[train_index]
            y_cv_train = y_train[train_index]

            X_cv_test = X_train[test_index]
            y_cv_test = y_train[test_index]

            # Fit the pipeline
            pipe.fit(X_cv_train, y_cv_train)

            # Get the accuracy scores
            cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
            cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

            wandb.log({'cv_train_acc': cv_train_scores[-1], 'fold_number': i + 1})
            wandb.log({'cv_test_acc': cv_test_scores[-1], 'fold_number': i + 1})

        wandb.log({'cv_train_acc_mean': np.mean(cv_train_scores)})
        wandb.log({'cv_test_acc_mean': np.mean(cv_test_scores)})


# Train the classifier.
wandb.agent(sweep_id=sweep_id, function=sweep_train)


# %%
cv_train_means = []
cv_test_means = []

for i in [20, 50, 100, 150, 200, 250, 300]:
    # Define the pipeline
    ss = StandardScaler()
    umap = UMAP(n_neighbors=i)
    svm = SVC(gamma='auto')

    pipe = Pipeline([('Scaler', ss), ('UMAP', umap), ('SVM', svm)])

    pipe.fit(X_train, y_train)

    # Transform data for plotting
    X_train_umap = pipe.named_steps['Scaler'].transform(X_train)
    X_train_umap = pipe.named_steps['UMAP'].transform(X_train_umap)

    # Plot the embeddings
    plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train)
    plt.title(f'Training Set (n_neighbors={i})')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

    # Reset the scores
    cv_train_scores = []
    cv_test_scores = []

    # Define the CV folds
    skf = StratifiedKFold(n_splits=3, shuffle=True)

    for train_index, test_index in skf.split(X_train, y_train):
        # Get the CV folds
        X_cv_train = X_train[train_index]
        y_cv_train = y_train[train_index]

        X_cv_test = X_train[test_index]
        y_cv_test = y_train[test_index]

        # Fit the pipeline
        pipe.fit(X_cv_train, y_cv_train)

        # Get the accuracy scores
        cv_train_scores.append(np.round(pipe.score(X_cv_train, y_cv_train), 3))
        cv_test_scores.append(np.round(pipe.score(X_cv_test, y_cv_test), 3))

        # Transform data for plotting
        X_cv_train_umap = pipe.named_steps['Scaler'].transform(X_cv_train)
        X_cv_train_umap = pipe.named_steps['UMAP'].transform(X_cv_train)

        X_cv_test_umap = pipe.named_steps['Scaler'].transform(X_cv_test)
        X_cv_test_umap = pipe.named_steps['UMAP'].transform(X_cv_test)

        # Plot embedding
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'UMAP (n_neighbors={i})')
        axs[0].scatter(X_cv_train_umap[:, 0], X_cv_train_umap[:, 1], c=y_cv_train)
        axs[0].set_title('Training Fold')
        axs[0].set_xlabel('UMAP Component 1')
        axs[0].set_ylabel('UMAP Component 2')

        axs[1].scatter(X_cv_test_umap[:, 0], X_cv_test_umap[:, 1], c=y_cv_test)
        axs[1].set_title('Test Fold')
        axs[1].set_xlabel('UMAP Component 1')
        axs[1].set_ylabel('UMAP Component 2')
        plt.show()

    # Save accuracy scores
    cv_train_means.append(np.mean(cv_train_scores))
    cv_test_means.append(np.mean(cv_test_scores))

# %%
plt.plot(np.arange(1, 8), cv_train_means, label='Training ACC')
plt.plot(np.arange(1, 8), cv_test_means, label='CV ACC')
plt.xticks(np.arange(1, 8))
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
# Define the pipeline
ss = StandardScaler()
umap = UMAP()
svm = SVC(gamma='auto')

pipe = Pipeline([('Scaler', ss), ('UMAP', umap), ('SVM', svm)])

pipe.fit(X_train, y_train)

# Transform data for plotting
X_train_umap = pipe.named_steps['Scaler'].transform(X_train)
X_train_umap = pipe.named_steps['UMAP'].transform(X_train_umap)

# Plot the embeddings
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train)
plt.title(f'Training Set')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()

pipe.score(X_train, y_train), pipe.score(X_test, y_test)
# %%
