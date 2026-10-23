"""Decode the gesture from single-trial time-frequency maps of the ROI sensor with
UMAP + SVM.

Input: derivatives/tf/tf-<run>-<axis>.npz (features/extract_tf.py)
"""

# %%
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from opm_rps.config import load_config
from opm_rps.decoding import (
    cross_validate,
    plot_confusion_matrices,
    plot_fold_embeddings,
    plot_sweep,
    run_umap_svm_sweep,
    sweep,
    umap_svm_pipeline,
)
from opm_rps.viz import set_style

set_style()
cfg = load_config()

SENSOR_AXES = ['X']  # axes whose TF maps are concatenated, e.g. ['X', 'Y', 'Z']
RUN_WANDB_SWEEP = False  # random search of UMAP + SVM hyperparameters

# %% Load the TF maps of all runs (trials x frequencies x times -> trials x features)
X, y = [], []
for run in cfg.runs:
    maps = [np.load(cfg.paths.tf(run, axis)) for axis in SENSOR_AXES]
    X.append(
        np.concatenate([m['power_db'].reshape(len(m['labels']), -1) for m in maps], 1)
    )
    y.append(maps[0]['labels'])
X, y = np.concatenate(X), np.concatenate(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Optional: Weights & Biases random search
if RUN_WANDB_SWEEP:
    run_umap_svm_sweep(
        X_train,
        y_train,
        name='UMAP+SVM on TF_dB Sweep 1',
        max_components=180,
        max_neighbors=180,
    )


# %% UMAP + SVM: n_neighbors sweep
def umap_svm(n_neighbors: int):
    return umap_svm_pipeline({'n_neighbors': n_neighbors}, {'gamma': 'auto'})


neighbors = [20, 50, 100, 150, 200, 250, 300]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_means, test_means = sweep(umap_svm, neighbors, X_train, y_train, cv=cv)
plot_sweep(neighbors, train_means, test_means, 'UMAP n_neighbors')

# %% Best n_neighbors: embeddings and confusion matrices
best_neighbors = neighbors[np.argmax(test_means)]
results = cross_validate(umap_svm(best_neighbors), X_train, y_train, cv=cv)
plot_fold_embeddings(results, X_train, y_train, n_steps=2, name='UMAP')
final = umap_svm(best_neighbors).fit(X_train, y_train)
print(f'Test accuracy: {final.score(X_test, y_test):.3f}')
plot_confusion_matrices(final, X_train, y_train, X_test, y_test)
