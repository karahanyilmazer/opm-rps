"""Decode the gesture from the hand-crafted feature matrices.

Classical pipelines on one feature set: PCA + SVM, LDA + SVM (shrinkage sweep) and
UMAP + SVM (n_neighbors sweep, optionally a Weights & Biases random search).

Input: derivatives/features/features.npz (features/extract_features.py)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from opm_rps.config import load_config
from opm_rps.decoding import (
    cross_validate,
    plot_confusion_matrices,
    plot_fold_embeddings,
    plot_sweep,
    run_umap_svm_sweep,
    scatter_classes,
    sweep,
    umap_svm_pipeline,
)
from opm_rps.viz import set_style

set_style()
cfg = load_config()

FEATURES = 'bp'  # 'bp' (band power), 'var' (variance) or 'welch' (Welch power)
SENSOR_AXES = 'xyz'  # axes whose features are concatenated
RUN_WANDB_SWEEP = False  # random search of UMAP + SVM hyperparameters

# %% Load the features
data = np.load(cfg.paths.features)
X = np.concatenate([data[f'{FEATURES}_{axis}'] for axis in SENSOR_AXES], axis=1)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% PCA of the (trial-wise z-scored) training set
pca = PCA().fit(zscore(X_train, axis=1))
embedding = pca.transform(zscore(X_train, axis=1))
n_shown = min(40, pca.n_components_)
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
components = np.arange(1, n_shown + 1)
axs[0].plot(components, pca.explained_variance_ratio_[:n_shown])
axs[0].set(xlabel='Principal Component', ylabel='Variance Explained')
axs[1].plot(components, np.cumsum(pca.explained_variance_ratio_[:n_shown]))
axs[1].set(xlabel='Principal Component', ylabel='Cumulative Variance Explained')
scatter_classes(axs[2], embedding, y_train)
axs[2].set(title='Training Set', xlabel='PCA Component 1', ylabel='PCA Component 2')
plt.show()

# %% PCA + SVM
pipe = Pipeline(
    [
        ('Scaler', StandardScaler()),
        ('PCA', PCA(n_components=2)),
        ('SVM', SVC(gamma='auto')),
    ]
)
results = cross_validate(pipe, X_train, y_train, cv=cv)
plot_fold_embeddings(results, X_train, y_train, n_steps=2, name='PCA')

# %% LDA embedding of the training set (automatic shrinkage)
scaled = StandardScaler().fit_transform(X_train)
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
fig, ax = plt.subplots()
scatter_classes(ax, lda.fit_transform(scaled, y_train), y_train)
ax.set(title='Training Set', xlabel='LDA Component 1', ylabel='LDA Component 2')
plt.show()


# %% LDA + SVM: shrinkage sweep
def lda_svm(shrinkage: float) -> Pipeline:
    return Pipeline(
        [
            ('Scaler', StandardScaler()),
            ('LDA', LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)),
            ('SVM', SVC(gamma='auto')),
        ]
    )


shrinkages = np.logspace(-4, 0, 20)
train_means, test_means = sweep(lda_svm, shrinkages, X_train, y_train, cv=cv)
plot_sweep(shrinkages, train_means, test_means, 'LDA shrinkage', logx=True)

best_shrinkage = shrinkages[np.argmax(test_means)]
results = cross_validate(lda_svm(best_shrinkage), X_train, y_train, cv=cv)
plot_fold_embeddings(results, X_train, y_train, n_steps=2, name='LDA')
final = lda_svm(best_shrinkage).fit(X_train, y_train)
plot_confusion_matrices(final, X_train, y_train, X_test, y_test)

# %% UMAP + SVM: n_neighbors sweep
if RUN_WANDB_SWEEP:
    run_umap_svm_sweep(
        X_train, y_train, name='UMAP+SVM Sweep 1', max_components=500, max_neighbors=400
    )


def umap_svm(n_neighbors: int) -> Pipeline:
    return umap_svm_pipeline({'n_neighbors': n_neighbors}, {'gamma': 'auto'})


neighbors = [20, 50, 100, 150, 200, 250, 300]
cv_umap = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_means, test_means = sweep(umap_svm, neighbors, X_train, y_train, cv=cv_umap)
plot_sweep(neighbors, train_means, test_means, 'UMAP n_neighbors')

best_neighbors = neighbors[np.argmax(test_means)]
results = cross_validate(umap_svm(best_neighbors), X_train, y_train, cv=cv_umap)
plot_fold_embeddings(results, X_train, y_train, n_steps=2, name='UMAP')
final = umap_svm(best_neighbors).fit(X_train, y_train)
plot_confusion_matrices(final, X_train, y_train, X_test, y_test)
