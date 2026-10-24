"""Cross-validation, sweeps and plots shared by the decoding scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    permutation_test_score,
)
from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .io import CONDITION_NAMES, EVENT_ID

#: Class labels (trigger codes) and their names
LABELS = {EVENT_ID[cond]: name for cond, name in CONDITION_NAMES.items()}


def cross_validate(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    verbose: bool = True,
) -> dict:
    """Cross-validate a pipeline, keeping the fitted folds for later inspection.

    Returns the output of :func:`sklearn.model_selection.cross_validate` with
    ``train_score``, ``test_score``, the fitted ``estimator`` of each fold and the
    fold ``indices``.
    """
    results = sk_cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
    )
    if verbose:
        print_scores(results['train_score'], results['test_score'])
    return results


def haufe_patterns(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Activation patterns of a linear model with a single output, A = Cov(X) w.

    Equals ``mne.decoding.LinearModel(...).patterns_`` (Haufe et al., 2014) but
    never forms the n_features x n_features covariance matrix, which does not fit
    in memory for all channels x time points.
    """
    X_centered = X - X.mean(axis=0)
    return X_centered.T @ (X_centered @ filters) / (len(X) - 1)


def print_scores(train_scores: Sequence[float], test_scores: Sequence[float]) -> None:
    """Print per-fold and mean accuracies."""
    for name, scores in (('Train', train_scores), ('CV', test_scores)):
        print(f'{name:5s} accuracy: {np.round(scores, 3)}, mean {np.mean(scores):.3f}')


def permutation_test(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    n_permutations: int = 1000,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """Cross-validated accuracy and its p-value under shuffled labels.

    The whole cross-validation is repeated with ``n_permutations`` shufflings of the
    labels (:func:`sklearn.model_selection.permutation_test_score`). The p-value is
    the fraction of shufflings, counting the true labels as one of them, that reach
    at least the accuracy of the true labels. Returns the accuracy and the p-value.
    """
    score, null_scores, pvalue = permutation_test_score(
        pipe,
        X,
        y,
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=0,
        scoring='accuracy',
    )
    print(
        f'Permutation test: CV accuracy {score:.3f}, shuffled labels '
        f'{null_scores.mean():.3f} +- {null_scores.std():.3f}, p = {pvalue:.4f} '
        f'({n_permutations} permutations)'
    )
    return score, pvalue


def chance_level(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class."""
    return np.unique(y, return_counts=True)[1].max() / len(y)


def binomial_pvalues(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """One-sided exact binomial p-values of the accuracy of predictions.

    Tests the number of correct predictions against :func:`chance_level`. Only valid
    for trials that played no part in fitting or selecting the model (a held-out
    test set). ``y_pred`` has one prediction per trial, or one per trial and time
    point (n_trials, n_times) for time-resolved decoding.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_correct = np.sum(y_pred == y_true.reshape(-1, *[1] * (y_pred.ndim - 1)), axis=0)
    return binom.sf(n_correct - 1, len(y_true), chance_level(y_true))


def evaluate_on_test_set(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    """Accuracy of a fitted model on the held-out test set and its binomial p-value."""
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    pvalue = binomial_pvalues(y_test, y_pred)
    print(
        f'Test accuracy {accuracy:.3f} (n = {len(y_test)}, chance '
        f'{chance_level(y_test):.3f}), p = {pvalue:.4f} (binomial test)'
    )
    return accuracy, pvalue


def plot_fold_embeddings(
    results: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_steps: int,
    name: str,
    labels: dict[int, str] = LABELS,
) -> None:
    """Scatter the first two embedding dimensions of each CV fold.

    The embedding is the output of the first ``n_steps`` steps of each fitted
    pipeline (e.g. scaler and UMAP). Outputs with a time axis (spatial filters) are
    averaged over time.
    """
    folds = zip(
        results['estimator'],
        results['indices']['train'],
        results['indices']['test'],
        strict=True,
    )
    for fold, (pipe, train_idx, test_idx) in enumerate(folds, start=1):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax, idx, title in (
            (axs[0], train_idx, 'Training fold'),
            (axs[1], test_idx, 'Test fold'),
        ):
            embedding = pipe[:n_steps].transform(X[idx])
            if embedding.ndim == 3:
                embedding = embedding.mean(axis=-1)
            scatter_classes(ax, embedding, y[idx], labels)
            ax.set(
                title=title,
                xlabel=f'{name} component 1',
                ylabel=f'{name} component 2',
            )
        fig.suptitle(f'{name} embedding, fold {fold}')
        plt.tight_layout()
        plt.show()


def scatter_classes(
    ax: plt.Axes, points: np.ndarray, y: np.ndarray, labels: dict[int, str] = LABELS
) -> None:
    """Scatter the first two columns of ``points`` colored by class."""
    for code, label in labels.items():
        in_class = y == code
        ax.scatter(points[in_class, 0], points[in_class, 1], s=12, label=label)
    ax.legend()


def sweep(
    make_pipe: Callable[[float], Pipeline],
    values: Sequence[float],
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean train and CV accuracy of the pipelines built for each parameter value."""
    train_means, test_means = [], []
    for value in values:
        results = cross_validate(make_pipe(value), X, y, cv=cv, verbose=False)
        train_means.append(results['train_score'].mean())
        test_means.append(results['test_score'].mean())
        print(f'{value:10.4g}: train {train_means[-1]:.3f}, CV {test_means[-1]:.3f}')
    return np.array(train_means), np.array(test_means)


def plot_sweep(
    values: Sequence[float],
    train_means: np.ndarray,
    test_means: np.ndarray,
    xlabel: str,
    logx: bool = False,
) -> None:
    """Plot mean train and CV accuracy against a hyperparameter."""
    fig, ax = plt.subplots()
    ax.plot(values, train_means, 'o-', label='Training accuracy')
    ax.plot(values, test_means, 'o-', label='CV accuracy')
    if logx:
        ax.set_xscale('log')
    ax.set(xlabel=xlabel, ylabel='Accuracy')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_confusion_matrices(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: dict[int, str] = LABELS,
) -> None:
    """Confusion matrices of a fitted estimator on the training and test sets."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, X, y, title in (
        (axs[0], X_train, y_train, 'Train'),
        (axs[1], X_test, y_test, 'Test'),
    ):
        ConfusionMatrixDisplay.from_estimator(
            estimator,
            X,
            y,
            labels=list(labels),
            display_labels=list(labels.values()),
            cmap='Blues',
            colorbar=False,
            ax=ax,
        )
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def umap_svm_pipeline(
    umap_params: dict | None = None, svm_params: dict | None = None
) -> Pipeline:
    """Standard scaling, UMAP embedding and an SVM classifier.

    UMAP is seeded (``random_state=42`` unless ``umap_params`` sets it), so that the
    results are reproducible; this also makes it run single-threaded.
    """
    from umap import UMAP

    return Pipeline(
        [
            ('Scaler', StandardScaler()),
            ('UMAP', UMAP(**{'random_state': 42, **(umap_params or {})})),
            ('SVM', SVC(**(svm_params or {}))),
        ]
    )


def run_umap_svm_sweep(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    max_components: int,
    max_neighbors: int,
    project: str = 'OPM Nottingham Classification',
    count: int | None = None,
) -> None:
    """Random hyperparameter search of UMAP + SVM with Weights & Biases.

    Every run is evaluated with 3-fold stratified cross-validation. Without
    ``count`` the agent keeps sampling until it is stopped.
    """
    import wandb

    sweep_config = {
        'method': 'random',
        'name': name,
        'metric': {'goal': 'maximize', 'name': 'cv_test_acc_mean'},
        'parameters': {
            'n_components': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': max_components,
            },
            # UMAP needs at least two neighbours
            'n_neighbors': {
                'distribution': 'int_uniform',
                'min': 2,
                'max': max_neighbors,
            },
            'min_dist': {'distribution': 'uniform', 'min': 0, 'max': 0.99},
            'kernel': {'values': ['linear', 'poly', 'rbf', 'sigmoid']},
            'C': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e3},
            'n_classes': {'value': len(np.unique(y))},
            'dataset': {'value': 'OPM Nottingham'},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    def train() -> None:
        with wandb.init() as run:
            params = run.config
            pipe = umap_svm_pipeline(
                umap_params={
                    'n_components': params['n_components'],
                    'n_neighbors': params['n_neighbors'],
                    'min_dist': params['min_dist'],
                },
                svm_params={'kernel': params['kernel'], 'C': params['C']},
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True)
            results = cross_validate(pipe, X, y, cv=cv, verbose=False)
            for fold, (train_acc, test_acc) in enumerate(
                zip(results['train_score'], results['test_score'], strict=True),
                start=1,
            ):
                run.log(
                    {
                        'cv_train_acc': train_acc,
                        'cv_test_acc': test_acc,
                        'fold_number': fold,
                    }
                )
            run.log(
                {
                    'cv_train_acc_mean': results['train_score'].mean(),
                    'cv_test_acc_mean': results['test_score'].mean(),
                }
            )

    wandb.agent(sweep_id, function=train, count=count)
