"""Common spatial patterns (CSP) for two classes, ported from the motor imagery
analyses of eeg-eog-hybrid-unicorn.

Both transformers follow the scikit-learn API, so they can be placed in a pipeline
and are only ever fitted on training data during cross-validation. For more than two
classes, see :mod:`opm_rps.classification`.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def mean_covariance(X: np.ndarray) -> np.ndarray:
    """Average the trace-normalized spatial covariance matrices of the trials.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)

    Returns
    -------
    cov : np.ndarray, shape (n_channels, n_channels)
    """
    covs = X @ X.transpose(0, 2, 1)
    covs /= np.trace(covs, axis1=1, axis2=2)[:, np.newaxis, np.newaxis]
    return covs.mean(axis=0)


class CSP(TransformerMixin, BaseEstimator):
    """Common spatial patterns with log-variance features (Ramoser et al., 2000).

    The spatial filters ``w`` solve the generalized eigenvalue problem
    ``C1 w = lambda (C1 + C2) w`` of the mean class covariances. The filters with the
    largest eigenvalues maximize the variance of the first class relative to the
    second one, the filters with the smallest eigenvalues do the opposite. The
    features are the log-variances of the ``n_pairs`` filters from each end.

    The problem is solved by whitening the composite covariance ``C1 + C2`` and
    diagonalizing the whitened ``C1``. Directions without variance are left out, so
    rank-deficient data yields fewer filters than channels: the homogeneous field
    correction removes 8 dimensions (order 2) from the OPM recordings.

    Parameters
    ----------
    n_pairs : int
        Number of filters taken from each end of the spectrum, giving ``2 * n_pairs``
        features.
    normalize : bool
        Divide the variances by their sum before taking the logarithm.

    Attributes
    ----------
    classes_ : np.ndarray, shape (2,)
        Class labels, the first one belongs to the largest eigenvalues.
    eigenvalues_ : np.ndarray, shape (n_components,)
        Eigenvalues in descending order.
    filters_ : np.ndarray, shape (n_components, n_channels)
        Spatial filters as rows, in the order of the eigenvalues.
    patterns_ : np.ndarray, shape (n_channels, n_components)
        Spatial patterns as columns, i.e. the pseudo-inverse of the filters.
    """

    def __init__(self, n_pairs: int = 2, normalize: bool = True):
        self.n_pairs = n_pairs
        self.normalize = normalize

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Compute the spatial filters from epochs of shape (n_trials, n_channels,
        n_times) and their labels."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f'CSP needs two classes, got {self.classes_}.')

        cov_1 = mean_covariance(X[y == self.classes_[0]])
        cov_2 = mean_covariance(X[y == self.classes_[1]])

        # Whiten the composite covariance in the subspace that carries variance
        variances, directions = eigh(cov_1 + cov_2)
        keep = variances > variances.max() * 1e-10
        whitening = directions[:, keep].T / np.sqrt(variances[keep])[:, np.newaxis]
        if 2 * self.n_pairs > keep.sum():
            raise ValueError(
                f'2 * n_pairs must not exceed the rank of the data ({keep.sum()}).'
            )

        # Diagonalize the whitened covariance of the first class
        eigenvalues, rotation = eigh(whitening @ cov_1 @ whitening.T)

        # Sort the filters by decreasing eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[order]
        self.filters_ = rotation[:, order].T @ whitening
        self.patterns_ = np.linalg.pinv(self.filters_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the log-variance features, shape (n_trials, 2 * n_pairs)."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)

        # Take the first and the last n_pairs filters
        filters = np.concatenate(
            [self.filters_[: self.n_pairs], self.filters_[-self.n_pairs :]]
        )
        variances = (filters @ X).var(axis=-1)
        if self.normalize:
            variances /= variances.sum(axis=1, keepdims=True)
        return np.log(variances)


class FilterBankCSP(TransformerMixin, BaseEstimator):
    """CSP applied to each band of a filter bank (sub-band filter bank CSP).

    Parameters
    ----------
    n_pairs : int
        Number of filter pairs per band.
    normalize : bool
        Normalize the variances, see :class:`CSP`.

    Notes
    -----
    The input has the shape (n_trials, n_bands, n_channels, n_times), see
    :func:`opm_rps.preprocessing.filter_bank_epochs`. The features of all bands are
    concatenated band by band.
    """

    def __init__(self, n_pairs: int = 1, normalize: bool = True):
        self.n_pairs = n_pairs
        self.normalize = normalize

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X, dtype=float)
        self.csps_ = [
            CSP(self.n_pairs, self.normalize).fit(X[:, band], y)
            for band in range(X.shape[1])
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the features, shape (n_trials, n_bands * 2 * n_pairs)."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return np.concatenate(
            [csp.transform(X[:, band]) for band, csp in enumerate(self.csps_)], axis=1
        )
