import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from opm_rps.csp import CSP, FilterBankCSP, mean_covariance


def thesis_csp_features(X, y, m):
    """CSP of the thesis scripts (cov, csp, apply_mix and logvar).

    The original code did not sort the eigenvalues, which is added here so that the
    first and last components are the most discriminative ones.
    """
    cov_1 = mean_covariance(X[y == 0])
    cov_2 = mean_covariance(X[y == 1])
    w, v = np.linalg.eig(cov_1 + cov_2)
    P = np.diag(w**-0.5) @ v.T
    psi, B = np.linalg.eig(P @ cov_1 @ P.T)
    B = B[:, np.argsort(psi)[::-1]]
    W = P.T @ B
    Z = np.einsum('ck,nct->nkt', W, X)[:, np.r_[0:m, -m:0]]
    var = np.var(Z, axis=2)
    return np.log(var / var.sum(axis=1, keepdims=True))


def test_mean_covariance(csp_data):
    X, _ = csp_data
    covs = [trial @ trial.T for trial in X]
    expected = np.mean([cov / np.trace(cov) for cov in covs], axis=0)
    np.testing.assert_allclose(mean_covariance(X), expected, rtol=1e-12)


def test_matches_the_thesis_implementation(csp_data):
    X, y = csp_data
    features = CSP(n_pairs=2).fit(X, y).transform(X)
    np.testing.assert_allclose(features, thesis_csp_features(X, y, m=2), rtol=1e-8)


def test_filters_solve_the_generalized_eigenvalue_problem(csp_data):
    X, y = csp_data
    csp = CSP().fit(X, y)
    cov_1 = mean_covariance(X[y == 0])
    cov_2 = mean_covariance(X[y == 1])

    for value, w in zip(csp.eigenvalues_, csp.filters_, strict=True):
        np.testing.assert_allclose(cov_1 @ w, value * (cov_1 + cov_2) @ w, atol=1e-12)
    assert np.all(np.diff(csp.eigenvalues_) <= 0)
    np.testing.assert_allclose(
        csp.filters_ @ csp.patterns_, np.eye(X.shape[1]), atol=1e-8
    )


def test_extreme_filters_separate_the_classes(csp_data):
    X, y = csp_data
    features = CSP(n_pairs=1, normalize=False).fit(X, y).transform(X)
    # The first filter favours class 0, the last one class 1
    assert features[y == 0, 0].mean() > features[y == 1, 0].mean() + 1
    assert features[y == 1, 1].mean() > features[y == 0, 1].mean() + 1


def test_rank_deficient_data(csp_data):
    X, y = csp_data
    # Projected out: two dimensions, as the homogeneous field correction removes 8
    basis, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(X.shape[1], 2)))
    X = X - np.einsum('cd,nct->ndt', basis @ basis.T, X)

    csp = CSP().fit(X, y)

    assert csp.filters_.shape == (X.shape[1] - 2, X.shape[1])
    features = csp.transform(X)
    assert np.isfinite(features).all()
    assert features[y == 0, 0].mean() > features[y == 1, 0].mean()


def test_in_a_pipeline(csp_data):
    X, y = csp_data
    pipeline = make_pipeline(CSP(), LinearDiscriminantAnalysis())
    assert cross_val_score(pipeline, X, y, cv=5).mean() > 0.9


def test_input_validation(csp_data):
    X, y = csp_data
    with pytest.raises(ValueError, match='two classes'):
        CSP().fit(X, np.arange(len(y)) % 3)
    with pytest.raises(ValueError, match='rank of the data'):
        CSP(n_pairs=5).fit(X, y)


def test_filter_bank_csp(csp_data, rng):
    X, y = csp_data
    # Second band without class information
    X_bands = np.stack([X, rng.normal(size=X.shape)], axis=1)

    fbcsp = FilterBankCSP(n_pairs=2).fit(X_bands, y)
    features = fbcsp.transform(X_bands)

    assert features.shape == (len(y), 8)
    np.testing.assert_allclose(features[:, :4], CSP(n_pairs=2).fit(X, y).transform(X))
