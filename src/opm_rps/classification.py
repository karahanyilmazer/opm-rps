"""Classification pipelines and the evaluation protocol, ported from the motor imagery
analyses of eeg-eog-hybrid-unicorn.

Each pipeline maps epochs (or a filter bank of epochs) of two classes to predictions,
so the spatial filters are learned on the training data only. For more classes, wrap
it in :class:`sklearn.multiclass.OneVsRestClassifier`: one copy of the pipeline per
class learns to tell it from all the others. The evaluation holds out a test set,
cross-validates on the remaining trials and finally tests a model fitted on all
training trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted, validate_data

from .csp import CSP, FilterBankCSP

CLASSIFIERS = ('lda', 'lr', 'svc', 'mlp')


def make_classifier(name: str, random_state: int = 0) -> BaseEstimator:
    """Return one of the classifiers compared in the thesis.

    Parameters
    ----------
    name : 'lda' | 'lr' | 'svc' | 'mlp'
        Linear discriminant analysis, logistic regression, support vector machine
        (RBF kernel) or multi-layer perceptron. LR and SVC weight the classes by
        their inverse frequency.
    random_state : int
        Seed of the MLP initialization.
    """
    if name == 'lda':
        return LinearDiscriminantAnalysis()
    if name == 'lr':
        return LogisticRegression(class_weight='balanced')
    if name == 'svc':
        return SVC(class_weight='balanced')
    if name == 'mlp':
        return MLPClassifier(
            early_stopping=True,
            n_iter_no_change=50,
            max_iter=2000,
            random_state=random_state,
        )
    raise ValueError(f'Unknown classifier {name!r}, choose from {CLASSIFIERS}.')


def csp_pipeline(classifier: str = 'lda', n_pairs: int = 2) -> Pipeline:
    """CSP log-variance features followed by a classifier."""
    return make_pipeline(CSP(n_pairs=n_pairs), make_classifier(classifier))


class LassoSelector(SelectorMixin, BaseEstimator):
    """Keep the features with non-zero weights in a cross-validated LASSO.

    The LASSO regresses the labels on the features, with the regularization chosen
    by ``cv``-fold cross-validation (:class:`sklearn.linear_model.LassoCV`). The
    features whose absolute weight reaches ``threshold`` are kept, as by
    ``SelectFromModel(LassoCV())`` in the thesis. If none does, the labels look
    unrelated to every feature; the feature that enters the LASSO first (the largest
    covariance with the labels) is then kept, so that the classifier still has an
    input.

    Attributes
    ----------
    estimator_ : LassoCV
        The fitted LASSO.
    support_ : np.ndarray of bool, shape (n_features,)
        The kept features.
    """

    def __init__(self, cv: int = 20, max_iter: int = 5000, threshold: float = 1e-5):
        self.cv = cv
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = validate_data(self, X, y, y_numeric=True)
        self.estimator_ = LassoCV(cv=self.cv, max_iter=self.max_iter).fit(X, y)
        support = np.abs(self.estimator_.coef_) >= self.threshold
        if not support.any():
            covariance = (X - X.mean(axis=0)).T @ (y - y.mean())
            support[np.argmax(np.abs(covariance))] = True
        self.support_ = support
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.support_


def sfbcsp_pipeline(selection: str = 'lasso', n_pairs: int = 1) -> Pipeline:
    """Sub-band filter bank CSP with feature selection and an SVM.

    Expects filter bank input, see :func:`opm_rps.preprocessing.filter_bank_epochs`.

    Parameters
    ----------
    selection : 'lasso' | 'pca'
        Keep the features with non-zero LASSO weights (:class:`LassoSelector`, used
        for the BCI Competition data of the thesis), or project them onto two
        principal components (used for the thesis's own recordings).
    n_pairs : int
        Number of CSP filter pairs per band.
    """
    if selection == 'lasso':
        selector = LassoSelector()
    elif selection == 'pca':
        selector = PCA(n_components=2)
    else:
        raise ValueError(f'Unknown selection {selection!r}, use "lasso" or "pca".')
    return make_pipeline(
        FilterBankCSP(n_pairs=n_pairs), selector, make_classifier('svc')
    )


def chance_level(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class."""
    return np.unique(y, return_counts=True)[1].max() / len(y)


def significance_threshold(y: np.ndarray, alpha: float = 0.05) -> float:
    """Lowest accuracy on trials with the labels ``y`` that is significantly above
    chance.

    Under the null hypothesis, each trial is classified correctly with the
    probability :func:`chance_level`, so the number of correct trials is binomially
    distributed. An accuracy is above chance (one-sided binomial test, p < alpha)
    from the returned value on, which is NaN if not even a perfect accuracy is.
    """
    n_trials = len(y)
    correct = np.arange(n_trials + 1)
    above = binom.sf(correct - 1, n_trials, chance_level(y)) < alpha
    return float(correct[above][0] / n_trials) if above.any() else np.nan


@dataclass
class Evaluation:
    """Outcome of :func:`evaluate`.

    Attributes
    ----------
    model : BaseEstimator
        Model fitted on the whole training split.
    cv_scores : np.ndarray
        Accuracies of the cross-validation folds on the training split.
    test_score : float
        Accuracy of ``model`` on the test split.
    X_test, y_test : np.ndarray
        Held-out data, used e.g. by :func:`cross_evaluate`.
    """

    model: BaseEstimator
    cv_scores: np.ndarray
    test_score: float
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def pvalue(self) -> float:
        """One-sided binomial p-value of the test accuracy, see
        :func:`significance_threshold`."""
        n_trials = len(self.y_test)
        n_correct = round(self.test_score * n_trials)
        return float(binom.sf(n_correct - 1, n_trials, chance_level(self.y_test)))

    def __str__(self) -> str:
        return (
            f'validation accuracy {self.cv_scores.mean():.1%} '
            f'± {self.cv_scores.std():.1%}, test accuracy {self.test_score:.1%} '
            f'(p = {self.pvalue:.3g}, {len(self.y_test)} trials)'
        )


def evaluate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 12,
    n_splits: int = 10,
) -> Evaluation:
    """Hold out a test set, cross-validate on the rest and test the final model.

    Unlike in the thesis, the test set and the folds are stratified, so they keep
    the class proportions of all trials.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=n_splits)
    cv_scores = cross_val_score(clone(model), X_train, y_train, cv=cv)
    fitted = clone(model).fit(X_train, y_train)
    return Evaluation(fitted, cv_scores, fitted.score(X_test, y_test), X_test, y_test)


def cross_evaluate(evaluations: dict[str, Evaluation]) -> pd.DataFrame:
    """Test every model on the test split of every dataset.

    Returns
    -------
    scores : pd.DataFrame
        Accuracies with the training dataset as rows and the test dataset as
        columns, the diagonal holds the regular test accuracies. Pairs with
        different channel sets are NaN.
    """
    scores = {}
    for test_name, test in evaluations.items():
        scores[test_name] = {}
        for train_name, train in evaluations.items():
            try:
                score = train.model.score(test.X_test, test.y_test)
            except ValueError:
                score = np.nan
            scores[test_name][train_name] = score
    return pd.DataFrame(scores)
