import numpy as np
import pytest
from scipy.stats import binom
from sklearn.multiclass import OneVsRestClassifier
from synthetic import simulate_csp_data

from opm_rps.classification import (
    CLASSIFIERS,
    LassoSelector,
    chance_level,
    cross_evaluate,
    csp_pipeline,
    evaluate,
    make_classifier,
    sfbcsp_pipeline,
    significance_threshold,
)


def test_chance_level():
    assert chance_level(np.array([1, 1, 2, 4])) == 0.5
    assert chance_level(np.repeat([1, 2, 4], 5)) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ('n_trials', 'expected'),
    # Two balanced classes, values of the thesis: 36 test trials, 6 and 60
    [(36, 24 / 36), (6, 1.0), (60, 37 / 60)],
)
def test_significance_threshold(n_trials, expected):
    y = np.arange(n_trials) % 2
    assert significance_threshold(y) == pytest.approx(expected)


def test_significance_threshold_of_few_or_unbalanced_trials():
    # Guessing four out of four trials right happens with p = 1/16 > 0.05
    assert np.isnan(significance_threshold(np.arange(4) % 2))
    assert significance_threshold(np.arange(4) % 2, alpha=0.1) == 1.0
    # With three classes, guessing is right a third of the time
    three = np.arange(30) % 3
    assert significance_threshold(three) < significance_threshold(np.arange(30) % 2)
    # Always predicting the larger of two unbalanced classes is right 2/3 of the time
    unbalanced = np.repeat([0, 1], [20, 10])
    assert significance_threshold(unbalanced) > significance_threshold(three)
    assert significance_threshold(unbalanced) > 2 / 3


def test_pvalue_agrees_with_the_threshold(csp_data):
    X, y = csp_data
    result = evaluate(csp_pipeline(), X, y)
    n_trials = len(result.y_test)
    for n_correct in range(n_trials + 1):
        result.test_score = n_correct / n_trials
        expected = binom.sf(n_correct - 1, n_trials, chance_level(result.y_test))
        assert result.pvalue == pytest.approx(expected)
        above = result.test_score >= significance_threshold(result.y_test)
        assert above == (result.pvalue < 0.05)
    assert 'p = ' in str(result)


@pytest.mark.parametrize('name', CLASSIFIERS)
def test_csp_pipelines_learn_the_simulated_classes(csp_data, name):
    X, y = csp_data
    result = evaluate(csp_pipeline(name), X, y, random_state=210)
    assert result.test_score > 0.8
    assert len(result.cv_scores) == 10
    assert len(result.y_test) == 18
    # Stratified: both classes keep their half of the test trials
    np.testing.assert_array_equal(np.bincount(result.y_test), [9, 9])


def test_one_vs_rest_classifies_three_classes(rng):
    X, y = simulate_csp_data(rng, n_trials=90, n_classes=3)
    result = evaluate(OneVsRestClassifier(csp_pipeline()), X, y)
    assert len(result.model.estimators_) == 3  # one CSP + LDA per class
    assert result.test_score > 0.8

    # Two classes: a single model, which predicts like the pipeline itself
    binary = y < 2
    model = OneVsRestClassifier(csp_pipeline()).fit(X[binary], y[binary])
    assert len(model.estimators_) == 1
    pipeline = csp_pipeline().fit(X[binary], y[binary])
    np.testing.assert_array_equal(model.predict(X), pipeline.predict(X))


def test_unknown_classifier_and_selection():
    with pytest.raises(ValueError, match='Unknown classifier'):
        make_classifier('random-forest')
    with pytest.raises(ValueError, match='Unknown selection'):
        sfbcsp_pipeline('mutual-information')


@pytest.mark.parametrize('selection', ['lasso', 'pca'])
def test_sfbcsp_pipeline(csp_data, rng, selection):
    X, y = csp_data
    X_bands = np.stack([X, rng.normal(size=X.shape), X], axis=1)
    result = evaluate(sfbcsp_pipeline(selection), X_bands, y)
    assert result.test_score > 0.8


def test_lasso_selector_keeps_the_informative_features(rng):
    y = np.repeat([0, 1], 50)
    X = rng.normal(size=(100, 6))
    X[:, 2] += 2 * y
    selector = LassoSelector().fit(X, y)
    assert selector.get_support()[2]
    assert selector.transform(X).shape == (100, selector.get_support().sum())


def test_lasso_selector_keeps_a_feature_if_the_lasso_keeps_none(rng):
    y = np.repeat([0, 1], 30)
    X = rng.normal(size=(60, 5))
    X[:, 3] += 2 * y
    # No weight reaches the threshold: the feature the LASSO would take first stays
    selector = LassoSelector(threshold=np.inf).fit(X, y)
    np.testing.assert_array_equal(selector.get_support(), np.arange(5) == 3)

    # So that a pipeline on pure noise can be fitted and scored
    noise = rng.normal(size=(60, 3, 8, 50))
    result = evaluate(sfbcsp_pipeline(), noise, y, n_splits=3)
    assert 0 <= result.test_score <= 1


def test_cross_evaluate(rng):
    results = {
        name: evaluate(csp_pipeline(), *simulate_csp_data(rng)) for name in ('A', 'B')
    }
    # Different channel set, e.g. a run with excluded sensors
    results['C'] = evaluate(csp_pipeline(), *simulate_csp_data(rng, n_channels=6))

    scores = cross_evaluate(results)

    assert list(scores.index) == list(scores.columns) == ['A', 'B', 'C']
    for name, result in results.items():
        assert scores.loc[name, name] == result.test_score
    assert np.isnan(scores.loc['A', 'C']) and np.isnan(scores.loc['C', 'B'])
