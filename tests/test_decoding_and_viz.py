import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from opm_rps import decoding, viz
from opm_rps.preprocessing import load_epochs


@pytest.fixture
def data():
    """Three separable classes labelled with the trigger codes."""
    rng = np.random.default_rng(0)
    y = np.repeat([1, 2, 4], 20)
    X = rng.normal(size=(len(y), 10)) + y[:, np.newaxis]
    return X, y


def test_cross_validation_and_plots(data):
    X, y = data
    pipe = Pipeline(
        [('Scaler', StandardScaler()), ('PCA', PCA(n_components=2)), ('SVM', SVC())]
    )
    results = decoding.cross_validate(pipe, X, y, cv=StratifiedKFold(3), verbose=False)
    assert len(results['test_score']) == 3
    assert results['test_score'].mean() > 0.9

    decoding.plot_fold_embeddings(results, X, y, n_steps=2, name='PCA')
    decoding.plot_confusion_matrices(pipe.fit(X, y), X, y, X, y)
    plt.close('all')


def test_sweep(data):
    X, y = data
    train, test = decoding.sweep(
        lambda c: Pipeline([('SVM', SVC(C=c))]), [0.1, 1.0], X, y, StratifiedKFold(3)
    )
    assert train.shape == test.shape == (2,)
    decoding.plot_sweep([0.1, 1.0], train, test, 'C', logx=True)
    plt.close('all')


def test_haufe_patterns_match_mne_linear_model(data):
    from mne.decoding import LinearModel
    from sklearn.linear_model import LogisticRegression

    X, y = data
    binary = y != 4
    X, y = X[binary], y[binary]
    model = LinearModel(LogisticRegression(solver='liblinear')).fit(X, y)
    np.testing.assert_allclose(
        decoding.haufe_patterns(X, model.filters_), model.patterns_, rtol=1e-10
    )


def test_umap_svm_pipeline():
    pipe = decoding.umap_svm_pipeline({'n_neighbors': 5}, {'gamma': 'auto'})
    assert pipe['UMAP'].n_neighbors == 5
    assert pipe['SVM'].gamma == 'auto'


def test_parula_cmap():
    assert viz.parula_cmap().N == 256


def test_tf_grid():
    times, freqs = np.linspace(-0.5, 2, 50), np.linspace(1, 40, 20)
    tf = np.random.default_rng(0).uniform(size=(len(freqs), len(times)))
    fig = viz.plot_tf_grid(tf, tf / 2, tf, tf / 2, times, freqs, title='test')
    assert len(fig.axes) == 12  # six panels and their colour bars
    plt.close(fig)


def test_result_figures():
    rows, pipelines = ['Rock vs paper', 'All three'], ['CSP', 'SFBCSP']
    accuracies = pd.DataFrame(
        [[0.6, np.nan], [0.4, 0.5]], index=rows, columns=pipelines
    )
    fig = viz.plot_test_accuracies(
        accuracies,
        chance=pd.Series([0.5, 1 / 3], index=rows),
        threshold=pd.Series([0.6, 0.4], index=rows),
    )
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == rows

    runs = ['Run 1', 'Run 2']
    table = pd.DataFrame([[0.5, 0.3], [0.4, np.nan]], index=runs, columns=runs)
    fig = viz.plot_cross_accuracies({'CSP': table, 'SFBCSP': table}, chance=1 / 3)
    assert len(fig.axes) == 3  # two tables and their colour bar
    # Every accuracy is written into its cell, except the missing one
    assert len(fig.axes[0].texts) == 3

    bands = [(7.0, 11.0), (9.0, 13.0), (11.0, 15.0)]
    coefs = {'Rock vs rest': np.arange(6.0), 'Paper vs rest': -np.arange(6.0)}
    fig = viz.plot_lasso_coefficients(coefs, bands)
    bars = fig.axes[1].patches
    assert len(bars) == 6  # two features per band
    assert sorted(bar.get_height() for bar in bars) == list(range(6))  # absolute
    centres = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    np.testing.assert_allclose(
        np.mean(np.reshape(centres, (3, 2)), axis=1), [9, 11, 13]
    )
    plt.close('all')


def test_topomaps_of_triaxial_sensors(cfg):
    evoked = load_epochs(cfg, 'run_1').average()
    viz.plot_topomaps_per_axis(
        evoked.data, evoked.info, evoked.tmin, 'test', times=[0.5]
    )
    assert len(plt.get_fignums()) == 3  # one figure per axis
    plt.close('all')
