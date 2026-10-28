"""Decode the gesture with the CSP and filter bank CSP pipelines of the motor imagery
analyses in eeg-eog-hybrid-unicorn.

- CSP + LDA: log-variances of two CSP filter pairs in the decoding band and linear
  discriminant analysis.
- SFBCSP + LASSO + SVM: sub-band filter bank CSP, one CSP filter pair in each 4 Hz
  band, the features with non-zero LASSO weights and a support vector machine.

CSP compares two classes, so each pipeline is trained once per gesture to tell it from
the other two (one-vs-rest), and the gesture whose model is the most confident wins; a
pair of gestures needs a single model. As in the thesis, every model is trained on
70 % of the trials, cross-validated on them and tested on the remaining 30 % (here
stratified by gesture). The test accuracy is compared with always predicting the most
frequent gesture (one-sided binomial test).

The pipelines are evaluated on each pair of gestures and on all three with the trials
of all runs, and on all three gestures within each run. The cross-run tables test the
model of each run on the test trials of the other runs.

Input: derivatives/features/filter_bank.npz (features/extract_filter_bank.py)
Output: derivatives/figures/fbcsp/
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier

from opm_rps.classification import (
    chance_level,
    cross_evaluate,
    csp_pipeline,
    evaluate,
    sfbcsp_pipeline,
    significance_threshold,
)
from opm_rps.config import load_config
from opm_rps.decoding import LABELS
from opm_rps.io import EVENT_ID
from opm_rps.viz import (
    plot_cross_accuracies,
    plot_lasso_coefficients,
    plot_test_accuracies,
    set_style,
)

set_style()
cfg = load_config()

TIME_WINDOW = (0.5, 2.0)  # s after the cue, as in decode_csp.py; (-0.5, 0) as a control
RANDOM_STATE = 12  # seed of the split into training and test trials
CONTRASTS = {
    'Rock vs paper': ['roc', 'pap'],
    'Rock vs scissors': ['roc', 'sci'],
    'Paper vs scissors': ['pap', 'sci'],
    'All three': ['roc', 'pap', 'sci'],
}
# Name: input of the pipeline and the pipeline for two classes
PIPELINES = {
    'CSP + LDA': ('broadband', lambda: csp_pipeline('lda')),
    'SFBCSP + LASSO + SVM': ('filter_bank', lambda: sfbcsp_pipeline('lasso')),
}

# %% Load the trials within the time window
data = np.load(cfg.paths.filter_bank)
times = data['times']
tol = (times[1] - times[0]) / 2  # half a sample
in_window = (times > TIME_WINDOW[0] - tol) & (times < TIME_WINDOW[1] + tol)
inputs = {key: data[key][..., in_window] for key, _ in PIPELINES.values()}
y, runs, bands = data['y'], data['run'], data['bands']


def evaluate_pipelines(trials: np.ndarray) -> dict:
    """Evaluate every pipeline on the selected trials (a boolean mask)."""
    results = {}
    for name, (key, make_pipeline) in PIPELINES.items():
        model = OneVsRestClassifier(make_pipeline())
        results[name] = evaluate(
            model, inputs[key][trials], y[trials], random_state=RANDOM_STATE
        )
        print(f'  {name}: {results[name]}')
    return results


def plot_results(results: dict[str, dict], fname: str) -> None:
    """Test accuracies of the pipelines against chance, one row per dataset."""
    evaluations = pd.DataFrame(results).T  # datasets x pipelines
    accuracies = evaluations.map(lambda result: result.test_score)
    print('\nTest accuracies:')
    print(accuracies.round(3))
    print('p-values (one-sided binomial test):')
    print(evaluations.map(lambda result: result.pvalue).to_string(float_format='%.2g'))

    # The pipelines of a dataset share their test trials: the split only depends on
    # the labels and the seed
    y_tests = evaluations.iloc[:, 0].map(lambda result: result.y_test)
    fig = plot_test_accuracies(
        accuracies,
        chance=y_tests.map(chance_level),
        threshold=y_tests.map(significance_threshold),
    )
    fig.savefig(cfg.paths.figure('fbcsp', fname))
    plt.show()


# %% Each pair of gestures and all three, with the trials of all runs
by_contrast = {}
for label, conditions in CONTRASTS.items():
    print(label)
    trials = np.isin(y, [EVENT_ID[condition] for condition in conditions])
    by_contrast[label] = evaluate_pipelines(trials)
plot_results(by_contrast, 'accuracy_contrasts.png')

# %% All three gestures within each run, and across runs
by_run = {}
for run in cfg.runs:
    label = run.replace('_', ' ').capitalize()
    print(label)
    by_run[label] = evaluate_pipelines(runs == run)
plot_results(by_run, 'accuracy_runs.png')

tables = {
    name: cross_evaluate(
        {run: evaluations[name] for run, evaluations in by_run.items()}
    )
    for name in PIPELINES
}
for name, table in tables.items():
    print(f'\n{name}: rows trained on, columns tested on')
    print(table.round(2))
fig = plot_cross_accuracies(tables, chance=1 / len(cfg.events))
fig.savefig(cfg.paths.figure('fbcsp', 'cross_runs.png'))
plt.show()

# %% LASSO weights of the filter bank features (all three gestures, all runs)
model = by_contrast['All three']['SFBCSP + LASSO + SVM'].model
coefs = {
    f'{LABELS[code]} vs rest': pipeline[1].estimator_.coef_
    for code, pipeline in zip(model.classes_, model.estimators_, strict=True)
}
fig = plot_lasso_coefficients(coefs, bands)
fig.savefig(cfg.paths.figure('fbcsp', 'lasso_weights.png'))
plt.show()
