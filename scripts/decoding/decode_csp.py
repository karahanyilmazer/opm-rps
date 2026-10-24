"""Decode the gesture from the cleaned epochs with CSP-based pipelines.

Input: derivatives/epochs/all_runs-filt_7_30-epo.fif.gz (preprocessing/clean_epochs.py)
"""

# %%
import mne
from mne.decoding import CSP, UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from opm_rps.config import load_config
from opm_rps.decoding import (
    cross_validate,
    evaluate_on_test_set,
    permutation_test,
    plot_fold_embeddings,
)
from opm_rps.viz import set_style

set_style()
cfg = load_config()

TIME_WINDOW = (0.5, 2.0)  # s after the cue
PIPELINE = 'pca_csp_svm'  # key of PIPELINES below
# Label shufflings of the permutation test (0 skips it). Each repeats the whole
# cross-validation, which takes 20-60 s for 'pca_csp_svm' on the real data, and
# N_JOBS of them run in parallel (each needs about 2 GB more memory)
N_PERMUTATIONS = 100
N_JOBS = 4


def csp():
    return CSP(n_components=4, reg=None, log=True, norm_trace=False)


PIPELINES = {
    'pca_csp_svm': Pipeline(
        [
            ('PCA', UnsupervisedSpatialFilter(PCA(0.99), average=False)),
            ('CSP', csp()),
            ('SVM', SVC(gamma='auto')),
        ]
    ),
    'csp_svm': Pipeline([('CSP', csp()), ('SVM', SVC(gamma='auto'))]),
    'csp_lda': Pipeline([('CSP', csp()), ('LDA', LinearDiscriminantAnalysis())]),
}

# %% Load the data
epochs = mne.read_epochs(cfg.paths.clean_epochs(*cfg.decoding_band))
X = epochs.copy().crop(*TIME_WINDOW).get_data(copy=True)
y = epochs.events[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)

# %% Cross-validate on the training set
pipe = PIPELINES[PIPELINE]
cv = KFold(n_splits=3, shuffle=True, random_state=42)
results = cross_validate(pipe, X_train, y_train, cv=cv)
if N_PERMUTATIONS:
    permutation_test(pipe, X_train, y_train, cv, N_PERMUTATIONS, n_jobs=N_JOBS)
if 'PCA' in pipe.named_steps:
    plot_fold_embeddings(results, X_train, y_train, n_steps=1, name='PCA')

# %% Test-set accuracy of the pipeline fitted on the whole training set
pipe.fit(X_train, y_train)
evaluate_on_test_set(pipe, X_test, y_test)
