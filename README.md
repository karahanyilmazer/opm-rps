# opm-rps

Decoding rock-paper-scissors hand gestures from optically pumped magnetometer (OPM)
MEG data: one participant (subject 11766), four runs recorded on 2023-06-23 with
triaxial OPMs. Every trial starts with a cue at t = 0 telling the participant which
gesture to make (`roc`, `pap`, `sci`).

## Setup

```bash
uv sync                  # core dependencies
uv sync --extra sweep    # + Weights & Biases hyperparameter sweeps
uv sync --extra viz3d    # + 3D brain plots and MRI headers for the source space
```

Tell the code where the data lives in a `.env` file at the repository root:

```bash
# Dataset folder containing "20230623/" (raw cMEG data) and "MRI/" (FreeSurfer)
OPM_RPS_DATA_DIR=/path/to/opm-data/2023-06-21_nottingham

# Optional
# OPM_RPS_DERIVATIVES_DIR=/path/to/derivatives  # default: ./derivatives
# OPM_RPS_CONFIG=/path/to/other_config.yaml     # default: ./config.yaml
```

Expected layout of the data folder:

```
$OPM_RPS_DATA_DIR/
├── 20230623/
│   └── 20230623_<HHMMSS>_cMEG_Data/20230623_<HHMMSS>_meg.cMEG  (+ .json/.tsv sidecars)
└── MRI/Segmentation/11766/                                   (FreeSurfer subject)
```

## Repository layout

```
config.yaml           analysis parameters, bad channels and bad trials
src/opm_rps/          shared code, imported by the scripts
  config.py           reads config.yaml and .env; defines every file path
  io.py               .cMEG reader -> MNE Raw + events
  preprocessing.py    bad channels/trials, HFC, notch, cropping, epoching, filters
  features.py         variance, band power and Welch power features
  time_frequency.py   Morlet wavelet power and inter-trial phase coherence
  coregistration.py   MEG-MRI coregistration and sensor-to-scalp distances
  decoding.py         cross-validation, sweeps, embeddings, confusion matrices
  viz.py              plotting style, parula colormap, TF grid, topomaps
scripts/              the analysis steps (see below)
tests/                pytest suite on synthetic recordings
derivatives/          outputs of the scripts (git-ignored, except features.npz)
```

## Pipeline

```mermaid
flowchart LR
    raw[(cMEG<br>recordings)] --> clean[preprocessing/<br>clean_epochs]
    raw --> feat[features/<br>extract_features]
    raw --> tf[features/<br>extract_tf]
    raw --> figs[figures/<br>plot_psd, plot_sensors,<br>plot_erps, plot_tf]
    clean --> dec[decoding/<br>decode_csp, decode_ems,<br>decode_logreg]
    clean --> coreg[source_space/<br>coregister]
    mri[(MRI)] --> coreg
    coreg --> src[source_space/<br>source_reconstruction]
    clean --> src
    mri --> src
    src --> plotsrc[source_space/<br>plot_source]
    src --> decsrc[decoding/<br>decode_source]
    clean --> decsrc
    feat --> decfeat[decoding/<br>decode_features]
    tf --> dectf[decoding/<br>decode_tf]
```

| Script (`scripts/...`) | Reads | Writes (`derivatives/...`) |
| --- | --- | --- |
| `preprocessing/clean_epochs.py` | raw runs 1-4 | `epochs/all_runs-filt_7_30-epo.fif.gz` |
| `features/extract_features.py` | raw runs 2-4 | `features/features.npz` |
| `features/extract_tf.py` | raw runs 1-4 | `tf/tf-<run>-<axis>.npz` |
| `decoding/decode_csp.py` | cleaned epochs | (scores, embeddings) |
| `decoding/decode_ems.py` | cleaned epochs | (EMS surrogates, filters, scores) |
| `decoding/decode_logreg.py` | cleaned epochs | (pattern and filter topomaps) |
| `decoding/decode_features.py` | `features.npz` | (PCA/LDA/UMAP + SVM results) |
| `decoding/decode_tf.py` | `tf/*.npz` | (UMAP + SVM results) |
| `decoding/decode_source.py` | cleaned epochs, inverse operator | `figures/decoding/` |
| `source_space/coregister.py` | cleaned epochs, SensorTransform sidecars, MRI | `source/<subject>-trans.fif`, `figures/source/`, `reports/coregistration.html` |
| `source_space/source_reconstruction.py` | cleaned epochs, MRI, `source/<subject>-trans.fif` | `source/`, `reports/source_space.html` |
| `source_space/plot_source.py` | `source/` | `figures/source/`, `reports/source_estimate.html` |
| `figures/plot_psd.py` | raw runs 1-4 | `figures/psd/` |
| `figures/plot_sensors.py` | raw run 1 | `figures/sensors/` |
| `figures/plot_erps.py` | raw runs 1-4 | `figures/erp/` |
| `figures/plot_tf.py` | raw runs 1-4 | `figures/tf/` |

Run a script with `uv run python scripts/<step>/<script>.py`, or open it in VS Code
and run it cell by cell (`# %%`). When editing `src/opm_rps` during an interactive
session, enable `%autoreload 2`.

Run the scripts one at a time. On the real data, the preprocessing, figure and
decoding scripts each need up to about 17 GB of memory, and an interactive window
keeps its data until its kernel is restarted.

### Preprocessing at a glance

Every analysis marks the bad channels (`bad_channels` in `config.yaml` plus those
flagged "Bad" in the recording's channels.tsv), applies homogeneous field correction
(HFC, order 2) and crops run 3 at 681 s, where run 4 begins. Every trial-level
analysis rejects the same trials: the union of the two bad-trial annotations in
`config.yaml`, `bad_epochs` (the cue sample of each rejected trial, per run) and
`bad_epoch_indices` (positions into the cue events of runs 1-4, per sensor axis);
see `opm_rps.preprocessing.bad_trials`. Both annotations mark the same 122 of the
720 trials (73, 18, 17 and 14 in runs 1-4). Beyond that:

| Used by | Runs | Filtering | Epochs (-0.5 to 2.1 s) |
| --- | --- | --- | --- |
| ERP, TF and PSD figures, TF features | 1-4, separately | none | linear detrend |
| Decoding epochs, source space | 1-4, concatenated | least-squares FIR 7-30 Hz | linear detrend, baseline -0.5 to 0 s |
| Feature matrices | 2-4, concatenated | notch + 1-400 Hz | none |

The time-frequency figures and features are computed on epochs extended by a buffer
of recorded data on both sides of -0.5 to 2.1 s: three Gaussian widths of the
lowest-frequency wavelet, 1.91 s for `plot_tf.py` (1-40 Hz) and 0.95 s for
`extract_tf.py` (2-50 Hz). Each trial is convolved separately with the wavelets and
the result is cropped to -0.5 to 2.1 s before the dB baseline normalisation, so
within the window the wavelets never reach past the recorded data. Trials whose
buffered epoch does not fit into the recording (or the crop of run 3) are dropped;
the linear detrend is fitted to the buffered epochs.

The band-power and total-power features come from Welch spectra with a resolution
of 1 Hz (1-s Hamming windows overlapping by half).

### Source space

The sensor positions come from the HelmConfig sidecar. Subject 11766 wore a helmet
designed on the MRI, so the positions are already in the scanner RAS frame of the
MRI: the SensorTransform sidecars are the identity and the MRI's `c_ras` is 0.
`source_space/coregister.py` writes the head -> MRI transform (`coregistration:
scanner_ras` in `config.yaml`) and checks it with the sensor-to-scalp distances
(5-24 mm, none inside the head) and a 3D view in `reports/coregistration.html`.
`coregistration: identity` skips the coregistration; the files are then labelled
"misaligned".

The noise covariance of the inverse operator is estimated from the pre-cue baseline
(-0.5 to 0 s) of the cleaned epochs; its rank is reduced by the 8 HFC projectors.

## Known caveats of the analysis

- The feature matrices leave out run 1, as the original feature script did from
  2024 on; no reason was recorded. Add `'run_1'` to `RUNS` in
  `features/extract_features.py` to include its 107 good trials.

## Tests

```bash
uv run pytest
```

The tests generate synthetic recordings in the cMEG format (`tests/synthetic.py`);
no real data is needed. The coregistration tests need nibabel (`viz3d` extra) and are
skipped without it.
