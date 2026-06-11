# SLANT Makespan Prediction - ML Pipeline

Nested cross-validation regression that predicts SLANT job makespan (completion
time in seconds) from MRI image features. This directory is the core modeling
pipeline. This README covers setup and running only; methodology and results are
reported in the paper.

## Setup

Requires Python 3.13. From the repository root:

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    # source .venv/bin/activate     # macOS / Linux
    pip install -r requirements.txt

Note: `requirements.txt` lists `nilearn=0.13.1` with a single `=`, which pip
rejects. If the install stops on that line, install nilearn separately:

    pip install nilearn==0.13.1

## Run

From the repository root:

    python ml_pipeline/main.py

This runs `run_with_manual_features()` - the confirmatory evaluation that
produces the paper's reported per-model numbers, using the fixed 18-feature
`DEFAULT_FEATURES` set. A results summary prints to the console and per-sample
predictions are written under `ml_pipeline/csv_files/outputs/`.

The exploratory stochastic feature-selection mode is the commented loop at the
bottom of `ml_pipeline/main.py` (`run_with_feature_selection`); uncomment it to
regenerate `experiment_log.csv`.

## Outputs

Written to `ml_pipeline/csv_files/outputs/`:

| File | Contents |
|------|----------|
| `cv_predictions_<model>_manual.csv` | Per-sample predictions (filename, makespan, predicted_makespan) for each model. These are the paper's reported results. |
| `experiment_log.csv` | Exploratory feature-selection log: per-run selected features and pipeline-stage counts (no model metrics), used to derive the feature-stability frequencies and tiers. |

## Feature extraction (`features.py`)

`ml_pipeline/features.py` is the deploy-time extractor for the 18-feature
`DEFAULT_FEATURES` set. It requires the `slant` conda environment and MRI volumes
that are not shipped with this repository, and writes `features_data_18.csv`.
Reproducing the modeling results does not require re-running it - the pipeline
reads the committed `features_data.csv`.
