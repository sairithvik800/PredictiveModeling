# SLANT Makespan Prediction - ML Pipeline

Nested cross-validation regression that predicts SLANT job makespan (completion
time in seconds) from MRI image features.

## Setup

Requires Python 3.13. From the repository root:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Run

From the repository root:

    python ml_pipeline/main.py

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
