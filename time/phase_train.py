"""Phase 3: time per-model nested-CV training + inference for the paper.

Reuses model_eval_pipeline.nested_cv_evaluation (random_state=42 hardcoded
inside, so results are reproducible) and models.DEFAULT_MODELS.
"""
import csv
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import constants
from models import DEFAULT_MODELS
from main import DEFAULT_FEATURES
import model_eval_pipeline


N_PREDICT_REPEATS = 1000


def time_training_per_model(features: list[str]) -> list[dict]:
    """Run nested CV for each model in DEFAULT_MODELS, recording wall-clock."""
    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    X_all = df[features].values
    y_all = df['makespan'].values

    results = []
    for model in DEFAULT_MODELS:
        pipeline, param_grid = model.get_pipeline_and_params()
        print(f'Timing training for {model.name}...', flush=True)

        start = time.perf_counter()
        model_eval_pipeline.nested_cv_evaluation(pipeline, param_grid, X_all, y_all)
        elapsed = time.perf_counter() - start

        # Fit on full data, then time predict() repeated calls
        fresh_pipeline, _ = model.get_pipeline_and_params()
        fresh_pipeline.fit(X_all, y_all)
        x_new = X_all[0:1]  # 1 x n_features
        predict_times_us: list[float] = []
        for _ in range(N_PREDICT_REPEATS):
            t0 = time.perf_counter()
            fresh_pipeline.predict(x_new)
            predict_times_us.append((time.perf_counter() - t0) * 1e6)

        results.append({
            'model': model.name,
            'train_seconds': round(elapsed, 2),
            'predict_us_median': round(median(predict_times_us), 2),
        })
        print(f'  train={elapsed:.1f}s  predict_med={median(predict_times_us):.1f}us', flush=True)

    return results


def write_train_predict_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'train_seconds', 'predict_us_median'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    out = Path(__file__).parent / 'outputs' / 'timing_train_predict.csv'
    rows = time_training_per_model(DEFAULT_FEATURES)
    write_train_predict_csv(rows, out)
    print(f'Wrote {out}')
