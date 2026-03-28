import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone

import constants


def nested_cv_evaluation(pipeline, param_grid, X, y):
    """
    Single nested CV that computes both R² and MAE from the same predictions.
    Includes overfitting diagnostics by comparing inner CV, train, and test scores.

    When param_grid is empty (e.g. GPR), skips RandomizedSearchCV and fits the
    pipeline directly via clone() for each outer fold.
    """
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_test_indices = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]

        if param_grid:
            inner_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=30,
                cv=inner_cv,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=1,
                verbose=0
            )
            inner_search.fit(X_train, y_train_raw)
            inner_cv_mae = -inner_search.best_score_
            best_params = inner_search.best_params_
            fitted_model = inner_search
        else:
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_train, y_train_raw)
            inner_cv_mae = np.nan
            best_params = {}
            fitted_model = fold_pipeline

        y_train_pred_fit = fitted_model.predict(X_train)
        y_pred_fit = fitted_model.predict(X_test)
        y_train_pred = y_train_pred_fit
        y_pred = y_pred_fit

        # All metrics in original scale
        train_r2 = r2_score(y_train_raw, y_train_pred)
        train_mae = mean_absolute_error(y_train_raw, y_train_pred)

        # Store for overall calculation (original scale)
        all_y_true.extend(y_test_raw)
        all_y_pred.extend(y_pred)
        all_test_indices.extend(test_idx)

        # Compute fold metrics from same predictions
        fold_r2 = r2_score(y_test_raw, y_pred)
        fold_mae = mean_absolute_error(y_test_raw, y_pred)

        # Overfitting diagnostics
        r2_overfit_gap = train_r2 - fold_r2
        mae_overfit_gap = fold_mae - train_mae
        inner_outer_gap = fold_mae - inner_cv_mae

        fold_results.append({
            'fold': fold_idx + 1,
            'r2': fold_r2,
            'mae': fold_mae,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'inner_cv_mae': inner_cv_mae,
            'r2_overfit_gap': r2_overfit_gap,
            'mae_overfit_gap': mae_overfit_gap,
            'inner_outer_gap': inner_outer_gap,
            'best_params': best_params,
            'n_test': len(y_test_raw)
        })

        print(f"Fold {fold_idx + 1}: R²={fold_r2:.3f}, MAE={fold_mae:.1f}, "
            f"Train R²={train_r2:.3f}, Overfit gap={r2_overfit_gap:.3f}")

    # Aggregate results
    r2_scores = [f['r2'] for f in fold_results]
    mae_scores = [f['mae'] for f in fold_results]
    train_r2_scores = [f['train_r2'] for f in fold_results]
    train_mae_scores = [f['train_mae'] for f in fold_results]
    r2_overfit_gaps = [f['r2_overfit_gap'] for f in fold_results]
    mae_overfit_gaps = [f['mae_overfit_gap'] for f in fold_results]
    inner_outer_gaps = [f['inner_outer_gap'] for f in fold_results]

    # Reorder predictions to match original input order
    y_pred_ordered = np.empty(len(all_y_true))
    y_true_ordered = np.empty(len(all_y_true))
    for i, (idx, pred, true) in enumerate(zip(all_test_indices, all_y_pred, all_y_true)):
        y_pred_ordered[idx] = pred
        y_true_ordered[idx] = true

    return {
        'fold_results': fold_results,
        # Test scores
        'r2_scores': r2_scores,
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'mae_scores': mae_scores,
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        # Train scores
        'train_r2_mean': np.mean(train_r2_scores),
        'train_r2_std': np.std(train_r2_scores),
        'train_mae_mean': np.mean(train_mae_scores),
        'train_mae_std': np.std(train_mae_scores),
        # Overfitting diagnostics
        'r2_overfit_gap_mean': np.mean(r2_overfit_gaps),
        'r2_overfit_gap_std': np.std(r2_overfit_gaps),
        'mae_overfit_gap_mean': np.mean(mae_overfit_gaps),
        'mae_overfit_gap_std': np.std(mae_overfit_gaps),
        'inner_outer_gap_mean': np.mean(inner_outer_gaps),
        'inner_outer_gap_std': np.std(inner_outer_gaps),
        # Overall metrics from all predictions combined
        'overall_r2': r2_score(all_y_true, all_y_pred),
        'overall_mae': mean_absolute_error(all_y_true, all_y_pred),
        # Ordered predictions
        'y_pred_ordered': y_pred_ordered,
        'y_true_ordered': y_true_ordered,
    }


def save_nested_cv_predictions(df, cv_results, output_path):
    """Save nested CV predictions to CSV with filename, makespan, and predicted_makespan columns.

    Args:
        df: DataFrame with 'filename' column (full dataset, same order as X used in nested_cv_evaluation)
        cv_results: Results dict returned from nested_cv_evaluation (contains 'y_pred_ordered')
        output_path: Path to save CSV file
    """
    if len(df) != len(cv_results['y_pred_ordered']):
        raise ValueError(f"DataFrame length ({len(df)}) doesn't match predictions length ({len(cv_results['y_pred_ordered'])})")

    df_copy = df[['filename', 'makespan']].copy()
    df_copy['predicted_makespan'] = cv_results['y_pred_ordered']
    df_copy.to_csv(output_path, index=False)
    print(f'Saved nested CV predictions for {len(df_copy)} samples to {output_path}')


def print_nested_cv_results(model_name, results):
    print("=" * 70)
    print(f"{model_name} - Nested Cross-Validation Results")
    print("=" * 70)

    print("\nPer-Fold Results:")
    print("-" * 70)
    print(f"  {'Fold':<6} {'Test R²':<10} {'Train R²':<10} {'R² Gap':<10} {'Test MAE':<10} {'Train MAE':<10}")
    print("-" * 70)
    for fold in results['fold_results']:
        print(f"  {fold['fold']:<6} {fold['r2']:<10.3f} {fold['train_r2']:<10.3f} "
              f"{fold['r2_overfit_gap']:<10.3f} {fold['mae']:<10.1f} {fold['train_mae']:<10.1f}")

    print("\nAggregated Test Scores (mean ± std):")
    print("-" * 50)
    print(f"  R²:  {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"  MAE: {results['mae_mean']:.2f} ± {results['mae_std']:.2f} seconds")

    print("\nAggregated Train Scores (mean ± std):")
    print("-" * 50)
    print(f"  R²:  {results['train_r2_mean']:.4f} ± {results['train_r2_std']:.4f}")
    print(f"  MAE: {results['train_mae_mean']:.2f} ± {results['train_mae_std']:.2f} seconds")

    print("\n" + "=" * 70)
    print("OVERFITTING DIAGNOSTICS")
    print("=" * 70)
    print(f"  R² Gap (Train - Test):      {results['r2_overfit_gap_mean']:.4f} ± {results['r2_overfit_gap_std']:.4f}")
    print(f"  MAE Gap (Test - Train):     {results['mae_overfit_gap_mean']:.2f} ± {results['mae_overfit_gap_std']:.2f} seconds")
    print(f"  Inner-Outer Gap (Test MAE - Inner CV MAE): {results['inner_outer_gap_mean']:.2f} ± {results['inner_outer_gap_std']:.2f} seconds")
    print(f" Fold variance (R² std):        {results['r2_std']:.4f}")

    print("\nOverall (all predictions pooled):")
    print("-" * 50)
    print(f"  R²:  {results['overall_r2']:.4f}")
    print(f"  MAE: {results['overall_mae']:.2f} seconds")

    print("\nFold Parameters:")
    for fold in results['fold_results']:
        params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in fold['best_params'].items())
        print(f"  Fold {fold['fold']}: {params_str}")

    print("=" * 70)


def print_results(model_name, results):
    print("=" * 70)
    print(f"{model_name} Results")
    print("=" * 70)
    print(f"Best Parameters: {results['best_params']}")
    print()
    print("Train Set:")
    print(f"  R²:  {results['train_r2']:.4f}")
    print(f"  MAE: {results['train_mae']:.2f} seconds")
    print(f"  MSE: {results['train_mse']:.2f}")
    print()
    print("Test Set:")
    print(f"  R²:  {results['test_r2']:.4f}")
    print(f"  MAE: {results['test_mae']:.2f} seconds")
    print(f"  MSE: {results['test_mse']:.2f}")
    print()
    print("Cross-Validation:")
    print(f"  R²:  {results['cv_r2']:.4f}")
    print(f"  MAE: {results['cv_mae']:.2f} seconds")
    print(f"  MSE: {results['cv_mse']:.2f}")
    print("=" * 70)
    print()


def log_experiment_run(
    selected_features,
    pipeline_stats,
    model_results_map,
    random_seed,
    output_path=constants.EXPERIMENT_LOG_CSV,
):
    """Append one row to the cumulative experiment log CSV.

    Auto-increments run_id based on existing rows. Creates the file with a header
    row on the first call.

    Args:
        selected_features: Final feature names from the selection pipeline.
        pipeline_stats: Dict with keys initial, after_variance, after_corr_filter,
            after_target_corr, after_stability, after_clustering, after_rfecv.
        model_results_map: Dict keyed by model key string (e.g. 'ridge', 'gpr')
            mapping to the results dict returned by nested_cv_evaluation().
        random_seed: The numpy random seed used for this run.
        output_path: Path to the experiment log CSV file.
    """
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            existing_df = pd.read_csv(output_path)
            run_id = int(existing_df['run_id'].max()) + 1
        except (pd.errors.ParserError, pd.errors.EmptyDataError, KeyError):
            run_id = 1
            file_exists = False
    else:
        run_id = 1

    ridge_results = model_results_map['ridge']
    gpr_results = model_results_map['gpr']
    bayesian_ridge_results = model_results_map['bayesian_ridge']
    ard_results = model_results_map['ard']
    gpr_matern25_results = model_results_map['gpr_matern25']
    gpr_rbf_results = model_results_map['gpr_rbf']

    required_keys = [
        'r2_mean', 'r2_std', 'mae_mean', 'mae_std',
        'overall_r2', 'overall_mae', 'train_r2_mean',
        'r2_overfit_gap_mean', 'inner_outer_gap_mean',
    ]
    for key in required_keys:
        if key not in ridge_results:
            raise KeyError(f"Missing key '{key}' in model_results_map['ridge']")
        if key not in ard_results:
            raise KeyError(f"Missing key '{key}' in model_results_map['ard']")

    # Determine best model across all evaluated models
    candidates = [
        ('ridge', ridge_results['overall_r2']),
        ('gpr', gpr_results['overall_r2']),
        ('bayesian_ridge', bayesian_ridge_results['overall_r2']),
        ('ard', ard_results['overall_r2']),
        ('gpr_matern25', gpr_matern25_results['overall_r2']),
        ('gpr_rbf', gpr_rbf_results['overall_r2']),
    ]

    best_model, best_r2 = max(candidates, key=lambda x: x[1])
    if best_r2 < 0:
        best_model = 'neither'

    row = {
        # -- Identity & reproducibility --
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'random_seed': random_seed,

        # -- Pipeline stage counts --
        'n_initial': pipeline_stats['initial'],
        'n_after_variance': pipeline_stats['after_variance'],
        'n_after_corr_filter': pipeline_stats['after_corr_filter'],
        'n_after_target_corr': pipeline_stats['after_target_corr'],
        'n_after_stability': pipeline_stats['after_stability'],
        'n_after_clustering': pipeline_stats['after_clustering'],
        'n_after_rfecv': pipeline_stats.get('after_rfecv'),

        # -- Final feature set --
        'n_selected_features': len(selected_features),
        'selected_features': json.dumps(selected_features),

        # -- Ridge nested-CV metrics --
        'ridge_r2_mean': ridge_results['r2_mean'],
        'ridge_r2_std': ridge_results['r2_std'],
        'ridge_mae_mean': ridge_results['mae_mean'],
        'ridge_mae_std': ridge_results['mae_std'],
        'ridge_overall_r2': ridge_results['overall_r2'],
        'ridge_overall_mae': ridge_results['overall_mae'],
        'ridge_train_r2_mean': ridge_results['train_r2_mean'],
        'ridge_r2_overfit_gap_mean': ridge_results['r2_overfit_gap_mean'],
        'ridge_inner_outer_gap_mean': ridge_results['inner_outer_gap_mean'],

        # -- GPR (Matérn 1.5) nested-CV metrics --
        'gpr_r2_mean': gpr_results['r2_mean'],
        'gpr_r2_std': gpr_results['r2_std'],
        'gpr_mae_mean': gpr_results['mae_mean'],
        'gpr_mae_std': gpr_results['mae_std'],
        'gpr_overall_r2': gpr_results['overall_r2'],
        'gpr_overall_mae': gpr_results['overall_mae'],
        'gpr_train_r2_mean': gpr_results['train_r2_mean'],
        'gpr_r2_overfit_gap_mean': gpr_results['r2_overfit_gap_mean'],

        # -- Bayesian Ridge nested-CV metrics --
        'bayesian_ridge_r2_mean': bayesian_ridge_results['r2_mean'],
        'bayesian_ridge_r2_std': bayesian_ridge_results['r2_std'],
        'bayesian_ridge_mae_mean': bayesian_ridge_results['mae_mean'],
        'bayesian_ridge_mae_std': bayesian_ridge_results['mae_std'],
        'bayesian_ridge_overall_r2': bayesian_ridge_results['overall_r2'],
        'bayesian_ridge_overall_mae': bayesian_ridge_results['overall_mae'],
        'bayesian_ridge_train_r2_mean': bayesian_ridge_results['train_r2_mean'],
        'bayesian_ridge_r2_overfit_gap_mean': bayesian_ridge_results['r2_overfit_gap_mean'],

        # -- ARD Regression nested-CV metrics --
        'ard_r2_mean': ard_results['r2_mean'],
        'ard_r2_std': ard_results['r2_std'],
        'ard_mae_mean': ard_results['mae_mean'],
        'ard_mae_std': ard_results['mae_std'],
        'ard_overall_r2': ard_results['overall_r2'],
        'ard_overall_mae': ard_results['overall_mae'],
        'ard_train_r2_mean': ard_results['train_r2_mean'],
        'ard_r2_overfit_gap_mean': ard_results['r2_overfit_gap_mean'],

        # -- GPR (Matérn 2.5) nested-CV metrics --
        'gpr_matern25_r2_mean': gpr_matern25_results['r2_mean'],
        'gpr_matern25_r2_std': gpr_matern25_results['r2_std'],
        'gpr_matern25_mae_mean': gpr_matern25_results['mae_mean'],
        'gpr_matern25_mae_std': gpr_matern25_results['mae_std'],
        'gpr_matern25_overall_r2': gpr_matern25_results['overall_r2'],
        'gpr_matern25_overall_mae': gpr_matern25_results['overall_mae'],
        'gpr_matern25_train_r2_mean': gpr_matern25_results['train_r2_mean'],
        'gpr_matern25_r2_overfit_gap_mean': gpr_matern25_results['r2_overfit_gap_mean'],

        # -- GPR (RBF) nested-CV metrics --
        'gpr_rbf_r2_mean': gpr_rbf_results['r2_mean'],
        'gpr_rbf_r2_std': gpr_rbf_results['r2_std'],
        'gpr_rbf_mae_mean': gpr_rbf_results['mae_mean'],
        'gpr_rbf_mae_std': gpr_rbf_results['mae_std'],
        'gpr_rbf_overall_r2': gpr_rbf_results['overall_r2'],
        'gpr_rbf_overall_mae': gpr_rbf_results['overall_mae'],
        'gpr_rbf_train_r2_mean': gpr_rbf_results['train_r2_mean'],
        'gpr_rbf_r2_overfit_gap_mean': gpr_rbf_results['r2_overfit_gap_mean'],

        # -- Best model indicator --
        'best_model': best_model,
        'best_overall_r2': best_r2,
    }

    row_df = pd.DataFrame([row])

    if file_exists:
        combined = pd.concat([existing_df, row_df], ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        row_df.to_csv(output_path, index=False)

    print(f'\nExperiment run #{run_id} logged to {output_path}')


def log_feature_selection_run(
    selected_features: list[str],
    pipeline_stats: dict[str, int],
    random_seed: int,
    output_path: str = constants.EXPERIMENT_LOG_CSV,
) -> None:
    """Append a feature-selection-only row to the experiment log CSV.

    Logs run identity, pipeline stage counts, and selected features.
    No model metrics are written. Compatible with rows from log_experiment_run()
    — model metric columns will be NaN for these rows when read back.

    Args:
        selected_features: Final feature names from the selection pipeline.
        pipeline_stats: Dict with keys initial, after_variance, after_corr_filter,
            after_target_corr, after_stability, after_clustering.
        random_seed: The numpy random seed used for this run.
        output_path: Path to the experiment log CSV file.
    """
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            existing_df = pd.read_csv(output_path)
            run_id = int(existing_df['run_id'].max()) + 1
        except (pd.errors.ParserError, pd.errors.EmptyDataError, KeyError):
            run_id = 1
            file_exists = False
    else:
        run_id = 1

    row = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'random_seed': random_seed,
        'n_initial': pipeline_stats['initial'],
        'n_after_variance': pipeline_stats['after_variance'],
        'n_after_corr_filter': pipeline_stats['after_corr_filter'],
        'n_after_target_corr': pipeline_stats['after_target_corr'],
        'n_after_stability': pipeline_stats['after_stability'],
        'n_after_clustering': pipeline_stats['after_clustering'],
        'n_selected_features': len(selected_features),
        'selected_features': json.dumps(selected_features),
    }

    row_df = pd.DataFrame([row])

    if file_exists:
        combined = pd.concat([existing_df, row_df], ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        row_df.to_csv(output_path, index=False)

    print(f'\nFeature selection run #{run_id} logged to {output_path}')
