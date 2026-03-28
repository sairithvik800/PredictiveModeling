import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import constants
import feature_selection_pipeline
import model_eval_pipeline
from models import DEFAULT_MODELS


# For feature selection runs
PREDICTION_OUTPUTS = {
    'ridge':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge.csv',
    'gpr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr.csv',
    'gpr_matern25':   constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_matern25.csv',
    'gpr_rbf':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_rbf.csv',
    'bayesian_ridge': constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge.csv',
    'ard':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ard.csv',
    'huber':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_huber.csv',
    'svr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_svr.csv',
    'elastic_net':    constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_elastic_net.csv',
    'rvm':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_rvm.csv',
    'kernel_ridge':   constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_kernel_ridge.csv',
    'hist_gb':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_hist_gb.csv',
    'random_forest':  constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_random_forest.csv',
}

MANUAL_PREDICTION_OUTPUTS = {
    'ridge':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge_manual.csv',
    'gpr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_manual.csv',
    'gpr_matern25':   constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_matern25_manual.csv',
    'gpr_rbf':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_rbf_manual.csv',
    'bayesian_ridge': constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge_manual.csv',
    'ard':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ard_manual.csv',
    'huber':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_huber_manual.csv',
    'svr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_svr_manual.csv',
    'elastic_net':    constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_elastic_net_manual.csv',
    'rvm':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_rvm_manual.csv',
    'kernel_ridge':   constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_kernel_ridge_manual.csv',
    'hist_gb':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_hist_gb_manual.csv',
    'random_forest':  constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_random_forest_manual.csv',
}


# Features selected by frequency threshold from 500 stochastic pipeline runs.
# Each tier is a strict subset of the next: FEATURES_96 ⊂ FEATURES_50 ⊂ FEATURES_15.
# Used for sensitivity analysis and as the primary feature set (FEATURES_15 = DEFAULT_FEATURES).
# Tiers reflect natural breaks in the frequency distribution:
#   >=96%: sharpest break in the distribution (96.2% -> 81.8%, 14pp drop)
#   >=50%: top of the gradual-decline plateau
#   >=15%: natural cliff into noise (24.2% -> 14.0%, 10pp drop)

FEATURES_96 = [
    # >=96% selection frequency — "virtually always selected"
    'load_time_ms',                                        # 100.0%
    'synthseg_total_components',                           # 100.0%
    'synthseg_fragmentation_index',                        # 100.0%
    'inferior_brain_offset',                               # 100.0%
    '7Networks_LH_Default_PFC_1_range',                   # 100.0%
    'ngtdm_complexity',                                    # 100.0%
    '7Networks_RH_Default_PFCdPFCm_2_med_abs_dev',        #  99.8%
    'bias_field_proxy',                                    #  99.8%
    'morpho_intensity_kurtosis',                           #  96.2%
]

FEATURES_50 = FEATURES_96 + [
    # >=50% selection frequency
    'asymmetry_hemispheric_volume_ratio',                  #  81.8%
    '7Networks_LH_SalVentAttn_PFCl_1_kurtosis',           #  76.0%
    'tile_gradient_load_imbalance',                        #  69.6%
    '7Networks_LH_Default_PFC_1_kurtosis',                 #  68.4%
    'registration_scale_difference',                       #  56.4%
    '7Networks_LH_Limbic_TempPole_1_moment_5',             #  50.0%
]

FEATURES_15 = FEATURES_50 + [
    # >=15% selection frequency (natural cliff at 24.2% -> 14.0% terminates the set)
    '7Networks_RH_SomMot_2_kurtosis',                     #  42.4%
    '7Networks_RH_Cont_PFCl_3_kurtosis',                  #  36.4%
    '7Networks_RH_DorsAttn_Post_3_kurtosis',               #  24.2%
]

# Primary feature set for paper results (>=15% threshold, n=18).
DEFAULT_FEATURES = FEATURES_15


def run_with_feature_selection(
    random_seed: int | None = None,
    output_path: str = constants.EXPERIMENT_LOG_CSV,
    evaluate_models: bool = False,
) -> tuple[list[str], dict | None, int]:
    """Run feature selection pipeline, optionally followed by model evaluation.

    Args:
        random_seed: Numpy random seed for reproducibility. If None, a random
            seed is generated and printed so the run can be replayed.
        output_path: Path to the experiment log CSV file. Defaults to
            constants.EXPERIMENT_LOG_CSV.
        evaluate_models: If False (default), only run feature selection and log
            selected features — no model evaluation. Set to True to also run
            nested CV for all models and log full metrics (note: those metrics
            have data leakage and should not be cited as paper results).

    Returns:
        Tuple of (selected_features, model_results_map, random_seed).
        model_results_map is None when evaluate_models=False.
    """
    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(random_seed)
    print(f"Random seed: {random_seed}")

    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")

    selected_features, stats = feature_selection_pipeline.full_feature_selection_pipeline(df, target_col='makespan')

    if evaluate_models:
        X_all = df[selected_features].values
        y_all = df['makespan'].values

        print("\n" + "=" * 70)
        print("MODEL EVALUATION WITH SELECTED FEATURES")
        print("=" * 70 + "\n")

        model_results_map = {}
        for model in DEFAULT_MODELS:
            pipeline, param_grid = model.get_pipeline_and_params()
            print(f"\n{'=' * 70}")
            print(f"Now running {model.name}...")
            print(f"{'=' * 70}\n")
            results = model_eval_pipeline.nested_cv_evaluation(pipeline, param_grid, X_all, y_all)
            model_eval_pipeline.print_nested_cv_results(model.name, results)
            model_eval_pipeline.save_nested_cv_predictions(df, results, PREDICTION_OUTPUTS[model.key])
            model_results_map[model.key] = results

        model_eval_pipeline.log_experiment_run(
            selected_features=selected_features,
            pipeline_stats=stats,
            model_results_map=model_results_map,
            random_seed=random_seed,
            output_path=output_path,
        )
        return selected_features, model_results_map, random_seed
    else:
        model_eval_pipeline.log_feature_selection_run(
            selected_features=selected_features,
            pipeline_stats=stats,
            random_seed=random_seed,
            output_path=output_path,
        )
        return selected_features, None, random_seed


def run_with_manual_features(features=None):
    """Run nested CV with a fixed feature set applied uniformly to all models.

    Produces the paper's reported performance numbers. Feature selection
    stochasticity is eliminated by using a pre-determined fixed set.

    Args:
        features: Feature list to use for all models. If None, uses
            DEFAULT_FEATURES (the >=15% consensus set, n=21).
    """
    if features is None:
        features = DEFAULT_FEATURES

    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    X_all = df[features].values
    y_all = df['makespan'].values

    model_results = []
    for model in DEFAULT_MODELS:
        pipeline, param_grid = model.get_pipeline_and_params()
        print(f"\n{'=' * 70}")
        print(f"Now running {model.name} with {len(features)} features...")
        print(f"{'=' * 70}\n")
        results = model_eval_pipeline.nested_cv_evaluation(pipeline, param_grid, X_all, y_all)
        model_eval_pipeline.print_nested_cv_results(model.name, results)
        model_eval_pipeline.save_nested_cv_predictions(df, results, MANUAL_PREDICTION_OUTPUTS[model.key])
        model_results.append((model.name, results))

    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY  (n_features={len(features)})")
    print("=" * 70)
    for name, res in model_results:
        print(f"  {name:<30} overall_r2={res['overall_r2']:.4f}  mae={res['overall_mae']:.1f}s")


def run_threshold_sensitivity():
    """Sensitivity analysis: evaluate all models at >=50%, >=30%, and >=15% thresholds.

    Intended for the paper's sensitivity table. Results are printed but not
    saved to the main prediction CSVs so they do not overwrite primary results.
    """
    thresholds = [
        ('>=96%',  FEATURES_96),
        ('>=50%',  FEATURES_50),
        ('>=15%',  FEATURES_15),
    ]

    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    y_all = df['makespan'].values

    # results[label][model_key] = overall_r2, overall_mae
    all_results = {}
    for label, features in thresholds:
        print(f"\n{'=' * 70}")
        print(f"THRESHOLD {label}  ({len(features)} features)")
        print(f"{'=' * 70}")
        X_all = df[features].values
        threshold_results = {}
        for model in DEFAULT_MODELS:
            pipeline, param_grid = model.get_pipeline_and_params()
            print(f"  {model.name}...", flush=True)
            cv_results = model_eval_pipeline.nested_cv_evaluation(pipeline, param_grid, X_all, y_all)
            threshold_results[model.key] = cv_results
        all_results[label] = threshold_results

    # Print comparison table
    col_w = 13
    print("\n" + "=" * (28 + len(thresholds) * col_w * 2))
    print("SENSITIVITY ANALYSIS — FREQUENCY THRESHOLD COMPARISON")
    print("=" * (28 + len(thresholds) * col_w * 2))

    header = f"{'Model':<28}"
    for label, features in thresholds:
        header += f"  {(label + ' R2'):>{col_w}}  {'MAE':>{col_w - 2}}"
    print(header)
    print("-" * (28 + len(thresholds) * col_w * 2))

    for model in DEFAULT_MODELS:
        row = f"{model.name:<28}"
        for label, _ in thresholds:
            r2  = all_results[label][model.key]['overall_r2']
            mae = all_results[label][model.key]['overall_mae']
            row += f"  {r2:>{col_w}.4f}  {mae:>{col_w - 2}.1f}s"
        print(row)

    print("=" * (28 + len(thresholds) * col_w * 2))
    n_row = f"{'n features':<28}"
    for _, features in thresholds:
        n_row += f"  {'n='+str(len(features)):>{col_w}}  {'':>{col_w - 2}}"
    print(n_row)
    print("=" * (28 + len(thresholds) * col_w * 2))

    return all_results


if __name__ == "__main__":
    #run_with_manual_features()

    run_threshold_sensitivity()
    """
    for i in range(500):
        run_with_feature_selection(output_path=constants.EXPERIMENT_LOG_CSV)
    """