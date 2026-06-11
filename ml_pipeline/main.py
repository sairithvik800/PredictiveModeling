import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import constants
import feature_selection_pipeline
import model_eval_pipeline
from models import DEFAULT_MODELS

FEATURE_SELECTION_PREDICTION_OUTPUTS = {
    'ridge':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge.csv',
    'bayesian_ridge': constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge.csv',
    'huber':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_huber.csv',
    'svr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_svr.csv',
    'random_forest':  constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_random_forest.csv',
    'xgboost':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_xgboost.csv',
}

MANUAL_PREDICTION_OUTPUTS = {
    'ridge':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge_manual.csv',
    'bayesian_ridge': constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge_manual.csv',
    'huber':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_huber_manual.csv',
    'svr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_svr_manual.csv',
    'random_forest':  constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_random_forest_manual.csv',
    'xgboost':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_xgboost_manual.csv',
}

FEATURES_96 = [                                            # frequecy % based off feature selection pipeline
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
    'asymmetry_hemispheric_volume_ratio',                  #  81.8%
    '7Networks_LH_SalVentAttn_PFCl_1_kurtosis',           #  76.0%
    'tile_gradient_load_imbalance',                        #  69.6%
    '7Networks_LH_Default_PFC_1_kurtosis',                 #  68.4%
    'registration_scale_difference',                       #  56.4%
    '7Networks_LH_Limbic_TempPole_1_moment_5',             #  50.0%
]

FEATURES_24 = FEATURES_50 + [
    '7Networks_RH_SomMot_2_kurtosis',                     #  42.4%
    '7Networks_RH_Cont_PFCl_3_kurtosis',                  #  36.4%
    '7Networks_RH_DorsAttn_Post_3_kurtosis',               #  24.2%
]

DEFAULT_FEATURES = FEATURES_24


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
            model_eval_pipeline.save_nested_cv_predictions(df, results, FEATURE_SELECTION_PREDICTION_OUTPUTS[model.key])
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
            DEFAULT_FEATURES (the >=24% consensus set, n=18).
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
        

if __name__ == "__main__":
    run_with_manual_features()

    """
    for i in range(500):
        run_with_feature_selection(output_path=constants.EXPERIMENT_LOG_CSV)
    """