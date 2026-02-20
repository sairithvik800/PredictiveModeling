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
}

MANUAL_PREDICTION_OUTPUTS = {
    'ridge':          constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge_manual.csv',
    'gpr':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_manual.csv',
    'gpr_matern25':   constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_matern25_manual.csv',
    'gpr_rbf':        constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_rbf_manual.csv',
    'bayesian_ridge': constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge_manual.csv',
    'ard':            constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ard_manual.csv',
}


def run_with_feature_selection(random_seed=None, output_path=constants.EXPERIMENT_LOG_CSV):
    """Main execution: run feature selection pipeline then evaluate models.

    Args:
        random_seed: Numpy random seed for reproducibility. If None, a random
            seed is generated and printed so the run can be replayed.
        output_path: Path to the experiment log CSV file. Defaults to
            constants.EXPERIMENT_LOG_CSV.
    """
    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(random_seed)
    print(f"Random seed: {random_seed}")

    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")

    selected_features, stats = feature_selection_pipeline.full_feature_selection_pipeline(df, target_col='makespan')

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


def run_with_manual_features():
    """Run nested CV with a manually specified SELECTED_FEATURES list.

    Evaluates Ridge, GPR (Matérn 1.5 and 2.5, RBF), Bayesian Ridge, and ARD
    using the same nested CV as the stochastic pipeline, but with a fixed
    feature set to eliminate Bootstrap Lasso stochasticity.
    """
    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    columns_to_keep = ['filename', 'makespan'] + SELECTED_FEATURES
    df = df[columns_to_keep]

    X_all = df.drop(['filename', 'makespan'], axis=1).values
    y_all = df['makespan'].values

    print(f"Using {len(SELECTED_FEATURES)} manually selected features")
    print(f"Features: {SELECTED_FEATURES}")

    model_results = []
    for model in DEFAULT_MODELS:
        pipeline, param_grid = model.get_pipeline_and_params()
        print(f"\n{'=' * 70}")
        print(f"Now running {model.name}...")
        print(f"{'=' * 70}\n")
        results = model_eval_pipeline.nested_cv_evaluation(pipeline, param_grid, X_all, y_all)
        model_eval_pipeline.print_nested_cv_results(model.name, results)
        model_eval_pipeline.save_nested_cv_predictions(df, results, MANUAL_PREDICTION_OUTPUTS[model.key])
        model_results.append((model.name, results))

    print("\n" + "=" * 70)
    print("MANUAL FEATURES — SUMMARY")
    print("=" * 70)
    for name, res in model_results:
        print(f"  {name:<30} overall_r2={res['overall_r2']:.4f}  mae={res['overall_mae']:.1f}s")


if __name__ == "__main__":
    SELECTED_FEATURES = [
    # 100% frequency
    'ngtdm_complexity',
    'bias_field_proxy',
    'morpho_intensity_kurtosis',
    '7Networks_RH_Default_PFCdPFCm_2_med_abs_dev',
    '7Networks_LH_Default_PFC_1_range',
    'load_time_ms',
    'synthseg_total_components',
    'synthseg_fragmentation_index',
    # 85–95% frequency
    'tile_gradient_load_imbalance',
    '7Networks_LH_SalVentAttn_PFCl_1_kurtosis',
    # 65% frequency
    '7Networks_LH_Default_PFC_1_kurtosis',
    'inferior_brain_offset',
    ]
    
    run_with_manual_features()
    
    """
    for i in range(100):
        run_with_feature_selection(output_path=constants.EXPERIMENT_LOG_CSV)
    """