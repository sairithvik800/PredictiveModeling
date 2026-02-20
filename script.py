import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, LassoCV, ARDRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.feature_selection import RFECV
from sklearn.base import clone
from xgboost import XGBRegressor
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import constants


def remove_low_variance_features(df, feature_cols, threshold=0.01):
    """Remove features with variance below a certain threshold

    Args:
        df (pandas.DataFrame): features dataframe
        feature_cols (list of str): list of feature column names
        threshold (float, optional): minimum variance threshold. Defaults to 0.01.

    Returns:
        list of str: list of feature column names that passed the filter
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    variances = np.var(X_scaled, axis=0)

    mask = variances > threshold
    kept_features = [f for f, keep in zip(feature_cols, mask) if keep]
    removed_count = len(feature_cols) - len(kept_features)

    print(f"  Low variance filter: {len(feature_cols)} -> {len(kept_features)} features (removed {removed_count})")
    return kept_features


def remove_highly_correlated_features(df, feature_cols, target_col, threshold=0.90):
    """Remove redundant features that are highly correlated with each other.
    When two features correlate above threshold, keep the one with higher target correlation.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Compute target correlations
    target_corrs = np.array([abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])

    # Find pairs above threshold
    to_remove = set()
    n_features = len(feature_cols)

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Remove the one with lower target correlation
                if target_corrs[i] >= target_corrs[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i is removed, move to next i

    kept_features = [f for idx, f in enumerate(feature_cols) if idx not in to_remove]
    print(f"  High correlation filter (>{threshold}): {len(feature_cols)} -> {len(kept_features)} features (removed {len(to_remove)})")
    return kept_features


def filter_by_target_correlation(df, feature_cols, target_col, min_corr=0.15):
    """Keep features with absolute correlation to target above threshold.
    Uses both Pearson and Spearman, keeping features that pass either.
    """
    y = df[target_col].values
    kept_features = []

    for feat in feature_cols:
        x = df[feat].values
        # Handle NaN
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() < 10:
            continue

        x_valid, y_valid = x[valid_mask], y[valid_mask]

        pearson_r = abs(pearsonr(x_valid, y_valid)[0])
        spearman_r = abs(spearmanr(x_valid, y_valid)[0])

        if pearson_r >= min_corr or spearman_r >= min_corr:
            kept_features.append(feat)

    print(f"  Target correlation filter (|r|>{min_corr}): {len(feature_cols)} -> {len(kept_features)} features")
    return kept_features


def stability_selection_lasso(df, feature_cols, target_col, n_bootstraps=100, selection_threshold=0.5):
    """Bootstrap-based stability selection using LassoCV.
    Features must be selected in >selection_threshold fraction of bootstraps.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    n_samples = len(y)

    # Track selection frequency
    selection_counts = np.zeros(len(feature_cols))

    print(f"  Running {n_bootstraps} bootstrap iterations...")

    for i in range(n_bootstraps):
        # Bootstrap sample (80% of data)
        indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_boot)

        # Fit LassoCV
        lasso = LassoCV(cv=3, random_state=i, max_iter=5000, n_alphas=50)
        lasso.fit(X_scaled, y_boot)

        # Track which features have non-zero coefficients
        selected = np.abs(lasso.coef_) > 1e-10
        selection_counts += selected

        if (i + 1) % 25 == 0:
            print(f"    Bootstrap {i + 1}/{n_bootstraps} complete")

    # Compute selection frequencies
    selection_freq = selection_counts / n_bootstraps

    # Keep features selected more than threshold fraction of time
    kept_indices = np.where(selection_freq >= selection_threshold)[0]
    kept_features = [feature_cols[i] for i in kept_indices]

    # Print top stable features
    top_indices = np.argsort(selection_freq)[::-1][:10]
    print(f"\n  Top 10 most stable features:")
    for idx in top_indices:
        print(f"    {feature_cols[idx]}: {selection_freq[idx]:.2%}")

    print(f"\n  Stability selection (>{selection_threshold:.0%}): {len(feature_cols)} -> {len(kept_features)} features")
    return kept_features, selection_freq


def cluster_and_select_features(df, feature_cols, target_col, n_clusters=None, corr_threshold=0.7):
    """Cluster correlated features and select one representative per cluster.
    Uses hierarchical clustering based on correlation distance.
    Selects the feature with highest target correlation from each cluster.
    """
    if len(feature_cols) <= 5:
        print(f"  Clustering skipped: only {len(feature_cols)} features")
        return feature_cols

    X = df[feature_cols].values
    y = df[target_col].values

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Convert to distance matrix (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry and handle numerical issues
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.clip(distance_matrix, 0, 2)

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Determine number of clusters if not specified
    if n_clusters is None:
        # Cut tree at distance corresponding to corr_threshold
        cut_distance = 1 - corr_threshold
        cluster_labels = fcluster(linkage_matrix, t=cut_distance, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
    else:
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    # Compute target correlations
    target_corrs = {feat: abs(pearsonr(df[feat].values, y)[0]) for feat in feature_cols}

    # Select best feature from each cluster
    selected_features = []
    for cluster_id in np.unique(cluster_labels):
        cluster_features = [f for f, c in zip(feature_cols, cluster_labels) if c == cluster_id]
        best_feature = max(cluster_features, key=lambda f: target_corrs[f])
        selected_features.append(best_feature)

    print(f"  Clustering: {len(feature_cols)} features -> {n_clusters} clusters -> {len(selected_features)} representatives")
    return selected_features


def rfecv_selection(df, feature_cols, target_col, min_features=5, cv=5):
    """Recursive Feature Elimination with Cross-Validation.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use simple RandomForest for stability
    estimator = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1
    )

    # RFECV
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='r2',
        min_features_to_select=min_features,
        n_jobs=1
    )

    rfecv.fit(X_scaled, y)

    # Get selected features
    selected_mask = rfecv.support_
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]

    print(f"  RFECV: {len(feature_cols)} -> {len(selected_features)} features (optimal by CV)")
    print(f"  CV scores by n_features: min={min(rfecv.cv_results_['mean_test_score']):.3f}, max={max(rfecv.cv_results_['mean_test_score']):.3f}")

    return selected_features, rfecv


def full_feature_selection_pipeline(df, target_col='makespan'):
    """Run complete multi-stage feature selection pipeline.

    Returns:
        selected_features: list of selected feature names
        pipeline_stats: dict with statistics from each stage
    """
    # Get all feature columns (exclude filename and target)
    all_features = [c for c in df.columns if c not in ['filename', target_col, 'bin']]

    print("=" * 70)
    print("MULTI-STAGE FEATURE SELECTION PIPELINE")
    print("=" * 70)
    print(f"Starting with {len(all_features)} features, {len(df)} samples")
    print()

    pipeline_stats = {'initial': len(all_features)}

    # Stage 1: Variance and correlation pre-filtering
    print("STAGE 1: Pre-filtering")
    print("-" * 40)
    features = remove_low_variance_features(df, all_features, threshold=0.01)
    pipeline_stats['after_variance'] = len(features)

    features = remove_highly_correlated_features(df, features, target_col, threshold=0.90)
    pipeline_stats['after_corr_filter'] = len(features)
    print()

    # Stage 2: Target correlation filtering
    print("STAGE 2: Target Correlation Filtering")
    print("-" * 40)
    features = filter_by_target_correlation(df, features, target_col, min_corr=0.15)
    pipeline_stats['after_target_corr'] = len(features)
    print()

    # Stage 3: Stability selection
    print("STAGE 3: Stability Selection (Bootstrapped Lasso)")
    print("-" * 40)
    features, stability_scores = stability_selection_lasso(
        df, features, target_col,
        n_bootstraps=100,
        selection_threshold=0.30
    )
    pipeline_stats['after_stability'] = len(features)
    print()

    # Stage 4: Clustering
    print("STAGE 4: Hierarchical Clustering")
    print("-" * 40)
    if len(features) > 5:
        features = cluster_and_select_features(df, features, target_col, corr_threshold=0.7)
    pipeline_stats['after_clustering'] = len(features)
    print()

    # Stage 5: RFECV
    print("STAGE 5: RFECV Refinement")
    print("-" * 40)
    if len(features) > 10:
        features, rfecv_model = rfecv_selection(df, features, target_col, min_features=5)
        pipeline_stats['after_rfecv'] = len(features)
    else:
        print(f"  RFECV skipped: only {len(features)} features remaining")
        pipeline_stats['after_rfecv'] = len(features)
    print()

    # Summary
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Initial features:        {pipeline_stats['initial']}")
    print(f"After variance filter:   {pipeline_stats['after_variance']}")
    print(f"After correlation filter:{pipeline_stats['after_corr_filter']}")
    print(f"After target corr:       {pipeline_stats['after_target_corr']}")
    print(f"After stability:         {pipeline_stats['after_stability']}")
    print(f"After clustering:        {pipeline_stats['after_clustering']}")
    print(f"After RFECV:             {pipeline_stats['after_rfecv']}")
    print()
    print("FINAL SELECTED FEATURES:")
    for i, feat in enumerate(features, 1):
        corr = pearsonr(df[feat].values, df[target_col].values)[0]
        print(f"  {i:2d}. {feat}: r={corr:.3f}")
    print("=" * 70)

    return features, pipeline_stats


def load_data_with_selected_features(csv_path, features):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Total features in CSV: {len(df.columns) - 2}")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"The following features are not in the CSV: {missing_features}")
    print(f"Selected features: {len(features)}")

    columns_to_keep = ['filename', 'makespan'] + features
    df = df[columns_to_keep]

    df['bin'] = pd.qcut(df['makespan'], q=5, labels=False)
    train_df, test_df = train_test_split(
        df,
        test_size=0.252,
        random_state=42,
        stratify=df['bin']
    )

    train_df = train_df.drop('bin', axis=1)
    test_df = test_df.drop('bin', axis=1)

    X_train = train_df.drop(['filename', 'makespan'], axis=1).values
    y_train = train_df['makespan'].values
    X_test = test_df.drop(['filename', 'makespan'], axis=1).values
    y_test = test_df['makespan'].values
    X_all = df.drop(['filename', 'makespan', 'bin'], axis=1).values
    y_all = df['makespan'].values

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, y_train, X_test, y_test, X_all, y_all


def create_ridge_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge())
    ])


def create_xgboost_pipeline():
    return Pipeline([
        ('xgboost', XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1))
    ])


def create_gpr_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42,
        ))
    ])

def create_gpr_matern25_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42,
        ))
    ])


def create_gpr_rbf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42,
        ))
    ])


def create_bayesian_ridge_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('bayesian_ridge', BayesianRidge(max_iter=500)),
    ])


def create_ard_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ard', ARDRegression(max_iter=500)),
    ])


def train_and_evaluate_model(model_name, pipeline, param_grid, X_train, y_train, X_test, y_test, X_all, y_all):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=kfold,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    y_cv_pred = cross_val_predict(best_model, X_all, y_all, cv=kfold)

    results = {
        'best_params': random_search.best_params_,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'cv_r2': r2_score(y_all, y_cv_pred),
        'cv_mae': mean_absolute_error(y_all, y_cv_pred),
        'cv_mse': mean_squared_error(y_all, y_cv_pred),
    }

    print(f"{model_name} training complete!\n")
    return results


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


def log_experiment_run(
    selected_features,
    pipeline_stats,
    ridge_results,
    gpr_results,
    bayesian_ridge_results,
    ard_results,
    gpr_matern25_results,
    gpr_rbf_results,
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
        ridge_results: Dict returned by nested_cv_evaluation() for Ridge.
        gpr_results: Dict returned by nested_cv_evaluation() for GPR (Matérn 1.5).
        bayesian_ridge_results: Dict returned by nested_cv_evaluation() for Bayesian Ridge.
        ard_results: Dict returned by nested_cv_evaluation() for ARD Regression.
        gpr_matern25_results: Dict returned by nested_cv_evaluation() for GPR (Matérn 2.5).
        gpr_rbf_results: Dict returned by nested_cv_evaluation() for GPR (RBF).
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

    required_keys = [
        'r2_mean', 'r2_std', 'mae_mean', 'mae_std',
        'overall_r2', 'overall_mae', 'train_r2_mean',
        'r2_overfit_gap_mean', 'inner_outer_gap_mean',
    ]
    for key in required_keys:
        if key not in ridge_results:
            raise KeyError(f"Missing key '{key}' in ridge_results")
        if key not in ard_results:
            raise KeyError(f"Missing key '{key}' in ard_results")

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
        'n_after_rfecv': pipeline_stats['after_rfecv'],

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

    # Load full dataset
    df = pd.read_csv(constants.FEATURES_CSV_OUTPUT)
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")

    # Run feature selection pipeline
    selected_features, stats = full_feature_selection_pipeline(df, target_col='makespan')

    # Prepare data with selected features
    X_all = df[selected_features].values
    y_all = df['makespan'].values

    print("\n" + "=" * 70)
    print("MODEL EVALUATION WITH SELECTED FEATURES")
    print("=" * 70 + "\n")

    # Ridge regression with polynomial features
    ridge_pipeline = create_ridge_pipeline()
    ridge_params = {
        'poly__degree': [1, 2],
        'ridge__alpha': [1, 10, 50, 100, 500, 1000, 2000],
    }

    ridge_results = nested_cv_evaluation(
        ridge_pipeline,
        ridge_params,
        X_all,
        y_all,
    )
    print_nested_cv_results('Ridge (Polynomial)', ridge_results)
    save_nested_cv_predictions(df, ridge_results, 'cv_predictions_ridge.csv')

    print("\n" + "=" * 70)
    print("Now running GPR (Matérn ν=1.5)...")
    print("=" * 70 + "\n")

    gpr_pipeline = create_gpr_pipeline()
    gpr_results = nested_cv_evaluation(
        gpr_pipeline,
        {},  # empty grid → clone + direct fit inside nested_cv_evaluation
        X_all,
        y_all,
    )
    print_nested_cv_results('GPR (Matérn ν=1.5)', gpr_results)
    save_nested_cv_predictions(df, gpr_results, 'cv_predictions_gpr.csv')

    print("\n" + "=" * 70)
    print("Now running GPR (Matérn ν=2.5)...")
    print("=" * 70 + "\n")

    gpr_matern25_pipeline = create_gpr_matern25_pipeline()
    gpr_matern25_results = nested_cv_evaluation(
        gpr_matern25_pipeline,
        {},
        X_all,
        y_all,
    )
    print_nested_cv_results('GPR (Matérn ν=2.5)', gpr_matern25_results)
    save_nested_cv_predictions(df, gpr_matern25_results, 'cv_predictions_gpr_matern25.csv')

    print("\n" + "=" * 70)
    print("Now running GPR (RBF)...")
    print("=" * 70 + "\n")

    gpr_rbf_pipeline = create_gpr_rbf_pipeline()
    gpr_rbf_results = nested_cv_evaluation(
        gpr_rbf_pipeline,
        {},
        X_all,
        y_all,
    )
    print_nested_cv_results('GPR (RBF)', gpr_rbf_results)
    save_nested_cv_predictions(df, gpr_rbf_results, 'cv_predictions_gpr_rbf.csv')

    print("\n" + "=" * 70)
    print("Now running Bayesian Ridge (Polynomial)...")
    print("=" * 70 + "\n")
    
    bayesian_ridge_pipeline = create_bayesian_ridge_pipeline()
    bayesian_ridge_params = {
        'poly__degree': [1, 2],
    }
    bayesian_ridge_results = nested_cv_evaluation(
        bayesian_ridge_pipeline,
        bayesian_ridge_params,
        X_all,
        y_all,
    )
    print_nested_cv_results('Bayesian Ridge (Polynomial)', bayesian_ridge_results)
    save_nested_cv_predictions(df, bayesian_ridge_results, 'cv_predictions_bayesian_ridge.csv')

    print("\n" + "=" * 70)
    print("Now running ARD Regression...")
    print("=" * 70 + "\n")

    ard_pipeline = create_ard_pipeline()
    ard_params = {'poly__degree': [1, 2]}
    ard_results = nested_cv_evaluation(
        ard_pipeline,
        ard_params,
        X_all,
        y_all,
    )
    print_nested_cv_results('ARD Regression (Polynomial)', ard_results)
    save_nested_cv_predictions(df, ard_results, 'cv_predictions_ard.csv')

    # Log experiment to cumulative CSV
    log_experiment_run(
        selected_features=selected_features,
        pipeline_stats=stats,
        ridge_results=ridge_results,
        gpr_results=gpr_results,
        bayesian_ridge_results=bayesian_ridge_results,
        ard_results=ard_results,
        gpr_matern25_results=gpr_matern25_results,
        gpr_rbf_results=gpr_rbf_results,
        random_seed=random_seed,
        output_path=output_path,
    )

    return (selected_features, ridge_results, gpr_results,
            bayesian_ridge_results, ard_results,
            gpr_matern25_results, gpr_rbf_results, random_seed)


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

    # Ridge
    ridge_pipeline = create_ridge_pipeline()
    ridge_params = {
        'poly__degree': [1, 2],
        'ridge__alpha': [1, 10, 50, 100, 500, 1000, 2000],
    }
    ridge_results = nested_cv_evaluation(ridge_pipeline, ridge_params, X_all, y_all)
    print_nested_cv_results('Ridge (Polynomial)', ridge_results)
    save_nested_cv_predictions(df, ridge_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ridge_manual.csv')

    # GPR Matérn 1.5
    print("\n" + "=" * 70)
    print("Now running GPR (Matérn ν=1.5)...")
    print("=" * 70 + "\n")
    gpr_results = nested_cv_evaluation(create_gpr_pipeline(), {}, X_all, y_all)
    print_nested_cv_results('GPR (Matérn ν=1.5)', gpr_results)
    save_nested_cv_predictions(df, gpr_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_manual.csv')

    # GPR Matérn 2.5
    print("\n" + "=" * 70)
    print("Now running GPR (Matérn ν=2.5)...")
    print("=" * 70 + "\n")
    gpr_matern25_results = nested_cv_evaluation(create_gpr_matern25_pipeline(), {}, X_all, y_all)
    print_nested_cv_results('GPR (Matérn ν=2.5)', gpr_matern25_results)
    save_nested_cv_predictions(df, gpr_matern25_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_matern25_manual.csv')

    # GPR RBF
    print("\n" + "=" * 70)
    print("Now running GPR (RBF)...")
    print("=" * 70 + "\n")
    gpr_rbf_results = nested_cv_evaluation(create_gpr_rbf_pipeline(), {}, X_all, y_all)
    print_nested_cv_results('GPR (RBF)', gpr_rbf_results)
    save_nested_cv_predictions(df, gpr_rbf_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_gpr_rbf_manual.csv')

    # Bayesian Ridge
    bayesian_ridge_params = {'poly__degree': [1, 2]}
    bayesian_ridge_results = nested_cv_evaluation(
        create_bayesian_ridge_pipeline(), bayesian_ridge_params, X_all, y_all
    )
    print_nested_cv_results('Bayesian Ridge (Polynomial)', bayesian_ridge_results)
    save_nested_cv_predictions(df, bayesian_ridge_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_bayesian_ridge_manual.csv')

    # ARD
    ard_results = nested_cv_evaluation(
        create_ard_pipeline(), {'poly__degree': [1, 2]}, X_all, y_all
    )
    print_nested_cv_results('ARD Regression (Polynomial)', ard_results)
    save_nested_cv_predictions(df, ard_results, constants.CSV_OUTPUT_DIR + '/outputs/cv_predictions_ard_manual.csv')

    print("\n" + "=" * 70)
    print("MANUAL FEATURES — SUMMARY")
    print("=" * 70)
    for name, res in [
        ('Ridge', ridge_results),
        ('GPR Matérn 1.5', gpr_results),
        ('GPR Matérn 2.5', gpr_matern25_results),
        ('GPR RBF', gpr_rbf_results),
        ('Bayesian Ridge', bayesian_ridge_results),
        ('ARD', ard_results),
    ]:
        print(f"  {name:<20} overall_r2={res['overall_r2']:.4f}  mae={res['overall_mae']:.1f}s")


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


if __name__ == "__main__":
    run_with_manual_features()

    """
    for i in range(100):
        run_with_feature_selection(output_path=constants.EXPERIMENT_LOG_CSV)
    """