import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


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
    print()
    print("FINAL SELECTED FEATURES:")
    for i, feat in enumerate(features, 1):
        corr = pearsonr(df[feat].values, df[target_col].values)[0]
        print(f"  {i:2d}. {feat}: r={corr:.3f}")
    print("=" * 70)

    return features, pipeline_stats
