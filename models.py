from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, ARDRegression, HuberRegressor, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn_rvm import EMRVR
from xgboost import XGBRegressor


class ModelConfig:
    name = ''
    key = ''
    param_grid = {}

    def build_pipeline(self):
        raise NotImplementedError

    def get_pipeline_and_params(self):
        return self.build_pipeline(), self.param_grid


class RidgeModel(ModelConfig):
    name = 'Ridge (Polynomial)'
    key = 'ridge'
    param_grid = {
        'poly__degree': [1, 2],
        'ridge__alpha': [1, 10, 50, 100, 500, 1000, 2000],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge()),
        ])


class GPRMatern15Model(ModelConfig):
    name = 'GPR (Matern nu=1.5)'
    key = 'gpr'
    param_grid = {}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gpr', GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )),
        ])


class GPRMatern25Model(ModelConfig):
    name = 'GPR (Matern nu=2.5)'
    key = 'gpr_matern25'
    param_grid = {}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gpr', GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )),
        ])


class GPRRBFModel(ModelConfig):
    name = 'GPR (RBF)'
    key = 'gpr_rbf'
    param_grid = {}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gpr', GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF() + WhiteKernel(),
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )),
        ])


class BayesianRidgeModel(ModelConfig):
    name = 'Bayesian Ridge (Polynomial)'
    key = 'bayesian_ridge'
    param_grid = {'poly__degree': [1, 2]}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('bayesian_ridge', BayesianRidge(max_iter=500)),
        ])


class ARDModel(ModelConfig):
    name = 'ARD Regression (Polynomial)'
    key = 'ard'
    param_grid = {'poly__degree': [1, 2]}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ard', ARDRegression(max_iter=500)),
        ])


class HuberModel(ModelConfig):
    name = 'Huber Regression'
    key = 'huber'
    param_grid = {
        'huber__epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],
        'huber__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('huber', HuberRegressor(max_iter=500)),
        ])


class SVRModel(ModelConfig):
    name = 'SVR (RBF)'
    key = 'svr'
    param_grid = {
        'svr__C': [1000, 2000, 3000],
        'svr__epsilon': [1.5, 2.0, 2.5],
        'svr__gamma': [0.003, 0.005, 0.007],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf')),
        ])


class ElasticNetModel(ModelConfig):
    name = 'Elastic Net'
    key = 'elastic_net'
    param_grid = {
        'elastic_net__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'elastic_net__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('elastic_net', ElasticNet(max_iter=10000)),
        ])


class RVMModel(ModelConfig):
    name = 'RVM (RBF)'
    key = 'rvm'
    param_grid = {}

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rvm', EMRVR(kernel='rbf')),
        ])


class KernelRidgeModel(ModelConfig):
    name = 'Kernel Ridge (RBF)'
    key = 'kernel_ridge'
    param_grid = {
        'kernel_ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel_ridge__gamma': [0.001, 0.01, 0.1, 0.5, 1.0],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_ridge', KernelRidge(kernel='rbf')),
        ])


class HistGBModel(ModelConfig):
    name = 'HistGradientBoosting'
    key = 'hist_gb'
    param_grid = {
        'hist_gb__max_iter': [100, 200, 300],
        'hist_gb__max_leaf_nodes': [15, 31, 63],
        'hist_gb__learning_rate': [0.01, 0.05, 0.1],
        'hist_gb__l2_regularization': [0, 0.1, 1.0, 10.0],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('hist_gb', HistGradientBoostingRegressor(random_state=42)),
        ])


class RandomForestModel(ModelConfig):
    name = 'Random Forest'
    key = 'random_forest'
    param_grid = {
        'random_forest__n_estimators': [500, 800],
        'random_forest__max_depth': [4, 5, 6],
        'random_forest__max_features': [0.25, 0.3, 0.4],
        'random_forest__min_samples_leaf': [2, 3],
        'random_forest__ccp_alpha': [0.05, 0.1, 0.15, 0.2],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('random_forest', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])


class XGBoostModel(ModelConfig):
    name = 'XGBoost'
    key = 'xgboost'
    param_grid = {
        'xgboost__n_estimators': [100, 200, 300],
        'xgboost__max_depth': [2, 3, 4],
        'xgboost__learning_rate': [0.01, 0.05, 0.1],
        'xgboost__subsample': [0.6, 0.8],
        'xgboost__reg_lambda': [1.0, 10.0, 100.0],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('xgboost', XGBRegressor(
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])


DEFAULT_MODELS = [
    RidgeModel(),
    # GPRMatern15Model(),   # explored; ~0.316 at n=9, threshold-insensitive but dominated by Matern 2.5
    # GPRMatern25Model(),   # explored; ~0.375, threshold-insensitive; dropped for model diversity
    # GPRRBFModel(),        # explored; ~0.375, near-identical to Matern 2.5; RBF smoothness not justified
    BayesianRidgeModel(),
    # ARDModel(),           # explored; consistently poor (<=0.247 overall R2), overfits at n=88
    HuberModel(),
    SVRModel(),
    # ElasticNetModel(),    # explored; redundant with Ridge (inner CV pushes l1_ratio toward 1.0)
    # RVMModel(),           # explored; ~0.216, high variance across folds
    # KernelRidgeModel(),   # explored; ~0.384; dropped in favor of SVR for kernel/SVM representation
    # HistGBModel(),        # explored; R2 degrades with more features (0.076 at n=18), severe overfit
    RandomForestModel(),
    XGBoostModel(),
]
