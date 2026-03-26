from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, ARDRegression, HuberRegressor, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn_rvm import EMRVR


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
        'svr__C': [0.1, 1, 10, 100, 500],
        'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
        'svr__gamma': ['scale', 'auto'],
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
        'random_forest__n_estimators': [100, 200, 500],
        'random_forest__max_depth': [None, 3, 5, 7],
        'random_forest__max_features': ['sqrt', 'log2', 0.5],
        'random_forest__min_samples_leaf': [1, 2, 4],
    }

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('random_forest', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])



DEFAULT_MODELS = [
    RidgeModel(),
    GPRMatern15Model(),
    GPRMatern25Model(),
    GPRRBFModel(),
    BayesianRidgeModel(),
    ARDModel(),
    HuberModel(),
    SVRModel(),
    ElasticNetModel(),
    RVMModel(),
    KernelRidgeModel(),
    HistGBModel(),
    RandomForestModel(),
]
