from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF


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
    name = 'GPR (Matérn ν=1.5)'
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
    name = 'GPR (Matérn ν=2.5)'
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


DEFAULT_MODELS = [
    RidgeModel(),
    GPRMatern15Model(),
    GPRMatern25Model(),
    GPRRBFModel(),
    BayesianRidgeModel(),
    ARDModel(),
]
