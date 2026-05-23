from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge, Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
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
    BayesianRidgeModel(),
    HuberModel(),
    SVRModel(),
    RandomForestModel(),
    XGBoostModel(),
]
