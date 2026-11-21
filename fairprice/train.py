"""Phases 5-7 — Train/test split, model training, hyperparameter tuning and K-fold CV.

Every model is wrapped in the same Pipeline (preprocessing + regressor), so the comparison
is fair and the winning Pipeline can be saved and used by the app as-is.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from fairprice import config
from fairprice.ann import KerasANNRegressor
from fairprice.features import build_preprocessor
from fairprice.logger import get_logger

log = get_logger(__name__)


def split_data(df: pd.DataFrame):
    """80/20 split, stratified on income bands so both sets cover rich and poor districts alike."""
    income_band = pd.cut(df["MedInc"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=False)
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=income_band
    )
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def build_pipeline(regressor) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", regressor)])


CONVENTIONAL, NEURAL = "Conventional ML", "Neural Network"


def get_model_candidates() -> dict:
    """name -> (regressor, hyperparameter search space or None, model family)."""
    rs = config.RANDOM_STATE
    return {
        "LinearRegression": (LinearRegression(), None, CONVENTIONAL),  # baseline: no hyperparameters
        "Ridge": (Ridge(), {"alpha": loguniform(1e-2, 1e3)}, CONVENTIONAL),  # L2 regularization
        "RandomForest": (
            RandomForestRegressor(random_state=rs, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.5, 0.7, 1.0],
                "max_samples": [0.6, 0.8, None],  # row subsampling also speeds training up
            },
            CONVENTIONAL,
        ),
        "XGBoost": (
            XGBRegressor(random_state=rs, n_jobs=-1, tree_method="hist", verbosity=0),
            {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [1, 5, 10],
            },
            CONVENTIONAL,
        ),
        "ANN": (
            KerasANNRegressor(random_state=rs, epochs=150, patience=10),
            {
                "hidden_layers": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                "activation": ["relu", "elu"],
                "kernel_initializer": ["he_normal", "glorot_uniform"],  # He suits ReLU, Glorot suits tanh/ELU
                "dropout": [0.0, 0.1, 0.2],
                "l2": [0.0, 1e-5, 1e-4],
                "optimizer": ["adam", "sgd"],
                "learning_rate": loguniform(3e-4, 3e-3),
                "batch_size": [64, 128, 256],
            },
            NEURAL,
        ),
    }


def _kfold() -> KFold:
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def train_one(name, regressor, search_space, family, X_train, y_train, n_iter) -> dict:
    """Tune (if there is a search space) with K-fold CV and return the fitted best pipeline."""
    start = time.time()
    pipeline = build_pipeline(regressor)

    if search_space is None:
        cv = cross_validate(pipeline, X_train, y_train, cv=_kfold(), scoring="neg_root_mean_squared_error")
        cv_scores = -cv["test_score"]
        fitted = pipeline.fit(X_train, y_train)
        best_params = {}
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions={f"model__{k}": v for k, v in search_space.items()},
            n_iter=n_iter,
            cv=_kfold(),
            scoring="neg_root_mean_squared_error",
            random_state=config.RANDOM_STATE,
            n_jobs=1,  # the models themselves already use all cores
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_index_
        cv_scores = np.array([
            -search.cv_results_[f"split{i}_test_score"][best] for i in range(config.CV_FOLDS)
        ])
        fitted = search.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    result = {
        "name": name,
        "family": family,
        "pipeline": fitted,
        "cv_rmse_mean": float(cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "best_params": best_params,
        "fit_seconds": round(time.time() - start, 1),
    }
    log.info("%-16s CV RMSE = %.4f +/- %.4f  (%.1fs)  %s", name, result["cv_rmse_mean"],
             result["cv_rmse_std"], result["fit_seconds"], best_params)
    return result


def search_iterations(family: str, fast: bool) -> int:
    if family == NEURAL:
        return config.ANN_N_ITER_SEARCH_FAST if fast else config.ANN_N_ITER_SEARCH
    return config.N_ITER_SEARCH_FAST if fast else config.N_ITER_SEARCH


def train_all(X_train, y_train, fast: bool = False) -> list[dict]:
    candidates = get_model_candidates()
    log.info("Training %d models with %d-fold CV (search iterations: %d conventional, %d ANN)",
             len(candidates), config.CV_FOLDS, search_iterations(CONVENTIONAL, fast),
             search_iterations(NEURAL, fast))
    return [
        train_one(name, regressor, space, family, X_train, y_train, search_iterations(family, fast))
        for name, (regressor, space, family) in candidates.items()
    ]
