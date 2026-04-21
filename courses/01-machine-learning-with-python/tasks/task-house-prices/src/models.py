from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from config import ProjectConfig


def build_estimator(name: str, config: ProjectConfig):
    if name == "dummy":
        return DummyRegressor(**config.model_settings["dummy"])
    if name == "linear_regression":
        return LinearRegression(**config.model_settings["linear_regression"])
    if name == "ridge":
        return Ridge(**config.model_settings["ridge"])
    if name == "lasso":
        return Lasso(**config.model_settings["lasso"])
    if name == "random_forest":
        return RandomForestRegressor(random_state=config.random_state, **config.model_settings["random_forest"])
    raise ValueError(f"Unsupported model name: {name}")


def wrap_target_transform(estimator, config: ProjectConfig):
    if config.target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return estimator
