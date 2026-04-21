from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline

from config import ProjectConfig
from models import build_estimator, wrap_target_transform
from preprocessing import build_preprocessor


def model_requires_scaling(model_name: str) -> bool:
    return model_name in {"linear_regression", "ridge", "lasso"}


def build_pipeline(model_name: str, config: ProjectConfig) -> Pipeline:
    estimator = wrap_target_transform(build_estimator(model_name, config), config)
    preprocessor = build_preprocessor(config=config, scale_numeric=model_requires_scaling(model_name))
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def get_model_names(config: ProjectConfig) -> list[str]:
    return list(config.model_settings.keys())


def fit_pipeline(model_name: str, X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> Pipeline:
    pipeline = build_pipeline(model_name=model_name, config=config)
    pipeline.fit(X, y)
    return pipeline


def fit_final_pipeline(model_name: str, X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> Pipeline:
    return fit_pipeline(model_name=model_name, X=X, y=y, config=config)


def build_reduced_config(config: ProjectConfig, dropped_features: Iterable[str]) -> ProjectConfig:
    dropped = set(dropped_features)
    return ProjectConfig(
        base_dir=config.base_dir,
        data_dir_name=config.data_dir_name,
        train_candidates=config.train_candidates,
        test_candidates=config.test_candidates,
        artifacts_dir_name=config.artifacts_dir_name,
        outputs_dir_name=config.outputs_dir_name,
        target_column=config.target_column,
        id_column=config.id_column,
        random_state=config.random_state,
        n_splits=config.n_splits,
        test_size=config.test_size,
        target_transform=config.target_transform,
        numeric_features=[column for column in config.numeric_features if column not in dropped],
        categorical_features=[column for column in config.categorical_features if column not in dropped],
        ordinal_features=[column for column in config.ordinal_features if column not in dropped],
        log_transform_columns=[column for column in config.log_transform_columns if column not in dropped],
        ordinal_mappings={k: v for k, v in config.ordinal_mappings.items() if k not in dropped},
        model_settings=config.model_settings,
    )
