from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from config import ProjectConfig
from train import build_pipeline


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_model_cv(model_name: str, X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> dict[str, float]:
    pipeline = build_pipeline(model_name=model_name, config=config)
    cv = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=config.n_jobs,
        return_train_score=False,
    )
    return {
        "model_name": model_name,
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
    }


def compare_models(model_names: Iterable[str], X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> pd.DataFrame:
    rows = [evaluate_model_cv(model_name=name, X=X, y=y, config=config) for name in model_names]
    comparison = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    return comparison


def holdout_evaluation(model_name: str, X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> dict[str, float]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    pipeline = build_pipeline(model_name=model_name, config=config)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)
    metrics = regression_metrics(y_true=y_valid, y_pred=predictions)
    metrics["validation_rows"] = float(len(X_valid))
    return metrics


def compute_permutation_importance(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
    n_repeats: int = 10,
) -> pd.DataFrame:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    pipeline = build_pipeline(model_name=model_name, config=config)
    pipeline.fit(X_train, y_train)
    result = permutation_importance(
        estimator=pipeline,
        X=X_valid,
        y=y_valid,
        n_repeats=n_repeats,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        scoring="neg_root_mean_squared_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_valid.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance_df.reset_index(drop=True)
