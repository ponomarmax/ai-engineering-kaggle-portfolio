from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import ProjectConfig


def ensure_directories(config: ProjectConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.experiments_dir.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(base_dir: Path, candidates: Iterable[str]) -> Path:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    candidate_list = ", ".join(str(base_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any of the expected files: {candidate_list}")


def load_datasets(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_path = resolve_existing_path(config.base_dir, config.train_candidates)
    train_df = pd.read_csv(train_path)

    test_df: pd.DataFrame | None = None
    try:
        test_path = resolve_existing_path(config.base_dir, config.test_candidates)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        test_df = None

    return train_df, test_df


def collect_configured_features(config: ProjectConfig) -> list[str]:
    return config.numeric_features + config.categorical_features + config.ordinal_features


def validate_feature_groups(config: ProjectConfig, columns: Iterable[str]) -> None:
    column_set = set(columns)
    configured = collect_configured_features(config)

    missing = sorted(set(configured) - column_set)
    if missing:
        raise ValueError(f"Configured features missing from the dataset: {missing}")

    duplicates = sorted({column for column in configured if configured.count(column) > 1})
    if duplicates:
        raise ValueError(f"Features appear in more than one feature group: {duplicates}")


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def split_features_target(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = collect_configured_features(config)
    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, config.target_column].copy()
    return X, y


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def infer_prediction_ids(test_df: pd.DataFrame, config: ProjectConfig) -> pd.Series:
    if config.id_column in test_df.columns:
        return test_df[config.id_column]
    return pd.Series(range(len(test_df)), name=config.id_column)
