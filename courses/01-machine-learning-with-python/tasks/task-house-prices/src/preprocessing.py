from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, RobustScaler, StandardScaler

from config import ProjectConfig
from processing_config import (
    CategoricalProcessingConfig,
    NumericProcessingConfig,
    OrdinalProcessingConfig,
    merge_categorical_config,
    merge_numeric_config,
    merge_ordinal_config,
)


VALID_IMPUTER_STRATEGIES = {"mean", "median", "most_frequent", "constant"}


def normalize_imputer_strategy(strategy: Any, context: str) -> str:
    if isinstance(strategy, tuple):
        if len(strategy) != 1:
            raise ValueError(f"{context} impute_strategy must be a single value, got tuple {strategy!r}.")
        strategy = strategy[0]

    if strategy == "none":
        raise ValueError(
            f"{context} impute_strategy='none' is not supported. "
            "Use one of: mean, median, most_frequent, constant."
        )

    if not isinstance(strategy, str) or strategy not in VALID_IMPUTER_STRATEGIES:
        raise ValueError(
            f"{context} impute_strategy must be one of {sorted(VALID_IMPUTER_STRATEGIES)}, got {strategy!r}."
        )
    return strategy


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = "median", fill_value: Any | None = None):
        self.strategy = normalize_imputer_strategy(strategy, "Numeric")
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=fill_value)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameSimpleImputer":
        self.feature_names_in_ = list(X.columns)
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.imputer.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class NumericValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "none"):
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericValueTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        if self.method == "none":
            return X_df
        if self.method == "log1p":
            return np.log1p(np.clip(X_df, a_min=0, a_max=None))
        if self.method == "sqrt":
            return np.sqrt(np.clip(X_df, a_min=0, a_max=None))
        raise ValueError(f"Unsupported numeric transform: {self.method}")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale: str = "none"):
        self.scale = scale
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameScaler":
        self.feature_names_in_ = list(X.columns)
        if self.scale == "none":
            self.scaler = None
            return self
        if self.scale == "standard":
            self.scaler = StandardScaler()
        elif self.scale == "robust":
            self.scaler = RobustScaler()
        elif self.scale == "quantile":
            self.scaler = QuantileTransformer()
        else:
            raise ValueError(f"Unsupported scaling strategy: {self.scale}")
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            return X.copy()
        transformed = self.scaler.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def validate_ordinal_mappings(
    ordinal_features: list[str],
    ordinal_mappings: dict[str, dict[Any, int]],
) -> None:
    missing = sorted(set(ordinal_features) - set(ordinal_mappings))
    if missing:
        raise ValueError(f"Missing ordinal mappings for: {missing}")

    extra = sorted(set(ordinal_mappings) - set(ordinal_features))
    if extra:
        raise ValueError(f"Ordinal mappings are defined for non-ordinal columns: {extra}")

    for feature_name, mapping in ordinal_mappings.items():
        ordered_values = list(mapping.values())
        if len(ordered_values) != len(set(ordered_values)):
            raise ValueError(f"Ordinal mapping for {feature_name} contains duplicate numeric codes.")
        if "__MISSING__" not in mapping:
            raise ValueError(
                f"Ordinal mapping for {feature_name} must define a '__MISSING__' category explicitly."
            )


def get_ordered_categories(
    ordinal_features: list[str],
    ordinal_mappings: dict[str, dict[Any, int]],
) -> list[list[Any]]:
    categories: list[list[Any]] = []
    for feature_name in ordinal_features:
        mapping = ordinal_mappings[feature_name]
        ordered = [category for category, _ in sorted(mapping.items(), key=lambda item: item[1])]
        categories.append(ordered)
    return categories


def effective_numeric_processing(config: ProjectConfig, column: str, scale_numeric: bool) -> NumericProcessingConfig:
    override = config.numeric_processing_overrides.get(column)
    effective = merge_numeric_config(config.default_numeric_processing, override)
    transform = effective.transform
    if column in config.log_transform_columns and transform == "none":
        transform = "log1p"
    scale = effective.scale
    explicit_scale_none = override is not None and override.scale == "none"
    if scale_numeric and scale == "none" and not explicit_scale_none:
        scale = "standard"
    return NumericProcessingConfig(
        impute_strategy=effective.impute_strategy,
        fill_value=effective.fill_value,
        transform=transform,
        scale=scale,
    )


def effective_categorical_processing(config: ProjectConfig, column: str) -> CategoricalProcessingConfig:
    return merge_categorical_config(config.default_categorical_processing, config.categorical_processing_overrides.get(column))


def effective_ordinal_processing(config: ProjectConfig, column: str) -> OrdinalProcessingConfig:
    return merge_ordinal_config(config.default_ordinal_processing, config.ordinal_processing_overrides.get(column))


def build_numeric_transformer(column_config: NumericProcessingConfig) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", DataFrameSimpleImputer(strategy=column_config.impute_strategy, fill_value=column_config.fill_value)),
        ("value_transform", NumericValueTransformer(method=column_config.transform)),
        ("scaler", DataFrameScaler(scale=column_config.scale)),
    ]
    return Pipeline(steps=numeric_steps)


def build_categorical_transformer(column_config: CategoricalProcessingConfig) -> Pipeline:
    imputer_strategy = normalize_imputer_strategy(column_config.impute_strategy, "Categorical")
    fill_value = "__MISSING__" if imputer_strategy == "constant" else None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy, fill_value=fill_value)),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=column_config.min_frequency,
                ),
            ),
        ]
    )


def build_ordinal_transformer(config: ProjectConfig, columns: list[str], column_config: OrdinalProcessingConfig) -> Pipeline:
    validate_ordinal_mappings(config.ordinal_features, config.ordinal_mappings)
    categories = get_ordered_categories(columns, config.ordinal_mappings)
    imputer_strategy = normalize_imputer_strategy(column_config.impute_strategy, "Ordinal")
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=imputer_strategy, fill_value=column_config.fill_value),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def group_columns_by_processing(columns: list[str], processing_lookup) -> dict[Any, list[str]]:
    grouped: dict[Any, list[str]] = {}
    for column in columns:
        column_processing = processing_lookup(column)
        grouped.setdefault(column_processing, []).append(column)
    return grouped


def build_preprocessor(config: ProjectConfig, scale_numeric: bool) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    numeric_groups = group_columns_by_processing(
        config.numeric_features,
        lambda column: effective_numeric_processing(config, column, scale_numeric),
    )
    for idx, (column_config, columns) in enumerate(numeric_groups.items()):
        transformers.append((f"numeric_{idx}", build_numeric_transformer(column_config), columns))

    categorical_groups = group_columns_by_processing(
        config.categorical_features,
        lambda column: effective_categorical_processing(config, column),
    )
    for idx, (column_config, columns) in enumerate(categorical_groups.items()):
        transformers.append((f"categorical_{idx}", build_categorical_transformer(column_config), columns))

    ordinal_groups = group_columns_by_processing(
        config.ordinal_features,
        lambda column: effective_ordinal_processing(config, column),
    )
    for idx, (column_config, columns) in enumerate(ordinal_groups.items()):
        transformers.append((f"ordinal_{idx}", build_ordinal_transformer(config, columns, column_config), columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
