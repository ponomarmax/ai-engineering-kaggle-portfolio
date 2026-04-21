from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class NumericProcessingConfig:
    impute_strategy: str = "median"
    fill_value: float | None = None
    transform: str = "none"
    scale: str = "none"


@dataclass(frozen=True)
class CategoricalProcessingConfig:
    impute_strategy: str = "most_frequent"
    encoding: str = "onehot"
    min_frequency: float | int | None = None


@dataclass(frozen=True)
class OrdinalProcessingConfig:
    impute_strategy: str = "constant"
    fill_value: str = "__MISSING__"
    encoding: str = "explicit_ordinal"


def merge_numeric_config(
    base_config: NumericProcessingConfig,
    override: NumericProcessingConfig | None,
) -> NumericProcessingConfig:
    return override if override is not None else base_config


def merge_categorical_config(
    base_config: CategoricalProcessingConfig,
    override: CategoricalProcessingConfig | None,
) -> CategoricalProcessingConfig:
    return override if override is not None else base_config


def merge_ordinal_config(
    base_config: OrdinalProcessingConfig,
    override: OrdinalProcessingConfig | None,
) -> OrdinalProcessingConfig:
    return override if override is not None else base_config


def processing_reference_tables() -> dict[str, pd.DataFrame]:
    return {
        "numeric_processing_reference": pd.DataFrame(
            [
                {
                    "setting": "impute_strategy",
                    "default": "median",
                    "alternatives": "mean, most_frequent, constant",
                    "when_to_use": "median is robust to outliers; mean is reasonable when distribution is fairly symmetric.",
                },
                {
                    "setting": "transform",
                    "default": "none",
                    "alternatives": "log1p, sqrt",
                    "when_to_use": "log1p helps right-skewed positive features; sqrt is a gentler skewness reduction.",
                },
                {
                    "setting": "scale",
                    "default": "none or standard depending on model",
                    "alternatives": "standard, robust, none",
                    "when_to_use": "standard for linear models; robust when outliers are strong; none for trees.",
                },
            ]
        ),
        "categorical_processing_reference": pd.DataFrame(
            [
                {
                    "setting": "impute_strategy",
                    "default": "most_frequent",
                    "alternatives": "constant",
                    "when_to_use": "most_frequent is simple; constant is useful when missing itself may carry signal.",
                },
                {
                    "setting": "encoding",
                    "default": "onehot",
                    "alternatives": "documented only",
                    "when_to_use": "one-hot is the safest default for nominal categories; target encoding is powerful but riskier and needs leakage control.",
                },
                {
                    "setting": "min_frequency",
                    "default": "None",
                    "alternatives": "float or int threshold",
                    "when_to_use": "helps merge rare levels when one-hot columns become too sparse.",
                },
            ]
        ),
        "ordinal_processing_reference": pd.DataFrame(
            [
                {
                    "setting": "impute_strategy",
                    "default": "constant",
                    "alternatives": "most_frequent",
                    "when_to_use": "constant with __MISSING__ is safest when missing means absence or unknown quality.",
                },
                {
                    "setting": "encoding",
                    "default": "explicit_ordinal",
                    "alternatives": "documented only",
                    "when_to_use": "ordinal features need an explicit order; wrong order creates misleading numeric structure.",
                },
            ]
        ),
    }
