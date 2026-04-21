from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd


BuilderFn = Callable[[pd.DataFrame, dict[str, dict[Any, int]] | None], pd.DataFrame]


def add_total_sf(df: pd.DataFrame, ordinal_mappings: dict[str, dict[Any, int]] | None = None) -> pd.DataFrame:
    result = df.copy()
    result["TotalSF"] = (
        result["TotalBsmtSF"].fillna(0)
        + result["1stFlrSF"].fillna(0)
        + result["2ndFlrSF"].fillna(0)
    )
    return result


def add_garage_area_per_car(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["GarageAreaPerCar"] = np.where(
        result["GarageCars"] > 0,
        result["GarageArea"] / result["GarageCars"],
        0,
    )
    return result


def add_has_garage(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["HasGarage"] = (result["GarageArea"] > 0).astype(int)
    return result


def add_house_age(df: pd.DataFrame, ordinal_mappings: dict[str, dict[Any, int]] | None = None) -> pd.DataFrame:
    result = df.copy()
    result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
    return result


def add_remod_age(df: pd.DataFrame, ordinal_mappings: dict[str, dict[Any, int]] | None = None) -> pd.DataFrame:
    result = df.copy()
    result["RemodAge"] = result["YrSold"] - result["YearRemodAdd"]
    return result


def add_total_bathrooms(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["TotalBathrooms"] = (
        result["FullBath"].fillna(0)
        + 0.5 * result["HalfBath"].fillna(0)
        + result["BsmtFullBath"].fillna(0)
        + 0.5 * result["BsmtHalfBath"].fillna(0)
    )
    return result


def add_total_porch_sf(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["TotalPorchSF"] = (
        result["OpenPorchSF"].fillna(0)
        + result["EnclosedPorch"].fillna(0)
        + result["3SsnPorch"].fillna(0)
        + result["ScreenPorch"].fillna(0)
        + result["WoodDeckSF"].fillna(0)
    )
    return result


def add_ordinal_numeric_interaction(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None,
    *,
    output_column: str,
    ordinal_column: str,
    numeric_column: str,
) -> pd.DataFrame:
    if ordinal_mappings is None:
        raise ValueError(
            f"Cannot build {output_column} without ordinal mappings. "
            "Pass ordinal_mappings when applying feature builders."
        )
    if ordinal_column not in ordinal_mappings:
        raise ValueError(f"Missing ordinal mapping for interaction source column: {ordinal_column}")

    result = df.copy()
    mapping = ordinal_mappings[ordinal_column]
    missing_code = mapping.get("__MISSING__", 0)

    ordinal_codes = (
        result[ordinal_column]
        .fillna("__MISSING__")
        .map(mapping)
        .fillna(missing_code)
        .astype(float)
    )
    numeric_values = pd.to_numeric(result[numeric_column], errors="coerce").fillna(0.0)
    result[output_column] = ordinal_codes * numeric_values
    return result


def build_ordinal_numeric_interaction(
    output_column: str,
    ordinal_column: str,
    numeric_column: str,
) -> BuilderFn:
    return partial(
        add_ordinal_numeric_interaction,
        output_column=output_column,
        ordinal_column=ordinal_column,
        numeric_column=numeric_column,
    )


def add_multi_ordinal_numeric_interaction(
    df: pd.DataFrame,
    ordinal_mappings: dict[str, dict[Any, int]] | None,
    *,
    output_column: str,
    ordinal_columns: list[str],
    numeric_column: str,
) -> pd.DataFrame:
    if ordinal_mappings is None:
        raise ValueError(
            f"Cannot build {output_column} without ordinal mappings. "
            "Pass ordinal_mappings when applying feature builders."
        )

    result = df.copy()
    numeric_values = pd.to_numeric(result[numeric_column], errors="coerce").fillna(0.0)
    interaction_values = numeric_values.copy()

    for ordinal_column in ordinal_columns:
        if ordinal_column not in ordinal_mappings:
            raise ValueError(f"Missing ordinal mapping for interaction source column: {ordinal_column}")
        mapping = ordinal_mappings[ordinal_column]
        missing_code = mapping.get("__MISSING__", 0)
        ordinal_codes = (
            result[ordinal_column]
            .fillna("__MISSING__")
            .map(mapping)
            .fillna(missing_code)
            .astype(float)
        )
        interaction_values = interaction_values * ordinal_codes

    result[output_column] = interaction_values
    return result


def build_multi_ordinal_numeric_interaction(
    output_column: str,
    ordinal_columns: list[str],
    numeric_column: str,
) -> BuilderFn:
    return partial(
        add_multi_ordinal_numeric_interaction,
        output_column=output_column,
        ordinal_columns=ordinal_columns,
        numeric_column=numeric_column,
    )


FEATURE_BUILDERS: dict[str, BuilderFn] = {
    "TotalSF": add_total_sf,
    "HouseAge": add_house_age,
    "RemodAge": add_remod_age,
    "TotalBathrooms": add_total_bathrooms,
    "TotalPorchSF": add_total_porch_sf,
    "GarageAreaPerCar": add_garage_area_per_car,
    "HasGarage": add_has_garage,
    "GarageQual_x_GarageArea": build_ordinal_numeric_interaction(
        "GarageQual_x_GarageArea",
        "GarageQual",
        "GarageArea",
    ),
    "GarageCond_x_GarageArea": build_ordinal_numeric_interaction(
        "GarageCond_x_GarageArea",
        "GarageCond",
        "GarageArea",
    ),
    "GarageQual_x_GarageCars": build_ordinal_numeric_interaction(
        "GarageQual_x_GarageCars",
        "GarageQual",
        "GarageCars",
    ),
    "GarageCond_x_GarageCars": build_ordinal_numeric_interaction(
        "GarageCond_x_GarageCars",
        "GarageCond",
        "GarageCars",
    ),
    "GarageQual_x_GarageCond_x_GarageArea": build_multi_ordinal_numeric_interaction(
        "GarageQual_x_GarageCond_x_GarageArea",
        ["GarageQual", "GarageCond"],
        "GarageArea",
    ),
    "GarageQual_x_GarageCond_x_GarageCars": build_multi_ordinal_numeric_interaction(
        "GarageQual_x_GarageCond_x_GarageCars",
        ["GarageQual", "GarageCond"],
        "GarageCars",
    ),
}


DERIVED_FEATURE_GROUPS: dict[str, list[str]] = {
    "engineered_core": ["TotalSF", "HouseAge", "TotalBathrooms", "HasGarage"],
    "engineered_extended": ["TotalSF", "HouseAge", "RemodAge", "TotalBathrooms", "TotalPorchSF", "HasGarage"],
    "garage_interactions": [
        "GarageQual_x_GarageArea",
        "GarageCond_x_GarageArea",
        "GarageQual_x_GarageCars",
        "GarageCond_x_GarageCars",
        "GarageQual_x_GarageCond_x_GarageArea",
        "GarageQual_x_GarageCond_x_GarageCars",
    ],
}
