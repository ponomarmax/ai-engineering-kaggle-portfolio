from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


BuilderFn = Callable[[pd.DataFrame], pd.DataFrame]


def add_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TotalSF"] = (
        result["TotalBsmtSF"].fillna(0)
        + result["1stFlrSF"].fillna(0)
        + result["2ndFlrSF"].fillna(0)
    )
    return result

def add_garage_area_per_car(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["GarageAreaPerCar"] = np.where(
        result["GarageCars"] > 0,
        result["GarageArea"] / result["GarageCars"],
        0
    )
    return result

def add_has_garage(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["HasGarage"] = (result["GarageArea"] > 0).astype(int)
    return result


def add_house_age(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
    return result


def add_remod_age(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["RemodAge"] = result["YrSold"] - result["YearRemodAdd"]
    return result


def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TotalBathrooms"] = (
        result["FullBath"].fillna(0)
        + 0.5 * result["HalfBath"].fillna(0)
        + result["BsmtFullBath"].fillna(0)
        + 0.5 * result["BsmtHalfBath"].fillna(0)
    )
    return result


def add_total_porch_sf(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TotalPorchSF"] = (
        result["OpenPorchSF"].fillna(0)
        + result["EnclosedPorch"].fillna(0)
        + result["3SsnPorch"].fillna(0)
        + result["ScreenPorch"].fillna(0)
        + result["WoodDeckSF"].fillna(0)
    )
    return result


FEATURE_BUILDERS: dict[str, BuilderFn] = {
    "TotalSF": add_total_sf,
    "HouseAge": add_house_age,
    "RemodAge": add_remod_age,
    "TotalBathrooms": add_total_bathrooms,
    "TotalPorchSF": add_total_porch_sf,
    "GarageAreaPerCar": add_garage_area_per_car,
    "HasGarage": add_has_garage,
}


DERIVED_FEATURE_GROUPS: dict[str, list[str]] = {
    "engineered_core": ["TotalSF", "HouseAge", "TotalBathrooms", "HasGarage"],
    "engineered_extended": ["TotalSF", "HouseAge", "RemodAge", "TotalBathrooms", "TotalPorchSF", "HasGarage"],
}
