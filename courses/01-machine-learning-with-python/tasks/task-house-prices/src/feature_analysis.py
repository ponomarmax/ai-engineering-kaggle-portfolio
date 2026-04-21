from __future__ import annotations

import pandas as pd


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": missing.div(len(df)).mul(100),
        }
    )
    return summary.loc[summary["missing_count"] > 0].sort_values("missing_percent", ascending=False)


def skewness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "feature": columns,
            "skewness": df[columns].skew(numeric_only=True).values,
        }
    )
    return stats.sort_values("skewness", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def categorical_cardinality_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": columns,
            "n_unique": [df[column].nunique(dropna=False) for column in columns],
            "missing_count": [df[column].isna().sum() for column in columns],
        }
    ).sort_values(["n_unique", "missing_count"], ascending=[False, False]).reset_index(drop=True)
