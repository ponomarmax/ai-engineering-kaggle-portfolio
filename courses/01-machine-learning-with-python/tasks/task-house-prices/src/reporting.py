from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_registry import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES
from feature_analysis import categorical_cardinality_table, missingness_table, skewness_table
from preprocessing import (
    build_preprocessor,
    effective_categorical_processing,
    effective_numeric_processing,
    effective_ordinal_processing,
)
from processing_config import processing_reference_tables
from train import model_requires_scaling


def build_before_processing_report(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    ordinal_features: list[str],
) -> dict[str, pd.DataFrame]:
    report: dict[str, pd.DataFrame] = {
        "head": df.head(10),
        "missingness": missingness_table(df),
    }

    if numeric_features:
        report["numeric_summary"] = df[numeric_features].describe().T
        report["numeric_skewness"] = skewness_table(df, numeric_features)

    if categorical_features:
        report["categorical_cardinality"] = categorical_cardinality_table(df, categorical_features)

    if ordinal_features:
        report["ordinal_head"] = df[ordinal_features].head(10)

    return report


def build_column_processing_summary(config, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in config.numeric_features:
        option = effective_numeric_processing(config, column, model_requires_scaling(model_name))
        rows.append(
            {
                "column": column,
                "feature_type": "numeric",
                "impute_strategy": option.impute_strategy,
                "transform": option.transform,
                "scale": option.scale,
                "encoding": "none",
                "notes": "Default numeric alternatives: mean, most_frequent, constant, sqrt, robust scaling.",
            }
        )

    for column in config.categorical_features:
        option = effective_categorical_processing(config, column)
        rows.append(
            {
                "column": column,
                "feature_type": "categorical",
                "impute_strategy": option.impute_strategy,
                "transform": "none",
                "scale": "none",
                "encoding": option.encoding,
                "notes": f"min_frequency={option.min_frequency}; alternatives include constant imputation or rare-category grouping.",
            }
        )

    for column in config.ordinal_features:
        option = effective_ordinal_processing(config, column)
        rows.append(
            {
                "column": column,
                "feature_type": "ordinal",
                "impute_strategy": option.impute_strategy,
                "transform": "none",
                "scale": "none",
                "encoding": option.encoding,
                "notes": "Requires explicit order mapping; alternative is revisiting the feature type if the order is questionable.",
            }
        )

    return pd.DataFrame(rows)


def build_feature_selection_report(config, model_name: str) -> dict[str, pd.DataFrame]:
    used_df = build_column_processing_summary(config, model_name).copy()
    used_df["is_used"] = True

    unused_numeric = sorted(set(NUMERIC_FEATURES) - set(config.numeric_features))
    unused_categorical = sorted(set(CATEGORICAL_FEATURES) - set(config.categorical_features))
    unused_ordinal = sorted(set(ORDINAL_FEATURES) - set(config.ordinal_features))

    unused_rows: list[dict[str, object]] = []
    for column in unused_numeric:
        unused_rows.append(
            {
                "column": column,
                "feature_type": "numeric",
                "is_used": False,
                "impute_strategy": "",
                "transform": "",
                "scale": "",
                "encoding": "",
                "notes": "Configured as unused in this experiment.",
            }
        )
    for column in unused_categorical:
        unused_rows.append(
            {
                "column": column,
                "feature_type": "categorical",
                "is_used": False,
                "impute_strategy": "",
                "transform": "",
                "scale": "",
                "encoding": "",
                "notes": "Configured as unused in this experiment.",
            }
        )
    for column in unused_ordinal:
        unused_rows.append(
            {
                "column": column,
                "feature_type": "ordinal",
                "is_used": False,
                "impute_strategy": "",
                "transform": "",
                "scale": "",
                "encoding": "",
                "notes": "Configured as unused in this experiment.",
            }
        )

    all_features_df = pd.concat([used_df, pd.DataFrame(unused_rows)], ignore_index=True)
    all_features_df = all_features_df.sort_values(["feature_type", "is_used", "column"], ascending=[True, False, True])

    return {
        "all_features_with_usage": all_features_df,
        "used_features_with_processing": used_df.sort_values(["feature_type", "column"]).reset_index(drop=True),
        "unused_numeric_features": pd.DataFrame({"column": unused_numeric}),
        "unused_categorical_features": pd.DataFrame({"column": unused_categorical}),
        "unused_ordinal_features": pd.DataFrame({"column": unused_ordinal}),
    }


def build_after_processing_preview(
    X: pd.DataFrame,
    config,
    model_name: str,
    n_rows: int = 10,
) -> pd.DataFrame:
    preprocessor = build_preprocessor(config=config, scale_numeric=model_requires_scaling(model_name))
    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    preview = pd.DataFrame(transformed[:n_rows], columns=feature_names)
    return preview


def build_full_transformed_dataframe(
    X: pd.DataFrame,
    config,
    model_name: str,
) -> pd.DataFrame:
    preprocessor = build_preprocessor(config=config, scale_numeric=model_requires_scaling(model_name))
    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def transformed_columns_for_source(source_column: str, transformed_columns: list[str]) -> list[str]:
    exact = [column for column in transformed_columns if column == source_column]
    expanded = [column for column in transformed_columns if column.startswith(f"{source_column}_")]
    return exact + expanded


def build_processing_reference_report() -> dict[str, pd.DataFrame]:
    return processing_reference_tables()


def save_report_tables(report_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in report_tables.items():
        df.to_csv(output_dir / f"{table_name}.csv", index=True)
