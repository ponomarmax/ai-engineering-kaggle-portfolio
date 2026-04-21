from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from config import ProjectConfig
from evaluate import compare_models, compute_permutation_importance, holdout_evaluation
from experiment_config import DEFAULT_EXPERIMENTS, ExperimentConfig, project_config_from_experiment, resolve_derived_features
from feature_builders import FEATURE_BUILDERS
from feature_registry import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES
from reporting import (
    build_after_processing_preview,
    build_before_processing_report,
    build_column_processing_summary,
    build_feature_selection_report,
    build_processing_reference_report,
    save_report_tables,
)
from train import fit_final_pipeline
from utils import (
    collect_configured_features,
    ensure_directories,
    infer_prediction_ids,
    load_datasets,
    save_json,
    split_features_target,
    validate_feature_groups,
    validate_required_columns,
)


def apply_feature_builders(
    df: pd.DataFrame,
    feature_names: list[str],
    ordinal_mappings: dict[str, dict[object, int]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    for feature_name in feature_names:
        if feature_name not in FEATURE_BUILDERS:
            raise ValueError(f"Unknown derived feature: {feature_name}")
        result = FEATURE_BUILDERS[feature_name](result, ordinal_mappings)
    return result


def save_predictions(model, test_df: pd.DataFrame, config: ProjectConfig, output_path: Path) -> None:
    prediction_features = test_df.loc[:, collect_configured_features(config)].copy()
    prediction_frame = pd.DataFrame(
        {
            config.id_column: infer_prediction_ids(test_df, config),
            config.target_column: model.predict(prediction_features),
        }
    )
    prediction_frame.to_csv(output_path, index=False)


def run_experiment(base_config: ProjectConfig, experiment: ExperimentConfig) -> pd.DataFrame:
    ensure_directories(base_config)
    train_df, test_df = load_datasets(base_config)

    derived_features = resolve_derived_features(
        experiment.derived_feature_groups,
        experiment.derived_features,
        experiment.add_columns,
    )
    train_df = apply_feature_builders(train_df, derived_features, base_config.ordinal_mappings)
    if test_df is not None:
        test_df = apply_feature_builders(test_df, derived_features, base_config.ordinal_mappings)

    experiment_config = project_config_from_experiment(base_config, experiment)

    validate_required_columns(
        train_df,
        [experiment_config.target_column, *collect_configured_features(experiment_config)],
        dataset_name="train dataset",
    )
    validate_feature_groups(experiment_config, train_df.columns)

    if test_df is not None:
        validate_required_columns(
            test_df,
            collect_configured_features(experiment_config),
            dataset_name="test dataset",
        )

    X, y = split_features_target(train_df, experiment_config)
    experiment_dir = experiment_config.outputs_dir / "experiments" / experiment.name
    report_dir = experiment_dir / "reports"

    before_report = build_before_processing_report(
        train_df[collect_configured_features(experiment_config) + [experiment_config.target_column]],
        experiment_config.numeric_features,
        experiment_config.categorical_features,
        experiment_config.ordinal_features,
    )
    save_report_tables(before_report, report_dir / "before_processing")
    save_report_tables(build_processing_reference_report(), report_dir / "processing_reference")

    comparison_df = compare_models(experiment.model_names, X, y, experiment_config)
    best_model_name = comparison_df.loc[0, "model_name"]
    holdout_metrics = holdout_evaluation(best_model_name, X, y, experiment_config)
    importance_df = compute_permutation_importance(best_model_name, X, y, experiment_config)
    transformed_preview = build_after_processing_preview(X, experiment_config, best_model_name)
    processing_summary = build_column_processing_summary(experiment_config, best_model_name)
    feature_selection_report = build_feature_selection_report(experiment_config, best_model_name)

    transformed_preview.to_csv(report_dir / "after_processing_preview.csv", index=False)
    processing_summary.to_csv(report_dir / "column_processing_summary.csv", index=False)
    save_report_tables(feature_selection_report, report_dir / "feature_selection")
    comparison_df.to_csv(experiment_dir / "model_comparison.csv", index=False)
    importance_df.to_csv(experiment_dir / "permutation_importance.csv", index=False)
    save_json(holdout_metrics, experiment_dir / "holdout_metrics.json")
    save_json(
        {
            "experiment_name": experiment.name,
            "description": experiment.description,
            "feature_groups": experiment.feature_groups,
            "add_columns": experiment.add_columns,
            "drop_columns": experiment.drop_columns,
            "derived_features": derived_features,
            "log_transform_columns": experiment_config.log_transform_columns,
            "target_transform": experiment_config.target_transform,
            "model_names": experiment.model_names,
            "numeric_processing_overrides": {
                column: vars(option) for column, option in experiment_config.numeric_processing_overrides.items()
            },
            "categorical_processing_overrides": {
                column: vars(option) for column, option in experiment_config.categorical_processing_overrides.items()
            },
            "ordinal_processing_overrides": {
                column: vars(option) for column, option in experiment_config.ordinal_processing_overrides.items()
            },
            "selected_numeric_features": experiment_config.numeric_features,
            "selected_ordinal_features": experiment_config.ordinal_features,
            "selected_categorical_features": experiment_config.categorical_features,
            "unused_numeric_features": sorted(
                set(NUMERIC_FEATURES) - set(experiment_config.numeric_features)
            ),
            "unused_ordinal_features": sorted(
                set(ORDINAL_FEATURES) - set(experiment_config.ordinal_features)
            ),
            "unused_categorical_features": sorted(
                set(CATEGORICAL_FEATURES) - set(experiment_config.categorical_features)
            ),
        },
        experiment_dir / "experiment_definition.json",
    )

    final_model = fit_final_pipeline(best_model_name, X, y, experiment_config)
    model_path = experiment_dir / "final_pipeline.joblib"
    joblib.dump(final_model, model_path)

    if test_df is not None:
        save_predictions(final_model, test_df, experiment_config, experiment_dir / "test_predictions.csv")

    experiment_results_df = comparison_df.copy()
    experiment_results_df.insert(0, "experiment_name", experiment.name)
    experiment_results_df.insert(1, "experiment_description", experiment.description)
    experiment_results_df["is_best_model"] = experiment_results_df["model_name"].eq(best_model_name)
    experiment_results_df["best_model_name"] = best_model_name
    experiment_results_df["experiment_dir"] = str(experiment_dir)
    return experiment_results_df


def run_experiment_suite(base_config: ProjectConfig, experiments: list[ExperimentConfig] | None = None) -> pd.DataFrame:
    suite = experiments if experiments is not None else DEFAULT_EXPERIMENTS
    results_df = pd.concat([run_experiment(base_config, experiment) for experiment in suite], ignore_index=True)
    results_df = results_df.sort_values(["experiment_name", "rmse_mean", "mae_mean"]).reset_index(drop=True)
    summary_dir = base_config.outputs_dir / "experiments"
    results_path = summary_dir / "experiment_summary_all_models.csv"
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    best_only_df = (
        results_df.loc[results_df["is_best_model"]]
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )
    best_only_df.to_csv(summary_dir / "experiment_summary_best_models.csv", index=False)

    for metric_name in ["rmse_mean", "mae_mean", "r2_mean"]:
        pivot_df = (
            results_df.pivot(index="experiment_name", columns="model_name", values=metric_name)
            .reset_index()
        )
        pivot_df.to_csv(summary_dir / f"experiment_pivot_{metric_name}.csv", index=False)
    return results_df
