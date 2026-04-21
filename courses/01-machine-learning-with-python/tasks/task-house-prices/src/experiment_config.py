from __future__ import annotations

from dataclasses import dataclass, field, replace

from config import ProjectConfig
from feature_builders import DERIVED_FEATURE_GROUPS, FEATURE_BUILDERS
from feature_registry import FEATURE_GROUPS
from processing_config import CategoricalProcessingConfig, NumericProcessingConfig, OrdinalProcessingConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str = ""
    feature_groups: list[str] = field(default_factory=lambda: ["numeric_all"])
    add_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    derived_feature_groups: list[str] = field(default_factory=list)
    derived_features: list[str] = field(default_factory=list)
    log_transform_columns: list[str] | None = None
    target_transform: str | None = None
    numeric_processing_overrides: dict[str, NumericProcessingConfig] = field(default_factory=dict)
    categorical_processing_overrides: dict[str, CategoricalProcessingConfig] = field(default_factory=dict)
    ordinal_processing_overrides: dict[str, OrdinalProcessingConfig] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=lambda: ["linear_regression", "ridge", "lasso", "random_forest"])


def resolve_columns(group_names: list[str], add_columns: list[str], drop_columns: list[str]) -> list[str]:
    resolved: list[str] = []
    for group_name in group_names:
        if group_name not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group_name}")
        for column in FEATURE_GROUPS[group_name]:
            if column not in resolved:
                resolved.append(column)

    for column in add_columns:
        if column not in resolved:
            resolved.append(column)

    return [column for column in resolved if column not in set(drop_columns)]


def resolve_derived_features(
    group_names: list[str],
    feature_names: list[str],
    additional_feature_names: list[str] | None = None,
) -> list[str]:
    resolved: list[str] = []
    for group_name in group_names:
        if group_name not in DERIVED_FEATURE_GROUPS:
            raise ValueError(f"Unknown derived feature group: {group_name}")
        for feature_name in DERIVED_FEATURE_GROUPS[group_name]:
            if feature_name not in resolved:
                resolved.append(feature_name)

    for feature_name in feature_names:
        if feature_name not in resolved:
            resolved.append(feature_name)

    if additional_feature_names is not None:
        for feature_name in additional_feature_names:
            if feature_name in FEATURE_BUILDERS and feature_name not in resolved:
                resolved.append(feature_name)

    return resolved


def project_config_from_experiment(base_config: ProjectConfig, experiment: ExperimentConfig) -> ProjectConfig:
    selected_numeric = resolve_columns(
        [name for name in experiment.feature_groups if name.startswith("numeric_")],
        experiment.add_columns,
        experiment.drop_columns,
    )
    selected_ordinal = resolve_columns(
        [name for name in experiment.feature_groups if name.startswith("ordinal_")],
        [],
        experiment.drop_columns,
    )
    selected_categorical = resolve_columns(
        [name for name in experiment.feature_groups if name.startswith("categorical_")],
        [],
        experiment.drop_columns,
    )

    log_columns = (
        experiment.log_transform_columns
        if experiment.log_transform_columns is not None
        else [column for column in base_config.log_transform_columns if column in selected_numeric]
    )
    selected_ordinal_mappings = {
        column: mapping
        for column, mapping in base_config.ordinal_mappings.items()
        if column in selected_ordinal
    }

    return replace(
        base_config,
        numeric_features=selected_numeric,
        ordinal_features=selected_ordinal,
        categorical_features=selected_categorical,
        log_transform_columns=log_columns,
        numeric_processing_overrides={
            **{column: config for column, config in base_config.numeric_processing_overrides.items() if column in selected_numeric},
            **experiment.numeric_processing_overrides,
        },
        categorical_processing_overrides={
            **{
                column: config
                for column, config in base_config.categorical_processing_overrides.items()
                if column in selected_categorical
            },
            **experiment.categorical_processing_overrides,
        },
        ordinal_processing_overrides={
            **{
                column: config
                for column, config in base_config.ordinal_processing_overrides.items()
                if column in selected_ordinal
            },
            **experiment.ordinal_processing_overrides,
        },
        ordinal_mappings=selected_ordinal_mappings,
        target_transform=experiment.target_transform if experiment.target_transform is not None else base_config.target_transform,
    )


DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="linear_numeric_baseline",
        description="Numeric-only baseline close to the simplest linear setup.",
        feature_groups=["numeric_all"],
        model_names=["linear_regression", "ridge", "lasso"],
    ),
    ExperimentConfig(
        name="linear_numeric_engineered",
        description="Numeric baseline plus a few engineered size and age features.",
        feature_groups=["numeric_all"],
        derived_feature_groups=["engineered_core"],
        add_columns=["TotalSF", "HouseAge", "TotalBathrooms"],
        model_names=["linear_regression", "ridge", "lasso"],
    ),
    ExperimentConfig(
        name="mixed_core_features",
        description="Core numeric, ordinal, and categorical groups for learner-friendly iteration.",
        feature_groups=["numeric_core", "ordinal_core", "categorical_core"],
        derived_feature_groups=["engineered_core"],
        add_columns=["TotalSF", "HouseAge", "TotalBathrooms"],
        model_names=["linear_regression", "ridge", "lasso", "random_forest"],
    ),
    ExperimentConfig(
        name="full_feature_workflow",
        description="All configured features with engineered additions.",
        feature_groups=["numeric_all", "ordinal_all", "categorical_all"],
        derived_feature_groups=["engineered_extended"],
        add_columns=["TotalSF", "HouseAge", "RemodAge", "TotalBathrooms", "TotalPorchSF"],
        model_names=["dummy", "linear_regression", "ridge", "lasso", "random_forest"],
    ),
]
