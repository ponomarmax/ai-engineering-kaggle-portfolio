from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "train.csv"
COMPETITION_TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

TARGET = "SalePrice"
ID_COLUMN = "Id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature group plan.
# Keep these lists explicit so forgotten features are easy to spot.
LOG1P_CONTINUOUS_FEATURES: list[str] = [
    "GrLivArea",
    "LotArea",
    # "LotFrontage",
]
LOG1P_ADD_HAS_FEATURES: list[str] = [
    "2ndFlrSF",
    "BsmtFinSF1",
    "BsmtUnfSF",
    # "GarageArea",
    "OpenPorchSF",
    "WoodDeckSF",
    # "TotalBsmtSF",
]
DROP_MANY_ZERO_LOW_SIGNAL_FEATURES: list[str] = [
    "3SsnPorch",
    "LowQualFinSF",
    "MiscVal",
    "PoolArea",
]

# Keep dropped-for-now groups separate so we can revisit them later.
CONVERT_TO_NUMBER_INVESTIGATE_AFTER: list[str] = [
    "Alley",
    "BsmtQual",
    "CentralAir",
    "ExterCond",
    "ExterQual",
    "Fence",
    "FireplaceQu",
    "GarageCond",
    "GarageFinish",
    "GarageQual",
    "HeatingQC",
    "KitchenQual",
    "LandSlope",
    "LotShape",
    "MSSubClass",
    "MasVnrArea",
    "OverallCond",
    "OverallQual",
    "PavedDrive",
    "SaleCondition",
    "Street",
    "Utilities",
]
DROP_NOT_NUMBER_FEATURES: list[str] = [
    "Bedroom",
    "BldgType",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Functional",
    "MasVnrType",
    "Neighborhood",
    "PoolQC",
    "RoofMatl",
    "RoofStyle",
]
DROP_NOT_DECIDED_YET_FEATURES: list[str] = [
    "BsmtFinSF2",
    "Condition1",
    "Condition2",
    "Electrical",
    "EnclosedPorch",
    "Exterior1st",
    "Exterior2nd",
    "Foundation",
    "GarageType",
    "Heating",
    "HouseStyle",
    "LandContour",
    "LotConfig",
    "MSZoning",
    "MiscFeature",
    "SaleType",
    "ScreenPorch",
]

# Baseline features that stay untouched on purpose.
KEEP_AS_IS_FEATURES: list[str] = [
    # "GarageYrBlt",
    "BedroomAbvGr",
    "BsmtFullBath",
    "BsmtHalfBath",
    "Fireplaces",
    "FullBath",
    "GarageCars",
    "HalfBath",
    "KitchenAbvGr",
    # "MoSold",
    "TotRmsAbvGrd",
    "YrSold",
    "YearBuilt",
    "YearRemodAdd",
]


@dataclass
class FeatureCombination:
    name: str
    columns: list[str]
    transform: str = "sum"
    log1p: bool = False
    drop_source_columns: bool = False


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    use_all_processed_features: bool = True
    keep_only: list[str] | None = None
    add_features: list[str] = field(default_factory=list)
    drop_features: list[str] = field(default_factory=list)
    log1p_features: list[str] = field(default_factory=list)
    add_has_features: list[str] = field(default_factory=list)
    combinations: list[FeatureCombination] = field(default_factory=list)


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline_processed",
        description="All currently processed baseline features",
    ),
    ExperimentConfig(
        name="only_1stFlrSF",
        description="Keep baseline context, drop TotalBsmtSF, log1p 1stFlrSF",
        add_features=["1stFlrSF"],
        drop_features=["TotalBsmtSF"],
        log1p_features=["1stFlrSF"],
    ),
    ExperimentConfig(
        name="only_TotalBsmtSF",
        description="Keep baseline context, drop 1stFlrSF, log1p TotalBsmtSF, add HasTotalBsmtSF",
        add_features=["TotalBsmtSF"],
        drop_features=["1stFlrSF"],
        log1p_features=["TotalBsmtSF"],
        add_has_features=["TotalBsmtSF"],
    ),
    ExperimentConfig(
        name="both_1stFlrSF_TotalBsmtSF",
        description="Keep baseline context, log1p both features, add HasTotalBsmtSF",
        add_features=["1stFlrSF", "TotalBsmtSF"],
        log1p_features=["1stFlrSF", "TotalBsmtSF"],
        add_has_features=["TotalBsmtSF"],
    ),
    ExperimentConfig(
        name="TotalArea_only",
        description="Keep baseline context, replace 1stFlrSF and TotalBsmtSF with log1p(TotalArea)",
        combinations=[
            FeatureCombination(
                name="TotalArea",
                columns=["1stFlrSF", "TotalBsmtSF"],
                transform="sum",
                log1p=True,
                drop_source_columns=True,
            )
        ],
    ),
]


def format_feature_list(columns: list[str]) -> str:
    return ", ".join(sorted(columns))


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def validate_input_files() -> None:
    if not TRAIN_PATH.exists() or not COMPETITION_TEST_PATH.exists():
        raise FileNotFoundError("Put train.csv and test.csv into data/raw/ before running.")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    competition_test_df = pd.read_csv(COMPETITION_TEST_PATH)
    return train_df, competition_test_df


def select_numeric_features(
    train_df: pd.DataFrame, competition_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    numeric_train_df = train_df.select_dtypes(include=["number"]).copy()
    numeric_competition_test_df = competition_test_df.select_dtypes(include=["number"]).copy()

    X = numeric_train_df.drop(columns=[TARGET, ID_COLUMN])
    y = numeric_train_df[TARGET]
    X_competition_test = numeric_competition_test_df.drop(columns=[ID_COLUMN])

    return X, y, X_competition_test


def build_feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    feature_groups = {
        "log1p_continuous": LOG1P_CONTINUOUS_FEATURES,
        "log1p_add_has": LOG1P_ADD_HAS_FEATURES,
        "drop_many_zero_low_signal": DROP_MANY_ZERO_LOW_SIGNAL_FEATURES,
        "convert_to_number_investigate_after": CONVERT_TO_NUMBER_INVESTIGATE_AFTER,
        "drop_not_number": DROP_NOT_NUMBER_FEATURES,
        "drop_not_decided_yet": DROP_NOT_DECIDED_YET_FEATURES,
        "keep_as_is": KEEP_AS_IS_FEATURES,
    }

    all_listed_features = set().union(*feature_groups.values()) if feature_groups else set()
    missing_features = sorted(set(X.columns) - all_listed_features)
    if missing_features:
        feature_groups["forgotten_features"] = missing_features
    else:
        feature_groups["forgotten_features"] = []

    return feature_groups


def validate_feature_groups(feature_groups: dict[str, list[str]], available_columns: pd.Index) -> None:
    available_set = set(available_columns)
    all_listed_features: list[str] = []

    for group_name, feature_list in feature_groups.items():
        if group_name == "forgotten_features":
            continue
        missing_from_dataset = sorted(set(feature_list) - available_set)
        if missing_from_dataset:
            print(f"[warning] {group_name} contains columns missing in the dataset:")
            print(missing_from_dataset)

        all_listed_features.extend(feature_list)

    if feature_groups["forgotten_features"]:
        print("[warning] Uncategorized numeric features will be excluded until you assign them:")
        print(feature_groups["forgotten_features"])

    duplicated_features = pd.Series(all_listed_features).value_counts()
    duplicated_features = duplicated_features[duplicated_features > 1]
    if not duplicated_features.empty:
        print("[warning] The following features were assigned to multiple groups:")
        print(duplicated_features)


def preprocess_features(
    X: pd.DataFrame,
    X_competition_test: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_feature_columns = sorted(
        set(feature_groups["log1p_continuous"])
        | set(feature_groups["log1p_add_has"])
        | set(feature_groups["keep_as_is"])
    )

    X_processed = X[baseline_feature_columns].copy()
    X_competition_test_processed = X_competition_test[baseline_feature_columns].copy()

    drop_features = sorted(
        set(feature_groups["drop_many_zero_low_signal"])
        | set(feature_groups["convert_to_number_investigate_after"])
        | set(feature_groups["drop_not_number"])
        | set(feature_groups["drop_not_decided_yet"])
    )

    for col in feature_groups["log1p_continuous"]:
        if col in X_processed.columns:
            X_processed[col] = np.log1p(X_processed[col])
            X_competition_test_processed[col] = np.log1p(X_competition_test_processed[col])

    for col in feature_groups["log1p_add_has"]:
        if col in X_processed.columns:
            has_feature_name = f"Has{col}"
            X_processed[has_feature_name] = (X_processed[col] > 0).astype(int)
            X_competition_test_processed[has_feature_name] = (
                X_competition_test_processed[col] > 0
            ).astype(int)

            X_processed[col] = np.log1p(X_processed[col])
            X_competition_test_processed[col] = np.log1p(X_competition_test_processed[col])

    X_processed = X_processed.drop(columns=drop_features, errors="ignore")
    X_competition_test_processed = X_competition_test_processed.drop(columns=drop_features, errors="ignore")

    return X_processed, X_competition_test_processed


def apply_feature_combination(
    X: pd.DataFrame,
    X_competition_test: pd.DataFrame,
    X_source: pd.DataFrame,
    X_competition_test_source: pd.DataFrame,
    combination: FeatureCombination,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_experiment = X.copy()
    X_competition_test_experiment = X_competition_test.copy()

    missing_columns = [col for col in combination.columns if col not in X_source.columns]
    if missing_columns:
        raise KeyError(
            f"Experiment combination '{combination.name}' is missing columns: {missing_columns}"
        )

    if combination.transform != "sum":
        raise ValueError(f"Unsupported transform: {combination.transform}")

    X_experiment[combination.name] = X_source[combination.columns].sum(axis=1)
    X_competition_test_experiment[combination.name] = X_competition_test_source[
        combination.columns
    ].sum(axis=1)

    if combination.log1p:
        X_experiment[combination.name] = np.log1p(X_experiment[combination.name])
        X_competition_test_experiment[combination.name] = np.log1p(
            X_competition_test_experiment[combination.name]
        )

    if combination.drop_source_columns:
        X_experiment = X_experiment.drop(columns=combination.columns, errors="ignore")
        X_competition_test_experiment = X_competition_test_experiment.drop(
            columns=combination.columns,
            errors="ignore",
        )

    return X_experiment, X_competition_test_experiment


def apply_experiment_config(
    X_baseline: pd.DataFrame,
    X_competition_test_baseline: pd.DataFrame,
    X_source: pd.DataFrame,
    X_competition_test_source: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.use_all_processed_features:
        X_experiment = X_baseline.copy()
        X_competition_test_experiment = X_competition_test_baseline.copy()
    else:
        X_experiment = pd.DataFrame(index=X_baseline.index)
        X_competition_test_experiment = pd.DataFrame(index=X_competition_test_baseline.index)

    for combination in config.combinations:
        X_experiment, X_competition_test_experiment = apply_feature_combination(
            X_experiment,
            X_competition_test_experiment,
            X_source,
            X_competition_test_source,
            combination,
        )

    for col in config.add_features:
        if col not in X_source.columns:
            raise KeyError(f"Experiment '{config.name}' references missing add_features column '{col}'")
        X_experiment[col] = X_source[col]
        X_competition_test_experiment[col] = X_competition_test_source[col]

    for col in config.log1p_features:
        if col not in X_experiment.columns:
            raise KeyError(f"Experiment '{config.name}' cannot log1p missing column '{col}'")
        X_experiment[col] = np.log1p(X_experiment[col])
        X_competition_test_experiment[col] = np.log1p(X_competition_test_experiment[col])

    for col in config.add_has_features:
        if col not in X_experiment.columns:
            raise KeyError(f"Experiment '{config.name}' cannot add Has column for missing '{col}'")
        has_feature_name = f"Has{col}"
        X_experiment[has_feature_name] = (X_experiment[col] > 0).astype(int)
        X_competition_test_experiment[has_feature_name] = (
            X_competition_test_experiment[col] > 0
        ).astype(int)

    if config.drop_features:
        X_experiment = X_experiment.drop(columns=config.drop_features, errors="ignore")
        X_competition_test_experiment = X_competition_test_experiment.drop(
            columns=config.drop_features,
            errors="ignore",
        )

    if config.keep_only is not None:
        selected_columns = list(dict.fromkeys(config.keep_only + config.add_features))
        missing_keep_columns = [col for col in selected_columns if col not in X_experiment.columns]
        if missing_keep_columns:
            raise KeyError(
                f"Experiment '{config.name}' references missing selected columns: {missing_keep_columns}"
            )
        X_experiment = X_experiment[selected_columns].copy()
        X_competition_test_experiment = X_competition_test_experiment[selected_columns].copy()

    if X_experiment.empty:
        raise ValueError(f"Experiment '{config.name}' produced an empty feature matrix")

    return X_experiment, X_competition_test_experiment


def run_experiments(
    X_baseline: pd.DataFrame,
    X_competition_test_baseline: pd.DataFrame,
    X_source: pd.DataFrame,
    X_competition_test_source: pd.DataFrame,
    y: pd.Series,
    experiments: list[ExperimentConfig],
) -> pd.DataFrame:
    experiment_results: list[dict[str, object]] = []

    for config in experiments:
        X_experiment, _ = apply_experiment_config(
            X_baseline,
            X_competition_test_baseline,
            X_source,
            X_competition_test_source,
            config,
        )
        _, _, validation_score = evaluate_model(
            build_baseline_model(),
            X_experiment,
            np.log1p(y),
        )

        experiment_results.append(
            {
                "experiment": config.name,
                "description": config.description,
                "n_features": X_experiment.shape[1],
                "validation_r2": validation_score,
            }
        )

    return pd.DataFrame(experiment_results).sort_values(
        by="validation_r2",
        ascending=False,
    )


def get_experiment_config_by_name(
    experiments: list[ExperimentConfig],
    experiment_name: str,
) -> ExperimentConfig:
    for config in experiments:
        if config.name == experiment_name:
            return config
    raise KeyError(f"Unknown experiment: {experiment_name}")


def build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    return X_validation, y_validation, model.score(X_validation, y_validation)


def train_full_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model.fit(X, y)
    return model


def save_submission(predictions: pd.Series, competition_test_df: pd.DataFrame) -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            ID_COLUMN: competition_test_df[ID_COLUMN],
            TARGET: predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


def main() -> None:
    validate_input_files()

    train_df, competition_test_df = load_datasets()
    X, y, X_competition_test = select_numeric_features(train_df, competition_test_df)
    feature_groups = build_feature_groups(X)
    validate_feature_groups(feature_groups, X.columns)
    X_baseline, X_competition_test_baseline = preprocess_features(
        X,
        X_competition_test,
        feature_groups,
    )

    print_section("Feature Groups")
    print({group_name: len(feature_list) for group_name, feature_list in feature_groups.items()})
    print(f"Baseline feature count: {X_baseline.shape[1]}")

    print_section("Baseline Features")
    print(format_feature_list(X_baseline.columns.tolist()))

    experiment_results = run_experiments(
        X_baseline,
        X_competition_test_baseline,
        X,
        X_competition_test,
        y,
        EXPERIMENTS,
    )
    print_section("Experiment Comparison")
    print(
        experiment_results[
            ["experiment", "validation_r2", "n_features", "description"]
        ].to_string(index=False)
    )

    best_experiment_name = experiment_results.iloc[0]["experiment"]
    best_experiment_config = get_experiment_config_by_name(EXPERIMENTS, best_experiment_name)
    X_best, X_competition_test_best = apply_experiment_config(
        X_baseline,
        X_competition_test_baseline,
        X,
        X_competition_test,
        best_experiment_config,
    )

    print_section("Best Experiment")
    print(f"Selected experiment: {best_experiment_name}")
    print(f"Best feature count: {X_best.shape[1]}")
    print(format_feature_list(X_best.columns.tolist()))

    validation_model = build_baseline_model()
    X_validation, y_validation, validation_score = evaluate_model(
        validation_model,
        X_best,
        np.log1p(y),
    )
    print_section("Best Experiment Validation")
    print(f"Validation R^2: {validation_score:.4f}")

    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(
        validation_model,
        X_validation,
        y_validation,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    impo_diff = pd.DataFrame(
        {
        "feature": X_validation.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    print_section("Permutation Importance")
    print(impo_diff)

    production_model = train_full_model(build_baseline_model(), X_best, np.log1p(y))
    predictions = np.expm1(production_model.predict(X_competition_test_best))
    save_submission(predictions=predictions, competition_test_df=competition_test_df)

    print_section("Submission")
    print(f"Saved {SUBMISSION_PATH.relative_to(BASE_DIR)} using experiment '{best_experiment_name}'")


if __name__ == "__main__":
    main()
