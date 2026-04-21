from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feature_registry import CATEGORICAL_FEATURES, LOG_TRANSFORM_COLUMNS, NUMERIC_FEATURES, ORDINAL_FEATURES
from processing_config import CategoricalProcessingConfig, NumericProcessingConfig, OrdinalProcessingConfig

OrdinalMapping = dict[str, dict[Any, int]]


@dataclass(frozen=True)
class ProjectConfig:
    base_dir: Path = Path(__file__).resolve().parents[1]
    data_dir_name: str = "data"
    train_candidates: tuple[str, ...] = ("data/train.csv", "data/raw/train.csv")
    test_candidates: tuple[str, ...] = ("data/test.csv", "data/raw/test.csv")
    artifacts_dir_name: str = "artifacts"
    outputs_dir_name: str = "outputs"
    experiment_dir_name: str = "experiments"
    target_column: str = "SalePrice"
    id_column: str = "Id"
    random_state: int = 42
    n_splits: int = 5
    test_size: float = 0.2
    n_jobs: int = 1
    target_transform: str | None = "log1p"
    numeric_features: list[str] = field(default_factory=lambda: NUMERIC_FEATURES.copy())
    categorical_features: list[str] = field(default_factory=lambda: CATEGORICAL_FEATURES.copy())
    ordinal_features: list[str] = field(default_factory=lambda: ORDINAL_FEATURES.copy())
    log_transform_columns: list[str] = field(default_factory=lambda: LOG_TRANSFORM_COLUMNS.copy())
    default_numeric_processing: NumericProcessingConfig = field(default_factory=lambda: NumericProcessingConfig())
    numeric_processing_overrides: dict[str, NumericProcessingConfig] = field(default_factory=dict)
    default_categorical_processing: CategoricalProcessingConfig = field(
        default_factory=lambda: CategoricalProcessingConfig()
    )
    categorical_processing_overrides: dict[str, CategoricalProcessingConfig] = field(default_factory=dict)
    default_ordinal_processing: OrdinalProcessingConfig = field(default_factory=lambda: OrdinalProcessingConfig())
    ordinal_processing_overrides: dict[str, OrdinalProcessingConfig] = field(default_factory=dict)
    ordinal_mappings: OrdinalMapping = field(
        default_factory=lambda: {
            "Street": {"__MISSING__": 0, "Grvl": 1, "Pave": 2},
            "LotShape": {"__MISSING__": 0, "IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
            "Utilities": {"__MISSING__": 0, "ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
            "LandSlope": {"__MISSING__": 0, "Sev": 1, "Mod": 2, "Gtl": 3},
            "ExterQual": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "ExterCond": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "BsmtQual": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "BsmtCond": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "BsmtExposure": {"__MISSING__": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
            "BsmtFinType1": {
                "__MISSING__": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "BsmtFinType2": {
                "__MISSING__": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "HeatingQC": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "CentralAir": {"__MISSING__": 0, "N": 1, "Y": 2},
            "KitchenQual": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "Functional": {
                "__MISSING__": 0,
                "Sal": 1,
                "Sev": 2,
                "Maj2": 3,
                "Maj1": 4,
                "Mod": 5,
                "Min2": 6,
                "Min1": 7,
                "Typ": 8,
            },
            "FireplaceQu": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "GarageFinish": {"__MISSING__": 0, "Unf": 1, "RFn": 2, "Fin": 3},
            "GarageQual": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "GarageCond": {"__MISSING__": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "PavedDrive": {"__MISSING__": 0, "N": 1, "P": 2, "Y": 3},
            "PoolQC": {"__MISSING__": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "Fence": {"__MISSING__": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
        }
    )
    model_settings: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "dummy": {"strategy": "median"},
            "linear_regression": {},
            "ridge": {"alpha": 10.0},
            "lasso": {"alpha": 0.001, "max_iter": 20000},
            "random_forest": {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 2,
                "n_jobs": 1,
            },
        }
    )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.data_dir_name

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / self.artifacts_dir_name

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / self.outputs_dir_name

    @property
    def experiments_dir(self) -> Path:
        return self.outputs_dir / self.experiment_dir_name


CONFIG = ProjectConfig()
