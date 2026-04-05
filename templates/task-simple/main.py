from pathlib import Path

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

TARGET = "target_column"
ID_COLUMN = "id_column"
RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    return model.score(X_validation, y_validation)


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

    validation_model = build_baseline_model()
    validation_score = evaluate_model(validation_model, X, y)
    print(f"Validation R^2: {validation_score:.4f}")

    production_model = train_full_model(build_baseline_model(), X, y)
    predictions = production_model.predict(X_competition_test)
    save_submission(predictions=predictions, competition_test_df=competition_test_df)

    print(f"Saved {SUBMISSION_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
