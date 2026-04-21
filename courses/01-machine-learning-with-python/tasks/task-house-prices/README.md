# House Price Regression Project

This task folder contains a local, configurable, and educational regression project for the Ames house price dataset. It is designed to help a learner study numeric, categorical, and ordinal features carefully while still following a production-structured workflow with reusable preprocessing, model comparison, validation, artifact saving, prediction generation, and repeatable feature experiments.

## Project Purpose

The project teaches how to:

- inspect numeric features and reason about skewness
- separate numeric, categorical, and ordinal columns explicitly
- define ordinal mappings safely instead of guessing
- build preprocessing with `ColumnTransformer` and `Pipeline`
- compare baseline and stronger models with cross-validation
- use permutation importance without leaking information
- decide which features deserve manual review instead of deleting them blindly
- experiment with feature groups, dropped columns, derived features, and preprocessing choices in a structured way

## Folder Structure

```text
task-house-prices/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
├── notebooks/
│   ├── experiment_report_dashboard.ipynb
│   ├── feature_processing_playground.ipynb
│   └── house_price_analysis.ipynb
├── notes/
│   ├── experiment-log.md
│   ├── learning-journal.md
│   └── submission-log.md
├── outputs/                       # Predictions and permutation importance outputs
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── experiment_config.py
│   ├── experiment_runner.py
│   ├── feature_builders.py
│   ├── feature_analysis.py
│   ├── feature_registry.py
│   ├── main.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── train.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Setup

1. Open a terminal in this directory:

```bash
cd /Users/maksymponomarenko/Documents/ai-engineering-kaggle-portfolio/courses/01-machine-learning-with-python/tasks/task-house-prices
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

## Data Location

The code checks these paths in order:

- `data/train.csv`
- `data/raw/train.csv`
- `data/test.csv`
- `data/raw/test.csv`

In the current folder layout, the default working files are:

- `data/raw/train.csv`
- `data/raw/test.csv`

## How To Run The Notebook

Start Jupyter from this task directory:

```bash
jupyter notebook
```

Then open:

- `notebooks/experiment_report_dashboard.ipynb`
- `notebooks/house_price_analysis.ipynb`
- `notebooks/feature_processing_playground.ipynb`

`experiment_report_dashboard.ipynb` is the report-review notebook for:

- seeing all saved experiments
- comparing every model across experiments
- tracking model metric changes over time
- drilling into one experiment or one model

`house_price_analysis.ipynb` is the guided learning notebook.

`feature_processing_playground.ipynb` is the hands-on sandbox for:

- choosing feature groups
- dropping columns
- adding derived features
- previewing data before preprocessing
- previewing transformed features after preprocessing
- running a single experiment and inspecting saved reports

The `notes/` folder keeps the experiment and learning history in a repo-friendly format:

- `experiment-log.md` for structured experiment summaries and conclusions
- `learning-journal.md` for broader learning notes
- `submission-log.md` for Kaggle submission tracking

## How To Run The Main Script

From this task directory:

```bash
python3 src/main.py
```

The script will:

- run the default experiment suite
- save per-experiment reports under `outputs/experiments/`
- compare models inside each experiment
- save a global summary table with all models across all experiments
- save a separate compact summary with only the best model per experiment
- save transformed previews, metrics, importances, and predictions

## Where To Edit Settings

The main configuration lives in:

- `src/config.py`

This is the place to edit:

- dataset paths
- target column
- random state
- train/test split size
- cross-validation folds
- feature groups
- ordinal mappings
- log-transform columns
- target transformation
- model hyperparameters

Experiment definitions live in:

- `src/experiment_config.py`

Feature groups live in:

- `src/feature_registry.py`

Derived feature builders live in:

- `src/feature_builders.py`

Column processing defaults and per-column overrides live in:

- `src/config.py`
- `src/processing_config.py`

This gives you three easy ways to experiment:

1. Change feature group membership in `src/feature_registry.py`
2. Add a new derived feature function in `src/feature_builders.py`
3. Create a new `ExperimentConfig(...)` in `src/experiment_config.py`
4. Override preprocessing for a specific column in `ExperimentConfig(..._processing_overrides=...)`

## Outputs And Artifacts

Running `python3 src/main.py` creates experiment folders under:

- `outputs/experiments/experiment_summary_all_models.csv`
- `outputs/experiments/experiment_summary_best_models.csv`
- `outputs/experiments/experiment_pivot_rmse_mean.csv`
- `outputs/experiments/experiment_pivot_mae_mean.csv`
- `outputs/experiments/experiment_pivot_r2_mean.csv`
- `outputs/experiments/<experiment_name>/model_comparison.csv`
- `outputs/experiments/<experiment_name>/holdout_metrics.json`
- `outputs/experiments/<experiment_name>/permutation_importance.csv`
- `outputs/experiments/<experiment_name>/final_pipeline.joblib`
- `outputs/experiments/<experiment_name>/test_predictions.csv`
- `outputs/experiments/<experiment_name>/reports/before_processing/*.csv`
- `outputs/experiments/<experiment_name>/reports/column_processing_summary.csv`
- `outputs/experiments/<experiment_name>/reports/processing_reference/*.csv`
- `outputs/experiments/<experiment_name>/reports/after_processing_preview.csv`

## Experiment Workflow

The project is now organized as a small feature lab:

- `feature_registry.py` defines reusable groups such as `numeric_core`, `ordinal_core`, and `categorical_core`
- `feature_builders.py` defines reusable derived features such as `TotalSF`, `HouseAge`, and `TotalBathrooms`
- `processing_config.py` documents default preprocessing options and their common alternatives
- `experiment_config.py` describes experiments declaratively
- `experiment_runner.py` loads data, applies feature builders, runs validation, and saves reports
- `reporting.py` saves before/after snapshots and column-processing summaries so you can inspect what preprocessing changed

The global summaries are meant for two different questions:

- `experiment_summary_all_models.csv`: how every model changed from experiment to experiment
- `experiment_summary_best_models.csv`: which single model won inside each experiment

The pivot summaries are meant for fast side-by-side comparison:

- `experiment_pivot_rmse_mean.csv`: experiments as rows, models as columns, RMSE as values
- `experiment_pivot_mae_mean.csv`: experiments as rows, models as columns, MAE as values
- `experiment_pivot_r2_mean.csv`: experiments as rows, models as columns, R² as values

## Column Processing Defaults

The default processing behavior is:

- Numeric: median imputation, optional `log1p` from the configured log list, standard scaling for linear models, no scaling for trees
- Categorical: most-frequent imputation and one-hot encoding
- Ordinal: constant imputation with `__MISSING__` plus explicit ordinal encoding from the mapping dictionary

Common alternatives already documented in `src/processing_config.py` and the playground notebook:

- Numeric imputation: `mean`, `most_frequent`, `constant`
- Numeric transforms: `none`, `log1p`, `sqrt`
- Numeric scaling: `none`, `standard`, `robust`
- Categorical imputation: `most_frequent`, `constant`
- Categorical rare-level grouping: set `min_frequency`
- Ordinal imputation: usually `constant`, sometimes `most_frequent`

Example experiment idea:

- start with `numeric_all`
- add `derived_feature_groups=["engineered_core"]`
- drop weak columns such as `["MiscVal", "PoolArea"]`
- compare `["linear_regression", "ridge", "random_forest"]`

## Notes For Learners

- Formal normality tests are rarely the main decision tool in predictive machine learning. In practice, skewness, outliers, leakage risk, and validation performance matter more.
- ANOVA and t-tests can be useful in explanatory analysis, but they are not the main feature selection method in predictive workflows.
- Cross-validation and permutation importance usually answer more practical questions: “Does this help prediction?” and “Does this still help after the rest of the pipeline is present?”
- A weak feature on its own can still help in combination with others, especially for nonlinear models.
- Multicollinearity matters more for interpreting linear coefficients than for tree-based models, although redundancy can still affect stability and efficiency.
## Research Notes

Use these files for the project story and experiment history:

- `notes/research-log.md` is the main canonical research log
- `notes/learning-journal.md` is for informal study notes and reflections
- `notes/submission-log.md` is for submission tracking

Recommended workflow:

- record each meaningful experiment and conclusion in `notes/research-log.md`
- treat notebook outputs and saved experiment folders as supporting evidence
- avoid creating new ad hoc decision files when the same information belongs in the research log
