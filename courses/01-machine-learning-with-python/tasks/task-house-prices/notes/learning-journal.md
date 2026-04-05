# Learning Journal

## Why this file exists

This file keeps the thinking process that used to live as large commented blocks inside `main.py`.
The goal is to preserve the learning trail without turning the training script into an archive of experiments.

## Early observations

### Missing values in numeric columns

One of the first checks was to count missing values in numeric features. The strongest gaps I noticed were:

- `LotFrontage`: 259 missing values
- `GarageYrBlt`: 81 missing values
- `MasVnrArea`: 8 missing values

This pushed me toward using median imputation in the first baseline instead of manually cleaning columns one by one.

### Strong correlations worth remembering

During the first pass over correlations, these pairs stood out:

- `SalePrice` and `OverallQual`
- `SalePrice` and `GrLivArea`
- `1stFlrSF` and `TotalBsmtSF`
- `GarageYrBlt` and `YearBuilt`
- `GrLivArea` and `TotRmsAbvGrd`
- `GarageCars` and `GarageArea`

Not all of them imply that a feature should be removed, but they are useful when thinking about redundancy, feature engineering, and model interpretation.

### Manual interpretation notes

- `1stFlrSF` and `TotalBsmtSF` may be close in some houses, but they represent different physical areas, so similarity does not automatically mean duplication.
- `GarageYrBlt` and `YearBuilt` can differ naturally because a garage may be added later.
- `GarageCars` and `GarageArea` likely carry overlapping information, so this pair may be useful to revisit in a future feature-selection pass.

## Why the current baseline is simple

For the first submission I intentionally kept the pipeline small:

- numeric columns only
- median imputation
- standard scaling
- linear regression

The point of this version is not to be the final model. The point is to have a readable, reproducible baseline that creates a valid submission and gives me a reference point for future iterations.

## What to try next

- compare this numeric-only baseline against a pipeline that also keeps categorical features
- evaluate models with Kaggle-oriented error metrics, not only local `R^2`
- move one-off EDA checks into separate notebooks or small analysis scripts instead of storing them in the training entrypoint
- track each new submission with a short note on what changed and why
