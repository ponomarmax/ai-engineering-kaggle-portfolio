# House Prices Research Log

This file is the main running log for experiments, decisions, and conclusions in the `task-house-prices` project.

Why this file exists:
- to keep the modeling journey in one place
- to show what was tested and why
- to record what worked, what did not, and what was intentionally not pursued further
- to help future readers follow the sequence of decisions without digging through many notebooks and temporary notes

Suggested rule going forward:
- each meaningful experiment block gets one short entry
- each entry should capture the hypothesis, setup, result, and decision
- if a notebook/dashboard image is especially useful, it can be referenced as supporting material, but the written conclusion should stand on its own

---

## Entry 001: Garage quality and condition interaction review

### Goal
Understand whether `GarageQual` and `GarageCond` add meaningful predictive value on top of the restored strong baseline, either directly or through interaction features.

### Baseline context
The comparison was made against the restored stronger baseline configuration built around:
- `numeric_core`, `ordinal_core`, `categorical_core`
- engineered features from `engineered_core`
- explicit processing overrides for key columns such as `LotArea`, `GrLivArea`, `HouseAge`, and `TotalSF`

### Tested feature variants
The following garage-related variants were compared:
- `only_GarageQual`
- `only_GarageCond`
- `garage_qual_x_garage_area`
- `garage_qual_x_garage_cars`
- `garage_cond_x_garage_area`
- `garage_cond_x_garage_cars`
- `garage_qual_x_garage_cond_x_garage_area`
- `garage_qual_x_garage_cond_x_garage_cars`

### Models compared
- `linear_regression`
- `ridge`
- `lasso`
- `random_forest`

### What was measured
The comparison focused on cross-validated regression metrics:
- `RMSE`
- `MAE`
- `R²`

The main question was not whether metrics moved slightly, but whether the movement was large and stable enough to justify more feature complexity.

### Result summary
The garage quality/condition experiments produced only small metric changes relative to the restored baseline.

Observed pattern:
- `random_forest` stayed very strong, but the improvements from garage interactions were marginal
- `linear_regression`, `ridge`, and `lasso` changed only slightly across variants
- no garage interaction feature produced a clearly dominant and stable win across models

Practical interpretation:
- `GarageQual` and `GarageCond` do contain some signal
- but that signal is mostly weak once `GarageArea`, `GarageCars`, and the stronger baseline features are already present
- multiplying these garage ordinal features by area/cars increased feature complexity more than it improved validation quality

### Decision
Do not prioritize further feature engineering around `GarageQual` / `GarageCond` interactions right now.

Current decision:
- keep the restored stronger baseline as the main reference state
- treat garage interaction exploration as completed for now
- move attention to higher-leverage directions such as model tuning, stronger feature groups, or different engineered aggregates

### Why this decision is reasonable
This is a good stopping point because:
- the experiments were broad enough to test both direct and interaction-based uses of garage quality/condition
- the observed gains were too small to justify extra complexity in the final pipeline
- the project benefits more from a clean, explainable baseline than from many weak interaction features

### Archive note
For this type of exploration, a written summary is sufficient as the main record.

Optional supporting artifacts:
- dashboard heatmaps or experiment tables can still be kept in `outputs/experiments/` for local inspection
- if needed later, a screenshot can be added as supporting evidence, but the project should not depend on images to explain the conclusion

### Next step
Move on from garage quality/condition interaction search and focus on:
- model tuning for the strongest candidates
- validation stability
- higher-value feature ideas with clearer expected upside

