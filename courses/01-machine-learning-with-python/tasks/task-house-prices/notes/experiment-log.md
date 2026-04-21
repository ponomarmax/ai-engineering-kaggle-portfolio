# Experiment Log

This log turns the saved experiment outputs into a readable story.
The goal is to preserve the reasoning behind each step, not only the final score.

## How to read this file

- `Hypothesis`: what the experiment was trying to test
- `Change`: which feature or preprocessing change was introduced
- `Models`: which models were compared
- `Best result`: the strongest result inside that experiment
- `Conclusion`: what to keep in mind for the next iteration

## Experiment: `Only_TotalSF_Without_its_featrues`

- Hypothesis: replacing `1stFlrSF`, `2ndFlrSF`, and `TotalBsmtSF` with `TotalSF` may keep most of the size signal while simplifying the feature set.
- Change: added `TotalSF`, `HouseAge`, and `TotalBathrooms`; dropped `TotalBsmtSF`, `1stFlrSF`, and `2ndFlrSF`; used `log1p` with robust scaling on `TotalSF`.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `30419.765`, MAE `17843.893`, R² `0.842961`
- Conclusion: `TotalSF` is a strong engineered summary feature and can carry most of the useful area signal on its own.

## Experiment: `1 - TotalSF-focused_experiment: transform=none, scale=quantile`

- Hypothesis: `TotalSF` may work better without `log1p`, with quantile scaling instead of the earlier robust-scaled version.
- Change: kept the same simplified feature idea, but changed `TotalSF` processing to `transform=none` and `scale=quantile`.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `30435.378`, MAE `17849.811`, R² `0.842819`
- Conclusion: this variant stayed very close to the previous `TotalSF` experiment, so the main gain came from the feature idea itself, not from this preprocessing tweak.

## Experiment: `2 - GarageAreaPerCar: added new feature`

- Hypothesis: `GarageAreaPerCar = GarageArea / GarageCars` may capture garage efficiency better than keeping both garage features separately.
- Change: added `GarageAreaPerCar`; dropped both `GarageArea` and `GarageCars`; used quantile scaling for the new feature.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `31741.907`, MAE `18252.878`, R² `0.827497`
- Conclusion: replacing `GarageArea` and `GarageCars` with a single ratio hurt all tested models. The ratio was not strong enough to preserve the original garage signal.

## Experiment: `3 - GarageAreaPerCar: transform=log1p`

- Hypothesis: the weaker `GarageAreaPerCar` result might come from skewness rather than from the feature definition itself.
- Change: kept the same setup as experiment 2, but changed `GarageAreaPerCar` to `log1p`.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `31741.907`, MAE `18252.878`, R² `0.827497`
- Conclusion: `log1p` did not improve the ratio feature. The issue appears to be the feature formulation, not the transform.

## Experiment: `4 - HasGarage: GarageArea >0`

- Hypothesis: a simple binary indicator for garage presence might add useful signal on top of `GarageArea`.
- Change: added `HasGarage`; kept `GarageArea`; dropped `GarageCars`; prevented scaling for the binary feature.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `31419.111`, MAE `18064.901`, R² `0.829876`
- Conclusion: `HasGarage` was weaker than the original `GarageCars` + `GarageArea` pair. It may still be useful as a compact indicator, but not as a replacement for the stronger garage features.

## Experiment: `6 - Add GarageAge(transform=none+quantile)-GarageYrBlt`

- Hypothesis: `GarageAge = YrSold - GarageYrBlt` may capture garage recency better than a binary garage-presence flag.
- Change: replaced `HasGarage` inside the engineered core with `GarageAge`; kept the rest of the stronger baseline structure; dropped raw `GarageYrBlt`; used constant imputation with quantile scaling for `GarageAge`.
- Models: `linear_regression`, `ridge`, `lasso`, `random_forest`
- Best result: `random_forest`, RMSE `30422.443`, MAE `17889.572`, R² `0.842771`
- Conclusion: `GarageAge` fit the stronger baseline much better than `HasGarage` and stayed close to the best `TotalSF`-focused setups. It is a more informative engineered garage feature than the earlier binary indicator.

## Experiment: `playground_custom_experiment`

- Hypothesis: keep the strongest current feature set together and inspect preprocessing behavior before locking changes into a more formal experiment.
- Change: used the notebook playground configuration with `numeric_core`, `ordinal_core`, `categorical_core`, plus `TotalSF`, `HouseAge`, and `TotalBathrooms`.
- Models: `linear_regression`, `ridge`, `random_forest`
- Best result: `random_forest`, RMSE `30385.802`, MAE `17995.027`, R² `0.843584`
- Conclusion: this is currently the strongest saved configuration among the reviewed experiment outputs and a good reference point for future feature tests.

## Cross-experiment takeaways

- `random_forest` has been the strongest model in every saved experiment so far.
- `TotalSF` is a useful engineered feature and has held up across multiple variants.
- `GarageCars` and `GarageArea` are stronger as separate signals than as a single ratio feature.
- `GarageAge` is a stronger garage-focused engineered feature than `HasGarage` and is a better fit for the current engineered core.
- Preprocessing changes matter, but in these runs the feature definition mattered more than the transform applied to that feature.

## Next experiments worth trying

- Keep `GarageArea` and `GarageCars` together, then add `GarageAreaPerCar` as an extra feature instead of a replacement.
- Compare `GarageAge` against the raw `GarageYrBlt` column directly to confirm whether the engineered age form is consistently more useful.
- Test one garage-focused experiment at a time while keeping the rest of the feature set fixed, so the result is easier to interpret.
