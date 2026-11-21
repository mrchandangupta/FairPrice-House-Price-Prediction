# Results

Full run: `python run_pipeline.py` (20 RandomizedSearch iterations × 5-fold CV per conventional model,
6 iterations for the ANN).
Source of truth: `reports/leaderboard.csv`, `reports/metrics.json`, `models/model_meta.json`.

## Data cleaning
| Step | Rows |
|---|---|
| Raw | 20,640 |
| Missing target / duplicates / out-of-range | 0 / 0 / 0 removed |
| Capped target (≥ $500k, censored) | 992 removed |
| **Clean dataset** | **19,648** → train 15,718 / test 3,930 |

## Model leaderboard (target in $100k units)
| Model | CV RMSE (mean ± std) | Test RMSE | Test MAE | Test R² | Train–test gap |
|---|---|---|---|---|---|
| **XGBoost** | **0.405 ± 0.012** | **0.381** | **0.256 (≈ $25,600)** | **0.841** | 0.18 |
| Random Forest | 0.439 ± 0.012 | 0.414 | 0.279 (≈ $27,900) | 0.813 | 0.21 |
| **ANN (Keras)** | 0.472 ± 0.010 | 0.446 | 0.306 (≈ $30,500) | 0.783 | 0.02 |
| Ridge | 0.574 ± 0.013 | 0.557 | 0.414 (≈ $41,400) | 0.661 | −0.02 |
| Linear Regression (baseline) | 0.574 ± 0.013 | 0.557 | 0.414 (≈ $41,400) | 0.660 | −0.02 |

Best XGBoost parameters: `n_estimators=500, learning_rate=0.03, max_depth=8, min_child_weight=5,
subsample=0.85, colsample_bytree=0.85, reg_lambda=1`.

Best ANN configuration: 3 hidden layers (256-128-64), ELU activation, Glorot initialisation, Adam,
batch size 64, L2 = 1e-5, no dropout (the search preferred L2 alone), up to 150 epochs with early stopping.

## Interpretation
- **Trees beat linear models by ~38% MAE.** Price depends on location and income non-linearly
  (coastal clusters), which straight lines cannot capture.
- **Ridge ≈ Linear.** Regularization doesn't help because the linear model is *underfitting*, not
  overfitting (train and test errors are equal). This is evidence, not a failure.
- **The ANN beats the linear baseline by a wide margin** (R² 0.78 vs 0.66), so it learned real non-linear
  structure, but trails the tree ensembles. That's typical for ~16k rows of tabular data: trees model
  thresholds and interactions directly, a network has to learn them from data.
- **The ANN barely overfits** (train-test gap 0.02). Its learning curve (`11_ann_training_curve.png`)
  shows validation loss still falling slowly at epoch 150, so early stopping never triggered: a longer
  training budget would likely improve it a little.
- **CV std is small (≈ 0.012)** → the scores are stable across folds, not luck of one split.
- **Train–test gap for trees (0.18)** is expected for boosted trees; what matters is that CV and test
  RMSE agree (0.405 vs 0.381), i.e. the model generalizes.

## What drives prices (SHAP, `reports/figures/10_shap_summary.png`)
1. `MedInc` — higher income → higher value (strongest driver).
2. `Latitude` / `Longitude` — location; southern-coastal districts are pricier.
3. `DistToCity` (engineered) — further from a major city → lower value. Correlation with target −0.40,
   second only to income.
4. `AveOccup` — crowded households → lower value (only visible after outlier clipping: correlation
   went from −0.02 raw to −0.25 clipped).

## Known limitations (stated honestly)
- **Residuals widen for expensive districts** and the model under-predicts near the $500k ceiling
  (`08_residual_analysis.png`) — because capped rows were removed, the model has few examples there.
- Predictions are **district medians in 1990 dollars**, not individual-house prices today.
- Fair-price band = ± test MAE ($25.6k) — a single global band; uncertainty actually varies by district.

## Next improvements
Train the ANN for more epochs (300+) with a learning-rate schedule · quantile regression for per-prediction intervals · log-transform the target · K-means location
clusters as features · inflation adjustment to present-day dollars.
