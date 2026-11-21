# Evaluation prep — criteria → evidence, and questions to be ready for

## How the project meets each criterion
| # | Criterion | Evidence to show |
|---|---|---|
| 1 | Problem-solving & design thinking | `01_design.md` decision table (why + trade-off for every choice); EDA finding → decision list |
| 2 | Working solution & code quality | `python run_pipeline.py` runs end to end; Streamlit demo; modular package; 33 pytest tests incl. edge cases; Git history |
| 3 | Individual ownership | Each member owns modules (table below) and commits their own work |
| 4 | AI fluency | We can explain every line; AI was used to speed up boilerplate, decisions are ours and justified with data (see "AI usage") |
| 5 | Learning agility & communication | Honest limitations section; "next improvements" list; readable code with docstrings |

## Suggested ownership split (edit to your team)
| Member | Owns | Must be able to explain |
|---|---|---|
| A | `data_ingestion.py`, `preprocessing.py`, tests in `test_data.py` | Why capped rows are dropped; schema checks; what each cleaning rule catches |
| B | `eda.py`, `features.py`, `test_features.py` | Each EDA plot → decision; DistToCity; why clipping happens inside the Pipeline |
| C | `train.py`, `evaluate.py`, `explain.py` | CV vs test; why XGBoost won; Ridge ≈ Linear (underfitting); reading SHAP |
| C or E | `ann.py`, `tests/test_ann.py`, ANN training curve | Layers, activations, He vs Glorot, dropout, early stopping, why the ANN doesn't beat XGBoost |
| D | `predict.py`, `app/`, `test_predict.py`, README | Validation rules; Fair band = ± MAE; training/serving consistency |

Commit in small steps with clear messages (e.g. `feat(features): add DistToCity feature`) so each
person's contribution is visible in `git log`.

## Questions judges are likely to ask
**Why did you drop the capped rows instead of keeping them?**
`5.00001` means "≥ $500k" — the real value is unknown. Keeping them teaches the model a fake ceiling.
Trade-off: the model can't predict above $500k; we show this in the residual plot.

**Why is preprocessing inside the Pipeline instead of done on the whole dataset?**
Imputer medians, clip bounds and scaler means are statistics. Computing them on all data would leak
test information into training. Inside the Pipeline they're fitted on train folds only.

**Why choose the model on CV RMSE instead of test RMSE?**
If we pick by test score, the test set stops being an unbiased estimate. CV picks; test confirms.

**Why does Ridge give the same score as Linear Regression?**
Linear Regression is underfitting (train error ≈ test error). Regularization fixes overfitting, so it
has nothing to fix here. The fix for underfitting is a more flexible model — the trees.

**XGBoost has train RMSE 0.20 but test 0.38 — isn't that overfitting?**
Some gap is normal for boosted trees. The key check is that CV RMSE (0.405) and test RMSE (0.381)
agree, so it generalizes. We also limited it with `min_child_weight`, subsampling and CV-tuned depth.

**Latitude and Longitude are −0.92 correlated — why keep both?**
That's California's diagonal coastline. Each alone can't locate a district; trees use them together.
We removed redundancy only where one feature is just a noisy copy (`AveBedrms` vs `AveRooms`).

**How does the Fair Price verdict work?**
Fair band = predicted ± test MAE ($25.6k). Inside = Fair, above = Overpriced, below = Good Deal.
We don't call something overpriced when the difference is within the model's own typical error.

**What happens if a user enters garbage?**
`validate_input` blocks missing/non-numeric/impossible values, locations outside California, and
bedrooms > rooms; it warns (but allows) values outside the training range. All covered by tests.

**Describe your neural network.**
A Keras feed-forward network: the 10 scaled features go through 2-3 Dense hidden layers (ReLU or ELU,
He or Glorot initialisation, L2 penalty), each followed by Dropout, and a single linear output neuron.
It minimises MSE with Adam or SGD+momentum on mini-batches and stops early when validation loss stops
improving. Layer sizes, activation, initialiser, dropout, L2, optimizer, learning rate and batch size
were tuned with RandomizedSearchCV under the same 5-fold CV as the other models.

**Why doesn't the neural network beat XGBoost?**
On small-to-medium tabular data (~16k rows, 10 features), gradient-boosted trees usually win: they
handle thresholds and feature interactions (e.g. "coastal *and* high income") directly, while a
network has to learn them from scratch and needs more data. The ANN (R² 0.78) still clearly beats the
linear baseline (0.66), which shows it learned real non-linear structure. We picked the production model
on evidence rather than on which algorithm sounds more advanced.

**Why He initialisation? Why dropout? Why early stopping?**
He init keeps activations from shrinking or exploding with ReLU layers (Glorot suits tanh/ELU-like
activations — both were in the search). Dropout randomly disables neurons during training so the
network can't memorise the training set. Early stopping halts training when validation loss stops
improving and restores the best weights (see `11_ann_training_curve.png`).

**Why wrap Keras in a scikit-learn class?**
So the ANN uses the exact same preprocessing pipeline, cross-validation and hyperparameter search as
the conventional models — a fair comparison — and is saved as a `.pkl` the app can load.

**What would you do with more time?**
Per-prediction intervals (quantile regression), log target, location clusters, inflation adjustment.

## AI usage (be upfront about it)
AI tools were used to accelerate scaffolding and boilerplate. Every design decision is backed by our
own EDA numbers, and we verified outputs by running the pipeline, reading the plots and writing tests.
Be ready to change a parameter live (e.g. `CLIP_QUANTILES` in `config.py`) and explain the effect.
