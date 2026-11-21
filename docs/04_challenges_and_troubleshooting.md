# Challenges, mitigations and troubleshooting

## Challenges from the problem statement → what we did
| Challenge | Mitigation | Where in code |
|---|---|---|
| Missing or inconsistent data | Schema validation (columns, numeric types, text→NaN); range rules drop impossible rows; median imputation inside the Pipeline | `data_ingestion.validate_schema`, `preprocessing.clean_data`, `features.build_preprocessor` |
| Overfitting on training data | 5-fold CV for model selection, Ridge (L2), tree limits (`min_samples_leaf`, `min_child_weight`, `max_depth`), subsampling, train-vs-test gap reported | `train.py`, `evaluate.build_leaderboard` |
| Highly correlated features | Correlation report (threshold 0.8); `AveBedrms` (0.85 with `AveRooms`) replaced by `BedroomRatio`; re-checked after clipping | `features.correlation_report`, `eda.run_eda` |
| Slow model training | XGBoost `hist` method, `n_jobs=-1`, RandomizedSearch instead of Grid, `--fast` flag | `train.py`, `run_pipeline.py` |

## Additional challenges we found in this dataset
| Challenge | Mitigation |
|---|---|
| Target censored at $500k (992 rows) | Dropped — their true value is unknown; limitation documented |
| Extreme outliers (`AveOccup` up to 1,243) | Winsorize at 1st/99th percentile, learned on train only |
| Data leakage risk | Split before fitting any statistic; everything stateful lives in the Pipeline |
| Training/serving skew | Same Pipeline object saved and loaded by the app |
| Unrepresentative test set | Split stratified on income bands |
| Bad user input | `validate_input`: required, numeric, valid range, inside California, bedrooms ≤ rooms; warnings when outside training range |
| Division by zero in engineered ratios | `safe_divide` → NaN → imputed (never `inf`) |
| Over-confident verdicts | "Fair" band = ± model's real test MAE |
| ANN overfitting | Dropout, L2 weight decay, early stopping on a validation split with best weights restored |
| ANN sensitive to feature scale | StandardScaler in the shared pipeline; outliers clipped before scaling |
| ANN training is slow (~40 s per fit) | Smaller search budget (6 configs), larger batch sizes, `--fast` mode uses 2 configs |
| ANN results vary between runs | Fixed seed via `keras.utils.set_random_seed` for weights and shuffling |
| Keras models can't be pickled like sklearn models | `__getstate__`/`__setstate__` store the model as `.keras` bytes inside the `.pkl` |

## Troubleshooting
| Symptom | Fix |
|---|---|
| App says "No trained model found" | Run `python run_pipeline.py --fast` first |
| `ModuleNotFoundError: fairprice` | Run commands from the project root (`fairprice-house-price/`) |
| Dataset download fails (offline) | Copy `data/raw/california_housing.csv` from a teammate; the loader uses the cache first |
| Pipeline is slow | Use `--fast` (4 search iterations, 2 for the ANN) or `--skip-eda`; the full run takes ~15 min, mostly the ANN |
| `ModuleNotFoundError: keras` / TensorFlow errors | `pip install tensorflow` (Python 3.9–3.12); restart the app |
| Lots of TensorFlow log lines | Harmless; `fairprice/ann.py` already sets `TF_CPP_MIN_LOG_LEVEL=2` |
| `AttributeError` when loading `.pkl` | Model was saved with different library versions — `pip install -r requirements.txt` and retrain |
| Windows: `can't open file` with a very long path | Move the project to a short path such as `C:\projects\` (260-char path limit) |
| Dollar amounts look like math in Streamlit | Markdown treats `$…$` as LaTeX; use the `md()` helper in the app |
