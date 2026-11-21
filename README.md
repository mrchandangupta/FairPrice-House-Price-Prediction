# FairPrice — California House Value Prediction

Predicts the median house value of a California district, explains **why** with SHAP, and checks
whether an asking price is **Overpriced / Fair / Good Deal**.

**Best model:** XGBoost — test **R² 0.841**, **MAE ≈ $25,600**, RMSE 0.381 ($100k units),
5-fold CV RMSE 0.405 ± 0.012.
**Neural network (Keras ANN):** test R² 0.783, MAE ≈ $30,500. Baseline Linear Regression: R² 0.660, MAE ≈ $41,400.

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py --fast        # ~1 min: ingest → clean → EDA → train → evaluate → save model
streamlit run app/streamlit_app.py   # open http://localhost:8501
python -m pytest -q                  # 33 tests
```
Use `python run_pipeline.py` (no flag) for the full hyperparameter search (~20 min, mostly the ANN).
VS Code: open this folder; run configurations are in `.vscode/launch.json`.

## Workflow (phases)
| Phase | What | Code |
|---|---|---|
| 0 | Design & trade-offs | `docs/01_design.md` |
| 1 | Data ingestion + schema validation | `fairprice/data_ingestion.py` |
| 2 | Cleaning (range rules, dedupe, drop capped target) | `fairprice/preprocessing.py` |
| 3 | EDA (distributions, outliers, heatmaps, map) | `fairprice/eda.py` |
| 4 | Feature engineering + correlation analysis | `fairprice/features.py` |
| 5 | Training: conventional ML (Linear, Ridge, Random Forest, XGBoost) | `fairprice/train.py` |
| 5b | Training: Artificial Neural Network (Keras) | `fairprice/ann.py` |
| 6 | Hyperparameter tuning (RandomizedSearchCV) + 5-fold CV | `fairprice/train.py` |
| 7 | Evaluation (RMSE, MAE, R², residuals) + SHAP | `fairprice/evaluate.py`, `fairprice/explain.py` |
| 8 | Inference + Streamlit UI | `fairprice/predict.py`, `app/streamlit_app.py` |

## Project structure
```
fairprice-house-price/
├── app/streamlit_app.py        # UI only
├── fairprice/                  # the ML package (one module per phase)
│   ├── config.py               # paths, ranges, hyperparameters
│   ├── data_ingestion.py  preprocessing.py  eda.py  features.py
│   ├── ann.py  train.py  evaluate.py  explain.py  predict.py  logger.py
├── data/processed/housing_clean.csv   # cleaned dataset (deliverable)
├── models/best_model.pkl + ann_model.pkl + model_meta.json
├── reports/figures/*.png  leaderboard.csv  metrics.json
├── notebooks/01_end_to_end_walkthrough.ipynb
├── tests/                      # pytest: data, features, predict + edge cases
├── docs/                       # design, architecture, results, challenges, evaluation prep
├── presentation/               # design review deck + build_deck.py
└── run_pipeline.py             # runs phases 1-8
```

## Dataset
scikit-learn `fetch_california_housing` — 20,640 districts from the 1990 US Census, 8 features,
target `MedHouseVal` in $100,000s. Predictions are **district medians in 1990 dollars**.

## Key design decisions
- Capped target rows (≥ $500k, 992 rows) removed — their true value is unknown.
- All learned preprocessing (imputation, outlier clipping, scaling) lives in one sklearn Pipeline
  fitted on train only → no leakage, and the app uses the exact same transforms.
- Engineered features: `BedroomRatio` (replaces the redundant `AveBedrms`), `RoomsPerPerson`,
  `DistToCity` (km to nearest major city — 2nd strongest predictor).
- An ANN (Keras: Dense + Dropout + L2, He/Glorot init, Adam/SGD, early stopping) is wrapped as a
  scikit-learn estimator, so it shares the pipeline, CV and tuning with the conventional models.
- Best model chosen on CV RMSE; the test set is only used for the final report. The app shows the
  ANN's estimate next to the best model's as a second opinion.
- Fair-price band = prediction ± test MAE.

## Deliverables
| Deliverable | Location |
|---|---|
| ML notebook / scripts for full workflow | `notebooks/01_end_to_end_walkthrough.ipynb`, `run_pipeline.py` |
| Cleaned dataset | `data/processed/housing_clean.csv` |
| Trained regression models (.pkl) | `models/best_model.pkl`, `models/ann_model.pkl` |
| UI for predictions | `app/streamlit_app.py` |
| Documentation (architecture, results, challenges) | `docs/` |
| README + architecture document | this file, `docs/02_architecture.md` |
| Design review deck | `presentation/FairPrice_Design_Review.pptx` |

## Documentation
- [Design & trade-offs](docs/01_design.md)
- [Architecture](docs/02_architecture.md)
- [Results](docs/03_results.md)
- [Challenges, mitigations & troubleshooting](docs/04_challenges_and_troubleshooting.md)
- [Evaluation prep & Q&A](docs/05_evaluation_prep.md)
