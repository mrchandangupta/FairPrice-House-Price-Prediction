# Phase 0 — Design: FairPrice (California House Value Prediction)

## 1. Problem statement
Buyers and sellers rarely know whether a property is priced fairly; prices are set by gut feel
or an agent's opinion. **FairPrice** predicts the median house value of a California district
from its census features, explains *why* it predicted that value, and tells the user whether
an asking price looks **Overpriced / Fair / Good Deal**.

## 2. Dataset — California Housing (scikit-learn, 1990 US Census)
| Item | Value |
|---|---|
| Rows | 20,640 census block groups ("districts") |
| Features | `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` |
| Target | `MedHouseVal` — median house value in **$100,000s** (1990 dollars) |
| Missing values | None in the source, but the pipeline still imputes (future / user data may have gaps) |

**Honest limitations (we say these out loud in the demo):**
- One row = a district, not a single house → we predict a *district-level* typical value.
- Prices are 1990 values, so the app shows 1990 dollars.
- Target is **capped at 5.00001 ($500k)** for 992 rows and `HouseAge` is capped at 52.

## 3. Objectives
1. End-to-end ML pipeline with at least one conventional ML model and one Artificial Neural Network:
   ingest → validate → clean → EDA → features → train → tune → evaluate → serve.
2. Compare several regression models and pick the best with evidence (CV + held-out test).
3. Explain predictions (feature importance + SHAP).
4. Streamlit UI with a Fair Price Checker and input validation.

## 4. Approach and trade-offs (why we chose what we chose)
| Decision | Why | Trade-off |
|---|---|---|
| Split train/test **first**, fit all cleaning steps on train only | Prevents data leakage → honest test score | Slightly more code (sklearn `Pipeline`) |
| All preprocessing + features inside **one sklearn Pipeline** saved as `.pkl` | App applies exactly the same transforms as training (no training/serving skew) | Custom transformers must be importable |
| **Drop capped target rows** (`MedHouseVal >= 5.0`) | True value is unknown (censored); keeping them teaches the model a fake ceiling | Model can't predict above ~$500k; documented |
| **Clip (winsorize) outliers** at 1st/99th percentile instead of deleting | `AveOccup` reaches 1,243 — data errors — but deleting rows loses information | Extreme districts are pulled to the edge |
| Engineer `BedroomRatio`, `RoomsPerPerson`, `DistToCity` | Domain knowledge: space per person & proximity to LA/SF drive prices | More features to explain |
| Drop `AveBedrms` after creating `BedroomRatio` | 0.85 correlation with `AveRooms` (multicollinearity) | Loses a raw feature (info kept in the ratio) |
| Conventional models: Linear (baseline) → Ridge → Random Forest → XGBoost | Baseline shows how much complexity buys; Ridge = regularization; trees capture non-linearity | Trees are less transparent → mitigated with SHAP |
| **ANN (Keras)**: Dense layers + Dropout + L2, He/Glorot init, Adam/SGD, early stopping | Required by the brief; learns non-linear interactions; compared on equal terms with conventional models | ~40 s per fit on CPU → smaller search budget (6 configs); less interpretable → permutation importance / SHAP |
| ANN wrapped as a scikit-learn estimator | Same Pipeline, same CV, same tuning, same `.pkl` saving as every other model | Custom pickling code for the Keras model |
| `RandomizedSearchCV` over `GridSearchCV` | Same quality in a fraction of the time | Not exhaustive |
| 5-fold cross-validation | Confirms the score isn't luck of one split | 5× training time |
| Fair/Over/Under band = ± model's test MAE | Verdict reflects how accurate the model really is | Band is wide where the model is uncertain |

## 5. Architecture
```
[sklearn fetch / local CSV]
        │  data_ingestion.py   (load + schema validation)
        ▼
[Raw data]  ──►  preprocessing.py  (range validation, drop capped target, dedupe)
        │                 └── saves data/processed/housing_clean.csv
        ▼
   eda.py  → reports/figures (distributions, heatmap, map, outliers)
        ▼
 train/test split (80/20)
        ▼
 ┌──────────── sklearn Pipeline (fit on train only) ─────────────┐
 │ FeatureEngineer → SimpleImputer(median) → OutlierClipper →    │
 │ StandardScaler → Regressor (Linear | Ridge | RF | XGB | ANN)  │
 └───────────────────────────────────────────────────────────────┘
        ▼  train.py  (5-fold CV + RandomizedSearchCV)
 evaluate.py → RMSE / MAE / R², residual plots, leaderboard
 explain.py  → SHAP summary + per-prediction contributions
        ▼
 models/best_model.pkl  +  models/model_meta.json
        ▼
 app/streamlit_app.py  →  predict.py (validate input → predict → verdict → explanation)
```

## 6. Evaluation metrics
RMSE (main selection metric), MAE, R², 5-fold CV mean ± std, residual plots.

## 7. Challenges & mitigations
| Challenge | Mitigation in this project |
|---|---|
| Missing / inconsistent data | Schema + range validation rules, median imputation inside the pipeline |
| Outliers (`AveOccup` 1,243, `AveRooms` 141) | Percentile clipping learned from train only |
| Censored target ($500k cap) | Drop capped rows; documented limitation |
| Overfitting | K-fold CV, Ridge regularization, tree depth/min-samples limits, train-vs-test gap reported |
| Highly correlated features | Correlation analysis; replace `AveBedrms` with `BedroomRatio` |
| Slow training | XGBoost (`hist`), `n_jobs=-1`, RandomizedSearch, `--fast` mode |
| Bad user input in the app | Validation (negatives, bedrooms > rooms, outside California) + extrapolation warnings |

## 8. Deliverables
Pipeline scripts + notebook · cleaned dataset · trained model `.pkl` · Streamlit UI ·
README · architecture doc · results & challenges doc · tests · Git repository.
