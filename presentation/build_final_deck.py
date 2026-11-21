"""Builds the 10-slide final presentation: python presentation/build_final_deck.py

Results are read from reports/leaderboard.csv and models/model_meta.json, so the deck always
matches the trained models. Shares its palette and helpers with build_deck.py (design review).
"""
import json

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from build_deck import (BODY, CARD, HEAD, ICE, INK, MUTED, NAVY, NAVY_LIGHT, ROOT, TEAL, TEAM, TEAM_NAME,
                        TERRA, WHITE, background, badge, card, text)

OUT = ROOT / "presentation" / "FairPrice_Final_Presentation.pptx"
FIGURES = ROOT / "reports" / "figures"
BOARD = pd.read_csv(ROOT / "reports" / "leaderboard.csv")
META = json.loads((ROOT / "models" / "model_meta.json").read_text())
TERRA_TINT = RGBColor(0xF7, 0xE4, 0xDF)
TEAL_TINT = RGBColor(0xDC, 0xEE, 0xEC)
NAMES = {"XGBoost": "XGBoost", "RandomForest": "Random Forest", "ANN": "ANN (Keras)",
         "Ridge": "Ridge", "LinearRegression": "Linear Regression"}


def row(model: str) -> pd.Series:
    return BOARD[BOARD["model"] == model].iloc[0]


BEST, ANN, LINEAR = row("XGBoost"), row("ANN"), row("LinearRegression")
RMSE_CUT = (LINEAR["test_rmse"] - BEST["test_rmse"]) / LINEAR["test_rmse"] * 100
MAE_CUT = (LINEAR["test_mae"] - BEST["test_mae"]) / LINEAR["test_mae"] * 100


# ---------------------------------------------------------------- helpers
def title(slide, heading, kicker):
    text(slide, 0.6, 0.4, 12, 0.3, kicker.upper(), size=11, color=TERRA, bold=True, name="Kicker")
    text(slide, 0.6, 0.68, 12.1, 0.7, heading, size=30, color=NAVY, bold=True, font=HEAD, name="Title")


def footer(slide, number):
    text(slide, 0.6, 7.05, 8, 0.25, f"FairPrice  |  Team {TEAM_NAME}  |  Final presentation", size=10, color=MUTED)
    text(slide, 11.73, 7.05, 1.0, 0.25, str(number), size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def label(slide, x, y, w, words):
    text(slide, x, y, w, 0.3, words.upper(), size=11, color=TERRA, bold=True)


def table(slide, x, y, rows, col_w, font=12, row_h=0.34, fills=None, bold_rows=(), name=None):
    """rows[0] is the header. fills: {row_index: RGBColor} overrides the zebra striping."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_w)), Inches(row_h * len(rows)))
    if name:
        shape.name = name
    grid = shape.table
    for j, width in enumerate(col_w):
        grid.columns[j].width = Inches(width)
    for i, values in enumerate(rows):
        grid.rows[i].height = Inches(row_h)
        for j, value in enumerate(values):
            cell = grid.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif fills and i in fills:
                cell.fill.fore_color.rgb = fills[i]
            else:
                cell.fill.fore_color.rgb = CARD if i % 2 == 0 else WHITE
            cell.text_frame.word_wrap = True
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = str(value)
            run.font.size, run.font.name = Pt(font), BODY
            run.font.bold = i == 0 or i in bold_rows
            run.font.color.rgb = WHITE if i == 0 else INK
    return shape


def numbered(slide, x, y, w, items, step, fill=TEAL, size=13, badge_size=0.34):
    """Numbered list; each item is a string or a list of (text, overrides) runs."""
    for i, item in enumerate(items):
        top = y + i * step
        badge(slide, x, top, str(i + 1), fill=fill, size=badge_size, font_size=11)
        text(slide, x + badge_size + 0.18, top + 0.01, w - badge_size - 0.18, step, [item], size=size, color=INK)


def picture(slide, name, x, y, w, alt):
    pic = slide.shapes.add_picture(str(FIGURES / f"{name}.png"), Inches(x), Inches(y), width=Inches(w))
    pic.name = alt
    return pic


def new_slide(prs, color=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background(slide, color)
    return slide


# ---------------------------------------------------------------- 1. title and team
def slide_title(prs):
    s = new_slide(prs, NAVY)
    text(s, 0.6, 0.4, 8, 0.3, f"TEAM {TEAM_NAME.upper()}", size=11, color=TERRA, bold=True)
    for i, (name, reg, branch) in enumerate(TEAM):
        x = 0.6 + i * 3.08
        card(s, x, 0.75, 2.88, 1.0, fill=NAVY_LIGHT, radius=0.08, name=f"Member {i + 1}")
        text(s, x + 0.2, 0.87, 2.55, 0.35, name, size=14, color=WHITE, bold=True)
        text(s, x + 0.2, 1.25, 2.55, 0.35, f"Reg. No. {reg}  |  {branch}", size=12, color=ICE)

    text(s, 0.6, 2.3, 10, 0.3, "DATA SCIENCE HACKATHON  |  REGRESSION TRACK  |  FINAL PRESENTATION",
         size=12, color=TERRA, bold=True)
    text(s, 0.6, 2.65, 10, 1.0, "FairPrice", size=56, color=WHITE, bold=True, font=HEAD)
    text(s, 0.6, 3.75, 10.5, 0.9, "California house value prediction with conventional machine learning "
         "and an artificial neural network, served through an explainable fair-price app.", size=20, color=ICE)

    stats = [(f"{BEST['test_r2']:.3f}", "test R² (XGBoost, best model)"),
             (f"${BEST['test_mae_dollars'] / 1000:.1f}k", "typical error (test MAE)"),
             ("5", "models: 4 conventional + 1 ANN"),
             (f"{RMSE_CUT:.0f}%", "lower RMSE than the linear baseline")]
    for i, (value, caption) in enumerate(stats):
        x = 0.6 + i * 3.08
        text(s, x, 5.25, 2.9, 0.7, value, size=34, color=WHITE, bold=True, font=HEAD)
        text(s, x, 6.0, 2.7, 0.6, caption, size=13, color=ICE)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team and the one-line idea: we predict a neighbourhood's median house value, explain "
        "the prediction, and check an asking price against it. The four numbers are the headline results.")


# ---------------------------------------------------------------- 2. problem and objectives
def slide_problem(prs):
    s = new_slide(prs)
    title(s, "Problem statement and objectives", "The brief")
    text(s, 0.6, 1.6, 5.9, 2.0, [
        "Industries rely on accurate numerical predictions, but a dependable regression system needs clean "
        "data, robust modelling, hyperparameter optimisation and clear evaluation.",
        [("Our application: ", {"bold": True, "color": NAVY}),
         ("buyers and sellers rarely know whether a home is fairly priced. FairPrice predicts a neighbourhood's "
          "median house value, explains why, and checks an asking price against it.", {})],
    ], size=14, spacing_after=8)

    label(s, 0.6, 3.75, 5.9, "Objectives from the brief")
    numbered(s, 0.6, 4.1, 5.9, [
        "Ingest and preprocess structured data",
        "Identify correlations and relevant features",
        "Train, tune and evaluate conventional ML and an ANN",
        "Expose the model through a simple UI",
        "Generate insights and visual explanations",
    ], step=0.55, fill=NAVY, size=14)

    label(s, 6.9, 1.6, 5.8, "Evaluation criteria and how we meet them")
    table(s, 6.9, 1.95, [
        ["Criterion", "Our approach"],
        ["Feature preparation accuracy", "Schema and range validation, censored-target removal, leakage-safe pipeline"],
        ["Model performance (MAE, RMSE, R²)", "5 models, randomized search, 5-fold CV, untouched test set"],
        ["Interpretability", "Feature importance, SHAP per prediction, residual analysis"],
        ["End-to-end automation", "One command runs all 8 phases: python run_pipeline.py"],
        ["UI usability", "Streamlit app with input validation and fair-price verdict"],
    ], col_w=[2.15, 3.68], font=12, row_h=0.5, name="Criteria table")

    label(s, 6.9, 5.2, 5.8, "Business use cases")
    uses = ["Buyers: avoid overpaying", "Sellers and agents: price realistically",
            "Lenders: sanity-check valuations", "Portals: flag outlier listings"]
    for i, use in enumerate(uses):
        x, y = 6.9 + (i % 2) * 2.95, 5.55 + (i // 2) * 0.68
        card(s, x, y, 2.85, 0.56, name=f"Use case {i + 1}")
        text(s, x + 0.15, y, 2.6, 0.56, use, size=12, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "The brief asks for an end-to-end regression system with at least one conventional model and one "
        "ANN. The table maps each evaluation criterion in the brief to what we built.")


# ---------------------------------------------------------------- 3. dataset and data quality
def slide_dataset(prs):
    s = new_slide(prs)
    title(s, "California Housing dataset and data quality", "Data")
    stats = [("20,640", "census block groups (1990 US census)"),
             ("8 + 1", "features + target MedHouseVal ($100k units)"),
             ("992", "censored rows removed (value capped at $500k)"),
             ("15,718 / 3,930", "train / test, 80/20 stratified by income band")]
    for i, (value, caption) in enumerate(stats):
        x = 0.6 + i * 3.08
        text(s, x, 1.6, 2.95, 0.6, value, size=28, color=TERRA if i == 2 else NAVY, bold=True, font=HEAD)
        text(s, x, 2.2, 2.85, 0.55, caption, size=12, color=MUTED)

    table(s, 0.6, 3.0, [
        ["Column", "Meaning", "Unit", "Skew"],
        ["MedInc", "Median household income", "$10,000s", "1.65"],
        ["HouseAge", "Median house age", "years (capped at 52)", "0.06"],
        ["AveRooms", "Average rooms per home", "rooms", "20.70"],
        ["AveBedrms", "Average bedrooms per home", "bedrooms", "31.32"],
        ["Population", "People in the block group", "people", "4.94"],
        ["AveOccup", "Average occupants per home", "people", "97.64"],
        ["Latitude / Longitude", "Location", "degrees", "0.47 / -0.30"],
        ["MedHouseVal (target)", "Median house value", "$100,000s", "-"],
    ], col_w=[1.95, 2.55, 1.75, 1.05], font=12, row_h=0.36, fills={8: TERRA_TINT}, bold_rows=(8,),
        name="Feature table")

    label(s, 8.1, 3.0, 4.6, "Data-quality pipeline")
    numbered(s, 8.1, 3.35, 4.63, [
        "Schema check: 9 numeric columns, non-empty",
        "Missing values: 0. Duplicates: 0",
        "Range rules (e.g. latitude 32-42.5): 0 rows out of range",
        "Censored target at 5.0: 992 rows removed (4.8%)",
        "Clean dataset: 19,648 rows, saved as a deliverable",
    ], step=0.63, size=13)
    text(s, 0.6, 6.4, 12, 0.5, "Each row is a neighbourhood, not a single house, and values are in 1990 dollars. "
         "Both limitations are stated in the app.", size=12, color=MUTED, italic=True)
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "Know the units: income in $10,000s and value in $100,000s. The only real data problem was the "
        "censored target: 992 values recorded as 5.00001 mean 'above $500k', so the true value is unknown.")


# ---------------------------------------------------------------- 4. EDA
def slide_eda(prs):
    s = new_slide(prs)
    title(s, "Exploratory analysis: every finding drove a decision", "EDA")
    findings = [
        ("Target spike at exactly 5.0 (992 rows): prices are capped at $500k",
         "Drop censored rows, disclose the ceiling"),
        ("Extreme skew: AveOccup 97.6, AveBedrms 31.3, AveRooms 20.7",
         "Clip to train P1/P99 instead of deleting rows"),
        ("AveOccup vs price: r = -0.02 raw, r = -0.25 after clipping",
         "Outliers were masking a real signal"),
        ("AveRooms and AveBedrms correlated at r = 0.85 (multicollinearity)",
         "Replace AveBedrms with BedroomRatio"),
        ("High values cluster near LA and the Bay Area; Lat/Long r = -0.93",
         "Add DistToCity, keep both coordinates"),
    ]
    for i, (finding, decision) in enumerate(findings):
        y = 1.6 + i * 0.98
        card(s, 0.6, y, 7.35, 0.86, name=f"Finding {i + 1}")
        badge(s, 0.78, y + 0.24, str(i + 1), fill=NAVY, size=0.38, font_size=12)
        text(s, 1.35, y, 3.75, 0.86, finding, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 5.25, y + 0.1, 2.55, 0.3, "DECISION", size=10, color=TEAL, bold=True)
        text(s, 5.25, y + 0.36, 2.55, 0.5, decision, size=12, color=NAVY, bold=True)

    picture(s, "05_geographic_value_map", 8.3, 1.6, 4.4, "Geographic value map")
    text(s, 8.3, 5.8, 4.4, 0.6, "Median house value by location. Stars mark the five reference cities "
         "used for DistToCity.", size=11, color=MUTED, italic=True)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Tell this as finding, then decision. The strongest point: clipping changed AveOccup's correlation "
        "from -0.02 to -0.25, which shows outliers were hiding a real relationship.")


# ---------------------------------------------------------------- 5. features and pipeline
def slide_pipeline(prs):
    s = new_slide(prs)
    title(s, "Feature engineering and a leakage-safe pipeline", "Features")
    steps = [("FeatureEngineer", "3 domain features, drops AveBedrms"),
             ("SimpleImputer", "median, learned on train"),
             ("OutlierClipper", "clip to train P1 / P99"),
             ("StandardScaler", "z = (x - mean) / std"),
             ("Regressor", "Linear, Ridge, RF, XGBoost, ANN")]
    for i, (head, body) in enumerate(steps):
        x = 0.6 + i * 2.5
        dark = i == 4
        card(s, x, 1.6, 2.1, 1.2, fill=NAVY if dark else CARD, name=f"Pipeline {head}")
        text(s, x + 0.17, 1.72, 1.85, 0.35, head, size=14, color=WHITE if dark else NAVY, bold=True)
        text(s, x + 0.17, 2.1, 1.8, 0.65, body, size=12, color=ICE if dark else MUTED)
        if i < 4:
            text(s, x + 2.1, 2.0, 0.4, 0.4, ">", size=20, color=MUTED, bold=True, align=PP_ALIGN.CENTER)

    label(s, 0.6, 3.15, 6.6, "Engineered features")
    corr = json.loads((ROOT / "reports" / "metrics.json").read_text())["eda"]["target_correlation_processed"]
    table(s, 0.6, 3.5, [
        ["Feature", "Definition", "r with price"],
        ["BedroomRatio", "AveBedrms / AveRooms", f"{corr['BedroomRatio']:+.2f}"],
        ["RoomsPerPerson", "AveRooms / AveOccup (living space per person)", f"{corr['RoomsPerPerson']:+.2f}"],
        ["DistToCity", "Haversine km to nearest of LA, SF, San Diego, San Jose, Sacramento",
         f"{corr['DistToCity']:+.2f}"],
        ["MedInc (for reference)", "Strongest raw predictor", f"{corr['MedInc']:+.2f}"],
    ], col_w=[1.75, 3.85, 1.0], font=12, row_h=0.45, name="Engineered features table")
    text(s, 0.6, 5.85, 6.6, 0.8, "DistToCity is the second strongest predictor after income. Division by zero "
         "gives NaN, which is imputed (never infinity). Median beats mean because the features are heavily skewed.",
         size=12, color=MUTED)

    label(s, 7.6, 3.15, 5.1, "Why this design")
    numbered(s, 7.6, 3.5, 5.13, [
        [("No data leakage: ", {"bold": True, "color": NAVY}),
         ("medians, percentiles, means and stds are fit on the training split only, and refit inside every CV fold.", {})],
        [("No training-serving skew: ", {"bold": True, "color": NAVY}),
         ("preprocessing and model are saved as one .pkl, so the app transforms input exactly like training.", {})],
        [("Fair comparison: ", {"bold": True, "color": NAVY}),
         ("all five models see identical features. Scaling is essential for the ANN and Ridge.", {})],
    ], step=1.08, size=13)
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Order matters: engineer features from raw values, impute gaps, clip outliers, then scale. Clipping "
        "before scaling stops outliers from inflating the standard deviation.")


# ---------------------------------------------------------------- 6. architecture and workflow
def slide_architecture(prs):
    s = new_slide(prs)
    title(s, "System architecture and end-to-end workflow", "Architecture")
    workflow = [("Data collection", "data_ingestion.py"), ("Preprocessing", "preprocessing.py"),
                ("EDA", "eda.py"), ("Feature engineering", "features.py"),
                ("Model training", "train.py, ann.py"), ("Tuning and CV", "randomized search, 5 folds"),
                ("Evaluation", "evaluate.py, explain.py"), ("Deployment", "predict.py, Streamlit")]
    for i, (step, module) in enumerate(workflow):
        x = 0.6 + i * 1.53
        card(s, x, 1.6, 1.4, 1.5, name=f"Step {i + 1}")
        badge(s, x + 0.14, 1.72, str(i + 1), fill=TERRA, size=0.32, font_size=11)
        text(s, x + 0.14, 2.12, 1.2, 0.55, step, size=12, color=NAVY, bold=True)
        text(s, x + 0.14, 2.65, 1.2, 0.45, module, size=10, color=MUTED)

    label(s, 0.6, 3.4, 6.3, "Two programs, one saved model")
    card(s, 0.6, 3.75, 6.3, 1.35, name="Training program")
    text(s, 0.82, 3.85, 5.9, 0.3, "Training (offline): python run_pipeline.py", size=14, color=NAVY, bold=True)
    text(s, 0.82, 4.2, 5.9, 0.85, "Runs phases 1-8 and writes best_model.pkl, ann_model.pkl, model_meta.json, "
         "the leaderboard and 11 figures. About 20 min full, 6 min with --fast.", size=12, color=INK)
    card(s, 0.6, 5.25, 6.3, 1.35, fill=NAVY, name="Serving program")
    text(s, 0.82, 5.35, 5.9, 0.3, "Serving (online): streamlit run app/streamlit_app.py", size=14,
         color=WHITE, bold=True)
    text(s, 0.82, 5.7, 5.9, 0.85, "Loads the saved pipelines once (cached). Each request validates input, "
         "predicts, explains with SHAP and returns a verdict in under a second.", size=12, color=ICE)

    label(s, 7.3, 3.4, 5.4, "Engineering practices")
    practices = [("33", "pytest tests: schema, cleaning, features, ANN, validation, verdicts"),
                 ("1", "config file holds every threshold and hyperparameter"),
                 ("42", "fixed random seed: reproducible splits, trees and ANN weights"),
                 ("13", "Git commits, one per project phase")]
    for i, (value, caption) in enumerate(practices):
        x, y = 7.3 + (i % 2) * 2.75, 3.75 + (i // 2) * 1.5
        card(s, x, y, 2.63, 1.35, name=f"Practice {i + 1}")
        text(s, x + 0.18, y + 0.1, 2.3, 0.5, value, size=24, color=TERRA, bold=True, font=HEAD)
        text(s, x + 0.18, y + 0.62, 2.3, 0.7, caption, size=11, color=INK)
    footer(s, 6)
    s.notes_slide.notes_text_frame.text = (
        "The eight boxes are the eight workflow steps in the brief, each mapped to one module. Training and "
        "serving are separate: train once offline, then the app only loads the saved pipeline.")


# ---------------------------------------------------------------- 7. models
def slide_models(prs):
    s = new_slide(prs)
    title(s, "Modelling: four conventional models and an ANN", "Models")
    ann = META["ann"]["best_params"]
    table(s, 0.6, 1.55, [
        ["Model", "How it works", "Hyperparameters searched", "Best configuration (5-fold CV)"],
        ["Linear Regression (baseline)", "Ordinary least squares: minimises the sum of squared errors",
         "None", "Closed-form fit"],
        ["Ridge", "Least squares + L2 penalty alpha * sum(w^2) that shrinks weights",
         "alpha ~ log-uniform(0.01, 1000)", "alpha = 1.44"],
        ["Random Forest", "Bagging: averages decorrelated trees grown on bootstrap samples",
         "n_estimators, max_depth, min_samples_leaf, max_features, max_samples",
         "300 trees, unlimited depth, min leaf 2, 50% of features per split"],
        ["XGBoost", "Gradient boosting: each tree fits the previous trees' residuals; L2 on leaf weights",
         "n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda",
         "500 trees, learning rate 0.03, depth 8, min_child_weight 5, subsample 0.85"],
        ["ANN (Keras)", "Feed-forward network trained by backpropagation on MSE loss",
         "Layers, activation, initialiser, dropout, L2, optimiser, learning rate, batch size",
         f"{'-'.join(map(str, ann['hidden_layers']))}, {ann['activation'].upper()}, Glorot, Adam "
         f"(lr {ann['learning_rate']:.1e}), batch {ann['batch_size']}, L2 {ann['l2']:.0e}, no dropout"],
    ], col_w=[1.9, 3.35, 3.6, 3.28], font=11, row_h=0.55, fills={4: TERRA_TINT, 5: TEAL_TINT},
        name="Model table")

    label(s, 0.6, 5.1, 6.6, "ANN architecture")
    layers = [("Input", "10 features"), ("Dense 256", "ELU"), ("Dense 128", "ELU"), ("Dense 64", "ELU"),
              ("Output", "1, linear")]
    for i, (head, body) in enumerate(layers):
        x = 0.6 + i * 1.36
        dark = i in (0, 4)
        card(s, x, 5.42, 1.2, 0.78, fill=TEAL if dark else TEAL_TINT, name=f"Layer {head}")
        text(s, x + 0.08, 5.47, 1.04, 0.35, head, size=12, color=WHITE if dark else NAVY, bold=True,
             align=PP_ALIGN.CENTER)
        text(s, x + 0.08, 5.8, 1.04, 0.3, body, size=11, color=WHITE if dark else INK, align=PP_ALIGN.CENTER)
    text(s, 0.6, 6.3, 6.6, 0.65, "MSE loss, early stopping (patience 10 on a 10% validation split, best weights "
         "restored), up to 150 epochs. ReLU/ELU, He/Glorot, Adam/SGD and dropout were all in the search.",
         size=11, color=MUTED)

    label(s, 7.6, 5.1, 5.1, "Tuning protocol")
    numbered(s, 7.6, 5.42, 5.13, [
        "RandomizedSearchCV: 20 candidates per model, 6 for the ANN",
        "5-fold shuffled K-fold on the training split, scored on RMSE",
        "Winner = lowest mean CV RMSE. Test set used once.",
    ], step=0.5, size=12, badge_size=0.3)
    footer(s, 7)
    s.notes_slide.notes_text_frame.text = (
        "Simple to complex: the linear baseline shows how much complexity buys. Random Forest reduces variance "
        "by averaging; boosting reduces bias by correcting residuals. The ANN uses the same pipeline and CV.")


# ---------------------------------------------------------------- 8. results
def slide_results(prs):
    s = new_slide(prs)
    title(s, f"Results: XGBoost explains {BEST['test_r2'] * 100:.0f}% of price variance", "Results")
    rows = [["#", "Model", "Type", "CV RMSE (mean ± sd)", "Train RMSE", "Test RMSE", "Test MSE",
             "Test MAE", "Test R²", "Gap"]]
    fills = {}
    for i, r in enumerate(BOARD.sort_values("cv_rmse_mean").itertuples(), start=1):
        rows.append([i, NAMES[r.model], "Neural network" if r.family == "Neural Network" else "Conventional",
                     f"{r.cv_rmse_mean:.3f} ± {r.cv_rmse_std:.3f}", f"{r.train_rmse:.3f}", f"{r.test_rmse:.3f}",
                     f"{r.test_rmse ** 2:.3f}", f"{r.test_mae:.3f} (${r.test_mae_dollars:,.0f})",
                     f"{r.test_r2:.3f}", f"{r.overfit_gap:+.3f}"])
        if r.model == "XGBoost":
            fills[i] = TERRA_TINT
        elif r.model == "ANN":
            fills[i] = TEAL_TINT
    table(s, 0.6, 1.55, rows, col_w=[0.4, 1.75, 1.45, 1.75, 1.05, 1.05, 1.0, 1.73, 0.9, 1.05],
          font=12, row_h=0.4, fills=fills, bold_rows=(1,), name="Leaderboard")
    text(s, 0.6, 4.05, 12.1, 0.3, "Errors in $100k units unless shown in dollars. Gap = test RMSE - train RMSE. "
         "MSE = RMSE squared. Ranked by CV RMSE, the selection metric.", size=10, color=MUTED, italic=True)

    label(s, 0.6, 4.45, 5.6, "Metric definitions")
    metric_runs = [
        [("MSE = mean((y - ŷ)²). ", {"bold": True, "color": NAVY}), ("Training loss; squares large errors.", {})],
        [("RMSE = √MSE. ", {"bold": True, "color": NAVY}),
         (f"Price units; selection metric ({BEST['test_rmse']:.3f} = ${BEST['test_rmse'] * 1e5:,.0f}).", {})],
        [("MAE = mean(|y - ŷ|). ", {"bold": True, "color": NAVY}),
         (f"Typical error; sets the ±${BEST['test_mae_dollars']:,.0f} fair band.", {})],
        [("R² = 1 - SS_res / SS_tot. ", {"bold": True, "color": NAVY}), ("Share of variance explained.", {})],
    ]
    for i, runs in enumerate(metric_runs):
        text(s, 0.6, 4.8 + i * 0.5, 5.7, 0.5, [runs], size=12, color=INK)

    label(s, 6.6, 4.45, 6.1, "What the numbers mean")
    numbered(s, 6.6, 4.8, 6.13, [
        f"XGBoost cuts RMSE by {RMSE_CUT:.1f}% and MAE by {MAE_CUT:.1f}% versus the linear baseline.",
        "Ridge = Linear, gap ≈ 0: the baseline underfits (high bias); L2 cannot help.",
        f"CV RMSE {BEST['cv_rmse_mean']:.3f} vs test {BEST['test_rmse']:.3f}, fold sd {BEST['cv_rmse_std']:.3f}: "
        "it generalises and is stable.",
        f"ANN: R² {ANN['test_r2']:.3f} with a gap of only {ANN['overfit_gap']:.2f} (low variance), still improving "
        "at epoch 150.",
    ], step=0.52, size=12, badge_size=0.3)
    footer(s, 8)
    s.notes_slide.notes_text_frame.text = (
        "Selection used CV RMSE, not the test set, so the test score is an unbiased estimate. Trees win because "
        "prices depend on thresholds and interactions (rich and coastal) that a linear model cannot represent. "
        "XGBoost's train-test gap is normal for boosting; CV and test agree, so it generalises.")


# ---------------------------------------------------------------- 9. diagnostics and explainability
def slide_diagnostics(prs):
    s = new_slide(prs)
    title(s, "Diagnostics and explainability", "Evaluation in depth")
    picture(s, "08_residual_analysis", 0.6, 1.55, 7.4, "Residual analysis")
    text(s, 0.6, 3.5, 7.4, 0.5, "Residuals are centred on zero, but the spread widens for expensive areas "
         "(heteroscedasticity) and the model under-predicts near the $500k ceiling.", size=11, color=MUTED,
         italic=True)
    picture(s, "11_ann_training_curve", 0.6, 4.1, 7.4, "ANN learning curve")
    text(s, 0.6, 6.42, 7.4, 0.5, "ANN train and validation loss fall together, so it does not overfit. The best "
         "epoch was the last (150): more epochs would likely help.", size=11, color=MUTED, italic=True)

    picture(s, "10_shap_summary", 8.35, 1.55, 4.38, "SHAP summary")
    label(s, 8.35, 4.6, 4.4, "What drives price (XGBoost gain)")
    importance = list(META["feature_importance"].items())[:5]
    pretty = {"MedInc": "Median income", "DistToCity": "Distance to nearest city", "Latitude": "Latitude",
              "Longitude": "Longitude", "AveOccup": "Occupants per home", "RoomsPerPerson": "Rooms per person"}
    for i, (feature, share) in enumerate(importance):
        y = 4.95 + i * 0.38
        text(s, 8.35, y, 3.3, 0.35, f"{i + 1}.  {pretty.get(feature, feature)}", size=13, color=INK)
        text(s, 11.7, y, 1.0, 0.35, f"{share * 100:.1f}%", size=13, color=NAVY, bold=True, align=PP_ALIGN.RIGHT)
    footer(s, 9)
    s.notes_slide.notes_text_frame.text = (
        "SHAP splits each prediction into a base value plus per-feature contributions. High income pushes "
        "predictions up; distance from a city and crowding push them down. Location features together rival income.")


# ---------------------------------------------------------------- 10. deployment, challenges, next steps
def slide_delivery(prs):
    s = new_slide(prs)
    title(s, "Deployment, challenges and next steps", "Delivery")
    label(s, 0.6, 1.55, 5.4, "Streamlit app")
    card(s, 0.6, 1.9, 5.4, 2.45, name="App features")
    text(s, 0.82, 2.02, 5.0, 2.3, [
        [("Input: ", {"bold": True, "color": NAVY}), ("8 neighbourhood details and an optional asking price.", {})],
        [("Validation: ", {"bold": True, "color": NAVY}),
         ("required, numeric, valid range, inside California, bedrooms ≤ rooms; warns outside the training range.", {})],
        [("Output: ", {"bold": True, "color": NAVY}),
         ("estimate ± MAE, top SHAP drivers, map, and the ANN's estimate as a second opinion.", {})],
        [("Insights tab: ", {"bold": True, "color": NAVY}),
         ("model leaderboard, EDA and diagnostic figures, data limitations.", {})],
    ], size=12, color=INK, spacing_after=6)
    mae = META["test_mae_dollars"]
    card(s, 0.6, 4.5, 5.4, 1.3, fill=NAVY, name="Fair price rule")
    text(s, 0.82, 4.6, 5.0, 0.4, f"Fair band = ŷ ± MAE (${mae:,.0f})", size=17, color=WHITE, bold=True, font=HEAD)
    text(s, 0.82, 5.05, 5.0, 0.7, "Above the band: Overpriced. Inside: Fair. Below: Good deal. A gap smaller than "
         "the model's own typical error is never called overpriced.", size=12, color=ICE)
    text(s, 0.6, 5.95, 5.4, 0.75, "Deliverables: notebook, cleaned dataset, .pkl models, Streamlit UI, docs, README "
         "and architecture document, GitHub repository.", size=11, color=MUTED)

    label(s, 6.4, 1.55, 6.3, "Challenges and mitigations")
    table(s, 6.4, 1.9, [
        ["Challenge", "Mitigation"],
        ["Missing or inconsistent data", "Schema checks, range rules, median imputation inside the pipeline"],
        ["Censored target ($500k cap)", "992 rows removed; ceiling disclosed in the app"],
        ["Overfitting", "5-fold CV, L2 (Ridge, XGBoost, ANN), tree limits, early stopping"],
        ["Highly correlated features", "Correlation audit; AveBedrms replaced by BedroomRatio"],
        ["Data leakage", "Every fitted statistic lives in one pipeline fit on train only"],
        ["Slow training", "XGBoost hist, all CPU cores, randomized search, --fast mode"],
    ], col_w=[2.2, 4.13], font=11, row_h=0.42, name="Challenges table")

    label(s, 6.4, 5.05, 6.3, "Limitations and next steps")
    numbered(s, 6.4, 5.4, 6.33, [
        "Neighbourhood medians in 1990 dollars: retrain on recent sales data",
        "One global ± MAE band: per-prediction intervals with quantile regression",
        "ANN still improving at 150 epochs: longer training with an LR schedule",
    ], step=0.48, size=12, badge_size=0.3)
    footer(s, 10)
    s.notes_slide.notes_text_frame.text = (
        "Close on the honest verdict rule and the limitations: they show we understand where the model can "
        "and cannot be trusted. Then switch to the live app demo.")


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for build in (slide_title, slide_problem, slide_dataset, slide_eda, slide_pipeline, slide_architecture,
                  slide_models, slide_results, slide_diagnostics, slide_delivery):
        build(prs)
    prs.core_properties.title = "FairPrice - Final Presentation"
    prs.core_properties.author = f"Team {TEAM_NAME}"
    prs.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
