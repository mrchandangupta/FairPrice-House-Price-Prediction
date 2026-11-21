"""Phase 7b — Explainability with SHAP.

SHAP splits one prediction into "base value + contribution of each feature", e.g.
  $180k = $200k (average) + $45k (high income) - $30k (far from a city) - ...
That turns a black-box tree model into something a user (or judge) can question.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from fairprice import config


def make_explainer(pipeline, background: pd.DataFrame):
    """TreeExplainer for trees (fast, exact), LinearExplainer for linear models,
    model-agnostic permutation explainer for anything else (e.g. the ANN)."""
    model = pipeline.named_steps["model"]
    processed_background = pipeline.named_steps["preprocess"].transform(background)
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, processed_background)
    return shap.Explainer(model.predict, processed_background)


def plot_shap_summary(pipeline, explainer, X_sample: pd.DataFrame) -> None:
    processed = pipeline.named_steps["preprocess"].transform(X_sample)
    shap_values = explainer(processed)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=12)
    plt.title("SHAP: how each feature pushes predictions up or down")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "10_shap_summary.png", dpi=110, bbox_inches="tight")
    plt.close("all")


def explain_prediction(pipeline, explainer, X_row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Return (base value in $, table of feature contributions in $) for a single row."""
    preprocess = pipeline.named_steps["preprocess"]
    processed = preprocess.transform(X_row)
    explanation = explainer(processed)

    # Show users the human-readable (engineered, unscaled) value of each feature.
    readable = preprocess.named_steps["features"].transform(X_row).iloc[0]
    table = pd.DataFrame({
        "feature": processed.columns,
        "value": [readable[c] for c in processed.columns],
        "contribution_dollars": explanation.values[0] * config.PRICE_UNIT,
    })
    table = table.reindex(table["contribution_dollars"].abs().sort_values(ascending=False).index)
    base = float(explanation.base_values[0]) * config.PRICE_UNIT
    return base, table.reset_index(drop=True)
