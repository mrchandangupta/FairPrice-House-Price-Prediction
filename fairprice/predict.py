"""Phase 8 — Inference: load the saved model, validate input, predict and give a verdict.

The Streamlit app only calls functions from this module, so all prediction logic is
testable without a UI.
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from fairprice import config

BACKGROUND_PATH = config.MODEL_PATH.parent / "shap_background.csv"


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the app is started before running the training pipeline."""


@lru_cache(maxsize=1)
def load_model():
    """Return (pipeline, metadata). Cached so the .pkl is read only once."""
    if not config.MODEL_PATH.exists() or not config.MODEL_META_PATH.exists():
        raise ModelNotTrainedError("No trained model found. Run `python run_pipeline.py` first.")
    pipeline = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.MODEL_META_PATH.read_text())
    return pipeline, meta


@lru_cache(maxsize=1)
def load_ann_model():
    """Return the saved ANN pipeline, or None if it hasn't been trained (the app then hides it)."""
    if not config.ANN_MODEL_PATH.exists():
        return None
    return joblib.load(config.ANN_MODEL_PATH)


def load_background() -> pd.DataFrame:
    return pd.read_csv(BACKGROUND_PATH)


def validate_input(inputs: dict, feature_ranges: dict | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).

    errors   -> the prediction must not be made (missing, non-numeric, impossible values)
    warnings -> prediction is allowed but less reliable (outside what the model has seen)
    """
    errors, warnings = [], []

    for col in config.FEATURES:
        value = inputs.get(col)
        if value is None or value == "":
            errors.append(f"{col} is required.")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number (got {value!r}).")
            continue
        if value != value:  # NaN
            errors.append(f"{col} must be a number.")
            continue
        low, high = config.VALID_RANGES[col]
        if not low <= value <= high:
            errors.append(f"{col}={value:g} is outside the valid range [{low:g}, {high:g}].")
        elif feature_ranges and col in feature_ranges:
            seen_low, seen_high = feature_ranges[col]
            if not seen_low <= value <= seen_high:
                warnings.append(f"{col}={value:g} is outside the training range "
                                f"[{seen_low:g}, {seen_high:g}] - prediction may be unreliable.")

    if not errors and float(inputs["AveBedrms"]) > float(inputs["AveRooms"]):
        errors.append("Average bedrooms cannot be greater than average rooms.")
    return errors, warnings


def predict_value(pipeline, inputs: dict) -> float:
    """Predicted median house value in dollars."""
    row = pd.DataFrame([{col: float(inputs[col]) for col in config.FEATURES}])
    value = float(pipeline.predict(row)[0])
    # The model was trained on values below the $500k cap; never return a negative price.
    return max(value, 0.0) * config.PRICE_UNIT


def fair_price_verdict(predicted: float, asking: float, tolerance: float) -> dict:
    """Compare an asking price to the prediction.

    tolerance = the model's typical error (test MAE in $). Inside predicted ± tolerance the
    difference is within model error, so we call it "Fair" rather than over-claiming.
    """
    if asking is None or asking <= 0:
        raise ValueError("Asking price must be a positive number.")
    low, high = predicted - tolerance, predicted + tolerance
    diff_pct = (asking - predicted) / predicted * 100 if predicted > 0 else float("inf")

    if asking > high:
        verdict = "Overpriced"
    elif asking < low:
        verdict = "Good Deal"
    else:
        verdict = "Fair"
    return {"verdict": verdict, "difference_pct": round(diff_pct, 1),
            "fair_low": round(max(low, 0)), "fair_high": round(high)}
