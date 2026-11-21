"""Phase 4 — Feature engineering, correlation analysis and the shared preprocessing pipeline.

The transformers here follow the scikit-learn fit/transform API so they can live inside
one Pipeline. That Pipeline is saved with the model, so the Streamlit app transforms user
input exactly like the training data was transformed.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairprice import config


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points (works on scalars or arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def distance_to_nearest_city(lat, lon) -> np.ndarray:
    distances = [haversine_km(lat, lon, c_lat, c_lon) for c_lat, c_lon in config.CITY_COORDS.values()]
    return np.min(np.vstack(distances), axis=0)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN (later imputed) instead of inf when the denominator is 0."""
    return numerator / denominator.replace(0, np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features and drops redundant ones.

    BedroomRatio   = AveBedrms / AveRooms   (share of rooms that are bedrooms)
    RoomsPerPerson = AveRooms / AveOccup    (living space per person)
    DistToCity     = km to nearest of LA / SF / San Diego / San Jose / Sacramento
    """

    def fit(self, X, y=None):
        return self  # stateless: nothing to learn

    def transform(self, X):
        X = pd.DataFrame(X, columns=config.FEATURES).copy()
        X["BedroomRatio"] = safe_divide(X["AveBedrms"], X["AveRooms"])
        X["RoomsPerPerson"] = safe_divide(X["AveRooms"], X["AveOccup"])
        X["DistToCity"] = distance_to_nearest_city(X["Latitude"].to_numpy(), X["Longitude"].to_numpy())
        return X.drop(columns=config.DROP_AFTER_ENGINEERING)

    def get_feature_names_out(self, input_features=None):
        engineered = config.FEATURES + ["BedroomRatio", "RoomsPerPerson", "DistToCity"]
        return np.array([f for f in engineered if f not in config.DROP_AFTER_ENGINEERING])


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorizes each column to percentiles learned on the training data only."""

    def __init__(self, quantiles=config.CLIP_QUANTILES):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.quantiles[0])
        self.upper_ = X.quantile(self.quantiles[1])
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor() -> Pipeline:
    """Feature engineering -> median imputation -> outlier clipping -> scaling.

    Outputs pandas DataFrames so feature names survive through to SHAP and importances.
    """
    return Pipeline([
        ("features", FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper()),
        ("scaler", StandardScaler()),
    ]).set_output(transform="pandas")


def correlation_report(df: pd.DataFrame, threshold: float = config.CORRELATION_THRESHOLD) -> list[dict]:
    """List feature pairs whose |correlation| exceeds the threshold (candidates for removal)."""
    corr = df.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": round(float(value), 3)})
    return pairs
