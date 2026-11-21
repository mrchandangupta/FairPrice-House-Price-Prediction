"""Phase 1 — Data ingestion and schema validation.

Loads the California Housing data (local CSV cache first, scikit-learn download otherwise)
and checks it has the shape we expect before anything else touches it.
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing

from fairprice import config
from fairprice.logger import get_logger

log = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def load_raw_data(use_cache: bool = True) -> pd.DataFrame:
    """Return the raw dataset as a DataFrame with the 8 features + target.

    A local CSV copy is kept so the project still runs offline after the first download.
    """
    if use_cache and config.RAW_DATA_PATH.exists():
        log.info("Loading cached raw data from %s", config.RAW_DATA_PATH.name)
        return pd.read_csv(config.RAW_DATA_PATH)

    log.info("Downloading California Housing dataset via scikit-learn")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df[config.TARGET] = housing.target

    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns exist, are numeric and the table is not empty.

    Returns the DataFrame restricted to the expected columns (extra columns are ignored).
    """
    if df is None or df.empty:
        raise DataValidationError("Dataset is empty.")

    expected = config.FEATURES + [config.TARGET]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df = df[expected].copy()
    # Non-numeric text (e.g. "n/a") becomes NaN here and is handled during cleaning.
    for col in expected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            log.warning("Column %s is not numeric - coercing (invalid values -> NaN)", col)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Schema OK: %d rows x %d columns", *df.shape)
    return df


def ingest(use_cache: bool = True) -> pd.DataFrame:
    """Load + validate in one call (used by the pipeline)."""
    return validate_schema(load_raw_data(use_cache))
