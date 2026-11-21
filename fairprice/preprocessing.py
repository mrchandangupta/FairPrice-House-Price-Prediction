"""Phase 2 — Data cleaning (row-level rules that don't learn anything from the data).

What happens here vs. inside the model Pipeline:
  * Here: remove rows that are *wrong* (impossible values, duplicates, unknown/capped target).
  * Pipeline (fit on train only): imputation, outlier clipping, scaling — these learn
    statistics from data, so doing them before the train/test split would leak information.
"""
import pandas as pd

from fairprice import config
from fairprice.logger import get_logger

log = get_logger(__name__)


def out_of_range_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows where any known column is outside its valid range (NaN is not flagged)."""
    mask = pd.Series(False, index=df.index)
    for col, (low, high) in config.VALID_RANGES.items():
        if col in df.columns:
            mask |= (df[col] < low) | (df[col] > high)
    return mask


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise data issues so cleaning decisions are visible and explainable."""
    return {
        "rows": int(len(df)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "out_of_range_rows": int(out_of_range_mask(df).sum()),
        "capped_target_rows": int((df[config.TARGET] >= config.TARGET_CAP).sum()),
    }


def clean_data(df: pd.DataFrame, drop_capped_target: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply row-level cleaning rules. Returns (clean_df, report of what was removed)."""
    before = data_quality_report(df)
    log.info("Data quality before cleaning: %s", before)
    removed = {}

    n = len(df)
    df = df.dropna(subset=[config.TARGET])          # can't learn from rows with no label
    removed["missing_target"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()
    removed["duplicates"] = n - len(df)

    n = len(df)
    df = df[~out_of_range_mask(df)]
    removed["out_of_range"] = n - len(df)

    if drop_capped_target:
        # 5.00001 means "$500k or more" - the true value is unknown (censored).
        n = len(df)
        df = df[df[config.TARGET] < config.TARGET_CAP]
        removed["capped_target"] = n - len(df)

    df = df.reset_index(drop=True)
    report = {"before": before, "removed": removed, "rows_after": int(len(df))}
    log.info("Removed rows: %s -> %d rows remain", removed, len(df))
    return df, report


def save_clean_data(df: pd.DataFrame) -> None:
    config.CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    log.info("Saved cleaned dataset -> %s", config.CLEAN_DATA_PATH.relative_to(config.ROOT_DIR))
