import pandas as pd
import pytest

from fairprice import config
from fairprice.data_ingestion import DataValidationError, validate_schema
from fairprice.preprocessing import clean_data, data_quality_report


def test_schema_rejects_empty_dataframe():
    with pytest.raises(DataValidationError):
        validate_schema(pd.DataFrame())


def test_schema_rejects_missing_columns(small_df):
    with pytest.raises(DataValidationError, match="Latitude"):
        validate_schema(small_df.drop(columns=["Latitude"]))


def test_schema_coerces_text_to_nan(small_df):
    df = small_df.astype({"HouseAge": object})
    df.loc[0, "HouseAge"] = "n/a"
    validated = validate_schema(df)
    assert pd.isna(validated.loc[0, "HouseAge"])
    assert pd.api.types.is_numeric_dtype(validated["HouseAge"])


def test_schema_drops_extra_columns(small_df):
    validated = validate_schema(small_df.assign(extra="x"))
    assert list(validated.columns) == config.FEATURES + [config.TARGET]


def test_quality_report_counts_problems(small_df):
    report = data_quality_report(small_df)
    assert report["duplicate_rows"] == 1
    assert report["out_of_range_rows"] == 1
    assert report["capped_target_rows"] == 1
    assert report["missing_values"] == {"HouseAge": 1, config.TARGET: 1}


def test_clean_data_removes_each_problem_once(small_df):
    clean, report = clean_data(small_df)
    assert report["removed"] == {"missing_target": 1, "duplicates": 1, "out_of_range": 1, "capped_target": 1}
    assert len(clean) == 2
    assert clean[config.TARGET].max() < config.TARGET_CAP


def test_clean_data_keeps_missing_features_for_imputation(small_df):
    clean, _ = clean_data(small_df)
    assert clean["HouseAge"].isna().sum() == 1
