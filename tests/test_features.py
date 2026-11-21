import numpy as np
import pandas as pd

from fairprice import config
from fairprice.features import (FeatureEngineer, OutlierClipper, build_preprocessor, correlation_report,
                                haversine_km)


def test_haversine_known_distance():
    # Los Angeles -> San Francisco is roughly 560 km
    assert 540 < haversine_km(34.05, -118.24, 37.77, -122.42) < 580


def test_feature_engineer_adds_and_drops_columns(valid_row):
    out = FeatureEngineer().transform(pd.DataFrame([valid_row]))
    assert {"BedroomRatio", "RoomsPerPerson", "DistToCity"} <= set(out.columns)
    assert "AveBedrms" not in out.columns
    assert out.loc[0, "BedroomRatio"] == 0.2
    assert out.loc[0, "DistToCity"] < 1  # coordinates are downtown LA
    assert list(out.columns) == list(FeatureEngineer().get_feature_names_out())


def test_feature_engineer_division_by_zero_gives_nan_not_inf(valid_row):
    out = FeatureEngineer().transform(pd.DataFrame([{**valid_row, "AveOccup": 0}]))
    assert np.isnan(out.loc[0, "RoomsPerPerson"])


def test_outlier_clipper_uses_train_percentiles_only():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    clipper = OutlierClipper(quantiles=(0.05, 0.95)).fit(train)
    out = clipper.transform(pd.DataFrame({"x": [-1000.0, 50.0, 1000.0]}))
    assert out["x"].tolist() == [clipper.lower_["x"], 50.0, clipper.upper_["x"]]


def test_preprocessor_handles_missing_values(valid_row):
    rows = [{**valid_row, "MedInc": m} for m in (2.0, 4.0, 6.0)]
    rows.append({**valid_row, "HouseAge": np.nan})
    out = build_preprocessor().fit_transform(pd.DataFrame(rows)[config.FEATURES])
    assert not out.isna().any().any()


def test_correlation_report_flags_redundant_pair():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8.1], "c": [4, 1, 3, 2]})
    pairs = correlation_report(df, threshold=0.9)
    assert [(p["feature_a"], p["feature_b"]) for p in pairs] == [("a", "b")]
