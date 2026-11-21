import pytest

from fairprice import config
from fairprice.predict import fair_price_verdict, load_model, predict_value, validate_input


# ---------------------------------------------------------------- input validation
def test_valid_input_has_no_errors(valid_row):
    assert validate_input(valid_row) == ([], [])


def test_missing_field_is_an_error(valid_row):
    del valid_row["Latitude"]
    errors, _ = validate_input(valid_row)
    assert errors == ["Latitude is required."]


def test_non_numeric_field_is_an_error(valid_row):
    errors, _ = validate_input({**valid_row, "MedInc": "abc"})
    assert "must be a number" in errors[0]


def test_negative_value_is_an_error(valid_row):
    errors, _ = validate_input({**valid_row, "HouseAge": -5})
    assert "outside the valid range" in errors[0]


def test_location_outside_california_is_an_error(valid_row):
    errors, _ = validate_input({**valid_row, "Latitude": 40.71, "Longitude": -74.0})  # New York
    assert len(errors) == 1 and "Longitude" in errors[0]


def test_more_bedrooms_than_rooms_is_an_error(valid_row):
    errors, _ = validate_input({**valid_row, "AveBedrms": 6, "AveRooms": 4})
    assert errors == ["Average bedrooms cannot be greater than average rooms."]


def test_value_outside_training_range_is_a_warning(valid_row):
    errors, warnings = validate_input({**valid_row, "MedInc": 18}, feature_ranges={"MedInc": [0.5, 15]})
    assert errors == [] and "training range" in warnings[0]


# ---------------------------------------------------------------- fair price verdict
@pytest.mark.parametrize("asking, expected", [
    (200_000, "Fair"),        # exactly the prediction
    (225_000, "Fair"),        # inside +/- tolerance
    (260_000, "Overpriced"),
    (150_000, "Good Deal"),
])
def test_fair_price_verdict(asking, expected):
    assert fair_price_verdict(200_000, asking, tolerance=30_000)["verdict"] == expected


def test_verdict_rejects_non_positive_asking_price():
    with pytest.raises(ValueError):
        fair_price_verdict(200_000, 0, tolerance=30_000)


# ---------------------------------------------------------------- trained model (skipped if not trained yet)
@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python run_pipeline.py` first")
def test_saved_model_predicts_sensible_value(valid_row):
    pipeline, meta = load_model()
    value = predict_value(pipeline, valid_row)
    assert 50_000 < value < 500_000
    assert meta["test_metrics"]["r2"] > 0.7


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python run_pipeline.py` first")
def test_higher_income_predicts_higher_value(valid_row):
    pipeline, _ = load_model()
    low = predict_value(pipeline, {**valid_row, "MedInc": 2.0})
    high = predict_value(pipeline, {**valid_row, "MedInc": 8.0})
    assert high > low
