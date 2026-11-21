import io

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from fairprice.ann import KerasANNRegressor


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)).astype("float32")
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2).astype("float32")  # includes a non-linear term
    return X, y


@pytest.fixture(scope="module")
def fitted_ann(toy_data):
    X, y = toy_data
    return KerasANNRegressor(hidden_layers=(32, 16), epochs=60, patience=10, batch_size=32).fit(X, y)


def test_ann_learns_better_than_predicting_the_mean(fitted_ann, toy_data):
    X, y = toy_data
    mse_model = np.mean((fitted_ann.predict(X) - y) ** 2)
    mse_mean = np.mean((y.mean() - y) ** 2)
    assert mse_model < 0.3 * mse_mean


def test_ann_prediction_shape(fitted_ann, toy_data):
    X, _ = toy_data
    assert fitted_ann.predict(X[:7]).shape == (7,)


def test_ann_records_training_history_and_early_stopping(fitted_ann):
    assert {"loss", "val_loss"} <= set(fitted_ann.history_)
    assert 1 <= fitted_ann.n_epochs_ <= 60


def test_ann_architecture_matches_parameters(fitted_ann):
    dense = [layer for layer in fitted_ann.model_.layers if layer.__class__.__name__ == "Dense"]
    assert [layer.units for layer in dense] == [32, 16, 1]


def test_ann_survives_pickle_round_trip(fitted_ann, toy_data):
    X, _ = toy_data
    buffer = io.BytesIO()
    joblib.dump(fitted_ann, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(X[:5]), fitted_ann.predict(X[:5]), rtol=1e-5)


def test_ann_is_a_valid_sklearn_estimator_for_tuning():
    ann = KerasANNRegressor(dropout=0.3, optimizer="sgd")
    copy = clone(ann)  # RandomizedSearchCV clones estimators like this
    assert copy.get_params()["dropout"] == 0.3 and copy.get_params()["optimizer"] == "sgd"
