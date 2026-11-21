"""Phase 5b — Artificial Neural Network (Keras) wrapped as a scikit-learn regressor.

Wrapping the Keras model in the scikit-learn estimator API means the ANN:
  * sits in the same Pipeline as the conventional models (same features, imputation,
    clipping and scaling - scaling matters a lot for neural networks),
  * is tuned with the same RandomizedSearchCV + K-fold CV,
  * is saved/loaded with joblib like every other model.

Architecture (layer sizes are tuned):
    Input -> [Dense(ReLU/ELU, He/Glorot init, L2) -> Dropout] x N -> Dense(1, linear)
Training: MSE loss, Adam or SGD(momentum), mini-batches, early stopping on a validation split.
"""
import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow start-up noise

import keras
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class KerasANNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden_layers=(128, 64, 32), activation="relu", kernel_initializer="he_normal",
                 dropout=0.1, l2=1e-4, optimizer="adam", learning_rate=1e-3, batch_size=64,
                 epochs=200, patience=15, validation_split=0.1, random_state=42, verbose=0):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.dropout = dropout
        self.l2 = l2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ model
    def _build_model(self, n_features: int) -> keras.Model:
        regularizer = keras.regularizers.l2(self.l2) if self.l2 else None
        model = keras.Sequential([keras.Input(shape=(n_features,))])
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation=self.activation,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=regularizer))
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1, activation="linear"))  # regression output

        if self.optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        else:
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        keras.utils.set_random_seed(self.random_state)  # reproducible weights and shuffling

        self.model_ = self._build_model(X.shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                   restore_best_weights=True)
        history = self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                                  validation_split=self.validation_split, callbacks=[early_stop],
                                  verbose=self.verbose)
        self.history_ = {k: [float(v) for v in values] for k, values in history.history.items()}
        self.n_epochs_ = len(self.history_["loss"])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        return self.model_.predict(X, verbose=0).ravel()

    # ------------------------------------------------------------------ pickling
    # Keras models are not plain Python objects, so they are stored as .keras bytes inside the pickle.
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.keras")
                state["model_"].save(path)
                with open(path, "rb") as f:
                    state["model_"] = f.read()
        return state

    def __setstate__(self, state):
        if isinstance(state.get("model_"), bytes):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.keras")
                with open(path, "wb") as f:
                    f.write(state["model_"])
                state["model_"] = keras.models.load_model(path)
        self.__dict__.update(state)
