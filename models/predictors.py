"""

Four predictive models for FinSentiment Lab.

Model 1 — Logistic Regression (baseline classifier)
  Predicts: direction_1d (up=1 / down=0)
  Why: establishes a linear baseline. If XGBoost doesn't beat this,
       the non-linear interactions aren't worth the complexity.

Model 2 — XGBoost Classifier
  Predicts: direction_1d (up=1 / down=0)
  Why: captures non-linear interactions (e.g., bearish sentiment + high vol
       = stronger signal than either alone). Handles missing values natively.

Model 3 — XGBoost Regressor
  Predicts: forward_return_1d (continuous)
  Why: directly answers "by how much does negative news move prices?"
       rather than just the direction.

Model 4 — LSTM
  Predicts: direction_1d (sequence → binary)
  Why: captures temporal dependencies — e.g., 3 consecutive bearish days
       may be more predictive than a single bearish day. The only model
       that sees the full sequence of past sentiment.

Each model wraps sklearn/xgboost/keras with a unified interface:
  .fit(X_train, y_train)
  .predict(X_test) → np.ndarray
  .predict_proba(X_test) → np.ndarray  (classifiers only)
  .feature_importance() → pd.Series    (tree models only)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Model 1 — Logistic Regression baseline
# ===========================================================================

class LogisticRegressionModel:
    """
    L2-regularised logistic regression.
    Serves as the interpretable linear baseline.

    Parameters
    ----------
    C         : inverse regularisation strength (smaller = stronger reg)
    max_iter  : solver iteration limit
    """

    def __init__(self, C: float = 0.1, max_iter: int = 1000):
        self.C        = C
        self.max_iter = max_iter
        self._model   = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            C        = self.C,
            max_iter = self.max_iter,
            solver   = "lbfgs",
            random_state = 42,
        )
        self._model.fit(X, y.astype(int))
        log.info("LogisticRegression fitted | train_size=%d", len(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]   # probability of class 1

    def feature_importance(self, feature_names: list) -> pd.Series:
        """Absolute coefficient values as proxy for feature importance."""
        coefs = np.abs(self._model.coef_[0])
        return pd.Series(coefs, index=feature_names).sort_values(ascending=False)


# ===========================================================================
# Model 2 — XGBoost Classifier
# ===========================================================================

class XGBoostClassifier:
    """
    Gradient-boosted tree classifier for return direction.

    Hyperparameters are conservative defaults that work well on small
    financial datasets (n < 200). Tune with walk-forward CV in production.

    Parameters
    ----------
    n_estimators  : number of boosting rounds
    max_depth     : tree depth (keep shallow to avoid overfitting on small data)
    learning_rate : step size shrinkage
    subsample     : row sampling ratio per tree
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        max_depth:     int   = 3,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self._model        = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Run: pip install xgboost")

        self._model = xgb.XGBClassifier(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            subsample     = self.subsample,
            use_label_encoder = False,
            eval_metric   = "logloss",
            random_state  = 42,
            verbosity     = 0,
        )
        self._model.fit(X, y.astype(int))
        log.info("XGBoostClassifier fitted | train_size=%d | features=%d", len(X), X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list) -> pd.Series:
        scores = self._model.feature_importances_
        return pd.Series(scores, index=feature_names).sort_values(ascending=False)


# ===========================================================================
# Model 3 — XGBoost Regressor
# ===========================================================================

class XGBoostRegressorModel:
    """
    Gradient-boosted tree regressor for return magnitude prediction.

    Predicts forward_return_1d as a continuous value.
    The sign of the prediction can also be used as a trading signal.
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        max_depth:     int   = 3,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self._model        = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressorModel":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Run: pip install xgboost")

        self._model = xgb.XGBRegressor(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            subsample     = self.subsample,
            random_state  = 42,
            verbosity     = 0,
        )
        self._model.fit(X, y)
        log.info("XGBoostRegressor fitted | train_size=%d", len(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert regression output to direction probability via sigmoid."""
        raw   = self._model.predict(X)
        scale = np.std(raw) + 1e-8
        return 1 / (1 + np.exp(-raw / scale))

    def feature_importance(self, feature_names: list) -> pd.Series:
        scores = self._model.feature_importances_
        return pd.Series(scores, index=feature_names).sort_values(ascending=False)


# ===========================================================================
# Model 4 — LSTM
# ===========================================================================

class LSTMModel:
    """
    LSTM sequence classifier for return direction.

    Architecture
    ------------
    Input  → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.2)
           → Dense(16, relu) → Dense(1, sigmoid)

    The two-layer LSTM captures short-term patterns in the first layer
    and longer-range dependencies in the second.

    Parameters
    ----------
    timesteps    : sequence length (number of past days)
    n_features   : number of input features per timestep
    lstm_units   : hidden units in first LSTM layer
    dropout      : dropout rate between LSTM layers
    epochs       : training epochs
    batch_size   : mini-batch size
    """

    def __init__(
        self,
        timesteps:  int   = 10,
        n_features: int   = 10,
        lstm_units: int   = 64,
        dropout:    float = 0.3,
        epochs:     int   = 50,
        batch_size: int   = 16,
    ):
        self.timesteps  = timesteps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch_size = batch_size
        self._model     = None
        self.history    = None

    def fit(
        self,
        X: np.ndarray,    # shape: (n, timesteps, features)
        y: np.ndarray,
        validation_split: float = 0.1,
    ) -> "LSTMModel":
        self._model = self._build_model(X.shape[2])

        callbacks = self._get_callbacks()

        self.history = self._model.fit(
            X, y.astype(np.float32),
            epochs           = self.epochs,
            batch_size       = self.batch_size,
            validation_split = validation_split,
            callbacks        = callbacks,
            verbose          = 0,
        )
        log.info(
            "LSTM fitted | train=%d | epochs=%d | final_val_loss=%.4f",
            len(X), self.epochs,
            self.history.history.get("val_loss", [0])[-1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict(X, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).flatten()

    def feature_importance(self, feature_names: list) -> pd.Series:
        """LSTM has no native feature importance — return uniform weights."""
        n = len(feature_names)
        return pd.Series([1/n] * n, index=feature_names)

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError(
                "TensorFlow required for LSTM. Run: pip install tensorflow"
            )

        tf.random.set_seed(42)

        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True,
                 input_shape=(self.timesteps, n_features)),
            Dropout(self.dropout),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout * 0.7),
            Dense(16, activation="relu"),
            Dense(1,  activation="sigmoid"),
        ])

        model.compile(
            optimizer = Adam(learning_rate=0.001),
            loss      = "binary_crossentropy",
            metrics   = ["accuracy"],
        )
        return model

    def _get_callbacks(self):
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            return [
                EarlyStopping(
                    monitor   = "val_loss",
                    patience  = 10,
                    restore_best_weights = True,
                    verbose   = 0,
                ),
                ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 5,
                    verbose  = 0,
                ),
            ]
        except ImportError:
            return []