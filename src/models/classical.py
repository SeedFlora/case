"""Baseline 0 and 1: naive predictor and classical feature models.

Baseline 0 (training-set mean) establishes the floor. Any model that cannot beat
it has learned nothing, and on subject-independent affect regression that floor
is deceptively strong: predicting the training mean yields CCC ~ 0 but a
competitive MAE, which is exactly why CCC is the selection metric.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class NaivePredictor:
    """Predicts the training-set mean valence/arousal for every window."""

    name = "naive"

    def __init__(self, **_):
        self.mu_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NaivePredictor":
        self.mu_ = y.mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.mu_[None, :], x.shape[0], axis=0).astype(np.float32)


class RidgeModel:
    """Ridge regression with alpha chosen on the fold's validation participants."""

    name = "ridge"

    def __init__(self, alphas=(1.0, 10.0, 100.0, 1000.0), **_):
        self.alphas = list(alphas)
        self.model = None
        self.alpha_: float | None = None

    def select(self, xtr, ytr, xva, yva) -> float:
        from sklearn.linear_model import Ridge
        from evaluate import metrics_block

        best, best_a = -np.inf, self.alphas[0]
        for a in self.alphas:
            m = Ridge(alpha=a).fit(xtr, ytr)
            s = metrics_block(yva, m.predict(xva))["mean_ccc"]
            if np.isfinite(s) and s > best:
                best, best_a = s, a
        self.alpha_ = best_a
        return best_a

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha_ or 10.0).fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float32)


class LightGBMModel:
    """Gradient-boosted trees, one regressor per target."""

    name = "lightgbm"

    def __init__(self, **params):
        self.params: Dict[str, Any] = {
            "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8,
            "verbose": -1,
        }
        self.params.update({k: v for k, v in params.items() if k != "enabled"})
        self.models = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        import lightgbm as lgb
        self.models = []
        for j in range(y.shape[1]):
            m = lgb.LGBMRegressor(**self.params)
            m.fit(x, y[:, j])
            self.models.append(m)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(x) for m in self.models], axis=1).astype(np.float32)
