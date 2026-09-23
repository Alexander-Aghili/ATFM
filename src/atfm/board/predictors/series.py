from __future__ import annotations

import numpy as np

from .base import SeriesPredictor


class ConstantSeries(SeriesPredictor):
    name = "B0_constant"

    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        last = float(history[-1]) if len(history) else 0.0
        return np.full(n, last)


class KalmanSeries(SeriesPredictor):
    """Local linear trend model; one-step-ahead predictive Gaussian, clipped at 0."""

    name = "B1_kalman"

    def __init__(self, q_level: float = 1.0, q_trend: float = 0.1, r: float = 10.0, min_points: int = 5):
        self.q_level, self.q_trend, self.r, self.min_points = q_level, q_trend, r, min_points

    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        y = np.asarray(history, float)
        if len(y) < self.min_points:
            last = float(y[-1]) if len(y) else 0.0
            return np.full(n, last)
        scale = max(float(np.var(y)), 1e-6)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Hm = np.array([[1.0, 0.0]])
        Q = np.diag([self.q_level, self.q_trend]) * scale * 0.01
        R = np.array([[self.r]]) * scale * 0.1
        x = np.array([y[0], 0.0])
        P = np.eye(2) * scale
        for obs in y:
            x = F @ x
            P = F @ P @ F.T + Q
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T / S
            x = x + (K * (obs - Hm @ x)).ravel()
            P = (np.eye(2) - K @ Hm) @ P
        x = F @ x
        P = F @ P @ F.T + Q
        mean = float(x[0])
        var = float((Hm @ P @ Hm.T + R)[0, 0])
        return np.clip(rng.normal(mean, np.sqrt(var), size=n), 0.0, None)
