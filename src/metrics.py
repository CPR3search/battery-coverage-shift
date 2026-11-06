"""Point metrics for battery life regression and metrics for prediction intervals.

Point metrics follow the BatteryLife benchmark conventions. Coverage (PICP) is
always reported together with width (MPIW) and, for weighted calibration, the
effective sample size N_eff.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

__all__ = [
    "mape", "rmse", "alpha_accuracy",
    "picp", "mpiw", "coverage_gap", "interval_score", "n_eff",
    "point_metrics", "interval_metrics", "restoration_ratio",
]


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error as a fraction (0.179 means 17.9%)."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error, in cycles."""
    return float(root_mean_squared_error(y_true, y_pred))


def alpha_accuracy(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.15) -> float:
    """Fraction of cells predicted within +/- alpha relative error.

    BatteryLife's "15%-Acc" corresponds to alpha=0.15. This is a point-accuracy
    metric, not interval coverage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= alpha * np.abs(y_true)))


# --------------------------------------------------------------------------
# interval metrics
# --------------------------------------------------------------------------
def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability: fraction of truths inside [lo, hi]."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= np.asarray(lower, dtype=float))
                         & (y_true <= np.asarray(upper, dtype=float))))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width. Lower is better at equal coverage."""
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def coverage_gap(observed_picp: float, nominal: float) -> float:
    """Absolute calibration error |PICP - nominal|.

    `nominal` is the target coverage itself (e.g. 0.90), not alpha.
    """
    return float(abs(observed_picp - nominal))


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   nominal: float) -> float:
    """Winkler interval score: width plus (2/alpha) times any miss distance. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    alpha = 1.0 - nominal
    width = upper - lower
    below = np.clip(lower - y_true, 0, None)
    above = np.clip(y_true - upper, 0, None)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def n_eff(weights: np.ndarray) -> float:
    """Effective sample size (sum w)^2 / sum(w^2) of a weighted calibration set."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s ** 2 / np.sum(w ** 2))


# --------------------------------------------------------------------------
# metric bundles
# --------------------------------------------------------------------------
def restoration_ratio(y_true: np.ndarray, y_pred_log: np.ndarray,
                      produced_mpiw: float, nominal: float) -> float:
    """Ratio of the width needed for nominal coverage to the width produced.

    The numerator is the mean raw width of the narrowest symmetric log-space
    interval that would have covered these cells at `nominal` (an oracle); the
    denominator is the produced MPIW on the same cells. The ratio is dimensionless.
    """
    y_true = np.asarray(y_true, float)
    y_pred_log = np.asarray(y_pred_log, float)
    if produced_mpiw <= 0 or not np.isfinite(produced_mpiw):
        return float("nan")
    s = np.sort(np.abs(np.log10(y_true) - y_pred_log))
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(max(int(np.ceil((n + 1) * nominal)) - 1, 0), n - 1)
    q = s[k]
    required = float(np.mean(10 ** y_pred_log * (10 ** q - 10 ** -q)))
    return required / float(produced_mpiw)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  alpha: float = 0.15) -> dict:
    return {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        f"acc_{int(alpha * 100)}pct": alpha_accuracy(y_true, y_pred, alpha),
        "n": int(len(y_true)),
    }


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     nominal: float, weights: np.ndarray | None = None) -> dict:
    """Coverage, width and calibration error at a single nominal level."""
    cov = picp(y_true, lower, upper)
    out = {
        "nominal": float(nominal),
        "picp": cov,
        "coverage_gap": coverage_gap(cov, nominal),
        "mpiw": mpiw(lower, upper),
        "interval_score": interval_score(y_true, lower, upper, nominal),
        "n": int(len(y_true)),
    }
    if weights is not None:
        out["n_eff"] = n_eff(weights)
    return out
