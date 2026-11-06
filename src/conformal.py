"""Split, weighted and nonexchangeable conformal prediction intervals.

Split conformal prediction guarantees marginal coverage only under exchangeability.
Weighted conformal prediction (Tibshirani et al., arXiv:1904.06019) covers known
covariate shift. Nonexchangeable conformal prediction (Barber et al., Ann. Statist.
51(2), arXiv:2202.13415) bounds the coverage loss by sum_i(w~_i * d_i) for weights
fixed a priori; here the weights come from a physical distance over aging factors.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "absolute_residual_score", "normalized_residual_score",
    "conformal_quantile", "weighted_conformal_quantile",
    "gower_distance", "PhysicsWeightedConformal",
]


# ---------------------------------------------------------------------------
# nonconformity scores
# ---------------------------------------------------------------------------
def absolute_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute residual |y - yhat|."""
    return np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))


def normalized_residual_score(y_true: np.ndarray, y_pred: np.ndarray,
                              sigma: np.ndarray) -> np.ndarray:
    """Locally adaptive score |y - yhat| / (1 + sigma).

    `sigma` is any per-point difficulty estimate (e.g. MC-dropout std).
    """
    return absolute_residual_score(y_true, y_pred) / (1.0 + np.asarray(sigma, float))


# ---------------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """Unweighted split-conformal quantile with the finite-sample (n+1) correction.

    Returns the ceil((n+1) * nominal)-th smallest calibration score, or +inf when
    n is too small to certify the level.
    """
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * nominal))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def weighted_conformal_quantile(scores: np.ndarray, weights: np.ndarray,
                                nominal: float) -> float:
    """Weighted conformal quantile (Tibshirani et al. construction).

    Normalises the calibration weights together with a unit mass on the test point
    and returns the smallest score whose cumulative normalised weight reaches
    `nominal`, or +inf if the level is never reached.
    """
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    if len(s) != len(w):
        raise ValueError(f"scores/weights length mismatch: {len(s)} vs {len(w)}")
    if np.any(w < 0):
        raise ValueError("conformal weights must be non-negative")
    if len(s) == 0:
        return float("inf")

    order = np.argsort(s)
    s, w = s[order], w[order]
    # Unit mass on the unknown test score, required for validity.
    total = w.sum() + 1.0
    if total <= 0:
        return float("inf")
    cum = np.cumsum(w) / total
    idx = np.searchsorted(cum, nominal, side="left")
    if idx >= len(s):
        return float("inf")
    return float(s[idx])


# ---------------------------------------------------------------------------
# physical distance over aging factors
# ---------------------------------------------------------------------------
def gower_distance(a: dict, b: dict, numeric_ranges: dict[str, float]) -> float:
    """Gower distance between two aging-factor records, in [0, 1].

    Categorical fields contribute 0/1 on mismatch; numeric fields contribute
    |a - b| / range, capped at 1. Fields missing (None) in either record are
    skipped and excluded from the denominator, so a missing factor does not pull
    the distance toward zero.

    `numeric_ranges` maps a numeric field name to its range; any other field is
    treated as categorical.
    """
    keys = set(a) & set(b)
    if not keys:
        raise ValueError("no shared aging-factor keys between the two records")
    total, n_compared = 0.0, 0
    for k in keys:
        av, bv = a[k], b[k]
        if av is None or bv is None:
            continue
        n_compared += 1
        rng = numeric_ranges.get(k)
        if rng:
            total += min(1.0, abs(float(av) - float(bv)) / rng)
        else:
            total += 0.0 if av == bv else 1.0
    if n_compared == 0:
        # Returning 0 would treat two records with nothing in common as a perfect match.
        raise ValueError(
            "no comparable aging factors: every shared key is missing in one record")
    return total / n_compared


class PhysicsWeightedConformal:
    """Nonexchangeable conformal intervals weighted by physical aging-factor distance.

    Each calibration cell receives weight `w_i = exp(-d_i / bandwidth)`, where d_i
    is the Gower distance between its aging factors and the target's. Weights
    depend only on target metadata, never on target labels or residuals, so
    `predict_interval` accepts metadata alone.
    """

    def __init__(self, bandwidth: float = 0.3,
                 numeric_ranges: dict[str, float] | None = None) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        self.bandwidth = bandwidth
        self.numeric_ranges = numeric_ranges or {}
        self.cal_scores_: np.ndarray | None = None
        self.cal_meta_: list[dict] | None = None

    def fit(self, cal_scores: np.ndarray, cal_meta: list[dict]) -> "PhysicsWeightedConformal":
        """Store calibration scores and their per-cell aging-factor metadata."""
        cal_scores = np.asarray(cal_scores, float)
        if len(cal_scores) != len(cal_meta):
            raise ValueError(
                f"cal_scores/cal_meta length mismatch: {len(cal_scores)} vs {len(cal_meta)}")
        self.cal_scores_, self.cal_meta_ = cal_scores, list(cal_meta)
        return self

    def weights_for(self, target_meta: dict) -> np.ndarray:
        """Return the calibration weights for one target domain."""
        if self.cal_meta_ is None:
            raise RuntimeError("call fit() before requesting weights")
        d = np.array([gower_distance(m, target_meta, self.numeric_ranges)
                      for m in self.cal_meta_], dtype=float)
        return np.exp(-d / self.bandwidth)

    def predict_interval(self, y_pred: np.ndarray, target_meta: dict,
                         nominal: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (lower, upper, diagnostics) for predictions in a shifted domain.

        Diagnostics include the quantile, weight sum and N_eff.
        """
        if self.cal_scores_ is None:
            raise RuntimeError("call fit() before predict_interval()")
        from .metrics import n_eff
        w = self.weights_for(target_meta)
        q = weighted_conformal_quantile(self.cal_scores_, w, nominal)
        y_pred = np.asarray(y_pred, float)
        diag = {
            "quantile": q,
            "weight_sum": float(w.sum()),
            "n_eff": n_eff(w),
            "n_cal": int(len(w)),
            "bandwidth": self.bandwidth,
            "nominal": float(nominal),
        }
        return y_pred - q, y_pred + q, diag
