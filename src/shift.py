"""Shift measures between a calibration set and a held-out target domain.

`physical_shift` (mean Gower distance over aging factors), `feature_shift` (energy
distance) and `centroid_shift` use no target labels. `score_mean_ratio` measures shift
in nonconformity-score space and therefore requires target labels.
"""
from __future__ import annotations

import numpy as np

from src.conformal import gower_distance

__all__ = ["physical_shift", "feature_shift", "centroid_shift", "score_mean_ratio"]


def physical_shift(cal_meta: list[dict], target_meta: dict,
                   numeric_ranges: dict[str, float]) -> float:
    """Mean Gower distance from each calibration cell to the target, in [0, 1]."""
    if not cal_meta:
        return float("nan")
    d = [gower_distance(m, target_meta, numeric_ranges) for m in cal_meta]
    return float(np.mean(d))


def _pairwise_mean_distance(A: np.ndarray, B: np.ndarray,
                            max_n: int = 400, seed: int = 0) -> float:
    """Mean Euclidean distance between rows of A and rows of B.

    Each side is subsampled (seeded) to at most `max_n` rows to bound the O(n*m) cost.
    """
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)))


def feature_shift(X_cal: np.ndarray, X_target: np.ndarray,
                  scale: np.ndarray | None = None) -> float:
    """Energy distance 2*E||x-y|| - E||x-x'|| - E||y-y'|| between the two feature sets.

    Features are divided by `scale`, which should be the training-set scale.
    """
    Xc = np.asarray(X_cal, float)
    Xt = np.asarray(X_target, float)
    if scale is not None:
        s = np.where(np.asarray(scale, float) > 0, scale, 1.0)
        Xc, Xt = Xc / s, Xt / s
    cross = _pairwise_mean_distance(Xc, Xt)
    within_c = _pairwise_mean_distance(Xc, Xc)
    within_t = _pairwise_mean_distance(Xt, Xt)
    return float(max(0.0, 2 * cross - within_c - within_t))


def centroid_shift(X_cal: np.ndarray, X_target: np.ndarray,
                   scale: np.ndarray | None = None) -> float:
    """Euclidean distance between calibration and target feature centroids, divided by `scale`."""
    Xc = np.asarray(X_cal, float)
    Xt = np.asarray(X_target, float)
    if scale is not None:
        s = np.where(np.asarray(scale, float) > 0, scale, 1.0)
        Xc, Xt = Xc / s, Xt / s
    return float(np.linalg.norm(Xc.mean(0) - Xt.mean(0)))


def score_mean_ratio(cal_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """Mean target nonconformity score divided by mean calibration score.

    Requires target labels, so it is a retrospective diagnostic, not a deployable monitor.
    Both arguments must be residuals in the same space (log10 in this pipeline).
    """
    c = np.asarray(cal_scores, float)
    t = np.asarray(target_scores, float)
    c, t = c[np.isfinite(c)], t[np.isfinite(t)]
    if len(c) == 0 or len(t) == 0:
        return float("nan")
    mc = float(np.mean(c))
    if mc <= 0:
        return float("nan")     # zero calibration residuals: ratio undefined
    return float(np.mean(t) / mc)
