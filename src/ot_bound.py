"""Optimal-transport coverage-gap bound of Correia and Louizos (Theorem 3.3).

Reference: arXiv 2507.10425. The bound uses only unlabeled test inputs:

    Delta_{P,Q} <= 0.5 * max_s p(s) * ( W1(P, Q_up) + W1(P, Q_down)
                                        + E_{Q_up}[S] - E_{Q_down}[S] )

where p is the calibration score density and Q_up / Q_down are the per-instance maximum
and minimum scores over candidate labels, which bracket the true test score distribution.
For continuous cycle life the candidate set is the observed life range. A total coverage
gap cannot exceed 1, so a bound of 1 or more is vacuous.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "w1_empirical",
    "kde_sup_density",
    "sandwich_scores",
    "total_coverage_gap",
    "ot_coverage_gap_bound",
]


def w1_empirical(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between two empirical distributions.

    Integrates |F_a - F_b| over the pooled support; exact for unequal sample sizes.
    """
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.union1d(a, b)
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    # CDFs are constant on [grid[k], grid[k+1]); the last atom contributes no width.
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(grid)))


def kde_sup_density(scores: np.ndarray, n_grid: int = 2048) -> float:
    """Supremum of a Gaussian-KDE (Scott's rule) estimate of the score density.

    The grid is padded by three bandwidths so the maximum is not clipped at the sample edge.
    """
    from scipy.stats import gaussian_kde

    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if len(s) < 2 or np.allclose(s, s[0]):
        return float("nan")     # degenerate sample: density supremum undefined
    kde = gaussian_kde(s)
    bw = float(np.sqrt(kde.covariance[0, 0]))
    grid = np.linspace(s.min() - 3 * bw, s.max() + 3 * bw, n_grid)
    return float(np.max(kde(grid)))


def sandwich_scores(pred_log: np.ndarray, y_lo: float, y_hi: float,
                    n_candidates: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-instance min and max of |log10(y') - pred_log| over y' in [y_lo, y_hi].

    Computed in closed form. If `n_candidates` is given, an explicit candidate grid is
    evaluated instead (used to check the closed form).
    """
    c = np.asarray(pred_log, float)
    lo, hi = np.log10(float(y_lo)), np.log10(float(y_hi))
    if n_candidates is not None:
        cand = np.linspace(lo, hi, int(n_candidates))
        d = np.abs(cand[None, :] - c[:, None])
        return d.min(axis=1), d.max(axis=1)
    s_min = np.clip(np.maximum(lo - c, c - hi), 0.0, None)
    s_max = np.maximum(np.abs(lo - c), np.abs(hi - c))
    return s_min, s_max


def total_coverage_gap(cal_scores: np.ndarray, target_scores: np.ndarray,
                       n_alpha: int = 199) -> float:
    """Realized total coverage gap: mean over alpha of |coverage(alpha) - (1 - alpha)|.

    Uses the conservative conformal quantile, matching the rest of the pipeline.
    """
    c = np.sort(np.asarray(cal_scores, float))
    t = np.asarray(target_scores, float)
    c, t = c[np.isfinite(c)], t[np.isfinite(t)]
    if len(c) == 0 or len(t) == 0:
        return float("nan")
    n = len(c)
    alphas = np.linspace(1.0 / (n_alpha + 1), n_alpha / (n_alpha + 1), n_alpha)
    gaps = []
    for a in alphas:
        k = int(np.ceil((n + 1) * (1.0 - a))) - 1          # conservative order statistic
        q = c[min(max(k, 0), n - 1)]
        gaps.append(abs(float(np.mean(t <= q)) - (1.0 - a)))
    return float(np.mean(gaps))


def ot_coverage_gap_bound(cal_scores: np.ndarray, pred_log: np.ndarray,
                          y_lo: float, y_hi: float,
                          n_candidates: int | None = None) -> dict:
    """Evaluate the Theorem 3.3 bound for one fold; return the bound and its components."""
    s_dn, s_up = sandwich_scores(pred_log, y_lo, y_hi, n_candidates)
    sup_p = kde_sup_density(cal_scores)
    w1_up = w1_empirical(cal_scores, s_up)
    w1_dn = w1_empirical(cal_scores, s_dn)
    spread = float(np.mean(s_up) - np.mean(s_dn))
    bound = 0.5 * sup_p * (w1_up + w1_dn + spread)
    return {
        "bound": float(bound),
        "sup_cal_density": float(sup_p),
        "w1_upper": float(w1_up),
        "w1_lower": float(w1_dn),
        "sandwich_spread": spread,
        "vacuous": bool(bound >= 1.0),
    }
