"""Location correction for conformal intervals under dataset shift.

The correction shifts the interval centre by the median signed residual on the target
domain, leaving the width unchanged. It requires a small number of target labels.
"""
from __future__ import annotations

import numpy as np

__all__ = ["evaluate_no_predictor", "estimate_offset", "apply_offset", "censored_split", "evaluate_at_budget",
           "evaluate_qualification", "evaluate_comparators"]


def estimate_offset(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Median signed residual in the predictor's (log) space.

    The median is used for robustness to a single large residual on an extrapolating fold.
    """
    y_true_log = np.asarray(y_true_log, float)
    y_pred_log = np.asarray(y_pred_log, float)
    if len(y_true_log) == 0:
        return float("nan")
    return float(np.median(y_true_log - y_pred_log))


def apply_offset(y_pred_log: np.ndarray, offset: float) -> np.ndarray:
    """Shift the interval centre by `offset`; the width is unchanged."""
    return np.asarray(y_pred_log, float) + float(offset)


def censored_split(y_true: np.ndarray, budget: float):
    """Split cells into those observed to fail by cycle `budget` and those still alive.

    Returns boolean masks (observed, alive). The offset is estimated on the first group and
    scored on the second, so no cell contributes to both.
    """
    y_true = np.asarray(y_true, float)
    observed = y_true <= budget
    return observed, ~observed


def evaluate_at_budget(y_true: np.ndarray, y_pred_log: np.ndarray, half_width: float,
                       budget: float, min_observed: int = 3) -> dict | None:
    """Out-of-sample coverage of the corrected interval at one cycle budget.

    Returns None when fewer than `min_observed` cells have failed or none remain to score.
    """
    y_true = np.asarray(y_true, float)
    y_pred_log = np.asarray(y_pred_log, float)
    obs, alive = censored_split(y_true, budget)
    if obs.sum() < min_observed or alive.sum() == 0:
        return None

    y_log = np.log10(y_true)
    offset = estimate_offset(y_log[obs], y_pred_log[obs])
    centred = apply_offset(y_pred_log[alive], offset)

    def _cov(centre):
        return float(np.mean(np.abs(y_log[alive] - centre) <= half_width))

    return {
        "budget": float(budget),
        "n_observed": int(obs.sum()),
        "n_scored": int(alive.sum()),
        "observed_fraction": float(obs.mean()),
        "offset": offset,
        "coverage_corrected": _cov(centred),
        "coverage_uncorrected": _cov(y_pred_log[alive]),
    }


def evaluate_qualification(y_true: np.ndarray, y_pred_log: np.ndarray,
                           half_width: float, k: int, n_draws: int = 200,
                           seed: int = 0) -> dict | None:
    """Coverage after labelling k randomly drawn target cells, scored out of sample.

    Unlike `evaluate_at_budget`, the label budget is a number of cells run to end of life
    rather than a cycle count. The k labelled cells are excluded from scoring.
    """
    y_true = np.asarray(y_true, float)
    y_pred_log = np.asarray(y_pred_log, float)
    n = len(y_true)
    if n <= k + 1:
        return None

    y_log = np.log10(y_true)
    rng = np.random.default_rng(seed)
    cor, unc, offs = [], [], []
    for _ in range(n_draws):
        idx = rng.choice(n, k, replace=False)
        held = np.ones(n, bool)
        held[idx] = False
        offset = estimate_offset(y_log[idx], y_pred_log[idx])
        centred = apply_offset(y_pred_log[held], offset)
        cor.append(float(np.mean(np.abs(y_log[held] - centred) <= half_width)))
        unc.append(float(np.mean(np.abs(y_log[held] - y_pred_log[held]) <= half_width)))
        offs.append(offset)
    return {"k": int(k), "n_scored": int(n - k),
            "coverage_corrected": float(np.mean(cor)),
            "coverage_uncorrected": float(np.mean(unc)),
            "offset_median": float(np.median(offs)),
            "offset_spread": float(np.std(offs))}


def _cycle_width(centre_log: np.ndarray, half_width: float) -> float:
    """Mean interval width in cycles for log-space centres and one log half-width."""
    if not np.isfinite(half_width):
        return float("inf")
    return float(np.mean(10 ** centre_log * (10 ** half_width - 10 ** -half_width)))


def evaluate_comparators(y_true: np.ndarray, y_pred_log: np.ndarray, half_width: float,
                         cal_abs_scores: np.ndarray, k: int, nominal: float,
                         n_draws: int = 200, seed: int = 0) -> dict | None:
    """Compare ways of using the same k target labels, scored on the same held-out cells.

    All arms share the draws of `evaluate_qualification` (same rng sequence):

    U   uncorrected.
    L   centre + median residual of the k cells; half-width unchanged.
    S   half-width * median|r_k| / median|r_cal|; centre unchanged.
    T   centre unchanged; half-width = conformal quantile of |r_k|.
    LT  centre + median(r_k); half-width = conformal quantile of |r_k - median(r_k)|.

    T and LT have an infinite half-width when k is too small for the nominal level; their
    widths are then reported as undefined. Widths are given as a log-space multiple of the
    source half-width and as a ratio of mean cycle widths to the uncorrected arm.
    """
    from src.conformal import conformal_quantile

    y_true = np.asarray(y_true, float)
    y_pred_log = np.asarray(y_pred_log, float)
    n = len(y_true)
    if n <= k + 1:
        return None

    y_log = np.log10(y_true)
    cal_scale = float(np.median(np.asarray(cal_abs_scores, float)))
    rng = np.random.default_rng(seed)
    acc = {a: {"cov": [], "log_mult": [], "cyc_ratio": [], "width": []}
           for a in ("U", "L", "S", "T", "LT")}
    for _ in range(n_draws):
        idx = rng.choice(n, k, replace=False)
        held = np.ones(n, bool)
        held[idx] = False
        r_k = y_log[idx] - y_pred_log[idx]
        m = float(np.median(r_k))
        centre = y_pred_log[held]
        arms = {
            "U": (centre, half_width),
            "L": (centre + m, half_width),
            "S": (centre, half_width * float(np.median(np.abs(r_k))) / cal_scale),
            "T": (centre, conformal_quantile(np.abs(r_k), nominal)),
            "LT": (centre + m, conformal_quantile(np.abs(r_k - m), nominal)),
        }
        base_width = _cycle_width(centre, half_width)
        for a, (c, h) in arms.items():
            acc[a]["cov"].append(float(np.mean(np.abs(y_log[held] - c) <= h)))
            acc[a]["log_mult"].append(h / half_width)
            acc[a]["cyc_ratio"].append(_cycle_width(c, h) / base_width)
            acc[a]["width"].append(_cycle_width(c, h))

    out = {"k": int(k), "n_scored": int(n - k)}
    for a, v in acc.items():
        defined = bool(np.all(np.isfinite(v["log_mult"])))
        out[a] = {"coverage": float(np.mean(v["cov"])),
                  "log_mult": float(np.mean(v["log_mult"])) if defined else float("inf"),
                  "cyc_ratio": float(np.mean(v["cyc_ratio"])) if defined else float("inf"),
                  "width": float(np.mean(v["width"])) if defined else float("inf"),
                  "defined": defined,
                  # Fraction of draws below 80% coverage: a single qualification is one draw.
                  "below80": float(np.mean(np.array(v["cov"]) < 0.80))}
    return out


def evaluate_no_predictor(y_true: np.ndarray, k: int, nominal: float,
                          n_draws: int = 200, seed: int = 0) -> dict | None:
    """Predictor-free intervals from k complete lifetimes, on the draws of `evaluate_comparators`.

    `median`: centred on the median of the k log lives, half-width the conformal quantile of
    their absolute deviations (no split-conformal guarantee). `range`: the shortest to the
    longest of the k lives, with marginal coverage at least (k-1)/(k+1) under exchangeability.
    """
    from src.conformal import conformal_quantile

    y_log = np.log10(np.asarray(y_true, float))
    n = len(y_log)
    if n <= k + 1:
        return None
    rng = np.random.default_rng(seed)
    acc = {"median": {"cov": [], "width": []}, "range": {"cov": [], "width": []}}
    for _ in range(n_draws):
        idx = rng.choice(n, k, replace=False)
        held = np.ones(n, bool)
        held[idx] = False
        lives = y_log[idx]
        centre = float(np.median(lives))
        half = conformal_quantile(np.abs(lives - centre), nominal)
        bounds = {"median": (centre - half, centre + half), "range": (lives.min(), lives.max())}
        for name, (lo, hi) in bounds.items():
            acc[name]["cov"].append(float(np.mean((y_log[held] >= lo) & (y_log[held] <= hi))))
            acc[name]["width"].append(float(10 ** hi - 10 ** lo))
    return {name: {"coverage": float(np.mean(v["cov"])), "width": float(np.mean(v["width"])),
                   "below80": float(np.mean(np.array(v["cov"]) < 0.80))}
            for name, v in acc.items()}
