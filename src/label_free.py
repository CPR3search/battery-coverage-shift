"""Label-free shift indicators computed per held-out dataset and seed.

- prediction shift: target minus source-validation median predicted log10 life.
- ML-physics gap: median target (predicted log10 life - log10 physics life).
- absolute gap: median absolute value of the same cellwise gaps.
- extrapolation shift: target minus validation median log10 physics life.
- curvature shift: target minus validation median quadratic SOH coefficient, fit by least
  squares against cycle index in the observation window. Capacities are finite per-cycle
  discharge maxima normalized by nominal capacity (or the window maximum when absent).
- cycle-time ratio: target / validation median cycle_time_mean.
- absolute cycle time: target median cycle_time_mean, in seconds.

Physics life = n_window + (soh_last - 0.8) / (-soh_slope). Cells with nonnegative slope or
nonfinite or nonpositive life are excluded and counted; physics-based indicators require
MIN_VALID valid cells on each side. No step uses target life labels.
"""
from __future__ import annotations

import numpy as np

MIN_VALID = 5
PHYSICS_INDICATORS = ("ML-physics gap", "absolute gap", "extrapolation shift")


def capacity_series(cycles, n_window, nominal_capacity):
    """Extractor's window cycle indices, normalized discharge maxima, denominator."""
    win = cycles[:n_window]
    caps, nums = [], []
    for c in win:
        dc = np.asarray(c.get('discharge_capacity_in_Ah', []), float)
        dc = dc[np.isfinite(dc)]
        caps.append(float(np.max(dc)) if dc.size else np.nan)
        nums.append(float(c.get('cycle_number', len(nums) + 1)))
    caps = np.asarray(caps)
    denom = nominal_capacity if nominal_capacity and nominal_capacity > 0 else float(np.nanmax(caps))
    return np.asarray(nums), caps / denom, denom


def curvature(cycles, n_window, nominal_capacity):
    """Least-squares quadratic coefficient on the extractor's finite SOH points."""
    nums, soh, _ = capacity_series(cycles, n_window, nominal_capacity)
    ok = np.isfinite(nums) & np.isfinite(soh)
    if ok.sum() < 3 or len(np.unique(nums[ok])) < 3:
        return float('nan')
    return float(np.polyfit(nums[ok], soh[ok], 2)[0])


def physics_log_life(features, names, n_window):
    """Return log10 extrapolated life, NaN for excluded cells, and exclusion count."""
    last = features[:, names.index('soh_last')]
    slope = features[:, names.index('soh_slope')]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        life = n_window + (last - 0.8) / (-slope)
    valid = (slope < 0) & np.isfinite(life) & (life > 0)
    logs = np.full(len(life), np.nan)
    logs[valid] = np.log10(life[valid])
    return logs, int((~valid).sum())


def indicators(target, source, names, target_pred_log, source_pred_log,
               target_curvature, source_curvature, n_window):
    """Seven per-run indicators and physics exclusions, using validation as source."""
    tp, nt = physics_log_life(target, names, n_window)
    sp, ns = physics_log_life(source, names, n_window)
    def med(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else float('nan')
    gap = np.asarray(target_pred_log) - tp
    ct = names.index('cycle_time_mean')
    enough = min(np.isfinite(tp).sum(), np.isfinite(sp).sum()) >= MIN_VALID
    return {
        'prediction shift': med(target_pred_log) - med(source_pred_log),
        'ML-physics gap': med(gap) if enough else float('nan'),
        'absolute gap': med(np.abs(gap)) if enough else float('nan'),
        'extrapolation shift': med(tp) - med(sp) if enough else float('nan'),
        'curvature shift': med(target_curvature) - med(source_curvature),
        'cycle-time ratio': med(target[:, ct]) / med(source[:, ct]),
        'absolute cycle time': med(target[:, ct]),
    }, {'target': nt, 'validation': ns}
