"""Paired permutation (sign-flip) test for matched per-fold or per-unit differences."""
import numpy as np


def paired_permutation_test(a, b, n_perm: int = 10000, seed: int = 0):
    """Two-sided paired test on a - b. Returns (mean_diff, p_value).

    Enumerates all sign flips exactly for up to 20 pairs, otherwise samples n_perm.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must be matched (same shape)")
    d = a - b
    obs = d.mean()
    n = len(d)
    if n == 0:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    if n <= 20:
        signs = np.array(np.meshgrid(*[[1, -1]] * n)).T.reshape(-1, n)
    else:
        signs = rng.choice([1, -1], size=(n_perm, n))
    perm_means = (signs * np.abs(d)).mean(1)
    p = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    return float(obs), p
