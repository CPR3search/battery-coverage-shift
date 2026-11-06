"""Group-disjoint splits.

No group (a subject, or a battery cell) appears on both sides of a split; the
helpers are group-generic. Splits are provided at the cell level (train /
validation / calibration) and at the domain level (leave-one-domain-out).
"""
from typing import Iterator

import numpy as np


def loso_splits(subjects: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-subject-out: each unique subject is the test fold once."""
    for s in np.unique(subjects):
        test = np.where(subjects == s)[0]
        train = np.where(subjects != s)[0]
        yield train, test


def grouped_kfold(
    subjects: np.ndarray, k: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """k-fold split that assigns whole subjects to folds."""
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    for f in folds:
        test = np.isin(subjects, f)
        yield np.where(~test)[0], np.where(test)[0]


# ---------------------------------------------------------------------------
# domain-level and cell-level splits
# ---------------------------------------------------------------------------
def leave_one_domain_out(
    domains: np.ndarray,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-domain-out: each domain is the held-out target once.

    `domains` gives one source-domain label per sample. Yields
    (held_out_domain, source_idx, target_idx). The source side must still be
    partitioned by cell with `train_val_cal_split`.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    domains = np.asarray(domains)
    logo = LeaveOneGroupOut()
    dummy = np.zeros(len(domains))
    for src, tgt in logo.split(dummy, groups=domains):
        held = str(np.unique(domains[tgt])[0])
        yield held, src, tgt


def train_val_cal_split(
    cells: np.ndarray,
    seed: int,
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition source-domain samples into train / validation / calibration by cell.

    Returns three index arrays; each cell is assigned to exactly one part.
    Calibration is disjoint from train and validation and is used only by the
    conformal layer.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must sum to 1.0, got {fracs} (sum {sum(fracs)})")
    cells = np.asarray(cells)
    uniq = np.unique(cells)
    if len(uniq) < 3:
        raise ValueError(
            f"need >=3 distinct cells to form train/val/calibration, got {len(uniq)}")

    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_tr = max(1, int(round(fracs[0] * n)))
    n_va = max(1, int(round(fracs[1] * n)))
    # Calibration takes the remainder so no cell is dropped by rounding.
    if n_tr + n_va >= n:
        n_tr, n_va = max(1, n - 2), 1
    parts = (shuffled[:n_tr], shuffled[n_tr:n_tr + n_va], shuffled[n_tr + n_va:])
    return tuple(np.where(np.isin(cells, p))[0] for p in parts)
