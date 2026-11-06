"""Compare two runs on one metric with a paired permutation test over matched folds.

Both runs must have the same number of folds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .significance import paired_permutation_test


def _per_fold(run, metric):
    files = sorted(Path(run).glob("fold_*/metrics.json"))
    return np.array([json.loads(f.read_text())[metric] for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="ours")
    ap.add_argument("--run-b", required=True, help="baseline")
    ap.add_argument("--metric", default="macro_f1")
    args = ap.parse_args()

    a, b = _per_fold(args.run_a, args.metric), _per_fold(args.run_b, args.metric)
    if a.shape != b.shape:
        raise SystemExit(f"fold mismatch: {a.shape} vs {b.shape}")
    diff, p = paired_permutation_test(a, b)
    print(f"metric: {args.metric}  (paired over {len(a)} folds)")
    print(f"  A {a.mean():.3f} ± {a.std():.3f}   B {b.mean():.3f} ± {b.std():.3f}")
    print(f"  mean diff (A-B) = {diff:+.3f}   p = {p:.4f}   "
          f"{'(significant, p<0.05)' if p < 0.05 else '(not significant at 0.05)'}")


if __name__ == "__main__":
    main()
