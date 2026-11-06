"""Example of a one-time, lock-protected held-out test evaluation.

Running twice with the same --tag is refused by the lock.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import SyntheticDataset
from .model import NearestCentroid  # noqa: F401 (registers)
from .registry import build_model
from .seed import set_all_seeds
from .test_lock import evaluate_final_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eev_heldout")
    ap.add_argument("--run", default="runs/final_test")
    ap.add_argument("--holdout-subjects", type=int, default=2)
    args = ap.parse_args()

    set_all_seeds(0)
    X, y, subj = SyntheticDataset().load()
    uniq = np.unique(subj)
    held = set(uniq[-args.holdout_subjects:].tolist())  # fixed held-out subjects
    te = np.isin(subj, list(held))
    tr = ~te

    def _run():
        model = build_model(
            {"name": "nearest_centroid", "args": {"normalize": True}}
        ).fit(X[tr], y[tr], subj[tr])
        acc = float((model.predict(X[te]) == y[te]).mean())
        out = Path(args.run); out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))
        print(f"held-out test accuracy = {acc:.3f}  (subjects {sorted(held)})")
        return acc

    evaluate_final_test(args.tag, run_id=args.run, fn=_run)
    print(f"test '{args.tag}' is now LOCKED — re-running is refused.")


if __name__ == "__main__":
    main()
