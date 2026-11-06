"""Aggregate per-fold metrics.json files into summary.json with mean and std."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    files = sorted(run.glob("fold_*/metrics.json"))
    if not files:
        raise SystemExit(f"no metrics found under {run}")
    rows = [json.loads(f.read_text()) for f in files]
    keys = rows[0].keys()
    summary = {
        k: {"mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows])),
            "n_folds": len(rows)}
        for k in keys
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.3f} ± {v['std']:.3f}  (n={v['n_folds']})")


if __name__ == "__main__":
    main()
