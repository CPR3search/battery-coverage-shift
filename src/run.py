"""Generic subject-grouped cross-validation runner.

Loads a config, builds subject-grouped splits, checks each fold for leakage,
trains and evaluates the model, and writes one metrics.json per fold plus
provenance.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .data import build_dataset
from .dataset_hash import dataset_fingerprint, verify_or_pin
from .model import NearestCentroid  # noqa: F401  (registers the model)
from .provenance import write_provenance
from .registry import build_model
from .seed import set_all_seeds
from .splits import grouped_kfold, loso_splits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.leakage_check import assert_no_leakage  # noqa: E402


def _metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in np.unique(y_true):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("seed", 0)
    set_all_seeds(seed)
    out = Path(args.out)

    X, y, subjects = build_dataset(cfg.get("data", {})).load()
    fp = dataset_fingerprint(X, y, subjects, name=cfg.get("dataset_name", "synthetic"))
    verify_or_pin(cfg.get("manifest", "dataset_manifest.json"), fp)
    write_provenance(out, {**cfg, "dataset": fp}, seed)

    proto = cfg.get("protocol", "loso")
    if proto == "loso":
        splits = list(loso_splits(subjects))
    elif proto == "grouped_kfold":
        splits = list(grouped_kfold(subjects, cfg.get("k", 5), seed))
    else:
        raise ValueError(f"unknown protocol '{proto}' "
                         "(subject-random splits are leakage-prone; not allowed)")

    fold_results = []
    for i, (tr, te) in enumerate(splits):
        # Halt if any subject appears on both sides of the split.
        assert_no_leakage(subjects, tr, te, fold=i)
        model = build_model(cfg["model"]).fit(X[tr], y[tr], subjects[tr])
        m = _metrics(y[te], model.predict(X[te]))
        fold_dir = out / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(m, indent=2))
        fold_results.append(m)
        print(f"fold {i}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")

    print(f"\n{len(fold_results)} folds complete → {out}")


if __name__ == "__main__":
    main()
