"""Re-derive a run's metrics from its saved checkpoints, without retraining.

  --verify        reload each fold's estimator, re-predict on its target and
                  validation cells, and require the stored predictions back.
  --rewrite       additionally recompute the conformal intervals with the current
                  interval code from the stored predictions, and rewrite
                  metrics.json and the interval arrays.
  --cross-device  report how far this device's predictions differ from the
                  stored ones; writes nothing.

Point predictions are never refitted, so the held-out set is not re-evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import conformal_quantile  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.metrics import interval_metrics  # noqa: E402
from src.paths import resolve  # noqa: E402
from src.preprocess import domain_constant_features  # noqa: E402
from src.provenance import sha256_file  # noqa: E402
from src.run_lodo import _interval_variants  # noqa: E402


def _load_dataset(prov: dict):
    """Rebuild the feature matrix the run fitted on, without domain-constant columns.

    Returns (dataset, X, feature_names, row index by cell id).
    """
    d = build_dataset(prov["config"]["data"]).load_full()
    keep = ~domain_constant_features(d.X, d.domains)
    X = d.X[:, keep]
    names = [n for n, k in zip(d.feature_names, keep) if k]
    return d, X, names, {str(c): i for i, c in enumerate(d.cells)}


def _rows(row_of: dict, cells) -> np.ndarray:
    missing = [str(c) for c in cells if str(c) not in row_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} cell(s) in the saved predictions are absent from the "
            f"rebuilt dataset (first: {missing[:3]}). The data changed since the run; "
            f"the checkpoint cannot be re-scored against it.")
    return np.array([row_of[str(c)] for c in cells], dtype=int)


def process(run: Path, rewrite: bool) -> int:
    prov = json.loads((run / "provenance.json").read_text())
    cfg = prov["config"]
    nominals = [float(x) for x in cfg.get("nominals", [0.90, 0.95])]
    bandwidth = float(cfg.get("bandwidth", 0.3))
    d, X, feature_names, row_of = _load_dataset(prov)
    if cfg.get("model") == "cpmlp":
        from src.cpmlp import load_curves
        X = load_curves(build_dataset(cfg["data"]).root, d.domains, d.cells)[0]

    n_ok = n_fold = 0
    mismatches: list[str] = []
    inexact: list[str] = []
    unexpected: list[str] = []
    for mp in sorted(run.glob("seed_*/fold_*/metrics.json")):
        n_fold += 1
        rec = json.loads(mp.read_text())
        key = "/".join(mp.parts[-3:-1])
        ck_path = resolve(rec["checkpoint"]["path"])
        pred_path = resolve(rec["predictions"]["path"])
        if not ck_path.exists() or not pred_path.exists():
            mismatches.append(f"{key}: checkpoint or predictions missing from disk")
            continue

        ck = joblib.load(ck_path)
        model = ck["model"]
        if list(ck.get("feature_names", feature_names)) != list(feature_names):
            mismatches.append(f"{key}: checkpoint was fitted on different features")
            continue
        z = dict(np.load(pred_path, allow_pickle=True))
        tgt = _rows(row_of, z["target_cells"])
        cal = _rows(row_of, z["cal_cells"])
        val = _rows(row_of, z["val_cells"])

        # Threaded accumulation in some estimators makes predict() reproducible only
        # to floating-point tolerance; such folds are reported separately from real
        # mismatches.
        checks = {
            "y_pred_log": (model.predict(X[tgt]), z["y_pred_log"]),
            "val_y_pred": (10 ** model.predict(X[val]), z["val_y_pred"]),
        }
        worst_rel = max(
            float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300)))
            for got, want in checks.values())
        if worst_rel > 1e-9:
            mismatches.append(f"{key}: predictions differ (max relative {worst_rel:.3e})")
            continue
        if worst_rel > 0:
            inexact.append(f"{key}: {worst_rel:.2e}")
        n_ok += 1
        if not rewrite:
            continue

        # Intervals are rebuilt from the stored calibration scores and predictions,
        # never from fresh predictions, so the reconstruction is exact. The in-domain
        # intervals use the unweighted quantile and are left unchanged.
        cal_scores, tgt_pred_log = z["cal_scores"], z["y_pred_log"]
        target_meta, cal_meta = d.meta[int(tgt[0])], [d.meta[i] for i in cal]
        rec["intervals"] = {}
        for nominal in nominals:
            for vname, (q, diag) in _interval_variants(
                    cal_scores, cal_meta, target_meta, nominal, bandwidth).items():
                lo, hi = 10 ** (tgt_pred_log - q), 10 ** (tgt_pred_log + q)
                m = interval_metrics(z["y_true"], lo, hi, nominal)
                m.update({f"diag_{k}": v for k, v in diag.items()})
                rec["intervals"].setdefault(vname, []).append(m)
                # split_cp does not depend on the weighting, so any change is unexpected.
                if vname == "split_cp" and not np.array_equal(
                        z.get(f"lo_{vname}_{nominal:.2f}"), lo):
                    unexpected.append(f"{key}: split_cp @ {nominal} changed")
                z[f"lo_{vname}_{nominal:.2f}"], z[f"hi_{vname}_{nominal:.2f}"] = lo, hi

        np.savez_compressed(pred_path, **z)
        rec["predictions"]["sha256"] = sha256_file(pred_path)
        rec["recomputed_from_checkpoint"] = True
        mp.write_text(json.dumps(rec, indent=2))

    verb = "recomputed" if rewrite else "verified"
    print(f"{n_ok}/{n_fold} folds {verb} from frozen checkpoints ({run})")
    if inexact:
        print(f"  {len(inexact)} fold(s) reproduce to floating-point tolerance rather "
              f"than bit-exactly (threaded accumulation in predict()):")
        for m in inexact[:10]:
            print("    " + m)
        print("    Interval reconstruction uses the stored predictions, so this does "
              "not propagate.")
    if unexpected:
        print(f"\n{len(unexpected)} unweighted interval(s) moved, which nothing in this "
              f"change should touch:")
        for m in unexpected[:10]:
            print("  " + m)
        return 1
    if mismatches:
        print(f"\n{len(mismatches)} fold(s) did NOT reproduce:")
        for m in mismatches[:20]:
            print("  " + m)
        print("\nA stored prediction that the saved model does not return means the "
              "artifact and the weights disagree. Nothing downstream was rewritten "
              "for these folds.")
        return 1
    return 0


def downstream_changes(y_log, stored_pred, new_pred, stored_q, new_q) -> dict:
    """Coverage and recentered coverage from stored and recomputed predictions on one fold."""
    def _cov(pred, q):
        return float(np.mean(np.abs(y_log - pred) <= q))

    def _recentered(pred, q):
        return float(np.mean(np.abs(y_log - pred - np.median(y_log - pred)) <= q))

    return {"max_prediction_difference": float(np.max(np.abs(new_pred - stored_pred))),
            "threshold_difference": abs(new_q - stored_q),
            "coverage_difference": abs(_cov(new_pred, new_q) - _cov(stored_pred, stored_q)),
            "recentered_difference": abs(_recentered(new_pred, new_q)
                                         - _recentered(stored_pred, stored_q))}


def cross_device(run: Path, nominal: float = 0.90) -> int:
    """Compare this device's predictions with the stored ones, writing nothing.

    Returns exit status 1 if any coverage value changes.
    """
    prov = json.loads((run / "provenance.json").read_text())
    cfg = prov["config"]
    d, X, _, row_of = _load_dataset(prov)
    if cfg.get("model") == "cpmlp":
        from src.cpmlp import load_curves
        X = load_curves(build_dataset(cfg["data"]).root, d.domains, d.cells)[0]
    rows = []
    for mp in sorted(run.glob("seed_*/fold_*/metrics.json")):
        rec = json.loads(mp.read_text())
        model = joblib.load(resolve(rec["checkpoint"]["path"]))["model"]
        z = dict(np.load(resolve(rec["predictions"]["path"]), allow_pickle=True))
        tgt, cal = _rows(row_of, z["target_cells"]), _rows(row_of, z["cal_cells"])
        new_pred = np.asarray(model.predict(X[tgt]), float)
        new_scores = np.abs(np.log10(np.asarray(d.y[cal], float))
                            - np.asarray(model.predict(X[cal]), float))
        rows.append(downstream_changes(
            np.log10(z["y_true"].astype(float)), z["y_pred_log"], new_pred,
            conformal_quantile(z["cal_scores"], nominal), conformal_quantile(new_scores, nominal)))
    summary = {k: max(r[k] for r in rows) for k in rows[0]}
    print(f"{len(rows)} folds of {run}, largest differences at nominal {nominal}:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3e}")
    return int(summary["coverage_difference"] > 0 or summary["recentered_difference"] > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--rewrite", action="store_true",
                    help="also recompute intervals and rewrite metrics.json")
    ap.add_argument("--cross-device", action="store_true",
                    help="report how far this device's predictions and coverage differ from the "
                         "stored ones; writes nothing")
    a = ap.parse_args()
    if not (a.run / "provenance.json").exists():
        raise SystemExit(f"no provenance.json under {a.run}")
    if a.cross_device:
        return cross_device(a.run)
    return process(a.run, a.rewrite)


if __name__ == "__main__":
    raise SystemExit(main())
