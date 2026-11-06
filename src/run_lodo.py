"""Leave-one-dataset-out runner for calibrated battery-life prediction.

For each seed and each held-out domain, the source domains are split by cell into
train, validation and calibration partitions; the predictor is fitted on train,
the conformal quantile on calibration, and point and interval metrics are computed
on the held-out domain. Each fold writes metrics.json, a checkpoint and
predictions.npz, with sha256 digests of both artifacts recorded in the metrics.
Aggregation and significance testing are separate steps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.leakage_check import (  # noqa: E402
    assert_domain_held_out,
    assert_partitions_disjoint,
)
from src.conformal import (  # noqa: E402
    PhysicsWeightedConformal,
    absolute_residual_score,
    conformal_quantile,
    weighted_conformal_quantile,
)
from src.data import build_dataset  # noqa: E402
from src.data_batterylife import AGING_FACTOR_NUMERIC_RANGES  # noqa: E402
from src.metrics import interval_metrics, n_eff, point_metrics  # noqa: E402
from src.preprocess import (  # noqa: E402
    RangeClippedRegressor,
    RobustFeatureScaler,
    domain_constant_features,
)
from src.paths import resolve, to_record  # noqa: E402
from src.provenance import sha256_file, write_provenance  # noqa: E402
from src.shift import (centroid_shift, feature_shift, physical_shift,  # noqa: E402
                       score_mean_ratio)
from src.seed import set_all_seeds  # noqa: E402
from src.splits import leave_one_domain_out, train_val_cal_split  # noqa: E402

DEFAULT_CKPT_ROOT = "checkpoints"  # relative to $BATTERY_ROOT (src/paths.py)


def _build_model(name: str, seed: int, groups=None):
    """Build the named predictor, wrapped to clip predictions to the observed range.

    `tuned` runs a cell-grouped randomized search over model class and
    hyperparameters on training-domain cells only.
    """
    if name == "cpmlp":
        from src.cpmlp import CPMLPRegressor
        return RangeClippedRegressor(CPMLPRegressor(seed=seed))
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, RandomizedSearchCV
    from sklearn.pipeline import Pipeline, make_pipeline

    if name == "dummy":
        base = DummyRegressor(strategy="mean")
    elif name == "ridge":
        base = make_pipeline(RobustFeatureScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 25)))
    elif name == "rf":
        base = make_pipeline(
            RobustFeatureScaler(),
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1))
    elif name == "tuned":
        pipe = Pipeline([("sc", RobustFeatureScaler()),
                         ("est", HistGradientBoostingRegressor(random_state=seed))])
        grid = [
            {"est": [HistGradientBoostingRegressor(random_state=seed)],
             "est__max_depth": [2, 3, 4, None],
             "est__learning_rate": [0.03, 0.06, 0.1],
             "est__max_iter": [150, 300],
             "est__min_samples_leaf": [5, 10, 20],
             "sc__clip": [5.0, 10.0, None]},
            {"est": [RandomForestRegressor(random_state=seed, n_jobs=-1)],
             "est__n_estimators": [300],
             "est__max_depth": [3, 5, 8, None],
             "est__min_samples_leaf": [1, 3, 5],
             "sc__clip": [5.0, 10.0, None]},
            {"est": [RidgeCV(alphas=np.logspace(-3, 3, 25))],
             "sc__clip": [5.0, 10.0, None],
             "sc__scale_floor": [1e-4, 1e-3, 1e-2]},
        ]
        # GroupKFold keeps the inner folds cell-disjoint, so selection cannot be
        # inflated by a cell appearing on both sides of an inner split.
        base = RandomizedSearchCV(
            pipe, grid, n_iter=40, cv=GroupKFold(n_splits=4),
            scoring="neg_mean_absolute_error", random_state=seed, n_jobs=-1,
            refit=True)
    else:
        raise ValueError(
            f"unknown model '{name}' (options: dummy, ridge, rf, tuned)")
    # Clipping prevents extrapolated lifetimes outside the range seen in the corpus.
    return RangeClippedRegressor(base)


def _interval_variants(cal_scores, cal_meta, target_meta, nominal, bandwidth):
    """Return {variant: (quantile, diagnostics)} for each interval construction.

    All variants share one predictor and one calibration set, so coverage
    differences are attributable to the interval construction alone.
    """
    out = {}
    q = conformal_quantile(cal_scores, nominal)
    out["split_cp"] = (q, {"quantile": float(q), "n_cal": int(len(cal_scores))})

    pw = PhysicsWeightedConformal(
        bandwidth=bandwidth, numeric_ranges=AGING_FACTOR_NUMERIC_RANGES
    ).fit(cal_scores, cal_meta)
    w = pw.weights_for(target_meta)
    qw = weighted_conformal_quantile(cal_scores, w, nominal)
    out["physics_weighted_cp"] = (
        qw, {"quantile": float(qw), "n_cal": int(len(cal_scores)),
             "n_eff": n_eff(w), "weight_sum": float(w.sum()),
             "weight_min": float(w.min()), "weight_max": float(w.max()),
             "bandwidth": float(bandwidth)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/lodo")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seeds = [int(s) for s in cfg.get("seeds", [cfg.get("seed", 0)])]
    nominals = [float(x) for x in cfg.get("nominals", [0.90, 0.95])]
    model_name = cfg.get("model", "ridge")
    bandwidth = float(cfg.get("bandwidth", 0.3))
    fracs = tuple(cfg.get("split_fracs", (0.6, 0.2, 0.2)))
    # Domains too small for a coverage estimate remain in the source pool but are
    # not scored as targets.
    min_target_cells = int(cfg.get("min_target_cells", 10))
    ckpt_root = resolve(cfg.get("checkpoint_dir", DEFAULT_CKPT_ROOT)) / Path(args.out).name

    out = Path(args.out)

    ds = build_dataset(cfg["data"])
    if not hasattr(ds, "load_full"):
        raise SystemExit(
            f"data.type '{cfg['data'].get('type')}' has no load_full(); the LODO "
            f"protocol needs per-cell domains and aging-factor metadata.")
    d = ds.load_full()
    domains = np.unique(d.domains)
    if len(domains) < 2:
        raise SystemExit(f"LODO needs >=2 domains, found {list(domains)}")

    # Pin the corpus against silent data drift. The feature version is recorded in
    # the manifest so a deliberate feature change is identifiable.
    from src.dataset_hash import dataset_fingerprint, verify_or_pin
    from src.data_batterylife import feature_version
    fp = dataset_fingerprint(d.X, d.y, d.cells,
                             name="BatteryLife_Processed",
                             version=f"S{cfg['data'].get('n_cycles', 100)}_"
                                     f"{len(np.unique(d.domains))}dom_"
                                     f"{feature_version(cfg['data'].get('capacity_reference', 'nominal'))}")
    verify_or_pin(cfg.get("manifest", "dataset_manifest.json"), fp)

    const_mask = domain_constant_features(d.X, d.domains)
    keep = ~const_mask
    dropped = [d.feature_names[i] for i in np.where(const_mask)[0]]
    X = d.X[:, keep]
    feature_names = [n for n, k in zip(d.feature_names, keep) if k]

    counts = {str(s): int((d.domains == s).sum()) for s in domains}
    # `target_subsets` replaces the rotation with a fixed source/target split.
    explicit_targets = [str(s) for s in (cfg.get("target_subsets") or [])]
    if explicit_targets:
        unknown = [s for s in explicit_targets if s not in {str(x) for x in domains}]
        if unknown:
            raise SystemExit(f"target_subsets not present in the corpus: {unknown}")
        scored = [s for s in domains if str(s) in explicit_targets]
        unscored = [s for s in domains if str(s) not in explicit_targets]
    else:
        scored = [s for s in domains if counts[str(s)] >= min_target_cells]
        unscored = [s for s in domains if counts[str(s)] < min_target_cells]

    write_provenance(out, {
        **cfg,
        "n_cells": int(len(d)),
        "domains": domains.tolist(),
        "domain_cell_counts": counts,
        "domains_scored_as_target": [str(s) for s in scored],
        "domains_source_only_too_small": [str(s) for s in unscored],
        "min_target_cells": min_target_cells,
        "features_used": feature_names,
        "features_dropped_domain_constant": dropped,
        "cells_skipped": ds.skipped_,
        "cells_excluded_short_life": ds.excluded_short_life_,
        "checkpoint_dir": to_record(ckpt_root),
    }, seeds[0], manifest_path=cfg.get("manifest", "dataset_manifest.json"))

    print(f"corpus: {len(d)} cells | {len(domains)} domains | {len(feature_names)} features")
    if dropped:
        print(f"  dropped domain-constant features: {dropped}")
    if unscored:
        why = ("not a designated target" if explicit_targets
               else f"fewer than {min_target_cells} cells, cannot support a "
                    f"coverage estimate")
        print(f"  source-only ({why}): {[str(s) for s in unscored]}")
    print(f"  scored as targets: {[str(s) for s in scored]}")
    print(f"model={model_name} seeds={seeds} nominals={nominals}")
    print(f"  checkpoints -> {ckpt_root}\n")

    y_log = np.log10(d.y)
    n_written = 0
    # CPMLP reads raw early-cycle curves; the shift measures use the tabular features.
    X_model = X
    if model_name == "cpmlp":
        from src.cpmlp import load_curves
        X_model, zero_filled = load_curves(ds.root, d.domains, d.cells)
        print(f"  CPMLP curves {X_model.shape}, zero-filled cycles: {zero_filled}")

    for seed in seeds:
        set_all_seeds(seed)
        for fold, (held, src_idx, tgt_idx) in enumerate(leave_one_domain_out(d.domains)):
            if str(held) not in [str(s) for s in scored]:
                continue
            assert_domain_held_out(d.domains, src_idx, tgt_idx, fold=fold)

            rel_tr, rel_va, rel_cal = train_val_cal_split(
                d.cells[src_idx], seed=seed, fracs=fracs)
            tr, va, cal = src_idx[rel_tr], src_idx[rel_va], src_idx[rel_cal]
            assert_partitions_disjoint(
                d.cells, {"train": tr, "val": va, "cal": cal, "target": tgt_idx},
                fold=f"{seed}/{fold}")

            model = _build_model(model_name, seed)
            # Grouped inner CV needs cell ids to keep its folds cell-disjoint.
            fit_kw = {"groups": d.cells[tr]} if model_name in ("tuned", "cpmlp") else {}
            model.fit(X_model[tr], y_log[tr], **fit_kw)
            cal_pred_log = model.predict(X_model[cal])
            cal_scores = absolute_residual_score(y_log[cal], cal_pred_log)
            tgt_pred_log = model.predict(X_model[tgt_idx])
            tgt_pred, tgt_true = 10 ** tgt_pred_log, d.y[tgt_idx]
            val_pred = 10 ** model.predict(X_model[va])

            target_meta = d.meta[int(tgt_idx[0])]
            cal_meta = [d.meta[i] for i in cal]

            tag = f"seed{seed}_fold{fold}_{held}"
            ck_dir = ckpt_root / tag
            ck_dir.mkdir(parents=True, exist_ok=True)

            # Save the fitted estimator so results reproduce without retraining.
            import joblib
            ck_path = ck_dir / "checkpoint.joblib"
            joblib.dump({"model": model, "feature_names": feature_names,
                         "seed": seed, "held_out_domain": str(held),
                         "log_target": True}, ck_path)
            ck_sha = sha256_file(ck_path)

            # Save everything needed to recompute every metric.
            pred_path = ck_dir / "predictions.npz"
            npz = {
                "target_cells": d.cells[tgt_idx],
                "y_true": tgt_true,
                "y_pred": tgt_pred,
                "y_pred_log": tgt_pred_log,
                "cal_cells": d.cells[cal],
                "cal_scores": cal_scores,
                # Signed residuals, since CPMLP checkpoints cannot be re-applied to
                # tabular features to recover the signs.
                "cal_residuals": y_log[cal] - cal_pred_log,
            }

            # Continuous shift magnitudes between calibration and target inputs.
            train_scale = None
            inner = getattr(model, "estimator_", None)
            for step in (inner, getattr(inner, "best_estimator_", None)):
                if step is not None and hasattr(step, "named_steps"):
                    sc = step.named_steps.get("sc") or step.named_steps.get(
                        "robustfeaturescaler")
                    if sc is not None and hasattr(sc, "scale_"):
                        train_scale = sc.scale_
                        break
            shift = {
                "physical_gower": physical_shift(
                    cal_meta, target_meta, AGING_FACTOR_NUMERIC_RANGES),
                "feature_energy": feature_shift(X[cal], X[tgt_idx], train_scale),
                "feature_centroid": centroid_shift(X[cal], X[tgt_idx], train_scale),
                # Uses target labels, so it is an analysis quantity rather than a
                # deployable monitor. Both sides are log10 residuals.
                "score_mean_ratio": score_mean_ratio(
                    cal_scores, absolute_residual_score(y_log[tgt_idx], tgt_pred_log)),
            }

            rec = {
                "seed": seed, "fold": fold, "held_out_domain": str(held),
                "shift": shift,
                "n_train_cells": int(len(np.unique(d.cells[tr]))),
                "n_val_cells": int(len(np.unique(d.cells[va]))),
                "n_cal_cells": int(len(np.unique(d.cells[cal]))),
                "n_target_cells": int(len(np.unique(d.cells[tgt_idx]))),
                "n_predictions_clipped": (model.count_clipped(X_model[tgt_idx])
                                          if hasattr(model, "count_clipped") else 0),
                "n_val_predictions_clipped": (model.count_clipped(X_model[va])
                                              if hasattr(model, "count_clipped") else 0),
                "point_target": point_metrics(tgt_true, tgt_pred),
                "point_val": point_metrics(d.y[va], val_pred),
                "intervals": {},
                "checkpoint": {"path": to_record(ck_path), "sha256": ck_sha},
            }

            # In-domain reference: the same calibration quantile applied to unseen
            # source-domain validation cells measures coverage without shift.
            val_pred_log = model.predict(X_model[va])
            npz["val_cells"] = d.cells[va]
            npz["val_y_true"] = d.y[va]
            npz["val_y_pred"] = val_pred

            for nominal in nominals:
                for vname, (q, diag) in _interval_variants(
                        cal_scores, cal_meta, target_meta, nominal, bandwidth).items():
                    lo = 10 ** (tgt_pred_log - q)
                    hi = 10 ** (tgt_pred_log + q)
                    m = interval_metrics(tgt_true, lo, hi, nominal)
                    m.update({f"diag_{k}": v for k, v in diag.items()})
                    rec["intervals"].setdefault(vname, []).append(m)
                    npz[f"lo_{vname}_{nominal:.2f}"] = lo
                    npz[f"hi_{vname}_{nominal:.2f}"] = hi

                # Unweighted quantile, as there is no shift for weighting to correct.
                q_in = conformal_quantile(cal_scores, nominal)
                lo_in = 10 ** (val_pred_log - q_in)
                hi_in = 10 ** (val_pred_log + q_in)
                rec.setdefault("intervals_in_domain", []).append(
                    interval_metrics(d.y[va], lo_in, hi_in, nominal))
                npz[f"val_lo_{nominal:.2f}"] = lo_in
                npz[f"val_hi_{nominal:.2f}"] = hi_in

            np.savez_compressed(pred_path, **npz)
            rec["predictions"] = {"path": to_record(pred_path),
                                  "sha256": sha256_file(pred_path)}

            fold_dir = out / f"seed_{seed}" / f"fold_{fold}_{held}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "metrics.json").write_text(json.dumps(rec, indent=2))
            n_written += 1

            pm = rec["point_target"]
            cov = " ".join(
                f"{v.split('_')[0]}@{m['nominal']:.2f}={m['picp']:.3f}"
                for v, ms in rec["intervals"].items() for m in ms)
            print(f"  seed {seed} fold {fold} {held:9s} n={rec['n_target_cells']:3d} "
                  f"MAPE={pm['mape']*100:7.2f}%  {cov}")

    print(f"\n{n_written} fold-runs written -> {out}")
    print(f"checkpoints + predictions -> {ckpt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
