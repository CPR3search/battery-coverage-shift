"""One-time, lock-protected evaluation on the held-out non-Li chemistries.

Targets are Na-ion, Zn-ion and CALB. Each fold holds out one target; the source pool is
every Li-ion sub-dataset plus the other two targets. No hyperparameters are tuned here;
the tuned arm's inner cross-validation uses source cells only. A second evaluation of the
same tag is refused by `src.test_lock`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.test_lock import evaluate_final_test  # noqa: E402

LI_ION = ["CALCE", "HNEI", "HUST", "MATR", "MICH", "MICH_EXP",
          "RWTH", "SNL", "Tongji", "UL_PUR", "XJTU"]
NON_LI = ["NA-ion", "ZN-coin", "CALB"]


def _frozen_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chemistry_holdout")
    ap.add_argument("--out-root", default="runs/final_chemistry")
    ap.add_argument("--root", default="BatteryLife_Processed")  # relative to $BATTERY_ROOT
    ap.add_argument("--models", nargs="+", default=["ridge", "tuned"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--capacity-reference", default="nominal", choices=("nominal", "window"),
                    help="window runs the PROTOCOL.md Amendment 11 sensitivity, the third scoring campaign")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and check the lock, but evaluate nothing")
    args = ap.parse_args()

    root = Path(args.root)
    missing = [s for s in NON_LI if not (root / s).is_dir()
               or not any((root / s).glob("*.pkl"))]
    if missing:
        raise SystemExit(
            f"held-out chemistry data not present: {missing}\n"
            f"Fetch with: python -m src.download_batterylife --tier heldout")

    sha = _frozen_sha()
    print(f"tag        : {args.tag}")
    print(f"frozen at  : {sha}")
    print(f"source     : {len(LI_ION)} Li-ion sub-datasets")
    print(f"targets    : {NON_LI}")
    print(f"models     : {args.models}   seeds: {args.seeds}")
    if sha.endswith("-dirty"):
        print("\n!! WORKING TREE IS DIRTY. A one-shot test should be run from a clean\n"
              "!! commit so the reported numbers trace to an exact state. Commit first.")
        if not args.dry_run:
            raise SystemExit(1)

    window = "window" if args.capacity_reference == "window" else None

    def _run():
        results = {}
        for model in args.models:
            cfg = {
                "seeds": args.seeds, "model": model,
                "nominals": [0.90, 0.95], "bandwidth": 0.3,
                "split_fracs": [0.6, 0.2, 0.2],
                "min_target_cells": 10,
                "checkpoint_dir": "checkpoints",
                "target_subsets": NON_LI,
                "manifest": ("dataset_manifest_chemistry.json" if window is None
                             else "dataset_manifest_chemistry_window.json"),
                "data": {"type": "batterylife", "root": str(root),
                         "n_cycles": 100, "ref_cycle": 9,
                         "subsets": LI_ION + NON_LI,
                         **({} if window is None else {"capacity_reference": window})},
            }
            cfg_path = Path(f"/tmp/final_chem_{model}.yaml")
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            out = f"{args.out_root}_{model}"
            print(f"\n=== {model} -> {out} ===")
            rc = subprocess.call([sys.executable, "-m", "src.run_lodo",
                                  "--config", str(cfg_path), "--out", out])
            if rc != 0:
                raise RuntimeError(f"run_lodo failed for model={model} (rc={rc})")
            results[model] = out
        return results

    if args.dry_run:
        from src.test_lock import assert_test_unused
        assert_test_unused(args.tag)
        print("\n[dry-run] lock is free; nothing evaluated.")
        return 0

    if window is not None:
        # Window-capacity sensitivity: an authorized further scoring, not a first evaluation.
        from report.amendment11_primary import prediction_only_gate
        from src.test_lock import assert_campaign_may_start, mark_campaign_started
        if args.models != ["ridge", "tuned"] or args.seeds != [0, 1, 2, 3, 4]:
            raise SystemExit("Amendment 11 scores exactly ridge and tuned on seeds 0-4")
        marker = Path("runs") / f".campaign_{args.tag}.started"
        assert_campaign_may_start(args.tag, marker)
        print("\nThird scoring campaign on these labels (PROTOCOL.md Amendment 11). Prediction-only check first.")
        # Mark the start before the check, so a failed check or crash consumes the authorization.
        mark_campaign_started(args.tag, marker, f"{args.out_root}@{sha}")
        prediction_only_gate()
    res = evaluate_final_test(args.tag, run_id=f"{args.out_root}@{sha}", fn=_run)
    if window is not None:
        from src.test_lock import append_ledger
        append_ledger("campaign_completed", args.tag, amendment="Amendment 11", run_id=f"{args.out_root}@{sha}")
    print(f"\ntest '{args.tag}' is now LOCKED. Re-running is refused.")
    print("Record the outcome in notes/research-log.md, including that this was the "
          + ("third scoring campaign." if window is not None else "single permitted evaluation."))
    for m, o in res.items():
        print(f"  {m}: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
