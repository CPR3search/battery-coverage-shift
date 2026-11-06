"""Tiered downloader for the BatteryLife_Processed corpus (Hugging Face, MIT licence).

The Li-ion data are split into tiers so that smaller sub-datasets can be fetched first.
The non-Li chemistries (CALB, NA-ion, ZN-coin) form the held-out test set and are fetched
only by `--tier heldout`.

Usage:
    python -m src.download_batterylife --tier pilot
    python -m src.download_batterylife --tier pilot --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import resolve  # noqa: E402

REPO_ID = "Battery-Life/BatteryLife_Processed"
# Pinned dataset revision; src/dataset_hash.py additionally verifies the downloaded files.
REVISION = "154a4026cd188960b9bfc029a9e28ad3f1091910"
DEFAULT_DEST = resolve("BatteryLife_Processed")  # $BATTERY_ROOT/BatteryLife_Processed

# Li-ion sub-datasets, smallest-useful first. Sizes are approximate (GB).
TIER_PILOT = {  # ~21.8 GB, 10 sub-datasets
    "MATR": 8.32,      # canonical Severson fast-charge LFP set
    "HUST": 4.14,
    "Tongji": 4.17,
    "RWTH": 2.59,
    "MICH": 0.79,
    "SNL": 0.63,
    "MICH_EXP": 0.50,
    "HNEI": 0.40,
    "CALCE": 0.21,
    "UL_PUR": 0.03,
}
TIER_EXTRA = {"XJTU": 8.06, "Stanford": 4.24, "SDU": 3.96}      # +16.3 GB
TIER_LARGE = {"ISU_ILCC": 29.65, "Stanford_2": 20.36}            # +50.0 GB
TIER_HELDOUT = {"ZN-coin": 3.19, "NA-ion": 0.70, "CALB": 0.05}   # held-out test only

# Small metadata always fetched with a development tier (labels + per-set READMEs).
META_PATTERNS = ["Life labels/*", "READMEs/*", "README.md", "*_labels.json"]

TIERS = {
    "pilot": TIER_PILOT,
    "extra": TIER_EXTRA,
    "large": TIER_LARGE,
    "heldout": TIER_HELDOUT,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tier", choices=sorted(TIERS),
                     help="which tier to fetch; 'heldout' is the one-shot test set")
    src.add_argument("--subsets", nargs="+", metavar="NAME",
                     help="fetch specific sub-datasets by name, e.g. --subsets XJTU")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched, download nothing")
    args = ap.parse_args()

    known = {k: v for tier in TIERS.values() for k, v in tier.items()}
    if args.subsets:
        unknown = [s for s in args.subsets if s not in known]
        if unknown:
            ap.error(f"unknown sub-dataset(s) {unknown}; known: {sorted(known)}")
        held = [s for s in args.subsets if s in TIER_HELDOUT]
        if held:
            print(f"!! {held} belong to the FINAL one-shot chemistry test set "
                  f"(PROTOCOL.md §7).")
            print("!! Fetch only after model and calibration choices are frozen, "
                  "and log the decision.\n")
        subsets = {s: known[s] for s in args.subsets}
        args.tier = "custom-" + "-".join(args.subsets)
    else:
        subsets = TIERS[args.tier]
    approx_gb = sum(subsets.values())

    if args.tier == "heldout":
        print("!! 'heldout' is the FINAL one-shot chemistry test set (PROTOCOL.md §7).")
        print("!! Fetch it only after all model and calibration choices are frozen,")
        print("!! and record the decision in notes/research-log.md.\n")

    patterns = [f"{name}/*" for name in subsets]
    if args.tier != "heldout":
        patterns += META_PATTERNS
    # `Life labels/*` already covers the per-set label JSONs; the bare *_labels.json
    # pattern also catches the repo-root Stanford_2_labels.json.

    print(f"repo   : {REPO_ID}")
    print(f"tier   : {args.tier}  ({len(subsets)} sub-datasets, ~{approx_gb:.1f} GB)")
    print(f"dest   : {args.dest}")
    for name, gb in sorted(subsets.items(), key=lambda kv: -kv[1]):
        print(f"    {name:12s} ~{gb:6.2f} GB")

    if args.dry_run:
        print("\n[dry-run] nothing downloaded.")
        return 0

    from huggingface_hub import snapshot_download

    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(args.dest),
        allow_patterns=patterns,
        max_workers=4,
    )
    print(f"\ndone -> {path}")

    # Record what is on disk.
    manifest = args.dest / f"_downloaded_{args.tier}.json"
    got = {}
    for name in subsets:
        d = args.dest / name
        files = sorted(p.name for p in d.glob("*")) if d.is_dir() else []
        got[name] = {"n_files": len(files), "bytes": sum(
            p.stat().st_size for p in d.glob("*") if p.is_file())}
    manifest.write_text(json.dumps({"tier": args.tier, "repo": REPO_ID,
                                    "subsets": got}, indent=2))
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
