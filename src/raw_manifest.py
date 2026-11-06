"""Pin the raw BatteryLife files read directly by analyses that bypass the feature cache.

Records a SHA-256 digest per raw pickle and label file in raw_input_manifest.json.

    python -m src.raw_manifest build   # write raw_input_manifest.json (hashes ~32 GB)
    python -m src.raw_manifest verify  # full re-hash against the pinned digests

`check_sizes` is a cheap guard on the file set and byte sizes; `verify` re-hashes
every file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SUBSETS = ["CALCE", "HNEI", "HUST", "MATR", "MICH", "MICH_EXP", "RWTH", "SNL", "Tongji",
           "UL_PUR", "XJTU", "NA-ion", "ZN-coin", "CALB"]
MANIFEST = Path(__file__).resolve().parents[1] / "raw_input_manifest.json"


def _files(root: Path) -> list[Path]:
    out = []
    for sub in SUBSETS:
        out += sorted((root / sub).glob("*.pkl"))
    out += sorted((root / "Life labels").glob("*_labels.json"))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def build(root: Path) -> dict:
    files = {str(p.relative_to(root)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
             for p in _files(root)}
    body = json.dumps(files, sort_keys=True)
    manifest = {"root_name": root.name, "n_files": len(files),
                "digest": hashlib.sha256(body.encode()).hexdigest(), "files": files}
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _load() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} missing: run `python -m src.raw_manifest build`")
    m = json.loads(MANIFEST.read_text())
    # Recompute the aggregate digest so an edited entry cannot keep a stale identifier.
    body = json.dumps(m["files"], sort_keys=True)
    if hashlib.sha256(body.encode()).hexdigest() != m["digest"]:
        raise SystemExit(f"{MANIFEST.name}: aggregate digest does not match its file entries")
    return m


def check_sizes(root: Path) -> str:
    """Check that the file set and byte sizes match; return the manifest digest."""
    m = _load()
    have = {str(p.relative_to(root)): p.stat().st_size for p in _files(root)}
    want = {k: v["bytes"] for k, v in m["files"].items()}
    if have != want:
        missing = sorted(set(want) - set(have))[:5]
        extra = sorted(set(have) - set(want))[:5]
        changed = sorted(k for k in set(have) & set(want) if have[k] != want[k])[:5]
        raise SystemExit(f"raw inputs differ from {MANIFEST.name}: missing {missing}, "
                         f"extra {extra}, size changed {changed}")
    return m["digest"]


def verify(root: Path) -> str:
    m = _load()
    check_sizes(root)
    bad = [k for k, v in m["files"].items() if _sha256(root / k) != v["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} raw files changed, e.g. {bad[:5]}")
    return m["digest"]


if __name__ == "__main__":
    from src.paths import resolve
    prov = Path("runs/final_chemistry_v4_ridge/provenance.json")
    root = resolve(json.loads(prov.read_text())["config"]["data"]["root"])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "build":
        m = build(root)
        print(f"pinned {m['n_files']} files, digest {m['digest']}")
    else:
        print(f"raw inputs verified, digest {verify(root)}")
