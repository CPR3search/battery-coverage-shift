"""Dataset content fingerprinting to detect drift.

The first run writes the manifest; later runs verify against it."""
import hashlib
import json
from pathlib import Path

import numpy as np


def dataset_fingerprint(X, y, subjects, name="dataset", version="") -> dict:
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(X), np.ascontiguousarray(y),
              np.ascontiguousarray(subjects)):
        h.update(a.tobytes())
        h.update(str(a.shape).encode())
    return {
        "name": name, "version": version, "sha256": h.hexdigest(),
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))), "dim": int(X.shape[1]),
    }


class DatasetDriftError(RuntimeError):
    pass


def verify_or_pin(manifest_path: str, fp: dict) -> None:
    p = Path(manifest_path)
    if not p.exists():
        p.write_text(json.dumps(fp, indent=2))
        print(f"pinned dataset manifest → {manifest_path} (sha {fp['sha256'][:12]})")
        return
    pinned = json.loads(p.read_text())
    if pinned["sha256"] != fp["sha256"]:
        raise DatasetDriftError(
            f"dataset drift: manifest sha {pinned['sha256'][:12]} != current "
            f"{fp['sha256'][:12]}. The data changed since pinning. If intentional, "
            f"delete/update {manifest_path} and note it in the research log."
        )
    print(f"dataset verified against manifest (sha {fp['sha256'][:12]})")
