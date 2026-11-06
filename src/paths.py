"""Configurable root for large on-disk inputs and outputs.

The BatteryLife corpus and per-fold checkpoints live outside the repository under
$BATTERY_ROOT:

    BATTERY_ROOT=/path/to/battery make diag-numbers

Some recorded paths are absolute under LEGACY_ROOT; `resolve` re-roots them on read.
New paths are recorded relative to the root (`to_record`).

Expected layout under the root:

    <root>/BatteryLife_Processed/<subset>/...      the corpus
    <root>/checkpoints/<run>/<seedN_foldM_DOMAIN>/  checkpoint.joblib, predictions.npz
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "BATTERY_ROOT"
LEGACY_ROOT = Path("/media/vthuynh/Working/battery")


def battery_root() -> Path:
    """Return $BATTERY_ROOT if set, else LEGACY_ROOT."""
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else LEGACY_ROOT


def resolve(p: str | os.PathLike) -> Path:
    """Map a recorded path onto the current root.

    Relative paths are taken relative to the root. Absolute paths under LEGACY_ROOT
    are re-rooted. Any other absolute path is returned unchanged.
    """
    path = Path(p).expanduser()
    if not path.is_absolute():
        return battery_root() / path
    try:
        return battery_root() / path.relative_to(LEGACY_ROOT)
    except ValueError:
        return path


def to_record(p: str | os.PathLike) -> str:
    """Return the path relative to the root when under it, else unchanged."""
    path = Path(p)
    for base in (battery_root(), LEGACY_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)
