"""Run provenance: commit, seed, environment, config, dataset manifest and file hashes."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    """Return the current commit SHA, or 'no-git' outside a repository.

    The SHA is suffixed '-dirty' if there are any staged, unstaged or untracked changes.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha + ("-dirty" if status else "")
    except Exception:
        return "no-git"


def write_provenance(run_dir: Path, config: dict, seed: int,
                     manifest_path: str | None = None) -> str:
    """Write provenance.json to run_dir and return the git SHA.

    If manifest_path exists, the dataset manifest's hash, version and sample count
    are embedded in the record.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }
    if manifest_path and Path(manifest_path).exists():
        m = json.loads(Path(manifest_path).read_text())
        prov["dataset_manifest"] = {"path": str(manifest_path), "sha256": m.get("sha256"),
                                    "version": m.get("version"),
                                    "n_samples": m.get("n_samples")}
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return sha
