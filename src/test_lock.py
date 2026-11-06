"""Held-out test-set lock.

Each held-out evaluation is keyed by a tag and may run only once. A lock file under
`.locks/` blocks a second evaluation of the same tag until it is explicitly unlocked.
Every evaluation and unlock is also appended to `.locks/ledger.jsonl`, and released locks
are archived, so the ledger is the complete record of how often a held-out set was used.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(".locks")
ARCHIVE_DIR = LOCK_DIR / "archive"
LEDGER = LOCK_DIR / "ledger.jsonl"


class TestLockError(RuntimeError):
    pass


def _lock_path(tag: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    return LOCK_DIR / f"test_{safe}.lock"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_ledger(event: str, tag: str, **fields) -> None:
    """Append one record to the append-only evaluation ledger."""
    LOCK_DIR.mkdir(exist_ok=True)
    rec = {"event": event, "tag": tag, "timestamp": _now(), **fields}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(ln) for ln in LEDGER.read_text().splitlines() if ln.strip()]


def evaluation_count(tag: str) -> int:
    """Number of recorded evaluations of `tag` in the ledger."""
    return sum(1 for r in read_ledger()
               if r.get("event") == "evaluated" and r.get("tag") == tag)


def unlock(tag: str, reason: str) -> Path:
    """Release a lock: archive the lock file and record the unlock and its reason in the ledger."""
    if not reason or not reason.strip():
        raise TestLockError("an unlock requires a stated reason; it goes in the ledger")
    lp = _lock_path(tag)
    if not lp.exists():
        raise TestLockError(f"no live lock for '{tag}' to unlock")
    info = json.loads(lp.read_text())
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _now().replace(":", "").replace("-", "")
    dest = ARCHIVE_DIR / f"{lp.stem}.{stamp}.lock"
    dest.write_text(lp.read_text())
    lp.unlink()
    append_ledger("unlocked", tag, reason=reason.strip(),
                  superseded_run_id=info.get("run_id"),
                  superseded_timestamp=info.get("timestamp"),
                  archived_to=str(dest))
    return dest


def assert_test_unused(tag: str) -> None:
    lp = _lock_path(tag)
    if lp.exists():
        info = json.loads(lp.read_text())
        raise TestLockError(
            f"test set '{tag}' was already evaluated on {info['timestamp']} "
            f"(run={info.get('run_id')}). Evaluating it again is leakage. "
            f"To override deliberately: `make unlock-test TAG={tag} REASON='...'` (logged)."
        )


def mark_test_used(tag: str, run_id: str = "") -> None:
    LOCK_DIR.mkdir(exist_ok=True)
    _lock_path(tag).write_text(json.dumps(
        {"tag": tag, "run_id": run_id, "timestamp": _now()}, indent=2))
    append_ledger("evaluated", tag, run_id=run_id,
                  evaluation_number=evaluation_count(tag) + 1)


def authorize_campaign(tag: str, amendment: str, prior_tags: list[str]) -> None:
    """Record in the ledger that a further scoring of held-out labels is authorized.

    `prior_tags` lists earlier scorings of the same labels, so the ledger keeps the full count.
    """
    append_ledger("authorized", tag, amendment=amendment, prior_tags=list(prior_tags))


def assert_campaign_may_start(tag: str, started_marker: Path) -> None:
    """Raise unless the campaign is authorized and has not already started (even if it crashed)."""
    if not any(r.get("event") == "authorized" and r.get("tag") == tag for r in read_ledger()):
        raise TestLockError(f"no authorization for '{tag}' in the ledger")
    if started_marker.exists():
        raise TestLockError(f"scoring of '{tag}' already started: {started_marker.read_text().strip()}")
    assert_test_unused(tag)


def mark_campaign_started(tag: str, started_marker: Path, detail: str) -> None:
    """Create the start marker exclusively; raise if it already exists.

    The marker lives outside the tracked tree so the run's provenance is not marked dirty.
    """
    started_marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with started_marker.open("x") as fh:
            fh.write(f"{_now()} {detail}\n")
    except FileExistsError as err:
        raise TestLockError(f"scoring of '{tag}' already started: {started_marker}") from err


def evaluate_final_test(tag: str, run_id: str, fn):
    """Run `fn()` as the one-time test evaluation, then lock the tag."""
    assert_test_unused(tag)
    result = fn()
    mark_test_used(tag, run_id)
    return result
