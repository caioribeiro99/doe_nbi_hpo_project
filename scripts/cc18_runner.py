#!/usr/bin/env python
"""Local runner skeleton for the OpenML-CC18 doctoral benchmark.

Reads a single SQLite shard, selects pending jobs, resolves the method
adapter, logs the dispatch decision, and **does not train models**.
This commit (29) intentionally stops short of calling ``adapter.run()``;
the next commit replaces the dispatch-only no-op with executable
adapters for the canary cell.

Two safety modes:
  --dry-run           open shard read-only; never modify the database;
  --no-train (def.)   may briefly claim a job, then release it back to
                      pending; no model is trained, no result is
                      persisted, no shard row is left modified.

A stage-3 sign-off guardrail refuses to claim any job whose ``notes``
contains ``requires_manual_signoff_before_stage3`` unless a sign-off
file exists at ``--signoff-file`` (default
``jobs/doctoral/openml_cc18/stage3_signoff.json``). This commit does
NOT create that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.methods import ADAPTERS, get_adapter  # noqa: E402

DEFAULT_SIGNOFF = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
SIGNOFF_NOTE = "requires_manual_signoff_before_stage3"

logger = logging.getLogger("cc18_runner")


# ---------------------------------------------------------------------------
# Sign-off
# ---------------------------------------------------------------------------


def signoff_present(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Job claim / release
# ---------------------------------------------------------------------------


def open_shard(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        # Open read-only via URI to make sure --dry-run cannot mutate the file.
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(path)


def select_pending(cx: sqlite3.Connection, *, max_jobs: int,
                   stage: str | None = None) -> list[sqlite3.Row]:
    cx.row_factory = sqlite3.Row
    sql = (
        "SELECT job_id, openml_task_id, openml_dataset_id, dataset_name, "
        "algorithm, method, replica, stage, notes "
        "FROM cc18_jobs WHERE status = 'pending'"
    )
    params: tuple = ()
    if stage:
        sql += " AND stage = ?"
        params = (stage,)
    sql += " ORDER BY job_id LIMIT ?"
    params = (*params, max_jobs)
    return list(cx.execute(sql, params).fetchall())


def claim_and_release(cx: sqlite3.Connection, job_id: str,
                      worker_id: str) -> None:
    """Mark a job ``claimed`` then revert to ``pending`` immediately so the
    on-disk row is unchanged (apart from the trigger-managed updated_at).

    The runner uses this to exercise the claim path without leaving
    state behind. We use a single transaction so a crash mid-call does
    not leak a half-claimed row.
    """
    cx.execute("BEGIN IMMEDIATE")
    try:
        cx.execute(
            "UPDATE cc18_jobs SET status='claimed', assigned_worker=? "
            "WHERE job_id=? AND status='pending'",
            (worker_id, job_id),
        )
        cx.execute(
            "UPDATE cc18_jobs SET status='pending', assigned_worker=NULL "
            "WHERE job_id=?",
            (job_id,),
        )
        cx.commit()
    except Exception:
        cx.rollback()
        raise


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_decision(row: sqlite3.Row, *, signoff_ok: bool,
                      train: bool) -> dict:
    """Compute (without executing) the dispatch decision for one row."""
    method = row["method"]
    notes = row["notes"] or ""
    if SIGNOFF_NOTE in notes and not signoff_ok:
        return {
            "job_id": row["job_id"],
            "method": method,
            "stage": row["stage"],
            "decision": "refused_stage3_signoff_missing",
            "would_run": False,
        }
    if method not in ADAPTERS:
        return {
            "job_id": row["job_id"],
            "method": method,
            "stage": row["stage"],
            "decision": "no_adapter",
            "would_run": False,
        }
    adapter = get_adapter(method)
    return {
        "job_id": row["job_id"],
        "method": method,
        "algorithm": row["algorithm"],
        "task_id": row["openml_task_id"],
        "stage": row["stage"],
        "run_status": adapter.run_status,
        "would_run": train and adapter.run_status in ("smoke_ready", "full_ready"),
        "decision": (
            "stub_only" if adapter.run_status == "stub_only" else
            "dispatch_only" if adapter.run_status == "dispatch_only" else
            "would_train" if train else "ready_but_no_train_flag"
        ),
    }


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_skeleton(*, shard: Path, max_jobs: int, dry_run: bool,
                 stage: str | None, worker_id: str, no_train: bool,
                 signoff_file: Path) -> dict:
    signoff_ok = signoff_present(signoff_file)
    cx = open_shard(shard, read_only=dry_run)
    decisions: list[dict] = []
    try:
        rows = select_pending(cx, max_jobs=max_jobs, stage=stage)
        for row in rows:
            d = dispatch_decision(row, signoff_ok=signoff_ok, train=not no_train)
            decisions.append(d)
            if dry_run:
                continue
            if d["decision"] == "refused_stage3_signoff_missing":
                continue
            if d["decision"] == "no_adapter":
                continue
            # Skeleton mode: claim + immediately release. No training.
            claim_and_release(cx, row["job_id"], worker_id)
    finally:
        cx.close()
    return {
        "shard": str(shard),
        "stage_filter": stage,
        "worker_id": worker_id,
        "dry_run": dry_run,
        "no_train": no_train,
        "signoff_file": str(signoff_file),
        "signoff_ok": signoff_ok,
        "n_inspected": len(decisions),
        "decisions": decisions,
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", default=None,
                        choices=(None, "stage0_replica_001",
                                 "stage1_topup_to_005",
                                 "stage2_topup_to_010",
                                 "stage3_topup_to_030"))
    parser.add_argument("--worker-id", default="local_dev_runner")
    parser.add_argument("--no-train", action="store_true", default=True,
                        help="(default true in Commit 29; set --train to "
                             "attempt training, but every adapter still "
                             "raises NotImplementedError)")
    parser.add_argument("--train", dest="no_train", action="store_false",
                        help="opposite of --no-train; training is currently "
                             "blocked because every adapter is stub_only or "
                             "dispatch_only")
    parser.add_argument("--signoff-file", type=Path, default=DEFAULT_SIGNOFF)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_skeleton(
        shard=args.shard,
        max_jobs=args.max_jobs,
        dry_run=args.dry_run,
        stage=args.stage,
        worker_id=args.worker_id,
        no_train=args.no_train,
        signoff_file=args.signoff_file,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"},
                     indent=2))
    print(f"inspected {summary['n_inspected']} jobs; "
          f"dry_run={summary['dry_run']}  signoff_ok={summary['signoff_ok']}")
    decision_counts: dict[str, int] = {}
    for d in summary["decisions"]:
        decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1
    for k, n in sorted(decision_counts.items()):
        print(f"  {k}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
