#!/usr/bin/env python
"""Local runner for the OpenML-CC18 doctoral benchmark.

Default mode is **skeleton**: open a shard, select pending jobs, resolve
the method adapter, log a dispatch decision, briefly claim+release
jobs, do not train. Promote to canary execution only when both
``--canary-only`` and ``--train`` are set.

Safety modes:
  --dry-run      open shard read-only; never modify the database;
  --no-train     (default) do not run any HPO; jobs revert to pending;
  --train        opposite of --no-train; only allowed with --canary-only;
  --canary-only  restrict execution to {default_gbdt, random_search,
                 tpe_optuna, doe_rsm_vrf_true_nbi}; non-canary methods
                 are still inspected but never run.

The runner refuses any job whose ``notes`` contains
``requires_manual_signoff_before_stage3`` unless ``--signoff-file``
points at a parseable JSON. Refusal returns the
``refused_stage3_signoff_missing`` decision and never claims the row.

Tests must always copy a shard to a tmp directory before passing it
here in non-dry-run mode; the committed shards under
``jobs/doctoral/openml_cc18/shards/`` MUST NOT be mutated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.methods import ADAPTERS, get_adapter  # noqa: E402

DEFAULT_SIGNOFF = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
SIGNOFF_NOTE = "requires_manual_signoff_before_stage3"

CANARY_METHODS: tuple[str, ...] = (
    "default_gbdt",
    "random_search",
    "tpe_optuna",
    "doe_rsm_vrf_true_nbi",
)

DEFAULT_CANARY_OUTPUT_ROOT = REPO / "experiments/_canary_runs"

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
# DB helpers
# ---------------------------------------------------------------------------


def open_shard(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
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
    """Skeleton mode: claim then immediately release back to pending."""
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


def claim_for_run(cx: sqlite3.Connection, job_id: str, worker_id: str) -> None:
    cx.execute("BEGIN IMMEDIATE")
    cx.execute(
        "UPDATE cc18_jobs SET status='running', assigned_worker=?, "
        "started_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
        "WHERE job_id=? AND status='pending'",
        (worker_id, job_id),
    )
    cx.commit()


def mark_success(cx: sqlite3.Connection, job_id: str, runtime_seconds: float) -> None:
    cx.execute("BEGIN IMMEDIATE")
    cx.execute(
        "UPDATE cc18_jobs SET status='success', runtime_seconds=?, "
        "finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE job_id=?",
        (float(runtime_seconds), job_id),
    )
    cx.commit()


def mark_failed(cx: sqlite3.Connection, job_id: str, err: str) -> None:
    cx.execute("BEGIN IMMEDIATE")
    cx.execute(
        "UPDATE cc18_jobs SET status='failed', last_error=?, "
        "retry_count=retry_count+1, "
        "finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE job_id=?",
        (err[:500], job_id),
    )
    cx.commit()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_decision(row: sqlite3.Row, *, signoff_ok: bool, train: bool,
                      canary_only: bool = False) -> dict:
    method = row["method"]
    notes = row["notes"] or ""
    if SIGNOFF_NOTE in notes and not signoff_ok:
        return {
            "job_id": row["job_id"], "method": method,
            "stage": row["stage"], "decision": "refused_stage3_signoff_missing",
            "would_run": False,
        }
    if method not in ADAPTERS:
        return {
            "job_id": row["job_id"], "method": method,
            "stage": row["stage"], "decision": "no_adapter", "would_run": False,
        }
    adapter = get_adapter(method)
    if canary_only and method not in CANARY_METHODS:
        return {
            "job_id": row["job_id"], "method": method,
            "algorithm": row["algorithm"], "task_id": row["openml_task_id"],
            "stage": row["stage"], "run_status": adapter.run_status,
            "decision": "refused_not_in_canary_set", "would_run": False,
        }
    if train and adapter.run_status not in ("smoke_ready", "full_ready"):
        return {
            "job_id": row["job_id"], "method": method,
            "algorithm": row["algorithm"], "task_id": row["openml_task_id"],
            "stage": row["stage"], "run_status": adapter.run_status,
            "decision": "refused_adapter_not_smoke_ready", "would_run": False,
        }
    return {
        "job_id": row["job_id"], "method": method,
        "algorithm": row["algorithm"], "task_id": row["openml_task_id"],
        "stage": row["stage"], "run_status": adapter.run_status,
        "would_run": train and adapter.run_status in ("smoke_ready", "full_ready"),
        "decision": (
            "stub_only" if adapter.run_status == "stub_only" else
            "dispatch_only" if adapter.run_status == "dispatch_only" else
            "would_train" if train else "ready_but_no_train_flag"
        ),
    }


# ---------------------------------------------------------------------------
# Canary execution
# ---------------------------------------------------------------------------


def _execute_canary_job(row: sqlite3.Row, *, max_evaluations: int,
                        n_folds: int, output_root: Path,
                        synthetic_task: bool, seed_base: int,
                        openml_cache_root: Path | None = None) -> dict:
    """Run one canary job. Caller is responsible for DB status
    transitions; this returns a manifest dict on success."""
    from doe_xgb.methods.canary import (
        MethodRunContext,
        TaskData,
        make_synthetic_binary_task,
    )

    method = row["method"]
    if synthetic_task:
        task = make_synthetic_binary_task(
            n_samples=300, n_features=6,
            seed=int(seed_base) + int(row["openml_task_id"]),
        )
        dataset_name = row["dataset_name"] or "synthetic"
    else:
        # Real CC18 task: load via the gitignored on-disk cache.
        from doe_xgb.datasets.openml_cc18_loader import load_cc18_task

        payload = load_cc18_task(
            int(row["openml_task_id"]),
            cache_root=openml_cache_root,
        )
        task = TaskData(
            X=payload.X,
            y=payload.y,
            task_type=payload.task_type,
            n_classes=payload.n_classes,
            feature_names=payload.feature_names,
        )
        dataset_name = payload.dataset_name
    ctx = MethodRunContext(
        method_id=method,
        algorithm=row["algorithm"],
        replica=int(row["replica"]),
        seed=int(seed_base),
        n_folds=int(n_folds),
        max_evaluations=int(max_evaluations),
        output_dir=output_root / row["job_id"],
        config_path=None,
        task_id=int(row["openml_task_id"]),
        dataset_name=dataset_name,
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(method)
    res = adapter.run(task=task, ctx=ctx)

    manifest = res.to_manifest()
    manifest["job_id"] = row["job_id"]
    manifest["openml_task_id"] = int(row["openml_task_id"])
    (ctx.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ctx.output_dir / "fold_metrics.json").write_text(
        json.dumps(res.fold_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run(*, shard: Path, max_jobs: int, dry_run: bool,
        stage: str | None, worker_id: str, no_train: bool,
        canary_only: bool, signoff_file: Path,
        max_evaluations: int, n_folds: int,
        output_root: Path, synthetic_task: bool, seed_base: int,
        openml_cache_root: Path | None = None) -> dict:
    if not no_train and not canary_only:
        raise SystemExit(
            "--train requires --canary-only. Refusing to run a "
            "non-canary method without explicit canary scope."
        )

    signoff_ok = signoff_present(signoff_file)
    cx = open_shard(shard, read_only=dry_run)
    decisions: list[dict] = []
    successes = 0
    failures = 0
    try:
        rows = select_pending(cx, max_jobs=max_jobs, stage=stage)
        for row in rows:
            d = dispatch_decision(
                row, signoff_ok=signoff_ok, train=not no_train,
                canary_only=canary_only,
            )
            decisions.append(d)
            if dry_run:
                continue
            if d["decision"] in {
                "refused_stage3_signoff_missing", "no_adapter",
                "refused_not_in_canary_set", "refused_adapter_not_smoke_ready",
            }:
                continue
            if not d["would_run"]:
                # Skeleton mode: claim + release.
                claim_and_release(cx, row["job_id"], worker_id)
                continue
            # Canary execution path.
            try:
                claim_for_run(cx, row["job_id"], worker_id)
                t0 = time.perf_counter()
                _execute_canary_job(
                    row,
                    max_evaluations=max_evaluations,
                    n_folds=n_folds,
                    output_root=output_root,
                    synthetic_task=synthetic_task,
                    seed_base=seed_base,
                    openml_cache_root=openml_cache_root,
                )
                mark_success(cx, row["job_id"], time.perf_counter() - t0)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                mark_failed(cx, row["job_id"],
                            f"{type(exc).__name__}: {exc}")
                failures += 1
                d["execution_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cx.close()
    return {
        "shard": str(shard),
        "stage_filter": stage,
        "worker_id": worker_id,
        "dry_run": dry_run,
        "no_train": no_train,
        "canary_only": canary_only,
        "synthetic_task": synthetic_task,
        "signoff_file": str(signoff_file),
        "signoff_ok": signoff_ok,
        "n_inspected": len(decisions),
        "n_success": successes,
        "n_failed": failures,
        "decisions": decisions,
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# Back-compat for the Commit 29 test that imported run_skeleton.
def run_skeleton(*, shard, max_jobs, dry_run, stage, worker_id,
                 no_train, signoff_file):
    return run(
        shard=shard, max_jobs=max_jobs, dry_run=dry_run, stage=stage,
        worker_id=worker_id, no_train=True,  # back-compat: never train
        canary_only=False, signoff_file=signoff_file,
        max_evaluations=5, n_folds=2,
        output_root=DEFAULT_CANARY_OUTPUT_ROOT, synthetic_task=False,
        seed_base=42,
    )


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
                        help="default true; do not train models")
    parser.add_argument("--train", dest="no_train", action="store_false",
                        help="opposite of --no-train; only allowed with "
                             "--canary-only")
    parser.add_argument("--canary-only", action="store_true",
                        help="restrict execution to the canary set "
                             f"{CANARY_METHODS}")
    parser.add_argument("--synthetic-task", action="store_true",
                        help="feed the canary on a synthetic binary task; "
                             "without this flag the runner loads real "
                             "OpenML CC18 task data via the gitignored "
                             "cache under data/source/openml_cc18/.")
    parser.add_argument("--max-evaluations", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_CANARY_OUTPUT_ROOT)
    parser.add_argument("--signoff-file", type=Path, default=DEFAULT_SIGNOFF)
    parser.add_argument("--openml-cache-root", type=Path, default=None,
                        help="root directory for the OpenML payload cache "
                             "(default: data/source/openml_cc18/).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run(
        shard=args.shard,
        max_jobs=args.max_jobs,
        dry_run=args.dry_run,
        stage=args.stage,
        worker_id=args.worker_id,
        no_train=args.no_train,
        canary_only=args.canary_only,
        signoff_file=args.signoff_file,
        max_evaluations=args.max_evaluations,
        n_folds=args.n_folds,
        output_root=args.output_root,
        synthetic_task=args.synthetic_task,
        seed_base=args.seed_base,
        openml_cache_root=args.openml_cache_root,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"},
                     indent=2))
    print(f"inspected {summary['n_inspected']} jobs; "
          f"success={summary['n_success']}  failed={summary['n_failed']}  "
          f"dry_run={summary['dry_run']}  signoff_ok={summary['signoff_ok']}")
    decision_counts: dict[str, int] = {}
    for d in summary["decisions"]:
        decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1
    for k, n in sorted(decision_counts.items()):
        print(f"  {k}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
