#!/usr/bin/env python
"""Run batch_04_stage0_shard00_only against one committed stage-0 shard.

This is the fourth real-OpenML batch in the doctoral pipeline. It is
the first run that combines all three pillars introduced in earlier
commits:

- the **result handoff protocol** (Commit 35) — execution copies of
  the committed shard live under
  ``runs/cc18/batch_04_stage0_shard00_only_latest/`` and never write
  back to ``jobs/``; published summary lives under
  ``experiments/_stage_runs/``;
- the **OpenML loader / cache** (Commit 34) — payloads stay
  gitignored under ``data/source/openml_cc18/<task>/``;
- the **heavy-task runtime guardrails** (Commit 38) — every row is
  classified by lane via
  ``src/doe_xgb/runtime_guardrails.py`` before dispatch. Extreme
  tasks (`letter`, `Devnagari-Script`) are deferred by default;
  heavy tasks run at a tighter ``max_evaluations`` cap.

Pre-flight refusals
-------------------
- ``experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json``
  must exist, be green (216 / 216 success, 0 failed / pending),
  report ``source_shards_unchanged: true``,
  ``stage3_signoff_present: false``, and be ≤ 7 days old by default
  (the operator may pass ``--max-age-days N`` when the dedicated
  Mac's package set has not drifted);
- ``stage3_signoff.json`` must NOT exist;
- ``--shard`` must resolve under ``jobs/doctoral/openml_cc18/shards/``
  (we copy it; never open the source for write);
- the only allowed source is shard 00 unless ``--shard`` overrides
  explicitly.

Row classification (in order)
-----------------------------
1. Task lane is ``extreme`` and ``--include-extreme-tasks`` is NOT
   set → status ``skipped``, ``last_error='deferred_extreme_lane'``.
2. Method is not in the canary set (4 smoke-ready adapters) →
   status ``skipped``,
   ``last_error='refused_not_in_canary_set'``.
3. Otherwise → dispatched through the existing ``cc18_runner.py``
   subprocess in lane-respecting groups (heavy uses
   ``gate_max_evaluations=3``, standard uses
   ``gate_max_evaluations=5``). A subprocess timeout converts any
   still-in-flight cells into ``status='failed'`` with
   ``last_error='failed_timeout'`` per
   ``runtime_guardrails.yaml``'s
   ``disposition_on_timeout: failed_timeout``.

What the runner does NOT do
---------------------------
- run any shard besides ``shard_00.sqlite``;
- run full stage 0;
- create ``stage3_signoff.json``;
- commit raw OpenML payloads (gitignored under
  ``data/source/openml_cc18/``);
- commit execution SQLite files (gitignored under ``runs/``);
- mutate the committed source shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from doe_xgb._versions import collect_package_versions  # noqa: E402
from doe_xgb.runtime_guardrails import RuntimeGuardrails  # noqa: E402

DEFAULT_SHARD = (
    REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"
)
DEFAULT_SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
DEFAULT_BATCH03_SUMMARY = (
    REPO / "experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json"
)
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/batch_04_stage0_shard00_only"
DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_OPENML_CACHE_ROOT = REPO / "data/source/openml_cc18"
DEFAULT_POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
DEFAULT_GUARDRAILS_YAML = (
    REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
)
RUNNER = REPO / "scripts/cc18_runner.py"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
CANARY_STAGE = "stage0_replica_001"
RUN_ID = "batch_04_stage0_shard00_only_latest"
BATCH_ID = "batch_04_stage0_shard00_only"
EXPECTED_BATCH03_CELLS = 216
BATCH_03_MAX_AGE_DAYS = 7
DEFAULT_REQUESTED_MAX_EVALUATIONS = 5
DEFAULT_N_FOLDS = 2

HIDE_WORKER_PREFIX = "__batch04_hidden_lane__"


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when the batch_03 pre-flight checks reject the run."""


def _summary_age_days(summary_path: Path) -> float:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ts = payload.get("exported_at") or payload.get("run_timestamp") or ""
    try:
        run_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise GateRefusalError(
            f"batch_03 summary has unparsable timestamp={ts!r}: {exc}"
        ) from exc
    return (datetime.now(timezone.utc) - run_dt).total_seconds() / 86400.0


def verify_batch03_summary(
    summary_path: Path = DEFAULT_BATCH03_SUMMARY,
    *, max_age_days: float = BATCH_03_MAX_AGE_DAYS,
) -> dict:
    if not summary_path.exists():
        raise GateRefusalError(
            f"batch_03 latest summary not found at {summary_path}; "
            "run scripts/run_batch_03_cc18_representative_18_tasks.py first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = int(payload.get("n_cells_expected", 0))
    success = int(payload.get("n_cells_success", 0))
    failed = int(payload.get("n_cells_failed", 0))
    pending = int(payload.get("n_cells_pending", 0))
    unchanged = bool(payload.get("source_shards_unchanged", False))
    signoff = bool(payload.get("stage3_signoff_present", False))
    if (
        expected != EXPECTED_BATCH03_CELLS
        or success != EXPECTED_BATCH03_CELLS
        or failed != 0
        or pending != 0
    ):
        raise GateRefusalError(
            f"batch_03 summary is not green: expected={expected} "
            f"success={success} failed={failed} pending={pending}"
        )
    if not unchanged:
        raise GateRefusalError(
            "batch_03 summary reports source_shards_unchanged=False; "
            "investigate before running batch_04."
        )
    if signoff:
        raise GateRefusalError(
            "batch_03 summary reports stage3_signoff_present=True; "
            "refusing to run batch_04 in pre-stage-0 territory."
        )
    age = _summary_age_days(summary_path)
    if age > float(max_age_days):
        raise GateRefusalError(
            f"batch_03 summary is {age:.2f} days old "
            f"(>{max_age_days:.0f}d); re-run batch_03 or pass "
            "--max-age-days to override (only safe when the worker "
            "machine's package set has not drifted)."
        )
    return {
        "n_cells_expected": expected,
        "n_cells_success": success,
        "n_cells_failed": failed,
        "n_cells_pending": pending,
        "source_shards_unchanged": unchanged,
        "stage3_signoff_present": signoff,
        "exported_at": payload.get("exported_at"),
        "age_days": age,
        "source_git_sha": payload.get("source_git_sha"),
        "run_id": payload.get("run_id"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _platform() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "uname": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "machine": platform.machine(),
    }


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _capability_audit_summary() -> dict:
    sys.path.insert(0, str(REPO / "scripts"))
    from audit_method_capabilities import audit  # type: ignore

    matrix = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"
    rep = audit(matrix)
    return {
        "n_benchmarked": rep["n_benchmarked"],
        "smoke_ready": rep["smoke_ready"],
        "dispatch_only": rep["dispatch_only"],
        "stub_only": rep["stub_only"],
        "missing_packages": rep["missing_packages_overall"],
    }


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------


def classify_rows(
    rows: list[tuple],
    guardrails: RuntimeGuardrails,
    *, include_extreme: bool,
) -> dict[str, list[dict]]:
    """Split the shard rows into four disjoint buckets.

    rows: iterable of (job_id, openml_task_id, method, algorithm).
    Returns a dict with keys:
      ``deferred``                — extreme-lane rows skipped now;
      ``refused_not_in_canary_set`` — non-canary methods;
      ``runnable_standard``      — canary methods × standard tasks;
      ``runnable_heavy``         — canary methods × heavy tasks.
    """
    buckets: dict[str, list[dict]] = {
        "deferred": [],
        "refused_not_in_canary_set": [],
        "runnable_standard": [],
        "runnable_heavy": [],
    }
    for job_id, task_id, method, algorithm in rows:
        lane = guardrails.get_task_lane(task_id)
        entry = {
            "job_id": job_id, "openml_task_id": int(task_id),
            "method": method, "algorithm": algorithm, "lane": lane,
        }
        if guardrails.should_defer_task(
            task_id, include_extreme=include_extreme,
        ):
            buckets["deferred"].append(entry)
            continue
        if method not in CANARY_METHODS:
            buckets["refused_not_in_canary_set"].append(entry)
            continue
        if lane == "heavy":
            buckets["runnable_heavy"].append(entry)
        else:
            buckets["runnable_standard"].append(entry)
    return buckets


def _set_row_status(
    cx: sqlite3.Connection, job_ids: list[str], *,
    status: str, last_error: str | None = None,
    assigned_worker: str | None = None,
) -> None:
    if not job_ids:
        return
    placeholders = ",".join("?" * len(job_ids))
    if last_error is not None:
        cx.execute(
            f"UPDATE cc18_jobs SET status=?, last_error=?, "
            f"assigned_worker=?, finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            f"WHERE job_id IN ({placeholders})",
            (status, last_error[:500], assigned_worker, *job_ids),
        )
    else:
        cx.execute(
            f"UPDATE cc18_jobs SET status=?, assigned_worker=? "
            f"WHERE job_id IN ({placeholders})",
            (status, assigned_worker, *job_ids),
        )
    cx.commit()


# ---------------------------------------------------------------------------
# Lane execution
# ---------------------------------------------------------------------------


def _invoke_runner_for_group(
    *,
    exec_shard: Path,
    out_root: Path,
    openml_cache_root: Path,
    max_evaluations: int,
    n_folds: int,
    max_jobs: int,
    timeout_seconds: float,
) -> dict:
    cmd = [
        sys.executable, str(RUNNER),
        "--shard", str(exec_shard),
        "--canary-only", "--train",
        "--max-evaluations", str(int(max_evaluations)),
        "--n-folds", str(int(n_folds)),
        "--max-jobs", str(int(max_jobs)),
        "--output-root", str(out_root),
        "--openml-cache-root", str(openml_cache_root),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            timeout=max(1.0, float(timeout_seconds)),
        )
        return {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "timed_out": False,
            "runtime_seconds": time.perf_counter() - t0,
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -9,
            "stdout_tail": (exc.stdout or b"").decode("utf-8", errors="replace")[-2000:]
                if isinstance(exc.stdout, bytes) else (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
                if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-2000:],
            "timed_out": True,
            "runtime_seconds": time.perf_counter() - t0,
            "cmd": cmd,
        }


def _resolve_group_timeout(
    *, lane_timeout: float, n_cells: int, hard_cap_hours: float,
) -> float:
    """Group-level subprocess timeout. We give each cell its lane
    timeout, but cap the total at ``hard_cap_hours`` so a true hang
    surfaces within a sane window."""
    total = float(lane_timeout) * max(1, int(n_cells))
    return min(total, float(hard_cap_hours) * 3600.0)


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_batch_04(
    *,
    shard: Path = DEFAULT_SHARD,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    batch03_summary: Path = DEFAULT_BATCH03_SUMMARY,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    max_age_days: float = BATCH_03_MAX_AGE_DAYS,
    requested_max_evaluations: int = DEFAULT_REQUESTED_MAX_EVALUATIONS,
    n_folds: int = DEFAULT_N_FOLDS,
    include_extreme_tasks: bool = False,
    run_id: str = RUN_ID,
    hard_cap_hours_per_group: float = 6.0,
    skip_train: bool = False,
    force_run_dir: bool = True,
) -> dict:
    """Run the batch_04 canary on shard_00 only and return the
    summary dict written to disk."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    batch03_gate = verify_batch03_summary(
        batch03_summary, max_age_days=max_age_days,
    )

    if SIGNOFF_FILE.exists():
        raise GateRefusalError(
            f"refusing to run batch_04: stage-3 sign-off file already "
            f"exists at {SIGNOFF_FILE}"
        )

    shard = shard.resolve()
    if not shard.exists():
        raise GateRefusalError(f"shard not found: {shard}")
    try:
        shard.relative_to((REPO / "jobs").resolve())
    except ValueError as exc:
        raise GateRefusalError(
            f"refusing: --shard must live under jobs/; got {shard}"
        ) from exc

    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )
    policy_version = _sha256(policy_csv)

    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_runs_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Materialize the run dir, copying ONLY shard_00.
    #    ``shards_root`` is the parent of the stage directory because
    #    create_run_dir composes ``shards_root / stage`` to find
    #    sources (matches the batch_02 / batch_03 convention).
    create_run_dir(
        run_id=run_id,
        stage=CANARY_STAGE,
        shard_files=[shard.name],
        run_root=run_root,
        shards_root=shards_dir.parent,
        force=force_run_dir,
    )
    run_dir = run_root / run_id
    exec_dir = run_dir / "shards" / CANARY_STAGE
    exec_shards = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_shards) == 1, exec_shards
    exec_shard = exec_shards[0]

    # 2. Classify every row in the execution shard.
    cx = sqlite3.connect(exec_shard)
    rows = list(cx.execute(
        "SELECT job_id, openml_task_id, method, algorithm "
        "FROM cc18_jobs ORDER BY job_id",
    ))
    cx.close()
    n_jobs_total = len(rows)
    buckets = classify_rows(
        rows, guardrails, include_extreme=include_extreme_tasks,
    )

    # 3. Mark deferred + refused rows in the execution SQLite.
    cx = sqlite3.connect(exec_shard)
    try:
        _set_row_status(
            cx, [e["job_id"] for e in buckets["deferred"]],
            status="skipped", last_error="deferred_extreme_lane",
            assigned_worker="batch_04_policy",
        )
        _set_row_status(
            cx,
            [e["job_id"] for e in buckets["refused_not_in_canary_set"]],
            status="skipped", last_error="refused_not_in_canary_set",
            assigned_worker="batch_04_policy",
        )
    finally:
        cx.close()

    # 4. Per-lane execution. For each lane group, temporarily hide the
    #    other lane's runnable rows (mark as 'claimed' with a sentinel
    #    worker) so cc18_runner only picks the current lane.
    md5_before = _md5(shard)
    runner_invocations: list[dict] = []
    if not skip_train:
        lane_order = ("heavy", "standard")  # heavier lane first
        for lane_name in lane_order:
            current = buckets[f"runnable_{lane_name}"]
            other_lane = "standard" if lane_name == "heavy" else "heavy"
            other = buckets[f"runnable_{other_lane}"]
            if not current:
                continue

            lane_spec = guardrails.get_lane_spec(lane_name)
            eff_max = max(1, int(min(
                requested_max_evaluations, lane_spec.gate_max_evaluations,
            )))
            group_timeout = _resolve_group_timeout(
                lane_timeout=lane_spec.timeout_seconds_per_cell,
                n_cells=len(current),
                hard_cap_hours=hard_cap_hours_per_group,
            )
            hidden_ids = [e["job_id"] for e in other]
            hide_worker = f"{HIDE_WORKER_PREFIX}{other_lane}"

            # Hide ONLY rows currently in 'pending' status. Rows that
            # completed (success/failed/skipped) in a previous lane
            # pass must not be reverted to 'claimed' or we lose their
            # terminal status when we restore them.
            cx = sqlite3.connect(exec_shard)
            try:
                if hidden_ids:
                    placeholders = ",".join("?" * len(hidden_ids))
                    cx.execute(
                        f"UPDATE cc18_jobs SET status='claimed', "
                        f"assigned_worker=? "
                        f"WHERE job_id IN ({placeholders}) "
                        f"AND status='pending'",
                        (hide_worker, *hidden_ids),
                    )
                    cx.commit()
            finally:
                cx.close()

            res = _invoke_runner_for_group(
                exec_shard=exec_shard,
                out_root=out_root,
                openml_cache_root=openml_cache_root,
                max_evaluations=eff_max,
                n_folds=n_folds,
                max_jobs=len(current),
                timeout_seconds=group_timeout,
            )
            runner_invocations.append({
                "lane": lane_name,
                "n_cells_in_group": len(current),
                "max_evaluations": eff_max,
                "timeout_seconds_for_group": group_timeout,
                **{k: v for k, v in res.items() if k != "cmd"},
            })

            # Restore the rows we actually hid (status='claimed' with
            # our sentinel worker) back to 'pending'. We match by
            # worker only so we never touch rows that completed in
            # the previous lane pass.
            cx = sqlite3.connect(exec_shard)
            try:
                cx.execute(
                    "UPDATE cc18_jobs SET status='pending', "
                    "assigned_worker=NULL "
                    "WHERE assigned_worker=? AND status='claimed'",
                    (hide_worker,),
                )
                cx.commit()
                # If the runner subprocess timed out, anything still
                # 'pending' or 'running' / 'claimed' for the current
                # lane is converted to failed_timeout.
                if res["timed_out"]:
                    in_flight_ids = [e["job_id"] for e in current]
                    placeholders = ",".join("?" * len(in_flight_ids))
                    cx.execute(
                        f"UPDATE cc18_jobs SET status='failed', "
                        f"last_error=? WHERE status IN "
                        f"('pending', 'running', 'claimed') "
                        f"AND job_id IN ({placeholders})",
                        ("failed_timeout", *in_flight_ids),
                    )
                    cx.commit()
            finally:
                cx.close()

    md5_after_run = _md5(shard)

    # 5. Collect cell statuses + per-cell aggregate metrics.
    cells = _collect_cells(exec_shard, out_root)
    status_counts = Counter(c["status"] for c in cells)
    last_error_counts = Counter(
        (c.get("last_error") or "") for c in cells if c.get("last_error")
    )
    runtimes = [
        float(c["runtime_seconds"]) for c in cells
        if c.get("runtime_seconds") is not None
    ]
    slowest = sorted(
        (c for c in cells if c.get("runtime_seconds") is not None),
        key=lambda c: float(c["runtime_seconds"]), reverse=True,
    )[:8]

    deferred_task_ids = sorted({
        e["openml_task_id"] for e in buckets["deferred"]
    })
    non_canary_methods = sorted({
        e["method"] for e in buckets["refused_not_in_canary_set"]
    })
    task_lane_counts = Counter()
    for tid in {int(c["openml_task_id"]) for c in cells}:
        task_lane_counts[guardrails.get_task_lane(tid)] += 1

    n_executed = int(status_counts.get("success", 0)) + sum(
        1 for c in cells
        if c["status"] == "failed"
        and c.get("last_error") != "failed_timeout"
    )
    n_failed_timeout = int(last_error_counts.get("failed_timeout", 0))
    n_deferred = int(last_error_counts.get("deferred_extreme_lane", 0))
    n_refused = int(last_error_counts.get("refused_not_in_canary_set", 0))
    n_jobs_pending_after = int(status_counts.get("pending", 0))
    n_jobs_failed = int(status_counts.get("failed", 0))

    # 6. Publish a stage-run summary via the protocol exporter.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID,
    )

    # 7. Augment the summary with batch_04-specific blocks.
    summary.update({
        "batch_id": BATCH_ID,
        "source_shard": _safe_rel(shard),
        "execution_shard": _safe_rel(exec_shard),
        "policy_version": policy_version,
        "policy_csv_path": _safe_rel(policy_csv),
        "guardrails_yaml_path": _safe_rel(guardrails_yaml),
        "include_extreme_tasks": bool(include_extreme_tasks),
        "n_jobs_total_in_shard": int(n_jobs_total),
        "n_jobs_executed": int(n_executed),
        "n_jobs_deferred": int(n_deferred),
        "n_jobs_refused": int(n_refused),
        "n_jobs_failed_timeout": int(n_failed_timeout),
        "n_jobs_failed_other": int(n_jobs_failed - n_failed_timeout),
        "n_jobs_pending_after": int(n_jobs_pending_after),
        "n_jobs_runnable_standard": len(buckets["runnable_standard"]),
        "n_jobs_runnable_heavy": len(buckets["runnable_heavy"]),
        "n_jobs_runnable_extreme_deferred": len(buckets["deferred"]),
        "status_counts_extended": {
            "success": int(status_counts.get("success", 0)),
            "failed": int(status_counts.get("failed", 0)),
            "pending": int(status_counts.get("pending", 0)),
            "running": int(status_counts.get("running", 0)),
            "claimed": int(status_counts.get("claimed", 0)),
            "skipped": int(status_counts.get("skipped", 0)),
            "deferred_extreme_lane": int(
                last_error_counts.get("deferred_extreme_lane", 0),
            ),
            "refused_not_in_canary_set": int(
                last_error_counts.get("refused_not_in_canary_set", 0),
            ),
            "failed_timeout": n_failed_timeout,
        },
        "task_lane_counts_in_shard": dict(task_lane_counts),
        "deferred_extreme_tasks": deferred_task_ids,
        "non_canary_methods_refused": non_canary_methods,
        "cells": cells,
        "slowest_cells": [
            {
                "openml_task_id": c["openml_task_id"],
                "method": c["method"], "algorithm": c["algorithm"],
                "runtime_seconds": float(c["runtime_seconds"]),
                "lane": guardrails.get_task_lane(c["openml_task_id"]),
            }
            for c in slowest
        ],
        "runtime_seconds_runner_total": float(sum(runtimes)),
        "runner_invocations": runner_invocations,
        "openml_cache_root": _safe_rel(openml_cache_root),
        "openml_payloads_committed": False,
        "execution_shards_committed": False,
        "batch_03_gate": batch03_gate,
        "source_shard_md5_before": {shard.name: md5_before},
        "source_shard_md5_after": {shard.name: md5_after_run},
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "capability_audit": _capability_audit_summary(),
        "run_dir": _safe_rel(run_dir),
        "stage": CANARY_STAGE,
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
    })

    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    _augment_md(summary_md, summary)
    return summary


def _collect_cells(exec_shard: Path, out_root: Path) -> list[dict]:
    cx = sqlite3.connect(exec_shard)
    try:
        rows = list(cx.execute(
            "SELECT job_id, openml_task_id, method, algorithm, status, "
            "runtime_seconds, last_error FROM cc18_jobs "
            "ORDER BY openml_task_id, method, algorithm"
        ))
    finally:
        cx.close()
    cells: list[dict] = []
    seen_manifests: set[Path] = set()
    for job_id, tid, method, algorithm, status, rt, err in rows:
        manifest_path = None
        agg: dict | None = None
        metric_keys: list[str] = []
        for mf in out_root.rglob("manifest.json"):
            if mf in seen_manifests:
                continue
            try:
                payload = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                payload.get("method_id") == method
                and payload.get("algorithm") == algorithm
                and int(payload.get("openml_task_id", -1)) == int(tid)
            ):
                manifest_path = _safe_rel(mf)
                agg = payload.get("aggregate_metrics") or {}
                metric_keys = sorted(agg.keys())
                seen_manifests.add(mf)
                break
        cells.append({
            "job_id": str(job_id),
            "openml_task_id": int(tid),
            "method": method,
            "algorithm": algorithm,
            "status": status,
            "runtime_seconds": rt,
            "last_error": err,
            "manifest": manifest_path,
            "aggregate_metrics": agg,
            "metric_keys": metric_keys,
        })
    return cells


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _augment_md(md_path: Path, summary: dict) -> None:
    extra: list[str] = []
    extra.append("\n---\n")
    extra.append("## batch_04-specific summary\n")
    extra.append(f"- batch_id: `{summary['batch_id']}`")
    extra.append(f"- source_shard: `{summary['source_shard']}`")
    extra.append(f"- execution_shard: `{summary['execution_shard']}` (gitignored)")
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(f"- policy_version: `{str(summary['policy_version'])[:16]}`")
    extra.append(
        f"- include_extreme_tasks: {summary['include_extreme_tasks']}"
    )
    extra.append(
        f"- n_jobs_total_in_shard: {summary['n_jobs_total_in_shard']}"
    )
    extra.append(
        f"- runnable: standard={summary['n_jobs_runnable_standard']}, "
        f"heavy={summary['n_jobs_runnable_heavy']}, "
        f"deferred (extreme)={summary['n_jobs_runnable_extreme_deferred']}"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"deferred: **{summary['n_jobs_deferred']}**, "
        f"refused: **{summary['n_jobs_refused']}**, "
        f"failed_timeout: **{summary['n_jobs_failed_timeout']}**, "
        f"failed (other): **{summary['n_jobs_failed_other']}**, "
        f"pending_after: {summary['n_jobs_pending_after']}"
    )
    extra.append(
        f"- runtime (runner only): "
        f"{summary['runtime_seconds_runner_total']:.1f} s\n"
    )

    extra.append(f"- task_lane_counts_in_shard: {dict(summary['task_lane_counts_in_shard'])}")
    extra.append(
        f"- deferred_extreme_tasks: {summary['deferred_extreme_tasks']}"
    )
    extra.append(
        f"- non_canary_methods_refused: {summary['non_canary_methods_refused']}\n"
    )

    g = summary["batch_03_gate"]
    extra.append("### batch_03 pre-flight\n")
    extra.append(f"- exported_at: `{g.get('exported_at')}`")
    extra.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
    extra.append(
        f"- success: {g.get('n_cells_success')}/"
        f"{g.get('n_cells_expected')} (failed={g.get('n_cells_failed')}, "
        f"pending={g.get('n_cells_pending', 0)})"
    )
    extra.append(
        f"- source_shards_unchanged: {g.get('source_shards_unchanged')}"
    )
    extra.append(f"- run_id: `{g.get('run_id')}`\n")

    extra.append("### Extended status counts\n")
    extra.append("| status | count |")
    extra.append("|---|---:|")
    for k, n in summary["status_counts_extended"].items():
        extra.append(f"| `{k}` | {n} |")
    extra.append("")

    extra.append("### Slowest executed cells\n")
    extra.append("| task_id | method | algorithm | lane | runtime_s |")
    extra.append("|---:|---|---|---|---:|")
    for c in summary["slowest_cells"]:
        extra.append(
            f"| {c['openml_task_id']} | `{c['method']}` | "
            f"`{c['algorithm']}` | `{c['lane']}` | "
            f"{c['runtime_seconds']:.2f} |"
        )
    extra.append("")

    if (
        summary["n_jobs_failed_timeout"] == 0
        and summary["n_jobs_failed_other"] == 0
        and summary["n_jobs_pending_after"] == 0
        and summary["source_shards_unchanged"]
        and not summary["stage3_signoff_present"]
    ):
        extra.append("### batch_04 verdict: **GATE PASS**\n")
        extra.append(
            "stage-0 standard / heavy / extreme split may now be "
            "planned per `docs/HEAVY_TASK_POLICY.md`. Do not run full "
            "stage 0 without explicit operator sign-off.\n"
        )
    else:
        extra.append("### batch_04 verdict: **GATE FAIL**\n")
        extra.append(
            "Resolve failures / timeouts, restore the committed shard, "
            "or remove the stage-3 sign-off file before attempting "
            "stage-0 lanes.\n"
        )

    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\n".join(extra),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--stage-runs-dir", type=Path, default=DEFAULT_STAGE_RUNS_DIR,
    )
    parser.add_argument(
        "--openml-cache-root", type=Path, default=DEFAULT_OPENML_CACHE_ROOT,
    )
    parser.add_argument(
        "--batch03-summary", type=Path, default=DEFAULT_BATCH03_SUMMARY,
    )
    parser.add_argument(
        "--policy-csv", type=Path, default=DEFAULT_POLICY_CSV,
    )
    parser.add_argument(
        "--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS_YAML,
    )
    parser.add_argument(
        "--max-age-days", type=float, default=BATCH_03_MAX_AGE_DAYS,
        help="Reject the batch_03 gate when older than this; the "
             "default 7d matches the suggested staleness window.",
    )
    parser.add_argument(
        "--max-evaluations", type=int,
        default=DEFAULT_REQUESTED_MAX_EVALUATIONS,
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--include-extreme-tasks", action="store_true",
        help="Opt into the extreme lane. Off by default; "
             "Devnagari-Script + letter cells stay deferred.",
    )
    parser.add_argument(
        "--hard-cap-hours-per-group", type=float, default=6.0,
        help="Per-lane subprocess timeout ceiling.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved configuration and exit.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Run pre-flight + run-dir + classification, but do NOT "
             "invoke the cc18 runner; used by the test suite.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "shard": str(args.shard),
            "run_root": str(args.run_root),
            "output_root": str(args.output_root),
            "stage_runs_dir": str(args.stage_runs_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "batch03_summary": str(args.batch03_summary),
            "policy_csv": str(args.policy_csv),
            "guardrails_yaml": str(args.guardrails_yaml),
            "max_age_days": args.max_age_days,
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
            "run_id": args.run_id,
            "include_extreme_tasks": args.include_extreme_tasks,
            "hard_cap_hours_per_group": args.hard_cap_hours_per_group,
        }, indent=2))
        return 0

    try:
        summary = run_batch_04(
            shard=args.shard,
            shards_dir=args.shards_dir,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            batch03_summary=args.batch03_summary,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            max_age_days=args.max_age_days,
            requested_max_evaluations=args.max_evaluations,
            n_folds=args.n_folds,
            include_extreme_tasks=args.include_extreme_tasks,
            run_id=args.run_id,
            hard_cap_hours_per_group=args.hard_cap_hours_per_group,
            skip_train=args.skip_train,
        )
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"executed={summary['n_jobs_executed']}  "
        f"deferred={summary['n_jobs_deferred']}  "
        f"refused={summary['n_jobs_refused']}  "
        f"failed_timeout={summary['n_jobs_failed_timeout']}  "
        f"failed_other={summary['n_jobs_failed_other']}  "
        f"pending_after={summary['n_jobs_pending_after']}"
    )
    print(f"json: {args.stage_runs_dir / (args.run_id + '_summary.json')}")
    print(f"md:   {args.stage_runs_dir / (args.run_id + '_summary.md')}")
    rc = 0 if (
        summary["n_jobs_failed_timeout"] == 0
        and summary["n_jobs_failed_other"] == 0
        and summary["n_jobs_pending_after"] == 0
        and summary["source_shards_unchanged"]
        and not summary["stage3_signoff_present"]
    ) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
