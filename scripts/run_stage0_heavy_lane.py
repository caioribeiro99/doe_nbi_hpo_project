#!/usr/bin/env python
"""Run the heavy-lane pass of stage 0 across all 10 committed shards.

Second of the three planned stage-0 passes (Commit 40 → standard;
**Commit 41 → heavy** [this script]; later → extreme with explicit
``--include-extreme-tasks``). Combines:

- the **result handoff protocol** (Commit 35) — execution copies
  of every committed stage-0 shard live under
  ``runs/cc18/stage0_heavy_lane_latest/`` and never write back to
  ``jobs/``;
- the **OpenML loader / cache** (Commit 34) — payloads stay
  gitignored under ``data/source/openml_cc18/``;
- the **heavy-task runtime guardrails** (Commit 38) — every row
  is classified by lane via
  ``src/doe_xgb/runtime_guardrails.py`` before dispatch.

Per-row dispatch (this pass owns the heavy lane only):
- ``lane == 'standard'`` → skipped with
  ``last_error='deferred_standard_lane'`` (already handled by the
  Commit 40 standard-lane pass);
- ``lane == 'extreme'`` → skipped with
  ``last_error='deferred_extreme_lane'`` (require explicit
  ``--include-extreme-tasks`` in a future commit);
- ``lane == 'heavy'`` and method NOT in canary set → skipped
  with ``last_error='refused_not_in_canary_set'``;
- ``lane == 'heavy'`` and method IS canary → executed at the
  heavy lane's stage-0 budget
  (``stage0_max_evaluations`` from
  ``runtime_guardrails.yaml`` = 5 by default) with the heavy
  per-cell timeout (7,200 s).

Pre-flight refusals
-------------------
- ``experiments/_stage_runs/stage0_standard_lane_latest_summary.json``
  must exist, report ``n_success >= 684`` (or
  ``n_jobs_executed >= 684``), ``n_failed == 0``,
  ``n_pending == 0``, ``n_running == 0``,
  ``source_shards_unchanged: true``,
  ``stage3_signoff_present: false``, and be ≤ 7 days old by
  default (``--max-age-days`` overrides);
- ``stage3_signoff.json`` must NOT exist.

Policy version pinning
----------------------
``isolet`` (task 3481) was observed in Commit 40 standard-lane
running at 1,078 s — close to the 900 s heavy-threshold. The
classifier would promote it on a re-build of
``heavy_task_policy.csv``, but Commit 41 deliberately uses the
**same policy_version** as Commit 40
(``47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36``)
so the standard and heavy passes belong to the same replica.
isolet stays in the standard lane for this replica. Future
recalibration would re-run the builder and bump the policy
version; do that between replicas, not mid-replica.

What the runner does NOT do
---------------------------
- run the standard lane (Commit 40 already did);
- run the extreme lane (deferred indefinitely without explicit
  opt-in);
- run anything outside the canary set;
- promote ``isolet`` (task 3481) or any other standard task to
  heavy mid-replica;
- regenerate ``heavy_task_policy.csv`` or any of the policy
  artifacts;
- create ``stage3_signoff.json``;
- commit raw OpenML payloads or execution SQLite files;
- mutate any committed source shard.
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
from collections import Counter, defaultdict
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

DEFAULT_SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
DEFAULT_STANDARD_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs/stage0_standard_lane_latest_summary.json"
)
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/stage0_heavy_lane"
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
RUN_ID = "stage0_heavy_lane_latest"
BATCH_ID = "stage0_heavy_lane"
LANE = "heavy"
EXPECTED_HEAVY_CANARY_CELLS = 156  # 13 heavy tasks * 4 canary * 3 algos
STANDARD_LANE_MAX_AGE_DAYS = 7
DEFAULT_REQUESTED_MAX_EVALUATIONS = 5
DEFAULT_N_FOLDS = 2

# Pinned in this commit. Same value the Commit 40 standard-lane pass
# recorded. The runner refuses to proceed if the live policy CSV
# hashes differently, because that would indicate someone rebuilt
# the policy mid-replica (e.g. promoted isolet) and the two passes
# would belong to different policy versions.
PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
ISOLET_NOTE = (
    "isolet (task 3481) was observed in Commit 40 standard-lane "
    "at 1078.6 s. It remains in the standard lane under this "
    "policy_version. Future recalibration may promote it to heavy "
    "via the observed-runtime>=900 rule; do that between replicas, "
    "not mid-replica."
)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when the standard-lane pre-flight checks reject the run."""


def _summary_age_days(summary_path: Path) -> float:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ts = payload.get("exported_at") or payload.get("run_timestamp") or ""
    try:
        run_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise GateRefusalError(
            f"standard-lane summary has unparsable timestamp={ts!r}: {exc}"
        ) from exc
    return (datetime.now(timezone.utc) - run_dt).total_seconds() / 86400.0


def verify_standard_lane_summary(
    summary_path: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    *, max_age_days: float = STANDARD_LANE_MAX_AGE_DAYS,
    expected_executed: int = 684,
) -> dict:
    """Refuse stage 0 heavy lane if the standard-lane summary is
    missing, failed, has unfinished work, or is stale."""
    if not summary_path.exists():
        raise GateRefusalError(
            f"stage0 standard-lane summary not found at {summary_path}; "
            "run scripts/run_stage0_standard_lane.py first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    # Accept either the stage0-style fields (n_jobs_executed,
    # n_jobs_failed) or the protocol-level fields (n_success,
    # n_failed). Both should be consistent.
    n_executed = int(payload.get(
        "n_jobs_executed", payload.get("n_success", 0),
    ))
    n_failed = int(payload.get(
        "n_jobs_failed", payload.get("n_failed", 0),
    ))
    failed_timeout = int(payload.get("n_jobs_failed_timeout", 0))
    failed_other = int(payload.get("n_jobs_failed_other", 0))
    pending = int(payload.get(
        "n_jobs_pending_after", payload.get("n_pending", 0),
    ))
    running = int(payload.get(
        "n_jobs_running_after", payload.get("n_running", 0),
    ))
    unchanged = bool(payload.get("source_shards_unchanged", False))
    signoff = bool(payload.get("stage3_signoff_present", False))
    heavy_deferred = int(payload.get("n_jobs_deferred_heavy", -1))

    if n_executed < expected_executed:
        raise GateRefusalError(
            f"stage0 standard-lane summary executed only {n_executed} "
            f"of the expected {expected_executed} cells; refusing to "
            "promote to the heavy lane."
        )
    if n_failed != 0 or failed_timeout != 0 or failed_other != 0:
        raise GateRefusalError(
            f"stage0 standard-lane summary is not green: "
            f"n_failed={n_failed}, failed_timeout={failed_timeout}, "
            f"failed_other={failed_other}"
        )
    if pending != 0 or running != 0:
        raise GateRefusalError(
            f"stage0 standard-lane has unfinished work: "
            f"pending={pending}, running={running}"
        )
    if not unchanged:
        raise GateRefusalError(
            "stage0 standard-lane summary reports "
            "source_shards_unchanged=False; investigate before "
            "running the heavy lane."
        )
    if signoff:
        raise GateRefusalError(
            "stage0 standard-lane summary reports "
            "stage3_signoff_present=True; refusing to run heavy lane "
            "in pre-signoff territory."
        )
    if heavy_deferred not in (-1, 423):
        # Soft anchor: if standard-lane deferred a different number of
        # heavy rows from the universe Commit 40 saw, the policy may
        # have shifted between runs.
        raise GateRefusalError(
            f"stage0 standard-lane summary reports "
            f"n_jobs_deferred_heavy={heavy_deferred}; expected 423. "
            "Heavy lane refuses to proceed against a drifted policy."
        )
    age = _summary_age_days(summary_path)
    if age > float(max_age_days):
        raise GateRefusalError(
            f"stage0 standard-lane summary is {age:.2f} days old "
            f"(>{max_age_days:.0f}d); re-run or pass --max-age-days."
        )
    return {
        "n_jobs_executed": n_executed,
        "n_jobs_failed": n_failed,
        "n_jobs_failed_timeout": failed_timeout,
        "n_jobs_failed_other": failed_other,
        "n_jobs_pending_after": pending,
        "n_jobs_running_after": running,
        "n_jobs_deferred_heavy": heavy_deferred,
        "source_shards_unchanged": unchanged,
        "stage3_signoff_present": signoff,
        "exported_at": payload.get("exported_at"),
        "age_days": age,
        "source_git_sha": payload.get("source_git_sha"),
        "run_id": payload.get("run_id"),
        "policy_version_standard_lane": payload.get("policy_version"),
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
) -> dict[str, list[dict]]:
    """Split shard rows into four disjoint buckets.

    rows: (job_id, openml_task_id, method, algorithm).
    Returns dict keys:
      ``runnable_heavy``           — canary methods × heavy tasks;
      ``deferred_standard_lane``   — any row whose task is standard
                                     (already handled by Commit 40);
      ``deferred_extreme_lane``    — any row whose task is extreme;
      ``refused_not_in_canary_set`` — heavy task × non-canary
                                     method.
    """
    buckets: dict[str, list[dict]] = {
        "runnable_heavy": [],
        "deferred_standard_lane": [],
        "deferred_extreme_lane": [],
        "refused_not_in_canary_set": [],
    }
    for job_id, task_id, method, algorithm in rows:
        lane = guardrails.get_task_lane(task_id)
        entry = {
            "job_id": job_id, "openml_task_id": int(task_id),
            "method": method, "algorithm": algorithm, "lane": lane,
        }
        if lane == "standard":
            buckets["deferred_standard_lane"].append(entry)
            continue
        if lane == "extreme":
            buckets["deferred_extreme_lane"].append(entry)
            continue
        if method not in CANARY_METHODS:
            buckets["refused_not_in_canary_set"].append(entry)
            continue
        buckets["runnable_heavy"].append(entry)
    return buckets


def _set_status_for_job_ids(
    cx: sqlite3.Connection, job_ids: list[str], *,
    status: str, last_error: str, assigned_worker: str,
) -> None:
    if not job_ids:
        return
    placeholders = ",".join("?" * len(job_ids))
    cx.execute(
        f"UPDATE cc18_jobs SET status=?, last_error=?, "
        f"assigned_worker=?, finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
        f"WHERE job_id IN ({placeholders})",
        (status, last_error[:500], assigned_worker, *job_ids),
    )
    cx.commit()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _invoke_runner_for_shard(
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


# ---------------------------------------------------------------------------
# Cell collection
# ---------------------------------------------------------------------------


def _collect_cells(exec_dir: Path, out_root: Path) -> list[dict]:
    cells: list[dict] = []
    seen_manifests: set[Path] = set()
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(exec_p)
        try:
            rows = list(cx.execute(
                "SELECT job_id, openml_task_id, method, algorithm, status, "
                "runtime_seconds, last_error FROM cc18_jobs "
                "ORDER BY openml_task_id, method, algorithm"
            ))
        finally:
            cx.close()
        shard_name = exec_p.name
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
                "shard": shard_name,
            })
    return cells


# ---------------------------------------------------------------------------
# Pre-run plan
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shards: list[Path], guardrails: RuntimeGuardrails,
) -> dict:
    """Inventory all shards under the policy without mutating anything."""
    plan = {
        "n_source_shards": len(shards),
        "n_jobs_total": 0,
        "n_runnable_heavy": 0,
        "n_deferred_standard_lane": 0,
        "n_deferred_extreme_lane": 0,
        "n_refused_not_in_canary_set": 0,
        "task_lane_counts_universe": Counter(),
        "method_counts": Counter(),
        "algorithm_counts": Counter(),
        "heavy_tasks_executed": set(),
        "standard_tasks_deferred": set(),
        "extreme_tasks_deferred": set(),
        "non_canary_methods_refused": set(),
        "per_shard": [],
    }
    seen_tasks: dict[int, str] = {}
    for sh in shards:
        cx = sqlite3.connect(f"file:{sh}?mode=ro", uri=True)
        try:
            rows = list(cx.execute(
                "SELECT openml_task_id, method, algorithm "
                "FROM cc18_jobs WHERE stage=? AND replica=1",
                (CANARY_STAGE,),
            ))
        finally:
            cx.close()
        buckets = classify_rows(
            [("", tid, m, a) for tid, m, a in rows], guardrails,
        )
        per_sh = {
            "shard": sh.name,
            "n_jobs": len(rows),
            "runnable_heavy": len(buckets["runnable_heavy"]),
            "deferred_standard_lane": len(buckets["deferred_standard_lane"]),
            "deferred_extreme_lane": len(buckets["deferred_extreme_lane"]),
            "refused_not_in_canary_set": len(
                buckets["refused_not_in_canary_set"],
            ),
        }
        plan["per_shard"].append(per_sh)
        plan["n_jobs_total"] += len(rows)
        plan["n_runnable_heavy"] += per_sh["runnable_heavy"]
        plan["n_deferred_standard_lane"] += per_sh["deferred_standard_lane"]
        plan["n_deferred_extreme_lane"] += per_sh["deferred_extreme_lane"]
        plan["n_refused_not_in_canary_set"] += per_sh["refused_not_in_canary_set"]
        for tid, m, a in rows:
            tid_int = int(tid)
            lane = guardrails.get_task_lane(tid_int)
            seen_tasks[tid_int] = lane
            plan["method_counts"][m] += 1
            plan["algorithm_counts"][a] += 1
            if lane == "heavy" and m in CANARY_METHODS:
                plan["heavy_tasks_executed"].add(tid_int)
            elif lane == "standard":
                plan["standard_tasks_deferred"].add(tid_int)
            elif lane == "extreme":
                plan["extreme_tasks_deferred"].add(tid_int)
            if m not in CANARY_METHODS and lane == "heavy":
                plan["non_canary_methods_refused"].add(m)
    plan["task_lane_counts_universe"] = Counter(seen_tasks.values())
    plan["heavy_tasks_executed"] = sorted(plan["heavy_tasks_executed"])
    plan["standard_tasks_deferred"] = sorted(plan["standard_tasks_deferred"])
    plan["extreme_tasks_deferred"] = sorted(plan["extreme_tasks_deferred"])
    plan["non_canary_methods_refused"] = sorted(plan["non_canary_methods_refused"])
    plan["method_counts"] = dict(plan["method_counts"])
    plan["algorithm_counts"] = dict(plan["algorithm_counts"])
    plan["task_lane_counts_universe"] = dict(plan["task_lane_counts_universe"])
    return plan


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_stage0_heavy_lane(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    max_age_days: float = STANDARD_LANE_MAX_AGE_DAYS,
    requested_max_evaluations: int = DEFAULT_REQUESTED_MAX_EVALUATIONS,
    n_folds: int = DEFAULT_N_FOLDS,
    run_id: str = RUN_ID,
    hard_cap_hours_per_shard: float = 12.0,
    skip_train: bool = False,
    force_run_dir: bool = True,
    expected_heavy_canary_cells: int = EXPECTED_HEAVY_CANARY_CELLS,
    enforce_pinned_policy_version: bool = True,
) -> dict:
    """Run the heavy-lane stage-0 pass and return the summary
    dict written to disk."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    standard_gate = verify_standard_lane_summary(
        standard_lane_summary, max_age_days=max_age_days,
    )
    if SIGNOFF_FILE.exists():
        raise GateRefusalError(
            f"refusing to run stage 0 heavy lane: stage-3 sign-off "
            f"file already exists at {SIGNOFF_FILE}"
        )

    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )
    policy_version = _sha256(policy_csv)
    if (
        enforce_pinned_policy_version
        and policy_version != PINNED_POLICY_VERSION
    ):
        raise GateRefusalError(
            f"refusing: live policy CSV hashes "
            f"{policy_version} but Commit 41 pins "
            f"{PINNED_POLICY_VERSION} (= Commit 40 standard-lane "
            "policy_version). Re-build the policy between replicas, "
            "not mid-replica."
        )
    if (
        standard_gate.get("policy_version_standard_lane")
        and standard_gate["policy_version_standard_lane"] != policy_version
    ):
        raise GateRefusalError(
            f"standard-lane summary recorded policy_version="
            f"{standard_gate['policy_version_standard_lane']} but the "
            f"live policy hashes {policy_version}; refusing to mix."
        )

    source_shards = sorted(shards_dir.glob("shard_*.sqlite"))
    if len(source_shards) != 10:
        raise GateRefusalError(
            f"expected 10 stage-0 shards under {shards_dir}, "
            f"found {len(source_shards)}"
        )

    plan = build_pre_run_plan(source_shards, guardrails)
    if plan["n_runnable_heavy"] != expected_heavy_canary_cells:
        raise GateRefusalError(
            f"pre-run plan inconsistency: expected "
            f"{expected_heavy_canary_cells} heavy-lane canary cells "
            f"but found {plan['n_runnable_heavy']}. Verify "
            "heavy_task_policy.csv classification before proceeding."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_runs_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Materialize the run dir, copying all 10 stage-0 shards.
    create_run_dir(
        run_id=run_id,
        stage=CANARY_STAGE,
        shard_files=[p.name for p in source_shards],
        run_root=run_root,
        shards_root=shards_dir.parent,
        force=force_run_dir,
    )
    run_dir = run_root / run_id
    exec_dir = run_dir / "shards" / CANARY_STAGE
    exec_shards = sorted(exec_dir.glob("*.execution.sqlite"))
    if len(exec_shards) != 10:
        raise GateRefusalError(
            f"run dir contains {len(exec_shards)} execution shards; "
            "expected 10"
        )

    # 2. Per-shard: classify, pre-mark deferred + refused, invoke
    #    cc18_runner with the heavy-lane stage-0 budget.
    md5_before = {sh.name: _md5(sh) for sh in source_shards}
    runner_invocations: list[dict] = []
    cells_total_before_run = 0
    cells_runnable_per_shard: dict[str, int] = {}
    for exec_p in exec_shards:
        cx = sqlite3.connect(exec_p)
        try:
            rows = list(cx.execute(
                "SELECT job_id, openml_task_id, method, algorithm "
                "FROM cc18_jobs ORDER BY job_id",
            ))
        finally:
            cx.close()
        cells_total_before_run += len(rows)
        buckets = classify_rows(rows, guardrails)
        cells_runnable_per_shard[exec_p.name] = len(buckets["runnable_heavy"])

        # Pre-mark deferred (standard + extreme) and refused rows.
        cx = sqlite3.connect(exec_p)
        try:
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_standard_lane"]],
                status="skipped", last_error="deferred_standard_lane",
                assigned_worker="stage0_heavy_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_extreme_lane"]],
                status="skipped", last_error="deferred_extreme_lane",
                assigned_worker="stage0_heavy_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["refused_not_in_canary_set"]],
                status="skipped", last_error="refused_not_in_canary_set",
                assigned_worker="stage0_heavy_lane_policy",
            )
        finally:
            cx.close()

        if skip_train or not buckets["runnable_heavy"]:
            continue

        lane_spec = guardrails.get_lane_spec(LANE)
        # stage-0 uses stage0_max_evaluations (5 by default for heavy),
        # NOT gate_max_evaluations.
        eff_max = max(1, int(min(
            requested_max_evaluations, lane_spec.stage0_max_evaluations,
        )))
        n_runnable = len(buckets["runnable_heavy"])
        timeout_s = min(
            lane_spec.timeout_seconds_per_cell * n_runnable * 1.5,
            hard_cap_hours_per_shard * 3600.0,
        )
        res = _invoke_runner_for_shard(
            exec_shard=exec_p,
            out_root=out_root,
            openml_cache_root=openml_cache_root,
            max_evaluations=eff_max,
            n_folds=n_folds,
            max_jobs=n_runnable,
            timeout_seconds=timeout_s,
        )
        runner_invocations.append({
            "shard": exec_p.name,
            "n_runnable": n_runnable,
            "max_evaluations": eff_max,
            "timeout_seconds_for_shard": timeout_s,
            **{k: v for k, v in res.items() if k != "cmd"},
        })

        if res["timed_out"]:
            in_flight_ids = [e["job_id"] for e in buckets["runnable_heavy"]]
            cx = sqlite3.connect(exec_p)
            try:
                placeholders = ",".join("?" * len(in_flight_ids))
                cx.execute(
                    f"UPDATE cc18_jobs SET status='failed', last_error=? "
                    f"WHERE status IN ('pending', 'running', 'claimed') "
                    f"AND job_id IN ({placeholders})",
                    ("failed_timeout", *in_flight_ids),
                )
                cx.commit()
            finally:
                cx.close()

    md5_after_run = {sh.name: _md5(sh) for sh in source_shards}

    # 3. Collect statuses + aggregate metrics.
    cells = _collect_cells(exec_dir, out_root)
    status_counts = Counter(c["status"] for c in cells)
    last_error_counts = Counter(
        (c.get("last_error") or "") for c in cells if c.get("last_error")
    )
    n_success = int(status_counts.get("success", 0))
    n_failed = int(status_counts.get("failed", 0))
    n_failed_timeout = int(last_error_counts.get("failed_timeout", 0))
    n_failed_other = n_failed - n_failed_timeout
    n_pending = int(status_counts.get("pending", 0))
    n_running = int(status_counts.get("running", 0))
    n_skipped = int(status_counts.get("skipped", 0))
    n_deferred_standard = int(
        last_error_counts.get("deferred_standard_lane", 0),
    )
    n_deferred_extreme = int(
        last_error_counts.get("deferred_extreme_lane", 0),
    )
    n_refused_non_canary = int(
        last_error_counts.get("refused_not_in_canary_set", 0),
    )

    runtimes = [
        float(c["runtime_seconds"]) for c in cells
        if c.get("runtime_seconds") is not None
    ]
    slowest = sorted(
        (c for c in cells if c.get("runtime_seconds") is not None),
        key=lambda c: float(c["runtime_seconds"]), reverse=True,
    )[:12]

    per_shard_status: list[dict] = []
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_shard[c["shard"]].append(c)
    for shard_name in sorted(by_shard.keys()):
        sh_cells = by_shard[shard_name]
        sc = Counter(c["status"] for c in sh_cells)
        lec = Counter(
            (c.get("last_error") or "") for c in sh_cells
            if c.get("last_error")
        )
        per_shard_status.append({
            "shard": shard_name,
            "n_total": len(sh_cells),
            "success": int(sc.get("success", 0)),
            "failed": int(sc.get("failed", 0)),
            "failed_timeout": int(lec.get("failed_timeout", 0)),
            "pending": int(sc.get("pending", 0)),
            "running": int(sc.get("running", 0)),
            "skipped": int(sc.get("skipped", 0)),
            "deferred_standard_lane": int(
                lec.get("deferred_standard_lane", 0),
            ),
            "deferred_extreme_lane": int(lec.get("deferred_extreme_lane", 0)),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
        })

    heavy_tasks_executed = sorted({
        c["openml_task_id"] for c in cells
        if c["status"] == "success"
    })

    # 4. Publish the stage-run summary via the protocol exporter.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID,
    )

    summary.update({
        "batch_id": BATCH_ID,
        "lane": LANE,
        "stage": CANARY_STAGE,
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "execution_shards": [_safe_rel(p) for p in exec_shards],
        "policy_version": policy_version,
        "policy_version_pinned": PINNED_POLICY_VERSION,
        "policy_csv_path": _safe_rel(policy_csv),
        "guardrails_yaml_path": _safe_rel(guardrails_yaml),
        "n_jobs_total": int(cells_total_before_run),
        "n_jobs_executed": n_success + n_failed_other,
        "n_jobs_deferred_standard": n_deferred_standard,
        "n_jobs_deferred_extreme": n_deferred_extreme,
        "n_jobs_refused_non_canary": n_refused_non_canary,
        "n_jobs_failed": n_failed,
        "n_jobs_failed_timeout": n_failed_timeout,
        "n_jobs_failed_other": n_failed_other,
        "n_jobs_pending_after": n_pending,
        "n_jobs_running_after": n_running,
        "status_counts_extended": {
            "success": n_success,
            "failed": n_failed,
            "failed_timeout": n_failed_timeout,
            "pending": n_pending,
            "running": n_running,
            "claimed": int(status_counts.get("claimed", 0)),
            "skipped": n_skipped,
            "deferred_standard_lane": n_deferred_standard,
            "deferred_extreme_lane": n_deferred_extreme,
            "refused_not_in_canary_set": n_refused_non_canary,
        },
        "task_lane_counts_universe": plan["task_lane_counts_universe"],
        "heavy_tasks_executed": heavy_tasks_executed,
        "standard_tasks_deferred": plan["standard_tasks_deferred"],
        "extreme_tasks_deferred": plan["extreme_tasks_deferred"],
        "non_canary_methods_refused": plan["non_canary_methods_refused"],
        "expected_heavy_canary_cells": expected_heavy_canary_cells,
        "per_shard_status": per_shard_status,
        "cells_runnable_per_shard": cells_runnable_per_shard,
        "method_counts_universe": plan["method_counts"],
        "algorithm_counts_universe": plan["algorithm_counts"],
        "slowest_cells": [
            {
                "openml_task_id": c["openml_task_id"],
                "method": c["method"], "algorithm": c["algorithm"],
                "shard": c["shard"],
                "runtime_seconds": float(c["runtime_seconds"]),
                "lane": guardrails.get_task_lane(c["openml_task_id"]),
            }
            for c in slowest
        ],
        "cells": cells,
        "runtime_seconds_runner_total": float(sum(runtimes)),
        "runner_invocations": runner_invocations,
        "openml_cache_root": _safe_rel(openml_cache_root),
        "openml_payloads_committed": False,
        "execution_shards_committed": False,
        "standard_lane_gate": standard_gate,
        "source_shard_md5_before": md5_before,
        "source_shard_md5_after": md5_after_run,
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "capability_audit": _capability_audit_summary(),
        "run_dir": _safe_rel(run_dir),
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "isolet_recalibration_note": ISOLET_NOTE,
    })

    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    _augment_md(summary_md, summary)
    return summary


def _augment_md(md_path: Path, summary: dict) -> None:
    extra: list[str] = []
    extra.append("\n---\n")
    extra.append("## stage0 heavy-lane summary\n")
    extra.append(f"- batch_id: `{summary['batch_id']}`")
    extra.append(f"- lane: `{summary['lane']}`")
    extra.append(f"- n_source_shards: {summary['n_source_shards']}")
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(
        f"- policy_version: `{str(summary['policy_version'])[:16]}` "
        f"(pinned from Commit 40)"
    )
    extra.append(
        f"- n_jobs_total (across shards): {summary['n_jobs_total']}"
    )
    extra.append(
        f"- expected heavy-lane canary cells: "
        f"{summary['expected_heavy_canary_cells']}"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"deferred_standard: **{summary['n_jobs_deferred_standard']}**, "
        f"deferred_extreme: **{summary['n_jobs_deferred_extreme']}**, "
        f"refused_non_canary: **{summary['n_jobs_refused_non_canary']}**, "
        f"failed_timeout: **{summary['n_jobs_failed_timeout']}**, "
        f"failed_other: **{summary['n_jobs_failed_other']}**, "
        f"pending_after: {summary['n_jobs_pending_after']}"
    )
    extra.append(
        f"- runtime (runner only): "
        f"{summary['runtime_seconds_runner_total']:.1f} s\n"
    )
    extra.append(
        f"- task_lane_counts_universe: "
        f"{dict(summary['task_lane_counts_universe'])}"
    )
    extra.append(
        f"- non_canary_methods_refused: "
        f"{summary['non_canary_methods_refused']}\n"
    )

    g = summary["standard_lane_gate"]
    extra.append("### stage0 standard-lane pre-flight\n")
    extra.append(f"- exported_at: `{g.get('exported_at')}`")
    extra.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
    extra.append(
        f"- n_executed={g.get('n_jobs_executed')}, "
        f"failed={g.get('n_jobs_failed')}, "
        f"pending={g.get('n_jobs_pending_after')}"
    )
    extra.append(
        f"- source_shards_unchanged: {g.get('source_shards_unchanged')}"
    )
    extra.append(f"- run_id: `{g.get('run_id')}`")
    extra.append(
        f"- policy_version (standard-lane): "
        f"`{str(g.get('policy_version_standard_lane'))[:16]}`\n"
    )

    extra.append("### Per-shard status\n")
    extra.append(
        "| shard | total | success | failed | failed_to | pending | "
        "skipped | def_std | def_extreme | refused |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_status"]:
        extra.append(
            f"| `{sh['shard']}` | {sh['n_total']} | {sh['success']} | "
            f"{sh['failed']} | {sh['failed_timeout']} | {sh['pending']} | "
            f"{sh['skipped']} | {sh['deferred_standard_lane']} | "
            f"{sh['deferred_extreme_lane']} | "
            f"{sh['refused_not_in_canary_set']} |"
        )
    extra.append("")

    extra.append("### Heavy tasks executed\n")
    extra.append(
        f"{len(summary['heavy_tasks_executed'])} tasks: "
        f"{summary['heavy_tasks_executed']}\n"
    )

    extra.append("### Standard tasks deferred (Commit 40 already ran them)\n")
    extra.append(
        f"{len(summary['standard_tasks_deferred'])} tasks. (See "
        f"stage0_standard_lane_latest_summary.md for the executed list.)\n"
    )

    extra.append("### Extreme tasks deferred (require explicit opt-in)\n")
    extra.append(
        f"{len(summary['extreme_tasks_deferred'])} tasks: "
        f"{summary['extreme_tasks_deferred']}\n"
    )

    extra.append("### Slowest executed cells\n")
    extra.append("| task_id | method | algorithm | shard | runtime_s |")
    extra.append("|---:|---|---|---|---:|")
    for c in summary["slowest_cells"]:
        extra.append(
            f"| {c['openml_task_id']} | `{c['method']}` | "
            f"`{c['algorithm']}` | `{c['shard']}` | "
            f"{c['runtime_seconds']:.2f} |"
        )
    extra.append("")

    extra.append("### isolet (task 3481) recalibration note\n")
    extra.append(summary["isolet_recalibration_note"])
    extra.append("")

    if (
        summary["n_jobs_failed_timeout"] == 0
        and summary["n_jobs_failed_other"] == 0
        and summary["n_jobs_pending_after"] == 0
        and summary["source_shards_unchanged"]
        and not summary["stage3_signoff_present"]
    ):
        extra.append("### stage0 heavy-lane verdict: **GATE PASS**\n")
        extra.append(
            "The extreme-lane pass remains gated behind explicit "
            "`--include-extreme-tasks` and operator review of "
            "`docs/HEAVY_TASK_POLICY.md`. Do NOT run the extreme "
            "lane without a planning step that anticipates "
            "Devnagari-Script runtime.\n"
        )
    else:
        extra.append("### stage0 heavy-lane verdict: **GATE FAIL**\n")
        extra.append(
            "Resolve failures / timeouts before considering the "
            "extreme lane.\n"
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
        "--standard-lane-summary", type=Path,
        default=DEFAULT_STANDARD_LANE_SUMMARY,
    )
    parser.add_argument(
        "--policy-csv", type=Path, default=DEFAULT_POLICY_CSV,
    )
    parser.add_argument(
        "--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS_YAML,
    )
    parser.add_argument(
        "--max-age-days", type=float, default=STANDARD_LANE_MAX_AGE_DAYS,
        help="Reject the standard-lane gate when older than this; "
             "default 7d.",
    )
    parser.add_argument(
        "--max-evaluations", type=int,
        default=DEFAULT_REQUESTED_MAX_EVALUATIONS,
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--hard-cap-hours-per-shard", type=float, default=12.0,
        help="Per-shard subprocess timeout ceiling.",
    )
    parser.add_argument(
        "--allow-policy-drift", action="store_true",
        help="Skip the PINNED_POLICY_VERSION check. Off by default; "
             "set only when knowingly running a fresh-policy replica.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the pre-run plan and exit without copying shards.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Run pre-flight + run-dir + classification + summary, "
             "but do NOT invoke the cc18 runner; used by the test "
             "suite.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        from doe_xgb.runtime_guardrails import RuntimeGuardrails as RG
        g = RG.load(yaml_path=args.guardrails_yaml, csv_path=args.policy_csv)
        shards = sorted(args.shards_dir.glob("shard_*.sqlite"))
        plan = build_pre_run_plan(shards, g)
        print(json.dumps({
            "shards_dir": str(args.shards_dir),
            "run_root": str(args.run_root),
            "output_root": str(args.output_root),
            "stage_runs_dir": str(args.stage_runs_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "standard_lane_summary": str(args.standard_lane_summary),
            "policy_csv": str(args.policy_csv),
            "guardrails_yaml": str(args.guardrails_yaml),
            "policy_version_pinned": PINNED_POLICY_VERSION,
            "max_age_days": args.max_age_days,
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
            "run_id": args.run_id,
            "pre_run_plan": {
                k: v for k, v in plan.items() if k != "per_shard"
            },
            "per_shard": plan["per_shard"],
            "isolet_recalibration_note": ISOLET_NOTE,
        }, indent=2))
        return 0

    try:
        summary = run_stage0_heavy_lane(
            shards_dir=args.shards_dir,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            standard_lane_summary=args.standard_lane_summary,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            max_age_days=args.max_age_days,
            requested_max_evaluations=args.max_evaluations,
            n_folds=args.n_folds,
            run_id=args.run_id,
            hard_cap_hours_per_shard=args.hard_cap_hours_per_shard,
            enforce_pinned_policy_version=not args.allow_policy_drift,
        )
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"executed={summary['n_jobs_executed']}/"
        f"{summary['expected_heavy_canary_cells']}  "
        f"deferred_standard={summary['n_jobs_deferred_standard']}  "
        f"deferred_extreme={summary['n_jobs_deferred_extreme']}  "
        f"refused={summary['n_jobs_refused_non_canary']}  "
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
