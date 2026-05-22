#!/usr/bin/env python
"""Run the Stage-3 / top-up replica_002 extreme-lane pass across all 10 shards.

This is **Commit 51**: the extreme-lane execution that completes
replica_002 on top of the Commit 45 sign-off, the Commit 46 plan,
the Commit 48 standard execution, the Commit 49 heavy execution,
and the Commit 50 extreme-lane plan.

The script chains **five** gates before copying any shard:

1. ``jobs/doctoral/openml_cc18/stage3_signoff.json`` must be signed
   with ``signoff_type == 'stage0_replica_001'`` and the same
   ``policy_version`` as the live SHA-256 of
   ``benchmarks/doctoral/openml_cc18/heavy_task_policy.csv``;
2. ``experiments/_stage_runs/stage3_topup_plan_latest_summary.json``
   must be ``planned_not_executed`` and list ``topup_to_5``
   covering ``replica = 2``;
3. ``experiments/_stage_runs/``
   ``stage3_replica_002_standard_lane_latest_summary.json``
   (Commit 48) must be ``executed`` for replica = 2 / standard,
   with 684 / 684 success and 0 failures;
4. ``experiments/_stage_runs/``
   ``stage3_replica_002_heavy_lane_latest_summary.json``
   (Commit 49) must be ``executed`` for replica = 2 / heavy, with
   156 / 156 success and 0 failures;
5. ``experiments/_stage_runs/``
   ``stage3_replica_002_extreme_lane_plan_latest_summary.json``
   (Commit 50) must be ``planned_not_executed`` for replica = 2 /
   extreme, project 24 runnable extreme canary cells, and pin the
   same ``policy_version``.

The scope is deliberately narrow:

- **all 10** source template shards;
- one replica (``replica = 2``);
- one lane (``extreme``);
- four canary methods × three algorithms only;
- 24 executable extreme-lane canary cells total
  (2 extreme tasks × 4 canary methods × 3 algorithms).

Extreme-lane policy budget under
``benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml``:

- per-cell timeout = 14,400 s (4 h);
- ``stage0_max_evaluations`` = 1 (same as Commit 43's stage-0
  extreme run — the policy-defined budget that lets
  Devnagari-Script fit inside the worker window).

Real execution requires **both** of these CLI flags. Without them
the script refuses real execution. Tests rely on
``--skip-train`` and a non-blocking dry-run path.

What this script does NOT do
----------------------------
- run any other replica (3 / 4 / 5);
- rerun the standard lane (Commit 48 stands);
- rerun the heavy lane (Commit 49 stands);
- run non-canary methods;
- run the full ``topup_to_5`` tier (3,456 canary cells across
  replicas 2–5);
- regenerate ``heavy_task_policy.csv`` or
  ``runtime_guardrails.yaml``;
- mutate any committed source shard;
- commit raw OpenML payloads or execution SQLite files.
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
DEFAULT_SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
DEFAULT_TOPUP_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs/stage3_topup_plan_latest_summary.json"
)
DEFAULT_STANDARD_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_standard_lane_latest_summary.json"
)
DEFAULT_HEAVY_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_heavy_lane_latest_summary.json"
)
DEFAULT_EXTREME_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_extreme_lane_plan_latest_summary.json"
)
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = (
    REPO / "experiments/_batch_runs/stage3_replica_002_extreme_lane"
)
DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_OPENML_CACHE_ROOT = REPO / "data/source/openml_cc18"
DEFAULT_POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
DEFAULT_GUARDRAILS_YAML = (
    REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
)
RUNNER = REPO / "scripts/cc18_runner.py"

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
SOURCE_STAGE = "stage0_replica_001"
TARGET_STAGE_LABEL = "stage1_topup_to_005"  # Commit 47/48/49 convention
TARGET_REPLICA = 2
RUN_ID = "stage3_replica_002_extreme_lane_latest"
BATCH_ID = "stage3_replica_002_extreme_lane"
LANE = "extreme"
TOPUP_TIER = "topup_to_5_partial"
SIGNOFF_TYPE_EXPECTED = "stage0_replica_001"
PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
EXPECTED_EXTREME_CANARY_CELLS = 24  # 2 extreme tasks * 4 canary * 3 algos
EXPECTED_EXTREME_TASK_IDS = (6, 167121)  # letter, Devnagari-Script
EXPECTED_HEAVY_SUCCESS = 156
EXPECTED_STANDARD_SUCCESS = 684
N_EXPECTED_SHARDS = 10
DEFAULT_N_FOLDS = 2
DEFAULT_HARD_CAP_HOURS = 24.0  # extreme lane needs a generous shard cap


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when a pre-flight gate rejects the run."""


# ---------------------------------------------------------------------------
# Hash + platform helpers
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
# Gate verification
# ---------------------------------------------------------------------------


def verify_signoff(
    signoff_path: Path, *, expected_policy_version: str,
) -> dict:
    if not signoff_path.exists():
        raise GateRefusalError(
            f"signoff file not found at {signoff_path}; this run requires "
            "the Commit 45 signoff to exist."
        )
    try:
        record = json.loads(signoff_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{signoff_path} is not valid JSON: {exc}"
        ) from exc
    if record.get("signoff_status") != "signed":
        raise GateRefusalError(
            f"{signoff_path} has signoff_status="
            f"{record.get('signoff_status')!r}; expected 'signed'."
        )
    if record.get("signoff_type") != SIGNOFF_TYPE_EXPECTED:
        raise GateRefusalError(
            f"{signoff_path} has signoff_type="
            f"{record.get('signoff_type')!r}; expected "
            f"{SIGNOFF_TYPE_EXPECTED!r}."
        )
    if record.get("policy_version") != expected_policy_version:
        raise GateRefusalError(
            f"{signoff_path} carries policy_version="
            f"{record.get('policy_version')!r}; live policy_version="
            f"{expected_policy_version!r}. Refusing run against drifted "
            "policy."
        )
    return record


def verify_topup_plan(
    plan_path: Path, *, expected_policy_version: str,
) -> dict:
    if not plan_path.exists():
        raise GateRefusalError(
            f"stage3 top-up plan summary not found at {plan_path}."
        )
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{plan_path} is not valid JSON: {exc}"
        ) from exc
    if record.get("execution_status") != "planned_not_executed":
        raise GateRefusalError(
            f"{plan_path} has execution_status="
            f"{record.get('execution_status')!r}; expected "
            "'planned_not_executed'."
        )
    if record.get("policy_version") != expected_policy_version:
        raise GateRefusalError(
            f"{plan_path} carries policy_version="
            f"{record.get('policy_version')!r}; live policy_version="
            f"{expected_policy_version!r}. Refusing run against drifted "
            "policy."
        )
    tiers = record.get("tier_plans") or []
    t5 = next((t for t in tiers if t.get("tier") == "topup_to_5"), None)
    if t5 is None:
        raise GateRefusalError(
            f"{plan_path} does not list a 'topup_to_5' tier."
        )
    rs = int(t5.get("replica_start") or 0)
    re_ = int(t5.get("replica_end") or 0)
    if not (rs <= TARGET_REPLICA <= re_):
        raise GateRefusalError(
            f"topup_to_5 tier covers replicas {rs}..{re_}; replica="
            f"{TARGET_REPLICA} is outside that range."
        )
    return record


def _verify_executed_lane_summary(
    summary_path: Path, *,
    expected_policy_version: str, expected_lane: str,
    expected_success: int, commit_label: str,
) -> dict:
    if not summary_path.exists():
        raise GateRefusalError(
            f"{commit_label} summary not found at {summary_path}."
        )
    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{summary_path} is not valid JSON: {exc}"
        ) from exc
    if record.get("execution_status") != "executed":
        raise GateRefusalError(
            f"{summary_path} has execution_status="
            f"{record.get('execution_status')!r}; expected 'executed'."
        )
    if int(record.get("replica") or 0) != TARGET_REPLICA:
        raise GateRefusalError(
            f"{summary_path} replica={record.get('replica')!r}; expected "
            f"{TARGET_REPLICA}."
        )
    if record.get("lane") != expected_lane:
        raise GateRefusalError(
            f"{summary_path} lane={record.get('lane')!r}; expected "
            f"{expected_lane!r}."
        )
    if record.get("policy_version") != expected_policy_version:
        raise GateRefusalError(
            f"{summary_path} carries policy_version="
            f"{record.get('policy_version')!r}; live policy_version="
            f"{expected_policy_version!r}. Refusing run against drifted "
            "policy."
        )
    if int(record.get("n_jobs_success") or -1) != int(expected_success):
        raise GateRefusalError(
            f"{summary_path} n_jobs_success="
            f"{record.get('n_jobs_success')}; expected {expected_success}."
        )
    for key in (
        "n_jobs_failed", "n_jobs_failed_timeout",
        "n_jobs_pending_after", "n_jobs_running_after",
    ):
        actual = int(record.get(key) or 0)
        if actual != 0:
            raise GateRefusalError(
                f"{summary_path} {key}={actual}; expected 0."
            )
    if not bool(record.get("source_shards_unchanged", False)):
        raise GateRefusalError(
            f"{summary_path} reports source_shards_unchanged=False."
        )
    return record


def verify_standard_lane_summary(
    summary_path: Path, *, expected_policy_version: str,
    expected_success: int = EXPECTED_STANDARD_SUCCESS,
) -> dict:
    return _verify_executed_lane_summary(
        summary_path,
        expected_policy_version=expected_policy_version,
        expected_lane="standard",
        expected_success=expected_success,
        commit_label="Commit 48 standard-lane",
    )


def verify_heavy_lane_summary(
    summary_path: Path, *, expected_policy_version: str,
    expected_success: int = EXPECTED_HEAVY_SUCCESS,
) -> dict:
    return _verify_executed_lane_summary(
        summary_path,
        expected_policy_version=expected_policy_version,
        expected_lane="heavy",
        expected_success=expected_success,
        commit_label="Commit 49 heavy-lane",
    )


def verify_extreme_plan_summary(
    plan_path: Path, *, expected_policy_version: str,
    expected_canary_cells: int = EXPECTED_EXTREME_CANARY_CELLS,
) -> dict:
    if not plan_path.exists():
        raise GateRefusalError(
            f"Commit 50 extreme-lane plan summary not found at {plan_path}; "
            "run scripts/plan_stage3_replica002_extreme_lane.py first."
        )
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{plan_path} is not valid JSON: {exc}"
        ) from exc
    if record.get("execution_status") != "planned_not_executed":
        raise GateRefusalError(
            f"{plan_path} has execution_status="
            f"{record.get('execution_status')!r}; expected "
            "'planned_not_executed'."
        )
    if int(record.get("replica") or 0) != TARGET_REPLICA:
        raise GateRefusalError(
            f"{plan_path} replica={record.get('replica')!r}; expected "
            f"{TARGET_REPLICA}."
        )
    if record.get("lane") != LANE:
        raise GateRefusalError(
            f"{plan_path} lane={record.get('lane')!r}; expected "
            f"{LANE!r}."
        )
    if int(record.get("n_runnable_extreme_canary") or -1) != int(
        expected_canary_cells,
    ):
        raise GateRefusalError(
            f"{plan_path} n_runnable_extreme_canary="
            f"{record.get('n_runnable_extreme_canary')}; expected "
            f"{expected_canary_cells}."
        )
    extreme_tasks = tuple(record.get("extreme_tasks_planned") or ())
    if tuple(sorted(int(t) for t in extreme_tasks)) != tuple(
        sorted(EXPECTED_EXTREME_TASK_IDS),
    ):
        raise GateRefusalError(
            f"{plan_path} extreme_tasks_planned={extreme_tasks!r}; "
            f"expected {tuple(sorted(EXPECTED_EXTREME_TASK_IDS))!r}."
        )
    if record.get("policy_version") != expected_policy_version:
        raise GateRefusalError(
            f"{plan_path} carries policy_version="
            f"{record.get('policy_version')!r}; live policy_version="
            f"{expected_policy_version!r}. Refusing run against drifted "
            "policy."
        )
    if not bool(record.get("no_training_run_by_this_script", True)):
        raise GateRefusalError(
            f"{plan_path} no_training_run_by_this_script is False; the "
            "plan must remain planning-only."
        )
    return record


# ---------------------------------------------------------------------------
# Row classification (extreme-lane variant)
# ---------------------------------------------------------------------------


def classify_rows(
    rows: list[tuple],
    guardrails: RuntimeGuardrails,
) -> dict[str, list[dict]]:
    """Split shard rows into four disjoint extreme-lane execution buckets.

    rows: (job_id, openml_task_id, method, algorithm).
    Returns dict keys:
      ``runnable_extreme``         — canary methods × extreme tasks;
      ``deferred_standard_lane``   — any row whose task is standard
                                    (Commit 48 stands);
      ``deferred_heavy_lane``      — any row whose task is heavy
                                    (Commit 49 stands);
      ``refused_not_in_canary_set`` — extreme task × non-canary method.
    """
    buckets: dict[str, list[dict]] = {
        "runnable_extreme": [],
        "deferred_standard_lane": [],
        "deferred_heavy_lane": [],
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
        if lane == "heavy":
            buckets["deferred_heavy_lane"].append(entry)
            continue
        # lane == "extreme"
        if method in CANARY_METHODS:
            buckets["runnable_extreme"].append(entry)
        else:
            buckets["refused_not_in_canary_set"].append(entry)
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
# Execution-copy rewrite (replica + stage label)
# ---------------------------------------------------------------------------


def rewrite_execution_copy_to_replica2(
    exec_shard: Path, *, target_stage: str = TARGET_STAGE_LABEL,
    target_replica: int = TARGET_REPLICA,
) -> dict:
    """Rewrite the copied execution SQLite so every row reflects the
    Stage-3 / topup_to_5 / replica_002 identity. Never writes to the
    committed source."""
    if not exec_shard.exists():
        raise FileNotFoundError(exec_shard)
    if "runs" not in exec_shard.resolve().parts:
        raise ValueError(
            f"refusing to rewrite shard outside runs/: {exec_shard}"
        )
    cx = sqlite3.connect(exec_shard)
    try:
        n_before = cx.execute("SELECT COUNT(*) FROM cc18_jobs").fetchone()[0]
        before_replicas = {
            r[0] for r in cx.execute("SELECT DISTINCT replica FROM cc18_jobs")
        }
        before_stages = {
            r[0] for r in cx.execute("SELECT DISTINCT stage FROM cc18_jobs")
        }
        cx.execute("BEGIN IMMEDIATE")
        cx.execute(
            "UPDATE cc18_jobs SET stage=?, replica=?, "
            "status='pending', assigned_worker=NULL, last_error=NULL, "
            "started_at=NULL, finished_at=NULL, runtime_seconds=NULL, "
            "retry_count=0, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (target_stage, int(target_replica)),
        )
        cx.commit()
        n_after = cx.execute("SELECT COUNT(*) FROM cc18_jobs").fetchone()[0]
        after_replicas = {
            r[0] for r in cx.execute("SELECT DISTINCT replica FROM cc18_jobs")
        }
        after_stages = {
            r[0] for r in cx.execute("SELECT DISTINCT stage FROM cc18_jobs")
        }
    finally:
        cx.close()
    return {
        "shard": exec_shard.name,
        "n_rows_before": int(n_before),
        "n_rows_after": int(n_after),
        "source_template_replicas": sorted(before_replicas),
        "source_template_stages": sorted(before_stages),
        "execution_replicas": sorted(after_replicas),
        "execution_stages": sorted(after_stages),
        "target_stage": target_stage,
        "target_replica": int(target_replica),
    }


# ---------------------------------------------------------------------------
# Pre-run plan (extreme-lane variant, all 10 shards)
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shards: list[Path], guardrails: RuntimeGuardrails,
) -> dict:
    """Inventory the committed source shards under the policy,
    read-only, projecting extreme-lane execution buckets."""
    n_jobs_total = 0
    n_runnable_extreme = 0
    n_deferred_standard = 0
    n_deferred_heavy = 0
    n_refused_non_canary = 0
    method_counts: Counter = Counter()
    algorithm_counts: Counter = Counter()
    non_canary_methods: set[str] = set()
    extreme_tasks_executed: set[int] = set()
    standard_tasks_deferred: set[int] = set()
    heavy_tasks_deferred: set[int] = set()
    source_template_stages: set[str] = set()
    source_template_replicas: set[int] = set()
    per_shard: list[dict] = []
    seen_tasks: dict[int, str] = {}
    for sh in shards:
        cx = sqlite3.connect(f"file:{sh}?mode=ro", uri=True)
        try:
            rows = list(cx.execute(
                "SELECT openml_task_id, method, algorithm, stage, replica "
                "FROM cc18_jobs",
            ))
        finally:
            cx.close()
        buckets = classify_rows(
            [("", r[0], r[1], r[2]) for r in rows],
            guardrails,
        )
        per_sh = {
            "shard": sh.name,
            "n_jobs": len(rows),
            "runnable_extreme": len(buckets["runnable_extreme"]),
            "deferred_standard_lane": len(buckets["deferred_standard_lane"]),
            "deferred_heavy_lane": len(buckets["deferred_heavy_lane"]),
            "refused_not_in_canary_set": len(
                buckets["refused_not_in_canary_set"],
            ),
        }
        per_shard.append(per_sh)
        n_jobs_total += len(rows)
        n_runnable_extreme += per_sh["runnable_extreme"]
        n_deferred_standard += per_sh["deferred_standard_lane"]
        n_deferred_heavy += per_sh["deferred_heavy_lane"]
        n_refused_non_canary += per_sh["refused_not_in_canary_set"]
        for tid, m, a, st, rep in rows:
            tid_int = int(tid)
            lane = guardrails.get_task_lane(tid_int)
            seen_tasks[tid_int] = lane
            method_counts[m] += 1
            algorithm_counts[a] += 1
            source_template_stages.add(st)
            source_template_replicas.add(int(rep))
            if lane == "extreme" and m in CANARY_METHODS:
                extreme_tasks_executed.add(tid_int)
            elif lane == "extreme" and m not in CANARY_METHODS:
                non_canary_methods.add(m)
            elif lane == "standard":
                standard_tasks_deferred.add(tid_int)
            elif lane == "heavy":
                heavy_tasks_deferred.add(tid_int)
    return {
        "n_source_shards": len(shards),
        "n_jobs_total": n_jobs_total,
        "n_runnable_extreme": n_runnable_extreme,
        "n_deferred_standard_lane": n_deferred_standard,
        "n_deferred_heavy_lane": n_deferred_heavy,
        "n_refused_not_in_canary_set": n_refused_non_canary,
        "task_lane_counts_universe": dict(Counter(seen_tasks.values())),
        "method_counts": dict(method_counts),
        "algorithm_counts": dict(algorithm_counts),
        "non_canary_methods_refused": sorted(non_canary_methods),
        "extreme_tasks_executed": sorted(extreme_tasks_executed),
        "standard_tasks_deferred": sorted(standard_tasks_deferred),
        "heavy_tasks_deferred": sorted(heavy_tasks_deferred),
        "source_template_stages": sorted(source_template_stages),
        "source_template_replicas": sorted(source_template_replicas),
        "per_shard": per_shard,
    }


# ---------------------------------------------------------------------------
# Runner invocation
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
    stage_filter: str,
) -> dict:
    cmd = [
        sys.executable, str(RUNNER),
        "--shard", str(exec_shard),
        "--canary-only", "--train",
        "--stage", stage_filter,
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
                "ORDER BY openml_task_id, method, algorithm",
            ))
        finally:
            cx.close()
        shard_name = exec_p.name
        for job_id, tid, method, algorithm, status, rt, err in rows:
            manifest_path = None
            agg: dict | None = None
            metric_keys: list[str] = []
            if out_root.exists():
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
# Top-level run
# ---------------------------------------------------------------------------


def run_replica002_extreme_lane(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    signoff_file: Path = DEFAULT_SIGNOFF_FILE,
    topup_plan_summary: Path = DEFAULT_TOPUP_PLAN_SUMMARY,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    heavy_lane_summary: Path = DEFAULT_HEAVY_LANE_SUMMARY,
    extreme_plan_summary: Path = DEFAULT_EXTREME_PLAN_SUMMARY,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    n_folds: int = DEFAULT_N_FOLDS,
    run_id: str = RUN_ID,
    target_stage_label: str = TARGET_STAGE_LABEL,
    target_replica: int = TARGET_REPLICA,
    hard_cap_hours_per_shard: float = DEFAULT_HARD_CAP_HOURS,
    skip_train: bool = False,
    force_run_dir: bool = True,
    expected_extreme_canary_cells: int = EXPECTED_EXTREME_CANARY_CELLS,
) -> dict:
    """Run replica_002 extreme lane across all 10 shards and return
    the summary dict written to disk."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    source_shards = sorted(shards_dir.glob("shard_*.sqlite"))
    if len(source_shards) != N_EXPECTED_SHARDS:
        raise GateRefusalError(
            f"expected {N_EXPECTED_SHARDS} stage-0 shards under {shards_dir}, "
            f"found {len(source_shards)}"
        )

    # 1. Five gates BEFORE we touch anything.
    live_policy_version = _sha256(policy_csv)
    signoff_record = verify_signoff(
        signoff_file, expected_policy_version=live_policy_version,
    )
    topup_plan_record = verify_topup_plan(
        topup_plan_summary, expected_policy_version=live_policy_version,
    )
    std_record = verify_standard_lane_summary(
        standard_lane_summary, expected_policy_version=live_policy_version,
    )
    hvy_record = verify_heavy_lane_summary(
        heavy_lane_summary, expected_policy_version=live_policy_version,
    )
    plan50_record = verify_extreme_plan_summary(
        extreme_plan_summary, expected_policy_version=live_policy_version,
    )
    signoff_sha256 = _sha256(signoff_file)
    topup_plan_sha256 = _sha256(topup_plan_summary)
    std_summary_sha256 = _sha256(standard_lane_summary)
    hvy_summary_sha256 = _sha256(heavy_lane_summary)
    plan50_sha256 = _sha256(extreme_plan_summary)

    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )
    # Defensive: extreme universe must be exactly (6, 167121).
    live_extreme_tids = tuple(sorted(
        tid for tid, p in guardrails.tasks.items() if p.lane == "extreme"
    ))
    if live_extreme_tids != tuple(sorted(EXPECTED_EXTREME_TASK_IDS)):
        raise GateRefusalError(
            f"live extreme task universe is {live_extreme_tids}; expected "
            f"{tuple(sorted(EXPECTED_EXTREME_TASK_IDS))}."
        )

    # 2. Pre-run plan and refuse if expected runnable count is wrong.
    plan = build_pre_run_plan(source_shards, guardrails)
    if plan["n_runnable_extreme"] != expected_extreme_canary_cells:
        raise GateRefusalError(
            f"pre-run plan inconsistency: expected "
            f"{expected_extreme_canary_cells} extreme-lane canary cells "
            f"across all 10 shards but found {plan['n_runnable_extreme']}."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_runs_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    # 3. Materialize the run dir, copying all 10 source shards.
    md5_before = {sh.name: _md5(sh) for sh in source_shards}
    create_run_dir(
        run_id=run_id,
        stage=SOURCE_STAGE,
        shard_files=[p.name for p in source_shards],
        run_root=run_root,
        shards_root=shards_dir.parent,
        force=force_run_dir,
    )
    run_dir = run_root / run_id
    exec_dir = run_dir / "shards" / SOURCE_STAGE
    exec_shards = sorted(exec_dir.glob("*.execution.sqlite"))
    if len(exec_shards) != N_EXPECTED_SHARDS:
        raise GateRefusalError(
            f"run dir contains {len(exec_shards)} execution shards; "
            f"expected {N_EXPECTED_SHARDS}"
        )
    md5_after_copy = {sh.name: _md5(sh) for sh in source_shards}
    if md5_after_copy != md5_before:
        raise GateRefusalError(
            "source shard MD5 changed during create_run_dir; refusing run."
        )

    # 4. Rewrite every execution copy.
    rewrite_infos: list[dict] = []
    for exec_p in exec_shards:
        rewrite_infos.append(
            rewrite_execution_copy_to_replica2(
                exec_p, target_stage=target_stage_label,
                target_replica=target_replica,
            )
        )
    md5_after_rewrite = {sh.name: _md5(sh) for sh in source_shards}
    if md5_after_rewrite != md5_before:
        raise GateRefusalError(
            "source shard MD5 changed during execution-copy rewrite."
        )

    # 5. Per-shard: classify, pre-mark deferred + refused, dispatch the
    #    extreme-lane canary rows via cc18_runner at the policy budget.
    lane_spec = guardrails.get_lane_spec(LANE)
    # Extreme lane uses the policy's stage0_max_evaluations (1 by
    # default) — same budget as Commit 43's stage-0 extreme run.
    eff_max = max(1, int(lane_spec.stage0_max_evaluations))
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
        n_runnable = len(buckets["runnable_extreme"])
        cells_runnable_per_shard[exec_p.name] = n_runnable

        cx = sqlite3.connect(exec_p)
        try:
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_standard_lane"]],
                status="skipped", last_error="deferred_standard_lane",
                assigned_worker="stage3_replica_002_extreme_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_heavy_lane"]],
                status="skipped", last_error="deferred_heavy_lane",
                assigned_worker="stage3_replica_002_extreme_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["refused_not_in_canary_set"]],
                status="skipped", last_error="refused_not_in_canary_set",
                assigned_worker="stage3_replica_002_extreme_lane_policy",
            )
        finally:
            cx.close()

        if skip_train or not n_runnable:
            continue

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
            stage_filter=target_stage_label,
        )
        runner_invocations.append({
            "shard": exec_p.name,
            "n_runnable": n_runnable,
            "max_evaluations": eff_max,
            "timeout_seconds_for_shard": timeout_s,
            "timeout_seconds_per_cell": float(
                lane_spec.timeout_seconds_per_cell,
            ),
            **res,
        })

        if res["timed_out"]:
            in_flight_ids = [e["job_id"] for e in buckets["runnable_extreme"]]
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

    # 6. Cell collection + status counts.
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
    n_deferred_standard = int(last_error_counts.get("deferred_standard_lane", 0))
    n_deferred_heavy = int(last_error_counts.get("deferred_heavy_lane", 0))
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

    # Per-shard breakdown
    per_shard_status: list[dict] = []
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_shard[c["shard"]].append(c)
    for shard_name_ in sorted(by_shard.keys()):
        sh_cells = by_shard[shard_name_]
        sc = Counter(c["status"] for c in sh_cells)
        lec = Counter(
            (c.get("last_error") or "") for c in sh_cells
            if c.get("last_error")
        )
        per_shard_status.append({
            "shard": shard_name_,
            "n_total": len(sh_cells),
            "success": int(sc.get("success", 0)),
            "failed": int(sc.get("failed", 0)),
            "failed_timeout": int(lec.get("failed_timeout", 0)),
            "pending": int(sc.get("pending", 0)),
            "running": int(sc.get("running", 0)),
            "skipped": int(sc.get("skipped", 0)),
            "deferred_standard_lane": int(lec.get("deferred_standard_lane", 0)),
            "deferred_heavy_lane": int(lec.get("deferred_heavy_lane", 0)),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
        })

    extreme_tasks_executed_observed = sorted({
        c["openml_task_id"] for c in cells if c["status"] == "success"
    })

    # Per-task / per-method / per-algorithm breakdowns over executed cells.
    per_task: dict[int, dict] = defaultdict(
        lambda: {"success": 0, "failed": 0, "failed_timeout": 0, "total": 0,
                 "runtime_seconds_total": 0.0},
    )
    per_method: dict[str, dict] = defaultdict(
        lambda: {"success": 0, "failed": 0, "failed_timeout": 0, "total": 0,
                 "runtime_seconds_total": 0.0},
    )
    per_algorithm: dict[str, dict] = defaultdict(
        lambda: {"success": 0, "failed": 0, "failed_timeout": 0, "total": 0,
                 "runtime_seconds_total": 0.0},
    )
    for c in cells:
        # Only include cells in the extreme universe (runnable + refused).
        if c["openml_task_id"] not in EXPECTED_EXTREME_TASK_IDS:
            continue
        per_task[c["openml_task_id"]]["total"] += 1
        per_method[c["method"]]["total"] += 1
        per_algorithm[c["algorithm"]]["total"] += 1
        if c["status"] == "success":
            per_task[c["openml_task_id"]]["success"] += 1
            per_method[c["method"]]["success"] += 1
            per_algorithm[c["algorithm"]]["success"] += 1
            if c.get("runtime_seconds") is not None:
                rt = float(c["runtime_seconds"])
                per_task[c["openml_task_id"]]["runtime_seconds_total"] += rt
                per_method[c["method"]]["runtime_seconds_total"] += rt
                per_algorithm[c["algorithm"]]["runtime_seconds_total"] += rt
        elif c["status"] == "failed":
            if c.get("last_error") == "failed_timeout":
                per_task[c["openml_task_id"]]["failed_timeout"] += 1
                per_method[c["method"]]["failed_timeout"] += 1
                per_algorithm[c["algorithm"]]["failed_timeout"] += 1
            else:
                per_task[c["openml_task_id"]]["failed"] += 1
                per_method[c["method"]]["failed"] += 1
                per_algorithm[c["algorithm"]]["failed"] += 1

    # 7. Publish via the protocol exporter and augment.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID,
    )

    execution_sqlite_sha256 = {
        p.name: _sha256(p) for p in exec_shards
    }

    summary.update({
        "batch_id": BATCH_ID,
        "stage": target_stage_label,
        "topup_tier": TOPUP_TIER,
        "execution_status": "executed",
        "replica": int(target_replica),
        "source_template_replica": 1,
        "lane": LANE,
        "policy_version": live_policy_version,
        "policy_version_pinned": PINNED_POLICY_VERSION,
        "policy_csv_path": _safe_rel(policy_csv),
        "guardrails_yaml_path": _safe_rel(guardrails_yaml),
        "signoff_path": _safe_rel(signoff_file),
        "signoff_sha256": signoff_sha256,
        "signoff_signed_at_utc": signoff_record.get("signed_at_utc"),
        "signoff_operator_handle": signoff_record.get("operator_handle"),
        "signoff_operator_name": signoff_record.get("operator_name"),
        "signoff_type": signoff_record.get("signoff_type"),
        "signoff_status": signoff_record.get("signoff_status"),
        "stage3_topup_plan_summary_path": _safe_rel(topup_plan_summary),
        "stage3_topup_plan_summary_sha256": topup_plan_sha256,
        "stage3_topup_plan_execution_status": topup_plan_record.get(
            "execution_status",
        ),
        "commit48_standard_lane_summary_path": _safe_rel(
            standard_lane_summary,
        ),
        "commit48_standard_lane_summary_sha256": std_summary_sha256,
        "commit48_standard_lane_n_jobs_success": int(
            std_record.get("n_jobs_success") or 0,
        ),
        "commit48_standard_lane_runtime_seconds": float(
            std_record.get("runtime_seconds_runner_total") or 0.0,
        ),
        "commit49_heavy_lane_summary_path": _safe_rel(heavy_lane_summary),
        "commit49_heavy_lane_summary_sha256": hvy_summary_sha256,
        "commit49_heavy_lane_n_jobs_success": int(
            hvy_record.get("n_jobs_success") or 0,
        ),
        "commit49_heavy_lane_runtime_seconds": float(
            hvy_record.get("runtime_seconds_runner_total") or 0.0,
        ),
        "commit50_extreme_plan_summary_path": _safe_rel(extreme_plan_summary),
        "commit50_extreme_plan_summary_sha256": plan50_sha256,
        "commit50_extreme_plan_execution_status": plan50_record.get(
            "execution_status",
        ),
        "commit50_extreme_plan_n_runnable_extreme_canary": int(
            plan50_record.get("n_runnable_extreme_canary") or 0,
        ),
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "execution_shards": [_safe_rel(p) for p in exec_shards],
        "execution_sqlite_sha256": execution_sqlite_sha256,
        "n_jobs_total": int(cells_total_before_run),
        "n_jobs_executed": n_success + n_failed_other + n_failed_timeout,
        "n_jobs_success": n_success,
        "n_jobs_deferred_standard": n_deferred_standard,
        "n_jobs_deferred_heavy": n_deferred_heavy,
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
            "deferred_heavy_lane": n_deferred_heavy,
            "refused_not_in_canary_set": n_refused_non_canary,
        },
        "task_lane_counts_universe": plan["task_lane_counts_universe"],
        "extreme_tasks_executed": extreme_tasks_executed_observed,
        "extreme_tasks_in_universe": plan["extreme_tasks_executed"],
        "extreme_task_meta": {
            6: {"dataset": "letter"},
            167121: {"dataset": "Devnagari-Script"},
        },
        "standard_tasks_deferred": plan["standard_tasks_deferred"],
        "heavy_tasks_deferred": plan["heavy_tasks_deferred"],
        "non_canary_methods_refused": plan["non_canary_methods_refused"],
        "expected_extreme_canary_cells": expected_extreme_canary_cells,
        "per_task_status_breakdown": {
            str(tid): val for tid, val in sorted(per_task.items())
        },
        "per_method_status_breakdown": dict(per_method),
        "per_algorithm_status_breakdown": dict(per_algorithm),
        "per_shard_status": per_shard_status,
        "per_shard_planned": plan["per_shard"],
        "cells_runnable_per_shard": cells_runnable_per_shard,
        "method_counts_universe": plan["method_counts"],
        "algorithm_counts_universe": plan["algorithm_counts"],
        "metric_keys": sorted({k for c in cells for k in (c.get("metric_keys") or [])}),
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
        "source_shard_md5_before": md5_before,
        "source_shard_md5_after_copy": md5_after_copy,
        "source_shard_md5_after_rewrite": md5_after_rewrite,
        "source_shard_md5_after": md5_after_run,
        "execution_copy_rewrite": rewrite_infos,
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "capability_audit": _capability_audit_summary(),
        "run_dir": _safe_rel(run_dir),
        "stage3_signoff_present": signoff_file.exists(),
        "extreme_lane_max_evaluations_used": int(eff_max),
        "extreme_lane_timeout_seconds_per_cell_used": float(
            lane_spec.timeout_seconds_per_cell,
        ),
        "extreme_lane_max_evaluations_note": (
            f"Extreme lane executed with policy-defined "
            f"max_evaluations={int(eff_max)}."
        ),
        "no_other_replica_executed_by_this_script": True,
        "no_full_topup_to_5_executed_by_this_script": True,
        "no_standard_lane_rerun_by_this_script": True,
        "no_heavy_lane_rerun_by_this_script": True,
        "no_committed_shard_modified_by_this_script": (
            md5_after_run == md5_before
        ),
        "no_raw_openml_payloads_staged_by_this_script": True,
        "no_execution_sqlite_staged_by_this_script": True,
        "only_replica_002_extreme_lane_executed": True,
        "operator_review_required_before_replica002_signoff": True,
        "next_recommended_step": (
            "After Commit 51 is green and operator-reviewed, Commit 52 "
            "should aggregate / review / signoff replica_002 by chaining "
            "the four executed-lane summaries (Commit 48 standard + "
            "Commit 49 heavy + Commit 50 plan + this Commit 51 extreme). "
            "Do NOT scale to replicas 003-005 until replica_002 standard + "
            "heavy + extreme has been reviewed end-to-end."
        ),
    })

    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    _augment_md(summary_md, summary)
    return summary


# ---------------------------------------------------------------------------
# Markdown augmentation
# ---------------------------------------------------------------------------


def _augment_md(md_path: Path, summary: dict) -> None:
    extra: list[str] = []
    extra.append("\n---\n")
    extra.append("## stage 3 / replica_002 extreme-lane summary (Commit 51)\n")
    extra.append(f"- run_id: `{summary['run_id']}`")
    extra.append(f"- batch_id: `{summary['batch_id']}`")
    extra.append(f"- stage: `{summary['stage']}`")
    extra.append(f"- topup_tier: `{summary['topup_tier']}`")
    extra.append(
        f"- replica: **{summary['replica']}** (source template replica = "
        f"{summary['source_template_replica']})"
    )
    extra.append(f"- lane: `{summary['lane']}`")
    extra.append(f"- n_source_shards: {summary['n_source_shards']}")
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(f"- policy_version: `{str(summary['policy_version'])[:16]}`")
    extra.append(
        f"- policy_version_pinned: `{str(summary['policy_version_pinned'])[:16]}`"
    )
    extra.append(
        f"- signoff_sha256: `{str(summary['signoff_sha256'])[:16]}`"
    )
    extra.append(
        f"- stage3_topup_plan_summary_sha256: "
        f"`{str(summary['stage3_topup_plan_summary_sha256'])[:16]}`"
    )
    extra.append(
        f"- commit48_standard_lane_summary_sha256: "
        f"`{str(summary['commit48_standard_lane_summary_sha256'])[:16]}`"
    )
    extra.append(
        f"- commit49_heavy_lane_summary_sha256: "
        f"`{str(summary['commit49_heavy_lane_summary_sha256'])[:16]}`"
    )
    extra.append(
        f"- commit50_extreme_plan_summary_sha256: "
        f"`{str(summary['commit50_extreme_plan_summary_sha256'])[:16]}`\n"
    )

    extra.append(
        f"- n_jobs_total (across 10 shards): {summary['n_jobs_total']}"
    )
    extra.append(
        f"- expected runnable extreme-lane canary cells: "
        f"**{summary['expected_extreme_canary_cells']}**"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"success: **{summary['n_jobs_success']}**, "
        f"deferred_standard: **{summary['n_jobs_deferred_standard']}**, "
        f"deferred_heavy: **{summary['n_jobs_deferred_heavy']}**, "
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
        f"- {summary['extreme_lane_max_evaluations_note']}"
    )
    extra.append(
        f"- extreme_lane_timeout_seconds_per_cell_used: "
        f"{summary['extreme_lane_timeout_seconds_per_cell_used']:.0f} s\n"
    )

    extra.append("### Per-shard status\n")
    extra.append(
        "| shard | total | success | failed | failed_to | pending | "
        "skipped | def_std | def_heavy | refused |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_status"]:
        extra.append(
            f"| `{sh['shard']}` | {sh['n_total']} | {sh['success']} | "
            f"{sh['failed']} | {sh['failed_timeout']} | {sh['pending']} | "
            f"{sh['skipped']} | {sh['deferred_standard_lane']} | "
            f"{sh['deferred_heavy_lane']} | "
            f"{sh['refused_not_in_canary_set']} |"
        )
    extra.append("")

    extra.append("### Per-task status breakdown (extreme universe)\n")
    extra.append("| task_id | dataset | total | success | failed | failed_to | runtime_total_s |")
    extra.append("|---:|---|---:|---:|---:|---:|---:|")
    meta = summary["extreme_task_meta"]
    for tid_str, val in sorted(
        summary["per_task_status_breakdown"].items(),
        key=lambda kv: int(kv[0]),
    ):
        tid = int(tid_str)
        ds = (meta.get(tid) or meta.get(str(tid)) or {}).get("dataset", "?")
        extra.append(
            f"| {tid} | `{ds}` | {val['total']} | {val['success']} | "
            f"{val['failed']} | {val['failed_timeout']} | "
            f"{val['runtime_seconds_total']:.1f} |"
        )
    extra.append("")

    extra.append("### Per-method status breakdown (extreme universe)\n")
    extra.append("| method | total | success | failed | failed_to | runtime_total_s |")
    extra.append("|---|---:|---:|---:|---:|---:|")
    for m, val in sorted(summary["per_method_status_breakdown"].items()):
        extra.append(
            f"| `{m}` | {val['total']} | {val['success']} | "
            f"{val['failed']} | {val['failed_timeout']} | "
            f"{val['runtime_seconds_total']:.1f} |"
        )
    extra.append("")

    extra.append("### Per-algorithm status breakdown (extreme universe)\n")
    extra.append("| algorithm | total | success | failed | failed_to | runtime_total_s |")
    extra.append("|---|---:|---:|---:|---:|---:|")
    for a, val in sorted(summary["per_algorithm_status_breakdown"].items()):
        extra.append(
            f"| `{a}` | {val['total']} | {val['success']} | "
            f"{val['failed']} | {val['failed_timeout']} | "
            f"{val['runtime_seconds_total']:.1f} |"
        )
    extra.append("")

    extra.append("### Extreme tasks executed (success only)\n")
    extra.append(
        f"{len(summary['extreme_tasks_executed'])} tasks: "
        f"{summary['extreme_tasks_executed']}\n"
    )
    extra.append("### Standard tasks deferred (Commit 48 stands)\n")
    extra.append(
        f"{len(summary['standard_tasks_deferred'])} tasks deferred."
    )
    extra.append("### Heavy tasks deferred (Commit 49 stands)\n")
    extra.append(
        f"{len(summary['heavy_tasks_deferred'])} tasks deferred."
    )
    extra.append("### Non-canary methods refused\n")
    extra.append(
        f"{summary['non_canary_methods_refused']}\n"
    )

    if summary["slowest_cells"]:
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

    invariants_ok = (
        summary["n_jobs_failed_timeout"] == 0
        and summary["n_jobs_failed_other"] == 0
        and summary["n_jobs_pending_after"] == 0
        and summary["source_shards_unchanged"]
        and summary["no_committed_shard_modified_by_this_script"]
        and summary["no_full_topup_to_5_executed_by_this_script"]
        and summary["no_standard_lane_rerun_by_this_script"]
        and summary["no_heavy_lane_rerun_by_this_script"]
        and summary["no_other_replica_executed_by_this_script"]
    )
    if invariants_ok:
        extra.append(
            "### stage 3 replica_002 extreme lane verdict: "
            "**GATE PASS — operator review required**\n"
        )
        extra.append(
            "Run finished cleanly: every extreme-lane canary cell on "
            "replica_002 across all 10 shards reached a terminal status, "
            "every committed source shard is byte-identical to its "
            "pre-run MD5, standard / heavy lanes were not rerun, no "
            "other replica was executed, and the full topup_to_5 tier "
            "was not triggered. Commit 52 may aggregate / review / "
            "signoff replica_002.\n"
        )
    else:
        extra.append(
            "### stage 3 replica_002 extreme lane verdict: **NOT GREEN**\n"
        )
        extra.append(
            "Investigate failures / timeouts / source-shard drift before "
            "any aggregate replica_002 review or further Stage-3 / "
            "top-up execution.\n"
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
    parser.add_argument(
        "--signoff-file", type=Path, default=DEFAULT_SIGNOFF_FILE,
    )
    parser.add_argument(
        "--topup-plan-summary", type=Path,
        default=DEFAULT_TOPUP_PLAN_SUMMARY,
    )
    parser.add_argument(
        "--standard-lane-summary", type=Path,
        default=DEFAULT_STANDARD_LANE_SUMMARY,
    )
    parser.add_argument(
        "--heavy-lane-summary", type=Path,
        default=DEFAULT_HEAVY_LANE_SUMMARY,
    )
    parser.add_argument(
        "--extreme-plan-summary", type=Path,
        default=DEFAULT_EXTREME_PLAN_SUMMARY,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--stage-runs-dir", type=Path, default=DEFAULT_STAGE_RUNS_DIR,
    )
    parser.add_argument(
        "--openml-cache-root", type=Path, default=DEFAULT_OPENML_CACHE_ROOT,
    )
    parser.add_argument(
        "--policy-csv", type=Path, default=DEFAULT_POLICY_CSV,
    )
    parser.add_argument(
        "--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS_YAML,
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--target-stage-label", default=TARGET_STAGE_LABEL,
        help="SQLite CHECK-constrained stage label written into every "
             "row of every execution copy. The default "
             "'stage1_topup_to_005' is the Commit 47 / 48 / 49 convention "
             "for the topup_to_5 tier (replicas 2..5).",
    )
    parser.add_argument(
        "--target-replica", type=int, default=TARGET_REPLICA,
    )
    parser.add_argument(
        "--hard-cap-hours-per-shard", type=float,
        default=DEFAULT_HARD_CAP_HOURS,
        help="Per-shard subprocess timeout ceiling.",
    )
    parser.add_argument(
        "--include-extreme-tasks", action="store_true",
        help="Required (with --execute-extreme-lane) for real "
             "execution. Mirrors the policy-defined extreme opt-in.",
    )
    parser.add_argument(
        "--execute-extreme-lane", action="store_true",
        help="Required (with --include-extreme-tasks) for real "
             "execution. Without both flags the script refuses to "
             "execute training and falls back to a planning-only "
             "report (equivalent to --dry-run).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the pre-run plan and gate-check results, then exit "
             "with execution_status='planned_not_executed'. Does not "
             "copy or rewrite any shard.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Run gates + run-dir + rewrite + classification + summary, "
             "but do NOT invoke the cc18 runner. Used by tests.",
    )
    args = parser.parse_args(argv)

    # The two explicit flags must both be passed for real execution.
    flags_satisfied = bool(args.include_extreme_tasks) and bool(
        args.execute_extreme_lane,
    )

    if args.dry_run or (not flags_satisfied and not args.skip_train):
        try:
            live_policy_version = _sha256(args.policy_csv)
            signoff = verify_signoff(
                args.signoff_file,
                expected_policy_version=live_policy_version,
            )
            plan_record = verify_topup_plan(
                args.topup_plan_summary,
                expected_policy_version=live_policy_version,
            )
            std = verify_standard_lane_summary(
                args.standard_lane_summary,
                expected_policy_version=live_policy_version,
            )
            hvy = verify_heavy_lane_summary(
                args.heavy_lane_summary,
                expected_policy_version=live_policy_version,
            )
            plan50 = verify_extreme_plan_summary(
                args.extreme_plan_summary,
                expected_policy_version=live_policy_version,
            )
        except GateRefusalError as exc:
            print(f"GATE REFUSAL: {exc}", file=sys.stderr)
            return 3
        g = RuntimeGuardrails.load(
            yaml_path=args.guardrails_yaml, csv_path=args.policy_csv,
        )
        shards = sorted(args.shards_dir.glob("shard_*.sqlite"))
        plan = build_pre_run_plan(shards, g)
        ext_spec = g.get_lane_spec("extreme")
        reason = (
            "dry-run requested" if args.dry_run
            else "real execution requires both --include-extreme-tasks and "
                 "--execute-extreme-lane"
        )
        print(json.dumps({
            "execution_status": "planned_not_executed",
            "reason_not_executed": reason,
            "shards_dir": str(args.shards_dir),
            "signoff_file": str(args.signoff_file),
            "topup_plan_summary": str(args.topup_plan_summary),
            "standard_lane_summary": str(args.standard_lane_summary),
            "heavy_lane_summary": str(args.heavy_lane_summary),
            "extreme_plan_summary": str(args.extreme_plan_summary),
            "policy_version": live_policy_version,
            "policy_version_pinned": PINNED_POLICY_VERSION,
            "signoff_status": signoff.get("signoff_status"),
            "signoff_type": signoff.get("signoff_type"),
            "topup_plan_execution_status": plan_record.get(
                "execution_status",
            ),
            "commit48_standard_lane_execution_status": std.get(
                "execution_status",
            ),
            "commit48_standard_lane_n_jobs_success": std.get(
                "n_jobs_success",
            ),
            "commit49_heavy_lane_execution_status": hvy.get(
                "execution_status",
            ),
            "commit49_heavy_lane_n_jobs_success": hvy.get(
                "n_jobs_success",
            ),
            "commit50_extreme_plan_execution_status": plan50.get(
                "execution_status",
            ),
            "commit50_extreme_plan_n_runnable_extreme_canary": plan50.get(
                "n_runnable_extreme_canary",
            ),
            "include_extreme_tasks_flag": bool(args.include_extreme_tasks),
            "execute_extreme_lane_flag": bool(args.execute_extreme_lane),
            "run_root": str(args.run_root),
            "output_root": str(args.output_root),
            "stage_runs_dir": str(args.stage_runs_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "n_folds": args.n_folds,
            "run_id": args.run_id,
            "target_stage_label": args.target_stage_label,
            "target_replica": args.target_replica,
            "lane": LANE,
            "topup_tier": TOPUP_TIER,
            "extreme_lane_max_evaluations_recommended": int(
                ext_spec.stage0_max_evaluations,
            ),
            "extreme_lane_timeout_seconds_per_cell_recommended": float(
                ext_spec.timeout_seconds_per_cell,
            ),
            "pre_run_plan": {
                k: v for k, v in plan.items() if k != "per_shard"
            },
            "per_shard": plan["per_shard"],
            "expected_extreme_canary_cells": EXPECTED_EXTREME_CANARY_CELLS,
            "expected_extreme_task_ids": list(
                sorted(EXPECTED_EXTREME_TASK_IDS),
            ),
        }, indent=2))
        return 0

    try:
        summary = run_replica002_extreme_lane(
            shards_dir=args.shards_dir,
            signoff_file=args.signoff_file,
            topup_plan_summary=args.topup_plan_summary,
            standard_lane_summary=args.standard_lane_summary,
            heavy_lane_summary=args.heavy_lane_summary,
            extreme_plan_summary=args.extreme_plan_summary,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            n_folds=args.n_folds,
            run_id=args.run_id,
            target_stage_label=args.target_stage_label,
            target_replica=args.target_replica,
            hard_cap_hours_per_shard=args.hard_cap_hours_per_shard,
            skip_train=args.skip_train,
        )
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"executed={summary['n_jobs_executed']}/"
        f"{summary['expected_extreme_canary_cells']}  "
        f"success={summary['n_jobs_success']}  "
        f"deferred_standard={summary['n_jobs_deferred_standard']}  "
        f"deferred_heavy={summary['n_jobs_deferred_heavy']}  "
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
        and summary["no_committed_shard_modified_by_this_script"]
    ) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
