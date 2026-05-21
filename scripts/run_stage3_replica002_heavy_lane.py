#!/usr/bin/env python
"""Run the Stage-3 / top-up replica_002 heavy-lane pass across all 10 shards.

This is **Commit 49**: the heavy-lane companion to Commit 48's
standard-lane pass. It chains four gates on top of Commit 45 / 46:

- ``jobs/doctoral/openml_cc18/stage3_signoff.json`` must be signed
  with ``signoff_type == 'stage0_replica_001'`` and the same
  ``policy_version`` as the live SHA-256 of
  ``benchmarks/doctoral/openml_cc18/heavy_task_policy.csv``;
- ``experiments/_stage_runs/stage3_topup_plan_latest_summary.json``
  must be ``planned_not_executed`` and list ``topup_to_5`` covering
  ``replica = 2``;
- ``experiments/_stage_runs/``
  ``stage3_replica_002_standard_lane_latest_summary.json``
  (Commit 48) must be ``executed`` for replica = 2 / standard,
  with 684 / 684 success, 0 failures, source shards unchanged, and
  the same pinned ``policy_version``.

The scope is deliberately narrow:

- **all 10** source template shards;
- one replica (``replica = 2``);
- one lane (``heavy``);
- four canary methods × three algorithms only;
- 156 executable heavy-lane canary cells total
  (13 heavy tasks × 4 canary methods × 3 algorithms).

Heavy-lane policy budget under
``benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml``:

- per-cell timeout = 7,200 s (2 h);
- ``stage0_max_evaluations`` = 5 (same as Commit 41's stage-0 heavy
  run — Stage-3 top-up reuses the signed-off stage-0 budget so the
  results stack across replicas under the same ``policy_version``).

What this script does
---------------------
- copies every shard
  ``jobs/doctoral/openml_cc18/shards/stage0_replica_001/
  shard_NN.sqlite`` (NN = 00..09) into
  ``runs/cc18/stage3_replica_002_heavy_lane_latest/
  shards/stage0_replica_001/shard_NN.execution.sqlite`` via the
  result-handoff helper;
- rewrites every **copy** so all rows carry ``replica = 2`` and
  ``stage = 'stage1_topup_to_005'`` (the SQLite-CHECK-constrained
  label for the ``topup_to_5`` tier — Commit 47 / 48 convention);
- pre-marks standard / extreme / non-canary rows as skipped under
  the live runtime guardrails;
- invokes ``scripts/cc18_runner.py`` for the runnable heavy-lane
  canary rows of each shard;
- emits
  ``experiments/_stage_runs/
  stage3_replica_002_heavy_lane_latest_summary.{json,md}``
  via the protocol exporter, augmented with all keys the Commit 49
  prompt anchors on.

What this script does NOT do
----------------------------
- run any other replica (3 / 4 / 5);
- run the standard or extreme lanes (standard was already done in
  Commit 48; extreme is gated to a later commit);
- run non-canary methods;
- run the full ``topup_to_5`` tier (3,456 canary cells across
  replicas 2–5);
- promote ``isolet`` (task 3481) into the heavy lane — isolet
  remains a *future* policy recalibration candidate under the
  signoff caveats, but the pinned ``policy_version`` keeps it
  standard for this commit;
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
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = (
    REPO / "experiments/_batch_runs/stage3_replica_002_heavy_lane"
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
TARGET_STAGE_LABEL = "stage1_topup_to_005"  # Commit 47/48 convention
TARGET_REPLICA = 2
RUN_ID = "stage3_replica_002_heavy_lane_latest"
BATCH_ID = "stage3_replica_002_heavy_lane"
LANE = "heavy"
TOPUP_TIER = "topup_to_5_partial"
SIGNOFF_TYPE_EXPECTED = "stage0_replica_001"
PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
EXPECTED_HEAVY_CANARY_CELLS = 156  # 13 heavy tasks * 4 canary * 3 algos
COMMIT48_EXPECTED_SUCCESS = 684
N_EXPECTED_SHARDS = 10
DEFAULT_REQUESTED_MAX_EVALUATIONS = 5
DEFAULT_N_FOLDS = 2
# Heavy lane: per-shard subprocess timeout ceiling. 156 cells / 10
# shards ~ 16 cells per shard. At 7,200 s per-cell timeout × 16 ×
# 1.5 safety factor ~ 48 hours per shard, capped at 12 h here. The
# observed Commit 41 stage-0 heavy run took ~34,889 s total for all
# 156 cells, so per-shard wall time is far below the cap.
DEFAULT_HARD_CAP_HOURS = 12.0
# Recalibration candidate carried forward from the signoff caveats.
# This task is NOT promoted to heavy in this commit; we only record
# its lane and dataset name in the summary for traceability.
ISOLET_TASK_ID = 3481


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when a pre-flight check rejects the run."""


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
# Gate verification: signoff
# ---------------------------------------------------------------------------


def verify_signoff(
    signoff_path: Path, *, expected_policy_version: str,
) -> dict:
    """Refuse the run unless the signoff is signed, the right type,
    and carries the expected policy_version."""
    if not signoff_path.exists():
        raise GateRefusalError(
            f"signoff file not found at {signoff_path}; this run "
            "requires the Commit 45 signoff to exist."
        )
    try:
        record = json.loads(signoff_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{signoff_path} is not valid JSON: {exc}"
        ) from exc
    status = record.get("signoff_status")
    if status != "signed":
        raise GateRefusalError(
            f"{signoff_path} has signoff_status={status!r}; expected 'signed'."
        )
    stype = record.get("signoff_type")
    if stype != SIGNOFF_TYPE_EXPECTED:
        raise GateRefusalError(
            f"{signoff_path} has signoff_type={stype!r}; expected "
            f"{SIGNOFF_TYPE_EXPECTED!r}."
        )
    signed_pv = record.get("policy_version")
    if signed_pv != expected_policy_version:
        raise GateRefusalError(
            f"{signoff_path} carries policy_version={signed_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing run "
            "against drifted policy."
        )
    return record


# ---------------------------------------------------------------------------
# Gate verification: top-up plan
# ---------------------------------------------------------------------------


def verify_topup_plan(
    plan_path: Path, *, expected_policy_version: str,
) -> dict:
    """Refuse the run unless the Stage-3 plan is fresh, planning-only,
    pinned to the same policy_version, and lists ``topup_to_5`` with
    ``replica = 2`` included."""
    if not plan_path.exists():
        raise GateRefusalError(
            f"stage3 top-up plan summary not found at {plan_path}; run "
            "scripts/plan_stage3_topup.py first (Commit 46)."
        )
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{plan_path} is not valid JSON: {exc}"
        ) from exc
    exec_status = record.get("execution_status")
    if exec_status != "planned_not_executed":
        raise GateRefusalError(
            f"{plan_path} has execution_status={exec_status!r}; expected "
            "'planned_not_executed'."
        )
    plan_pv = record.get("policy_version")
    if plan_pv != expected_policy_version:
        raise GateRefusalError(
            f"{plan_path} carries policy_version={plan_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing run "
            "against drifted policy."
        )
    tiers = record.get("tier_plans") or []
    tier_topup_5 = next(
        (t for t in tiers if t.get("tier") == "topup_to_5"), None,
    )
    if tier_topup_5 is None:
        raise GateRefusalError(
            f"{plan_path} does not list a 'topup_to_5' tier; refusing."
        )
    rs = int(tier_topup_5.get("replica_start") or 0)
    re_ = int(tier_topup_5.get("replica_end") or 0)
    if not (rs <= TARGET_REPLICA <= re_):
        raise GateRefusalError(
            f"topup_to_5 tier covers replicas {rs}..{re_}; replica="
            f"{TARGET_REPLICA} is outside that range."
        )
    return record


# ---------------------------------------------------------------------------
# Gate verification: Commit 48 standard-lane summary
# ---------------------------------------------------------------------------


def verify_standard_lane_summary(
    summary_path: Path, *, expected_policy_version: str,
    expected_success: int = COMMIT48_EXPECTED_SUCCESS,
) -> dict:
    """Refuse the run unless Commit 48's standard-lane pass is green,
    against the same policy, and exercised replica = 2 / standard
    only."""
    if not summary_path.exists():
        raise GateRefusalError(
            f"Commit 48 standard-lane summary not found at {summary_path}; "
            "run scripts/run_stage3_replica002_standard_lane.py first "
            "(Commit 48)."
        )
    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateRefusalError(
            f"{summary_path} is not valid JSON: {exc}"
        ) from exc
    exec_status = record.get("execution_status")
    if exec_status != "executed":
        raise GateRefusalError(
            f"{summary_path} has execution_status={exec_status!r}; expected "
            "'executed'. The standard-lane pass must have completed "
            "successfully before heavy execution."
        )
    if int(record.get("replica") or 0) != TARGET_REPLICA:
        raise GateRefusalError(
            f"{summary_path} replica={record.get('replica')!r}; expected "
            f"{TARGET_REPLICA}."
        )
    if record.get("lane") != "standard":
        raise GateRefusalError(
            f"{summary_path} lane={record.get('lane')!r}; expected 'standard'."
        )
    pv = record.get("policy_version")
    if pv != expected_policy_version:
        raise GateRefusalError(
            f"{summary_path} carries policy_version={pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing run "
            "against drifted policy."
        )
    if int(record.get("n_jobs_success") or -1) != int(expected_success):
        raise GateRefusalError(
            f"{summary_path} n_jobs_success={record.get('n_jobs_success')}; "
            f"expected {expected_success}. The standard-lane pass must be "
            "fully green before heavy execution."
        )
    for key, expected in (
        ("n_jobs_failed", 0),
        ("n_jobs_failed_timeout", 0),
        ("n_jobs_pending_after", 0),
        ("n_jobs_running_after", 0),
    ):
        actual = int(record.get(key) or 0)
        if actual != expected:
            raise GateRefusalError(
                f"{summary_path} {key}={actual}; expected {expected}."
            )
    if not bool(record.get("source_shards_unchanged", False)):
        raise GateRefusalError(
            f"{summary_path} reports source_shards_unchanged=False; "
            "refusing heavy execution."
        )
    if not bool(
        record.get("no_full_topup_to_5_executed_by_this_script", False)
    ):
        raise GateRefusalError(
            f"{summary_path} no_full_topup_to_5_executed_by_this_script "
            "is not true; standard pass scope must have stayed narrow."
        )
    if not bool(
        record.get("no_heavy_lane_executed_by_this_script", False)
    ):
        raise GateRefusalError(
            f"{summary_path} no_heavy_lane_executed_by_this_script is "
            "not true; standard pass must not have run heavy cells."
        )
    return record


# ---------------------------------------------------------------------------
# Row classification (heavy-lane variant)
# ---------------------------------------------------------------------------


def classify_rows(
    rows: list[tuple],
    guardrails: RuntimeGuardrails,
) -> dict[str, list[dict]]:
    """Split shard rows into four disjoint heavy-lane buckets.

    rows: (job_id, openml_task_id, method, algorithm).
    Returns dict keys:
      ``runnable_heavy``           — canary methods × heavy tasks;
      ``deferred_standard_lane``   — any row whose task is standard
                                     (already done by Commit 48);
      ``deferred_extreme_lane``    — any row whose task is extreme
                                     (a later commit handles it);
      ``refused_not_in_canary_set`` — heavy task × non-canary method.
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
# Pre-run plan (heavy-lane variant, all 10 shards)
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shards: list[Path], guardrails: RuntimeGuardrails,
) -> dict:
    """Inventory the committed source shards under the policy,
    read-only, projecting heavy-lane buckets."""
    n_jobs_total = 0
    n_runnable_heavy = 0
    n_deferred_standard = 0
    n_deferred_extreme = 0
    n_refused_non_canary = 0
    method_counts: Counter = Counter()
    algorithm_counts: Counter = Counter()
    non_canary_methods: set[str] = set()
    heavy_tasks_executed: set[int] = set()
    standard_tasks_deferred: set[int] = set()
    extreme_tasks_deferred: set[int] = set()
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
            "runnable_heavy": len(buckets["runnable_heavy"]),
            "deferred_standard_lane": len(buckets["deferred_standard_lane"]),
            "deferred_extreme_lane": len(buckets["deferred_extreme_lane"]),
            "refused_not_in_canary_set": len(
                buckets["refused_not_in_canary_set"],
            ),
        }
        per_shard.append(per_sh)
        n_jobs_total += len(rows)
        n_runnable_heavy += per_sh["runnable_heavy"]
        n_deferred_standard += per_sh["deferred_standard_lane"]
        n_deferred_extreme += per_sh["deferred_extreme_lane"]
        n_refused_non_canary += per_sh["refused_not_in_canary_set"]
        for tid, m, a, st, rep in rows:
            tid_int = int(tid)
            lane = guardrails.get_task_lane(tid_int)
            seen_tasks[tid_int] = lane
            method_counts[m] += 1
            algorithm_counts[a] += 1
            source_template_stages.add(st)
            source_template_replicas.add(int(rep))
            if lane == "heavy" and m in CANARY_METHODS:
                heavy_tasks_executed.add(tid_int)
            elif lane == "standard":
                standard_tasks_deferred.add(tid_int)
            elif lane == "extreme":
                extreme_tasks_deferred.add(tid_int)
            if lane == "heavy" and m not in CANARY_METHODS:
                non_canary_methods.add(m)
    return {
        "n_source_shards": len(shards),
        "n_jobs_total": n_jobs_total,
        "n_runnable_heavy": n_runnable_heavy,
        "n_deferred_standard_lane": n_deferred_standard,
        "n_deferred_extreme_lane": n_deferred_extreme,
        "n_refused_not_in_canary_set": n_refused_non_canary,
        "task_lane_counts_universe": dict(Counter(seen_tasks.values())),
        "method_counts": dict(method_counts),
        "algorithm_counts": dict(algorithm_counts),
        "non_canary_methods_refused": sorted(non_canary_methods),
        "heavy_tasks_executed": sorted(heavy_tasks_executed),
        "standard_tasks_deferred": sorted(standard_tasks_deferred),
        "extreme_tasks_deferred": sorted(extreme_tasks_deferred),
        "source_template_stages": sorted(source_template_stages),
        "source_template_replicas": sorted(source_template_replicas),
        "per_shard": per_shard,
        "isolet_task_id": ISOLET_TASK_ID,
        "isolet_lane_under_pinned_policy": (
            guardrails.get_task_lane(ISOLET_TASK_ID)
        ),
        "isolet_promoted_to_heavy_in_this_commit": False,
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


def run_replica002_heavy_lane(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    signoff_file: Path = DEFAULT_SIGNOFF_FILE,
    topup_plan_summary: Path = DEFAULT_TOPUP_PLAN_SUMMARY,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    requested_max_evaluations: int = DEFAULT_REQUESTED_MAX_EVALUATIONS,
    n_folds: int = DEFAULT_N_FOLDS,
    run_id: str = RUN_ID,
    target_stage_label: str = TARGET_STAGE_LABEL,
    target_replica: int = TARGET_REPLICA,
    hard_cap_hours_per_shard: float = DEFAULT_HARD_CAP_HOURS,
    skip_train: bool = False,
    force_run_dir: bool = True,
    expected_heavy_canary_cells: int = EXPECTED_HEAVY_CANARY_CELLS,
) -> dict:
    """Run replica_002 heavy lane across all 10 shards and return
    the summary dict written to disk."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    source_shards = sorted(shards_dir.glob("shard_*.sqlite"))
    if len(source_shards) != N_EXPECTED_SHARDS:
        raise GateRefusalError(
            f"expected {N_EXPECTED_SHARDS} stage-0 shards under {shards_dir}, "
            f"found {len(source_shards)}"
        )

    # 1. Three gates BEFORE we touch anything.
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
    signoff_sha256 = _sha256(signoff_file)
    topup_plan_sha256 = _sha256(topup_plan_summary)
    std_summary_sha256 = _sha256(standard_lane_summary)

    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )

    # Defensive: refuse if the pinned policy somehow moved isolet to heavy.
    if guardrails.get_task_lane(ISOLET_TASK_ID) != "standard":
        raise GateRefusalError(
            f"isolet (task {ISOLET_TASK_ID}) is no longer 'standard' under "
            "the live policy; refusing because this commit must not "
            "promote isolet to heavy."
        )

    # 2. Pre-run plan and refuse if expected runnable count is wrong.
    plan = build_pre_run_plan(source_shards, guardrails)
    if plan["n_runnable_heavy"] != expected_heavy_canary_cells:
        raise GateRefusalError(
            f"pre-run plan inconsistency: expected "
            f"{expected_heavy_canary_cells} heavy-lane canary cells "
            f"across all 10 shards but found "
            f"{plan['n_runnable_heavy']}. Verify "
            "heavy_task_policy.csv classification before proceeding."
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
            "source shard MD5 changed during execution-copy rewrite; "
            "refusing run."
        )

    # 5. Per-shard: classify, pre-mark deferred + refused, dispatch the
    #    heavy-lane canary rows via cc18_runner.
    lane_spec = guardrails.get_lane_spec(LANE)
    # Heavy lane uses the policy's stage0_max_evaluations (5) so the
    # top-up cells stack with the signed-off stage-0 cells under the
    # same budget; the gate context (3) is narrower.
    eff_max = max(1, int(min(
        requested_max_evaluations, lane_spec.stage0_max_evaluations,
    )))
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
        n_runnable = len(buckets["runnable_heavy"])
        cells_runnable_per_shard[exec_p.name] = n_runnable

        cx = sqlite3.connect(exec_p)
        try:
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_standard_lane"]],
                status="skipped", last_error="deferred_standard_lane",
                assigned_worker="stage3_replica_002_heavy_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["deferred_extreme_lane"]],
                status="skipped", last_error="deferred_extreme_lane",
                assigned_worker="stage3_replica_002_heavy_lane_policy",
            )
            _set_status_for_job_ids(
                cx, [e["job_id"] for e in buckets["refused_not_in_canary_set"]],
                status="skipped", last_error="refused_not_in_canary_set",
                assigned_worker="stage3_replica_002_heavy_lane_policy",
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
            **res,
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
    n_deferred_extreme = int(last_error_counts.get("deferred_extreme_lane", 0))
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
            "deferred_extreme_lane": int(lec.get("deferred_extreme_lane", 0)),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
        })

    heavy_tasks_executed_observed = sorted({
        c["openml_task_id"] for c in cells if c["status"] == "success"
    })

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
        "commit48_standard_lane_summary_path": _safe_rel(standard_lane_summary),
        "commit48_standard_lane_summary_sha256": std_summary_sha256,
        "commit48_standard_lane_n_jobs_success": int(
            std_record.get("n_jobs_success") or 0,
        ),
        "commit48_standard_lane_runtime_seconds": float(
            std_record.get("runtime_seconds_runner_total") or 0.0,
        ),
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "execution_shards": [_safe_rel(p) for p in exec_shards],
        "execution_sqlite_sha256": execution_sqlite_sha256,
        "n_jobs_total": int(cells_total_before_run),
        "n_jobs_executed": n_success + n_failed_other,
        "n_jobs_success": n_success,
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
        "heavy_tasks_executed": heavy_tasks_executed_observed,
        "heavy_tasks_in_universe": plan["heavy_tasks_executed"],
        "standard_tasks_deferred": plan["standard_tasks_deferred"],
        "extreme_tasks_deferred": plan["extreme_tasks_deferred"],
        "non_canary_methods_refused": plan["non_canary_methods_refused"],
        "expected_heavy_canary_cells": expected_heavy_canary_cells,
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
        "isolet_task_id": ISOLET_TASK_ID,
        "isolet_lane_under_pinned_policy": guardrails.get_task_lane(
            ISOLET_TASK_ID,
        ),
        "isolet_promoted_to_heavy_in_this_commit": False,
        "isolet_note": (
            "isolet/task 3481 remains a future policy-recalibration "
            "candidate under signoff caveat 1 but is NOT promoted to "
            "the heavy lane in this commit; the pinned policy_version "
            "keeps it standard. A future commit (post-replica_002 "
            "review) may decide whether to rebuild heavy_task_policy.csv "
            "between replicas."
        ),
        "no_other_replica_executed_by_this_script": True,
        "no_full_topup_to_5_executed_by_this_script": True,
        "no_standard_lane_rerun_by_this_script": True,
        "no_extreme_lane_executed_by_this_script": True,
        "no_committed_shard_modified_by_this_script": (
            md5_after_run == md5_before
        ),
        "no_raw_openml_payloads_staged_by_this_script": True,
        "no_execution_sqlite_staged_by_this_script": True,
        "only_replica_002_heavy_lane_executed": True,
        "operator_review_required_before_replica002_extreme": True,
        "next_recommended_step": (
            "After Commit 49 is green and operator-reviewed, Commit 50 "
            "should plan the replica_002 extreme lane (Devnagari-Script + "
            "any other extreme task under the pinned policy). Do NOT run "
            "extreme lane in Commit 49. Do NOT scale to replica_003-005 "
            "until replica_002 standard + heavy + extreme has been "
            "reviewed end-to-end."
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
    extra.append("## stage 3 / replica_002 heavy-lane summary (Commit 49)\n")
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
    extra.append(f"- signoff_path: `{summary['signoff_path']}`")
    extra.append(f"- signoff_sha256: `{str(summary['signoff_sha256'])[:16]}`")
    extra.append(
        f"- stage3_topup_plan_summary_sha256: "
        f"`{str(summary['stage3_topup_plan_summary_sha256'])[:16]}`"
    )
    extra.append(
        f"- commit48_standard_lane_summary_sha256: "
        f"`{str(summary['commit48_standard_lane_summary_sha256'])[:16]}`\n"
    )

    extra.append(
        f"- n_jobs_total (across 10 shards): {summary['n_jobs_total']}"
    )
    extra.append(
        f"- expected runnable heavy-lane canary cells: "
        f"**{summary['expected_heavy_canary_cells']}**"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"success: **{summary['n_jobs_success']}**, "
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

    extra.append("### Heavy tasks executed (success only)\n")
    extra.append(
        f"{len(summary['heavy_tasks_executed'])} tasks: "
        f"{summary['heavy_tasks_executed']}\n"
    )
    extra.append("### Standard tasks deferred (already handled by Commit 48)\n")
    extra.append(
        f"{len(summary['standard_tasks_deferred'])} tasks: "
        f"{summary['standard_tasks_deferred']}\n"
    )
    extra.append("### Extreme tasks deferred (later commit)\n")
    extra.append(
        f"{len(summary['extreme_tasks_deferred'])} tasks: "
        f"{summary['extreme_tasks_deferred']}\n"
    )
    extra.append("### Non-canary methods refused\n")
    extra.append(
        f"{summary['non_canary_methods_refused']}\n"
    )

    extra.append("### isolet recalibration note\n")
    extra.append(
        f"- isolet/task {summary['isolet_task_id']} is currently "
        f"`{summary['isolet_lane_under_pinned_policy']}` under the pinned "
        "policy_version and is NOT promoted to heavy by this commit.\n"
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
        and summary["no_extreme_lane_executed_by_this_script"]
        and summary["no_other_replica_executed_by_this_script"]
        and not summary["isolet_promoted_to_heavy_in_this_commit"]
    )
    if invariants_ok:
        extra.append(
            "### stage 3 replica_002 heavy lane verdict: "
            "**GATE PASS — operator review required**\n"
        )
        extra.append(
            "Run finished cleanly: every heavy-lane canary cell on "
            "replica_002 across all 10 shards reached a terminal status, "
            "every committed source shard is byte-identical to its "
            "pre-run MD5, standard lane was not rerun, extreme lane was "
            "not executed, and isolet was not promoted to heavy. "
            "Commit 50 may plan the replica_002 extreme lane; do NOT "
            "scale to replica_003-005 without an aggregate review of "
            "replica_002 standard + heavy + extreme.\n"
        )
    else:
        extra.append(
            "### stage 3 replica_002 heavy lane verdict: **NOT GREEN**\n"
        )
        extra.append(
            "Investigate failures / timeouts / source-shard drift before "
            "any further Stage-3 / top-up execution.\n"
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
    parser.add_argument(
        "--max-evaluations", type=int,
        default=DEFAULT_REQUESTED_MAX_EVALUATIONS,
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--target-stage-label", default=TARGET_STAGE_LABEL,
        help="SQLite CHECK-constrained stage label written into every "
             "row of every execution copy. The default "
             "'stage1_topup_to_005' is the Commit 47 / 48 convention "
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
        "--dry-run", action="store_true",
        help="Print the pre-run plan and the gate check results, then "
             "exit with execution_status='planned_not_executed'.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Run gates + run-dir + rewrite + classification + summary, "
             "but do NOT invoke the cc18 runner. Used by tests.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
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
        except GateRefusalError as exc:
            print(f"GATE REFUSAL: {exc}", file=sys.stderr)
            return 3
        g = RuntimeGuardrails.load(
            yaml_path=args.guardrails_yaml, csv_path=args.policy_csv,
        )
        shards = sorted(args.shards_dir.glob("shard_*.sqlite"))
        plan = build_pre_run_plan(shards, g)
        print(json.dumps({
            "execution_status": "planned_not_executed",
            "shards_dir": str(args.shards_dir),
            "signoff_file": str(args.signoff_file),
            "topup_plan_summary": str(args.topup_plan_summary),
            "standard_lane_summary": str(args.standard_lane_summary),
            "policy_version": live_policy_version,
            "policy_version_pinned": PINNED_POLICY_VERSION,
            "signoff_status": signoff.get("signoff_status"),
            "signoff_type": signoff.get("signoff_type"),
            "topup_plan_execution_status": plan_record.get(
                "execution_status",
            ),
            "commit48_standard_lane_execution_status":
                std.get("execution_status"),
            "commit48_standard_lane_n_jobs_success":
                std.get("n_jobs_success"),
            "run_root": str(args.run_root),
            "output_root": str(args.output_root),
            "stage_runs_dir": str(args.stage_runs_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
            "run_id": args.run_id,
            "target_stage_label": args.target_stage_label,
            "target_replica": args.target_replica,
            "lane": LANE,
            "topup_tier": TOPUP_TIER,
            "pre_run_plan": {
                k: v for k, v in plan.items() if k != "per_shard"
            },
            "per_shard": plan["per_shard"],
            "expected_heavy_canary_cells": EXPECTED_HEAVY_CANARY_CELLS,
            "isolet_task_id": ISOLET_TASK_ID,
            "isolet_lane_under_pinned_policy": (
                g.get_task_lane(ISOLET_TASK_ID)
            ),
            "isolet_promoted_to_heavy_in_this_commit": False,
        }, indent=2))
        return 0

    try:
        summary = run_replica002_heavy_lane(
            shards_dir=args.shards_dir,
            signoff_file=args.signoff_file,
            topup_plan_summary=args.topup_plan_summary,
            standard_lane_summary=args.standard_lane_summary,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            requested_max_evaluations=args.max_evaluations,
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
        f"{summary['expected_heavy_canary_cells']}  "
        f"success={summary['n_jobs_success']}  "
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
        and summary["no_committed_shard_modified_by_this_script"]
    ) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
