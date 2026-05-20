#!/usr/bin/env python
"""Run the tiny Stage-3 / top-up pilot on shard_00 / replica 002 / standard lane.

This is **Commit 47**: the very first real Stage-3 / top-up execution
on top of the Commit 45 sign-off and the Commit 46 plan. It is
deliberately *tiny*:

- one source shard: ``shard_00``;
- one replica: ``2`` (the first replica of the ``topup_to_5`` tier);
- one lane: ``standard``;
- four canary methods only:
  ``default_gbdt``, ``random_search``, ``tpe_optuna``,
  ``doe_rsm_vrf_true_nbi``;
- three algorithms: ``xgboost``, ``lightgbm``, ``catboost``.

The pilot exists so an operator can review the *shape* of a top-up
cell on this worker (runtime per cell, manifest format, summary JSON
schema, source-shard byte-equality) before scaling to the full
``topup_to_5`` tier (10 shards × 4 replicas × 864 cells/replica =
3,456 canary cells).

Pre-flight refusals
-------------------
- ``jobs/doctoral/openml_cc18/stage3_signoff.json`` must exist, be
  signed (``signoff_status == 'signed'``), and carry
  ``signoff_type == 'stage0_replica_001'``;
- ``experiments/_stage_runs/stage3_topup_plan_latest_summary.json``
  must exist, report
  ``execution_status == 'planned_not_executed'``, list
  ``topup_to_5`` as a tier covering ``replica = 2``, and pin the
  same ``policy_version`` as the live SHA-256 of
  ``benchmarks/doctoral/openml_cc18/heavy_task_policy.csv``;
- if either gate is missing / stale / drifted, the runner exits 3
  *before* copying any shard.

What this script does
---------------------
- copies ``jobs/doctoral/openml_cc18/shards/stage0_replica_001/
  shard_00.sqlite`` into
  ``runs/cc18/stage3_pilot_replica_002_shard00_standard_lane_latest/
  shards/stage0_replica_001/shard_00.execution.sqlite`` via the
  result-handoff helper;
- rewrites the **copy only** so every row carries
  ``replica = 2`` and ``stage = 'stage1_topup_to_005'`` (the
  ``topup_to_5`` tier's existing label in the SQLite CHECK
  constraint), and every status starts from ``pending``;
- classifies the copy under the live runtime guardrails and
  pre-marks heavy / extreme / non-canary rows as skipped;
- invokes ``scripts/cc18_runner.py`` for the runnable
  standard-lane canary rows with the standard-lane policy budget
  (max_evaluations = 5, per-cell timeout = 1,800 s);
- emits
  ``experiments/_stage_runs/
  stage3_pilot_replica_002_shard00_standard_lane_latest_summary.{json,md}``
  via the protocol exporter, augmented with all keys the prompt
  for Commit 47 anchors on.

What this script does NOT do
----------------------------
- run the full ``topup_to_5`` tier (Commit 48+);
- run heavy or extreme lanes;
- run non-canary methods;
- create or modify ``stage3_signoff.json``;
- create or modify
  ``stage3_topup_plan_latest_summary.{json,md}``;
- regenerate ``heavy_task_policy.csv`` or
  ``runtime_guardrails.yaml``;
- promote or demote any task between lanes;
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
DEFAULT_SHARD_NAME = "shard_00.sqlite"
DEFAULT_SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
DEFAULT_TOPUP_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs/stage3_topup_plan_latest_summary.json"
)
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = (
    REPO / "experiments/_batch_runs/stage3_pilot_replica_002_shard00_standard_lane"
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
SOURCE_STAGE = "stage0_replica_001"          # subdir under shards/ we copy from
TARGET_STAGE_LABEL = "stage1_topup_to_005"   # SQLite CHECK enum for topup_to_5
TARGET_REPLICA = 2
RUN_ID = "stage3_pilot_replica_002_shard00_standard_lane_latest"
BATCH_ID = "stage3_pilot_replica_002_shard00_standard_lane"
LANE = "standard"
TOPUP_TIER = "topup_to_5_pilot"
SIGNOFF_TYPE_EXPECTED = "stage0_replica_001"
PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
EXPECTED_STANDARD_CANARY_CELLS_SHARD_00 = 68
DEFAULT_REQUESTED_MAX_EVALUATIONS = 5
DEFAULT_N_FOLDS = 2
DEFAULT_HARD_CAP_HOURS = 6.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PilotRefusalError(RuntimeError):
    """Raised when a pre-flight check rejects the pilot."""


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
    """Refuse the pilot unless the signoff is signed, the right type,
    and carries the expected policy_version."""
    if not signoff_path.exists():
        raise PilotRefusalError(
            f"signoff file not found at {signoff_path}; the Stage-3 pilot "
            "requires the Commit 45 signoff to exist."
        )
    try:
        record = json.loads(signoff_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotRefusalError(
            f"{signoff_path} is not valid JSON: {exc}"
        ) from exc
    status = record.get("signoff_status")
    if status != "signed":
        raise PilotRefusalError(
            f"{signoff_path} has signoff_status={status!r}; expected 'signed'."
        )
    stype = record.get("signoff_type")
    if stype != SIGNOFF_TYPE_EXPECTED:
        raise PilotRefusalError(
            f"{signoff_path} has signoff_type={stype!r}; expected "
            f"{SIGNOFF_TYPE_EXPECTED!r}."
        )
    signed_pv = record.get("policy_version")
    if signed_pv != expected_policy_version:
        raise PilotRefusalError(
            f"{signoff_path} carries policy_version={signed_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing pilot "
            "against drifted policy."
        )
    return record


def verify_topup_plan(
    plan_path: Path, *, expected_policy_version: str,
) -> dict:
    """Refuse the pilot unless the Stage-3 plan is fresh, planning-only,
    pinned to the same policy_version, and lists ``topup_to_5`` with
    replica = 2 included."""
    if not plan_path.exists():
        raise PilotRefusalError(
            f"stage3 top-up plan summary not found at {plan_path}; run "
            "scripts/plan_stage3_topup.py first (Commit 46)."
        )
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotRefusalError(
            f"{plan_path} is not valid JSON: {exc}"
        ) from exc
    exec_status = record.get("execution_status")
    if exec_status != "planned_not_executed":
        raise PilotRefusalError(
            f"{plan_path} has execution_status={exec_status!r}; expected "
            "'planned_not_executed'. The planner is a read-only artifact; "
            "another script appears to have mutated it."
        )
    plan_pv = record.get("policy_version")
    if plan_pv != expected_policy_version:
        raise PilotRefusalError(
            f"{plan_path} carries policy_version={plan_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing pilot "
            "against drifted policy."
        )
    tiers = record.get("tier_plans") or []
    tier_topup_5 = None
    for t in tiers:
        if t.get("tier") == "topup_to_5":
            tier_topup_5 = t
            break
    if tier_topup_5 is None:
        raise PilotRefusalError(
            f"{plan_path} does not list a 'topup_to_5' tier; refusing."
        )
    rs = int(tier_topup_5.get("replica_start") or 0)
    re_ = int(tier_topup_5.get("replica_end") or 0)
    if not (rs <= TARGET_REPLICA <= re_):
        raise PilotRefusalError(
            f"topup_to_5 tier covers replicas {rs}..{re_}; replica="
            f"{TARGET_REPLICA} is outside that range."
        )
    return record


# ---------------------------------------------------------------------------
# Row classification (same buckets as the Stage-0 standard-lane runner,
# kept local so this script does not depend on it).
# ---------------------------------------------------------------------------


def classify_rows(
    rows: list[tuple],
    guardrails: RuntimeGuardrails,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "runnable_standard": [],
        "deferred_heavy_lane": [],
        "deferred_extreme_lane": [],
        "refused_not_in_canary_set": [],
    }
    for job_id, task_id, method, algorithm in rows:
        lane = guardrails.get_task_lane(task_id)
        entry = {
            "job_id": job_id, "openml_task_id": int(task_id),
            "method": method, "algorithm": algorithm, "lane": lane,
        }
        if lane == "extreme":
            buckets["deferred_extreme_lane"].append(entry)
            continue
        if lane == "heavy":
            buckets["deferred_heavy_lane"].append(entry)
            continue
        if method not in CANARY_METHODS:
            buckets["refused_not_in_canary_set"].append(entry)
            continue
        buckets["runnable_standard"].append(entry)
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
    Stage-3 pilot identity.

    The committed shard_00 carries ``(stage='stage0_replica_001',
    replica=1)`` for all 219 rows. We:

    - flip every row's ``stage`` to ``target_stage``;
    - flip every row's ``replica`` to ``target_replica``;
    - reset ``status`` to 'pending', clear assigned_worker, last_error,
      started_at, finished_at, runtime_seconds, retry_count;
    - leave ``job_id`` (the PRIMARY KEY) and the ``UNIQUE (task, algo,
      method, replica)`` row identity unchanged.

    We never write to the committed source.
    """
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
# Pre-run plan (single shard)
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shard_path: Path, guardrails: RuntimeGuardrails,
) -> dict:
    """Inventory the source shard under the policy, read-only."""
    cx = sqlite3.connect(f"file:{shard_path}?mode=ro", uri=True)
    try:
        rows = list(cx.execute(
            "SELECT openml_task_id, method, algorithm, stage, replica "
            "FROM cc18_jobs",
        ))
    finally:
        cx.close()
    raw = [("", r[0], r[1], r[2]) for r in rows]
    buckets = classify_rows(raw, guardrails)

    task_lane_counts = Counter()
    seen_tasks: dict[int, str] = {}
    method_counts: Counter = Counter()
    algorithm_counts: Counter = Counter()
    non_canary_methods: set[str] = set()
    standard_tasks_executed: set[int] = set()
    heavy_tasks_deferred: set[int] = set()
    extreme_tasks_deferred: set[int] = set()
    source_template_stages: set[str] = set()
    source_template_replicas: set[int] = set()
    for tid, m, a, st, rep in rows:
        tid_int = int(tid)
        lane = guardrails.get_task_lane(tid_int)
        seen_tasks[tid_int] = lane
        method_counts[m] += 1
        algorithm_counts[a] += 1
        source_template_stages.add(st)
        source_template_replicas.add(int(rep))
        if lane == "standard" and m in CANARY_METHODS:
            standard_tasks_executed.add(tid_int)
        elif lane == "heavy":
            heavy_tasks_deferred.add(tid_int)
        elif lane == "extreme":
            extreme_tasks_deferred.add(tid_int)
        if lane == "standard" and m not in CANARY_METHODS:
            non_canary_methods.add(m)
    task_lane_counts = Counter(seen_tasks.values())
    return {
        "shard_path": _safe_rel(shard_path),
        "n_jobs_total": len(rows),
        "n_runnable_standard": len(buckets["runnable_standard"]),
        "n_deferred_heavy_lane": len(buckets["deferred_heavy_lane"]),
        "n_deferred_extreme_lane": len(buckets["deferred_extreme_lane"]),
        "n_refused_not_in_canary_set": len(
            buckets["refused_not_in_canary_set"],
        ),
        "task_lane_counts_universe": dict(task_lane_counts),
        "method_counts": dict(method_counts),
        "algorithm_counts": dict(algorithm_counts),
        "non_canary_methods_refused": sorted(non_canary_methods),
        "standard_tasks_executed": sorted(standard_tasks_executed),
        "heavy_tasks_deferred": sorted(heavy_tasks_deferred),
        "extreme_tasks_deferred": sorted(extreme_tasks_deferred),
        "source_template_stages": sorted(source_template_stages),
        "source_template_replicas": sorted(source_template_replicas),
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
# Cell collection (mirrors run_stage0_standard_lane._collect_cells)
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
# Top-level pilot
# ---------------------------------------------------------------------------


def run_pilot(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    shard_name: str = DEFAULT_SHARD_NAME,
    signoff_file: Path = DEFAULT_SIGNOFF_FILE,
    topup_plan_summary: Path = DEFAULT_TOPUP_PLAN_SUMMARY,
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
    hard_cap_hours: float = DEFAULT_HARD_CAP_HOURS,
    skip_train: bool = False,
    force_run_dir: bool = True,
    expected_standard_canary_cells: int = EXPECTED_STANDARD_CANARY_CELLS_SHARD_00,
) -> dict:
    """Run the Stage-3 pilot end-to-end and return the summary dict."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    source_shard = shards_dir / shard_name
    if not source_shard.exists():
        raise PilotRefusalError(
            f"source shard not found: {source_shard}"
        )

    # 1. Gate verification BEFORE we touch anything.
    live_policy_version = _sha256(policy_csv)
    signoff_record = verify_signoff(
        signoff_file, expected_policy_version=live_policy_version,
    )
    topup_plan_record = verify_topup_plan(
        topup_plan_summary, expected_policy_version=live_policy_version,
    )
    signoff_sha256 = _sha256(signoff_file)
    topup_plan_sha256 = _sha256(topup_plan_summary)

    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )

    # 2. Build the pre-run plan against the source shard (read-only)
    #    and refuse if the expected runnable count is wrong.
    plan = build_pre_run_plan(source_shard, guardrails)
    if plan["n_runnable_standard"] != expected_standard_canary_cells:
        raise PilotRefusalError(
            f"pre-run plan inconsistency on {source_shard.name}: expected "
            f"{expected_standard_canary_cells} standard-lane canary cells "
            f"but found {plan['n_runnable_standard']}. Verify "
            "heavy_task_policy.csv classification before proceeding."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_runs_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    # 3. Materialize the run dir via the handoff helper. We copy the
    #    single shard under the source template's stage subdir so the
    #    helper's path checks accept it.
    source_md5_before = _md5(source_shard)
    create_run_dir(
        run_id=run_id,
        stage=SOURCE_STAGE,
        shard_files=[shard_name],
        run_root=run_root,
        shards_root=shards_dir.parent,
        force=force_run_dir,
    )
    run_dir = run_root / run_id
    exec_dir = run_dir / "shards" / SOURCE_STAGE
    exec_shard = exec_dir / shard_name.replace(".sqlite", ".execution.sqlite")
    if not exec_shard.exists():
        raise PilotRefusalError(
            f"execution copy not found after create_run_dir: {exec_shard}"
        )
    source_md5_after_copy = _md5(source_shard)
    if source_md5_after_copy != source_md5_before:
        raise PilotRefusalError(
            "source shard MD5 changed during create_run_dir; refusing pilot."
        )

    # 4. Rewrite the execution copy: replica -> 2, stage -> topup label,
    #    all statuses back to pending. Source shard remains untouched.
    rewrite_info = rewrite_execution_copy_to_replica2(
        exec_shard, target_stage=target_stage_label,
        target_replica=target_replica,
    )
    source_md5_after_rewrite = _md5(source_shard)
    if source_md5_after_rewrite != source_md5_before:
        raise PilotRefusalError(
            "source shard MD5 changed during execution-copy rewrite; "
            "refusing pilot."
        )
    execution_sha256_after_rewrite = _sha256(exec_shard)

    # 5. Classify + pre-mark deferred / refused rows. Read the rewritten
    #    copy directly so the buckets reflect the new stage label.
    cx = sqlite3.connect(exec_shard)
    try:
        rows = list(cx.execute(
            "SELECT job_id, openml_task_id, method, algorithm "
            "FROM cc18_jobs ORDER BY job_id",
        ))
    finally:
        cx.close()
    buckets = classify_rows(rows, guardrails)
    n_runnable = len(buckets["runnable_standard"])
    if n_runnable != expected_standard_canary_cells:
        raise PilotRefusalError(
            f"post-rewrite classification mismatch: "
            f"runnable_standard={n_runnable} but expected "
            f"{expected_standard_canary_cells}."
        )

    cx = sqlite3.connect(exec_shard)
    try:
        _set_status_for_job_ids(
            cx, [e["job_id"] for e in buckets["deferred_heavy_lane"]],
            status="skipped", last_error="deferred_heavy_lane",
            assigned_worker="stage3_pilot_replica002_policy",
        )
        _set_status_for_job_ids(
            cx, [e["job_id"] for e in buckets["deferred_extreme_lane"]],
            status="skipped", last_error="deferred_extreme_lane",
            assigned_worker="stage3_pilot_replica002_policy",
        )
        _set_status_for_job_ids(
            cx, [e["job_id"] for e in buckets["refused_not_in_canary_set"]],
            status="skipped", last_error="refused_not_in_canary_set",
            assigned_worker="stage3_pilot_replica002_policy",
        )
    finally:
        cx.close()

    # 6. Dispatch the runnable standard-lane canary cells.
    lane_spec = guardrails.get_lane_spec(LANE)
    eff_max = max(1, int(min(
        requested_max_evaluations, lane_spec.gate_max_evaluations,
    )))
    timeout_s = min(
        lane_spec.timeout_seconds_per_cell * max(1, n_runnable) * 1.5,
        hard_cap_hours * 3600.0,
    )
    runner_invocations: list[dict] = []
    if not skip_train and n_runnable > 0:
        res = _invoke_runner_for_shard(
            exec_shard=exec_shard,
            out_root=out_root,
            openml_cache_root=openml_cache_root,
            max_evaluations=eff_max,
            n_folds=n_folds,
            max_jobs=n_runnable,
            timeout_seconds=timeout_s,
            stage_filter=target_stage_label,
        )
        runner_invocations.append({
            "shard": exec_shard.name,
            "n_runnable": n_runnable,
            "max_evaluations": eff_max,
            "timeout_seconds_for_shard": timeout_s,
            **res,
        })
        if res["timed_out"]:
            in_flight_ids = [e["job_id"] for e in buckets["runnable_standard"]]
            cx = sqlite3.connect(exec_shard)
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

    source_md5_after_run = _md5(source_shard)

    # 7. Collect cells + status counts.
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
    n_deferred_heavy = int(last_error_counts.get("deferred_heavy_lane", 0))
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

    # Per-shard status (single-shard pilot, so a one-row table).
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
            "deferred_heavy_lane": int(lec.get("deferred_heavy_lane", 0)),
            "deferred_extreme_lane": int(lec.get("deferred_extreme_lane", 0)),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
        })

    standard_tasks_executed_observed = sorted({
        c["openml_task_id"] for c in cells if c["status"] == "success"
    })

    # 8. Publish the stage-run summary via the protocol exporter.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID,
    )

    # 9. Augment with Stage-3-pilot-specific blocks.
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
        "n_source_shards": 1,
        "source_shards": [_safe_rel(source_shard)],
        "execution_shards": [_safe_rel(exec_shard)],
        "execution_sqlite_sha256_after_rewrite": execution_sha256_after_rewrite,
        "n_jobs_total": int(plan["n_jobs_total"]),
        "n_jobs_executed": n_success + n_failed_other,
        "n_jobs_success": n_success,
        "n_jobs_deferred_heavy": n_deferred_heavy,
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
            "deferred_heavy_lane": n_deferred_heavy,
            "deferred_extreme_lane": n_deferred_extreme,
            "refused_not_in_canary_set": n_refused_non_canary,
        },
        "task_lane_counts_universe": plan["task_lane_counts_universe"],
        "standard_tasks_executed": standard_tasks_executed_observed,
        "standard_tasks_in_shard": plan["standard_tasks_executed"],
        "heavy_tasks_deferred": plan["heavy_tasks_deferred"],
        "extreme_tasks_deferred": plan["extreme_tasks_deferred"],
        "non_canary_methods_refused": plan["non_canary_methods_refused"],
        "expected_standard_canary_cells": expected_standard_canary_cells,
        "per_shard_status": per_shard_status,
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
        "source_shard_md5_before": {shard_name: source_md5_before},
        "source_shard_md5_after": {shard_name: source_md5_after_run},
        "source_shard_md5_after_copy": {shard_name: source_md5_after_copy},
        "source_shard_md5_after_rewrite": {
            shard_name: source_md5_after_rewrite,
        },
        "execution_copy_rewrite": rewrite_info,
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
        "no_full_topup_to_5_executed_by_this_script": True,
        "no_heavy_lane_executed_by_this_script": True,
        "no_extreme_lane_executed_by_this_script": True,
        "no_committed_shard_modified_by_this_script": (
            source_md5_after_run == source_md5_before
        ),
        "no_raw_openml_payloads_staged_by_this_script": True,
        "no_execution_sqlite_staged_by_this_script": True,
        "operator_review_required_before_scaling": True,
        "next_recommended_step": (
            "After Commit 47 is green and operator-reviewed, plan or run a "
            "slightly larger Stage-3 pilot in Commit 48 (e.g. replica_002 "
            "across all 10 standard-lane shards, or shard_00 standard + a "
            "selected heavy probe). Do NOT scale directly to the full "
            "topup_to_5 tier without reviewing the Commit 47 summary."
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
    extra.append("## stage 3 / top-up pilot summary (Commit 47)\n")
    extra.append(f"- run_id: `{summary['run_id']}`")
    extra.append(f"- batch_id: `{summary['batch_id']}`")
    extra.append(f"- stage: `{summary['stage']}`")
    extra.append(f"- topup_tier: `{summary['topup_tier']}`")
    extra.append(f"- replica: **{summary['replica']}** (source template replica = "
                 f"{summary['source_template_replica']})")
    extra.append(f"- lane: `{summary['lane']}`")
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(f"- policy_version: `{str(summary['policy_version'])[:16]}`")
    extra.append(
        f"- policy_version_pinned: `{str(summary['policy_version_pinned'])[:16]}`"
    )
    extra.append(f"- signoff_path: `{summary['signoff_path']}`")
    extra.append(f"- signoff_sha256: `{str(summary['signoff_sha256'])[:16]}`")
    extra.append(
        f"- stage3_topup_plan_summary_path: "
        f"`{summary['stage3_topup_plan_summary_path']}`"
    )
    extra.append(
        f"- stage3_topup_plan_summary_sha256: "
        f"`{str(summary['stage3_topup_plan_summary_sha256'])[:16]}`"
    )
    extra.append(
        f"- execution_sqlite_sha256_after_rewrite: "
        f"`{str(summary['execution_sqlite_sha256_after_rewrite'])[:16]}`\n"
    )

    extra.append(
        f"- n_jobs_total (shard_00): {summary['n_jobs_total']}"
    )
    extra.append(
        f"- expected runnable standard-lane canary cells: "
        f"**{summary['expected_standard_canary_cells']}**"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"success: **{summary['n_jobs_success']}**, "
        f"deferred_heavy: **{summary['n_jobs_deferred_heavy']}**, "
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

    extra.append("### Per-shard status (single-shard pilot)\n")
    extra.append(
        "| shard | total | success | failed | failed_to | pending | "
        "skipped | def_heavy | def_extreme | refused |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_status"]:
        extra.append(
            f"| `{sh['shard']}` | {sh['n_total']} | {sh['success']} | "
            f"{sh['failed']} | {sh['failed_timeout']} | {sh['pending']} | "
            f"{sh['skipped']} | {sh['deferred_heavy_lane']} | "
            f"{sh['deferred_extreme_lane']} | "
            f"{sh['refused_not_in_canary_set']} |"
        )
    extra.append("")

    extra.append("### Standard tasks executed (success only)\n")
    extra.append(
        f"{len(summary['standard_tasks_executed'])} tasks: "
        f"{summary['standard_tasks_executed']}\n"
    )
    extra.append("### Heavy tasks deferred in this shard\n")
    extra.append(
        f"{len(summary['heavy_tasks_deferred'])} tasks: "
        f"{summary['heavy_tasks_deferred']}\n"
    )
    extra.append("### Extreme tasks deferred in this shard\n")
    extra.append(
        f"{len(summary['extreme_tasks_deferred'])} tasks: "
        f"{summary['extreme_tasks_deferred']}\n"
    )
    extra.append("### Non-canary methods refused in this shard\n")
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
        and summary["no_heavy_lane_executed_by_this_script"]
        and summary["no_extreme_lane_executed_by_this_script"]
    )
    if invariants_ok:
        extra.append("### stage 3 pilot verdict: **GATE PASS — operator review required**\n")
        extra.append(
            "Pilot finished cleanly: every standard-lane canary cell on "
            "shard_00 reached a terminal status, the committed source "
            "shard is byte-identical to its pre-pilot MD5, no heavy / "
            "extreme cell ran, and no full topup_to_5 dispatch was "
            "triggered. Commit 48 may plan or run a slightly larger "
            "Stage-3 pilot, but **not** the full topup_to_5 tier without "
            "explicit operator sign-off.\n"
        )
    else:
        extra.append("### stage 3 pilot verdict: **NOT GREEN**\n")
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
    parser.add_argument("--shard-name", default=DEFAULT_SHARD_NAME)
    parser.add_argument(
        "--signoff-file", type=Path, default=DEFAULT_SIGNOFF_FILE,
    )
    parser.add_argument(
        "--topup-plan-summary", type=Path,
        default=DEFAULT_TOPUP_PLAN_SUMMARY,
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
             "row of the execution copy. The default 'stage1_topup_to_005' "
             "is the closest existing convention for replica = 2 "
             "(the topup_to_5 tier).",
    )
    parser.add_argument(
        "--target-replica", type=int, default=TARGET_REPLICA,
    )
    parser.add_argument(
        "--hard-cap-hours", type=float, default=DEFAULT_HARD_CAP_HOURS,
        help="Per-shard subprocess timeout ceiling.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the pre-run plan, the gate check results, and exit "
             "with execution_status='planned_not_executed'. Does not "
             "copy or rewrite any shard.",
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
        except PilotRefusalError as exc:
            print(f"GATE REFUSAL: {exc}", file=sys.stderr)
            return 3
        g = RuntimeGuardrails.load(
            yaml_path=args.guardrails_yaml, csv_path=args.policy_csv,
        )
        shard = args.shards_dir / args.shard_name
        plan = build_pre_run_plan(shard, g)
        print(json.dumps({
            "execution_status": "planned_not_executed",
            "shards_dir": str(args.shards_dir),
            "shard_name": args.shard_name,
            "signoff_file": str(args.signoff_file),
            "topup_plan_summary": str(args.topup_plan_summary),
            "policy_version": live_policy_version,
            "policy_version_pinned": PINNED_POLICY_VERSION,
            "signoff_status": signoff.get("signoff_status"),
            "signoff_type": signoff.get("signoff_type"),
            "topup_plan_execution_status": plan_record.get(
                "execution_status",
            ),
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
            "pre_run_plan": plan,
            "expected_standard_canary_cells": (
                EXPECTED_STANDARD_CANARY_CELLS_SHARD_00
            ),
        }, indent=2))
        return 0

    try:
        summary = run_pilot(
            shards_dir=args.shards_dir,
            shard_name=args.shard_name,
            signoff_file=args.signoff_file,
            topup_plan_summary=args.topup_plan_summary,
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
            hard_cap_hours=args.hard_cap_hours,
            skip_train=args.skip_train,
        )
    except PilotRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"executed={summary['n_jobs_executed']}/"
        f"{summary['expected_standard_canary_cells']}  "
        f"success={summary['n_jobs_success']}  "
        f"deferred_heavy={summary['n_jobs_deferred_heavy']}  "
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
