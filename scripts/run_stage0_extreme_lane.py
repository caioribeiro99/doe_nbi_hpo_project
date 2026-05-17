#!/usr/bin/env python
"""Plan (and in a later commit, run) the extreme-lane pass of stage 0.

**Commit 42 ships this script in PLANNING-ONLY mode.** Without the
explicit ``--execute-extreme-lane`` flag, the script refuses to copy
any shard, refuses to contact OpenML, and refuses to invoke
``cc18_runner.py``. It only reads the committed metadata, classifies
the stage-0 rows under the heavy-task policy, and publishes a
dry-run plan under
``experiments/_stage_runs/stage0_extreme_lane_plan_latest_summary.{json,md}``
with ``execution_status = "planned_not_executed"``.

A future Commit 43 may invoke this same script with
``--execute-extreme-lane`` to actually run the 24 extreme-lane
canary cells (2 tasks × 4 canary methods × 3 algorithms). Before
doing that, read ``docs/EXTREME_LANE_PLAN.md`` end-to-end:
Devnagari-Script alone contributed ~92 % of batch_03's runtime, and
the worst single cell (``167121 / doe_rsm_vrf_true_nbi / xgboost``)
ran 11,091 s — over the 14,400 s extreme cell timeout's tolerance
band when you factor in OS jitter.

Pre-flight refusals (Commit 42 dry-run AND Commit 43 execution)
---------------------------------------------------------------
- ``experiments/_stage_runs/stage0_standard_lane_latest_summary.json``
  must be green: ``n_jobs_executed >= 684``, ``n_jobs_failed == 0``,
  ``n_jobs_failed_timeout == 0``, ``n_jobs_failed_other == 0``,
  ``n_jobs_pending_after == 0``, ``n_jobs_running_after == 0``,
  ``source_shards_unchanged: true``, ``stage3_signoff_present: false``.
- ``experiments/_stage_runs/stage0_heavy_lane_latest_summary.json``
  must be green by the same criteria with ``n_jobs_executed >= 156``.
- Both summaries must record the **same** ``policy_version`` as the
  live ``heavy_task_policy.csv``, pinned to:
  ``47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36``.
- Both summaries must be ≤ 7 days old by default (``--max-age-days``
  overrides for a known-safe stale gate).
- ``stage3_signoff.json`` must NOT exist.

What the runner does NOT do in Commit 42
----------------------------------------
- run any canary cells (executable mode requires
  ``--execute-extreme-lane``, and Commit 42 never passes it);
- copy any committed shard into ``runs/cc18/`` (the run-dir is
  materialized only in execution mode);
- contact OpenML (no payloads loaded; no cache writes);
- create ``stage3_signoff.json``;
- regenerate ``heavy_task_policy.csv``, the policy report, or
  ``runtime_guardrails.yaml``;
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
DEFAULT_HEAVY_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs/stage0_heavy_lane_latest_summary.json"
)
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/stage0_extreme_lane"
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
DRY_RUN_ID = "stage0_extreme_lane_plan_latest"
RUN_ID = "stage0_extreme_lane_latest"
BATCH_ID_PLAN = "stage0_extreme_lane_plan"
BATCH_ID_EXEC = "stage0_extreme_lane"
LANE = "extreme"
EXPECTED_EXTREME_CANARY_CELLS = 24  # 2 extreme × 4 canary × 3 algos
LANE_SUMMARY_MAX_AGE_DAYS = 7
DEFAULT_REQUESTED_MAX_EVALUATIONS = 5
DEFAULT_N_FOLDS = 2

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
PLAN_DOC = "docs/EXTREME_LANE_PLAN.md"


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when the standard or heavy gate rejects the run."""


def _summary_age_days(summary_path: Path) -> float:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ts = payload.get("exported_at") or payload.get("run_timestamp") or ""
    try:
        run_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise GateRefusalError(
            f"summary has unparsable timestamp={ts!r}: {exc}"
        ) from exc
    return (datetime.now(timezone.utc) - run_dt).total_seconds() / 86400.0


def _verify_lane_summary(
    summary_path: Path,
    *, lane_name: str, expected_executed: int,
    max_age_days: float,
) -> dict:
    if not summary_path.exists():
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary not found at "
            f"{summary_path}; run scripts/run_stage0_{lane_name}_lane.py "
            "first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
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
    policy_version = payload.get("policy_version")

    if n_executed < expected_executed:
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary executed only "
            f"{n_executed} of the expected {expected_executed} cells; "
            "refusing to promote to the extreme lane."
        )
    if n_failed != 0 or failed_timeout != 0 or failed_other != 0:
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary is not green: "
            f"n_failed={n_failed}, failed_timeout={failed_timeout}, "
            f"failed_other={failed_other}"
        )
    if pending != 0 or running != 0:
        raise GateRefusalError(
            f"stage0 {lane_name}-lane has unfinished work: "
            f"pending={pending}, running={running}"
        )
    if not unchanged:
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary reports "
            "source_shards_unchanged=False; investigate."
        )
    if signoff:
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary reports "
            "stage3_signoff_present=True; refusing to run extreme "
            "lane in pre-signoff territory."
        )
    age = _summary_age_days(summary_path)
    if age > float(max_age_days):
        raise GateRefusalError(
            f"stage0 {lane_name}-lane summary is {age:.2f} days old "
            f"(>{max_age_days:.0f}d); re-run or pass --max-age-days."
        )
    return {
        "lane": lane_name,
        "n_jobs_executed": n_executed,
        "n_jobs_failed": n_failed,
        "n_jobs_failed_timeout": failed_timeout,
        "n_jobs_failed_other": failed_other,
        "n_jobs_pending_after": pending,
        "n_jobs_running_after": running,
        "source_shards_unchanged": unchanged,
        "stage3_signoff_present": signoff,
        "exported_at": payload.get("exported_at"),
        "age_days": age,
        "source_git_sha": payload.get("source_git_sha"),
        "run_id": payload.get("run_id"),
        "policy_version": policy_version,
    }


def verify_prior_stages(
    *,
    standard_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    heavy_summary: Path = DEFAULT_HEAVY_LANE_SUMMARY,
    live_policy_version: str | None = None,
    max_age_days: float = LANE_SUMMARY_MAX_AGE_DAYS,
) -> tuple[dict, dict]:
    """Verify standard + heavy lanes are green AND share the policy
    version. Returns the two gate dicts."""
    std_gate = _verify_lane_summary(
        standard_summary, lane_name="standard",
        expected_executed=684, max_age_days=max_age_days,
    )
    hvy_gate = _verify_lane_summary(
        heavy_summary, lane_name="heavy",
        expected_executed=156, max_age_days=max_age_days,
    )
    pv_std = std_gate.get("policy_version")
    pv_hvy = hvy_gate.get("policy_version")
    if pv_std != pv_hvy:
        raise GateRefusalError(
            f"standard-lane policy_version={pv_std} != heavy-lane "
            f"policy_version={pv_hvy}; the prior stage-0 passes do "
            "not share a policy."
        )
    if live_policy_version is not None and pv_std != live_policy_version:
        raise GateRefusalError(
            f"prior-stage policy_version={pv_std} != live "
            f"policy_version={live_policy_version}; mid-replica "
            "policy drift detected. Run the extreme lane with the "
            "same policy_version the earlier stages used, or restart "
            "the replica."
        )
    return std_gate, hvy_gate


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
    """Split shard rows into four disjoint buckets for the extreme
    lane planning / execution.

    Returns:
      ``runnable_extreme_canary``                — canary × extreme;
      ``skipped_standard_lane_already_completed`` — any standard row;
      ``skipped_heavy_lane_already_completed``    — any heavy row;
      ``refused_not_in_canary_set``               — extreme × non-canary.
    """
    buckets: dict[str, list[dict]] = {
        "runnable_extreme_canary": [],
        "skipped_standard_lane_already_completed": [],
        "skipped_heavy_lane_already_completed": [],
        "refused_not_in_canary_set": [],
    }
    for job_id, task_id, method, algorithm in rows:
        lane = guardrails.get_task_lane(task_id)
        entry = {
            "job_id": job_id, "openml_task_id": int(task_id),
            "method": method, "algorithm": algorithm, "lane": lane,
        }
        if lane == "standard":
            buckets["skipped_standard_lane_already_completed"].append(entry)
            continue
        if lane == "heavy":
            buckets["skipped_heavy_lane_already_completed"].append(entry)
            continue
        # lane == "extreme"
        if method not in CANARY_METHODS:
            buckets["refused_not_in_canary_set"].append(entry)
            continue
        buckets["runnable_extreme_canary"].append(entry)
    return buckets


# ---------------------------------------------------------------------------
# Pre-run plan
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shards: list[Path], guardrails: RuntimeGuardrails,
) -> dict:
    plan = {
        "n_source_shards": len(shards),
        "n_jobs_total": 0,
        "n_runnable_extreme_canary": 0,
        "n_skipped_standard_lane": 0,
        "n_skipped_heavy_lane": 0,
        "n_refused_not_in_canary_set": 0,
        "task_lane_counts_universe": Counter(),
        "method_counts": Counter(),
        "algorithm_counts": Counter(),
        "extreme_tasks_to_execute": set(),
        "standard_tasks_already_executed": set(),
        "heavy_tasks_already_executed": set(),
        "non_canary_methods_refused_on_extreme": set(),
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
            "runnable_extreme_canary": len(buckets["runnable_extreme_canary"]),
            "skipped_standard_lane_already_completed": len(
                buckets["skipped_standard_lane_already_completed"],
            ),
            "skipped_heavy_lane_already_completed": len(
                buckets["skipped_heavy_lane_already_completed"],
            ),
            "refused_not_in_canary_set": len(
                buckets["refused_not_in_canary_set"],
            ),
        }
        plan["per_shard"].append(per_sh)
        plan["n_jobs_total"] += len(rows)
        plan["n_runnable_extreme_canary"] += per_sh["runnable_extreme_canary"]
        plan["n_skipped_standard_lane"] += per_sh[
            "skipped_standard_lane_already_completed"
        ]
        plan["n_skipped_heavy_lane"] += per_sh[
            "skipped_heavy_lane_already_completed"
        ]
        plan["n_refused_not_in_canary_set"] += per_sh[
            "refused_not_in_canary_set"
        ]
        for tid, m, a in rows:
            tid_int = int(tid)
            lane = guardrails.get_task_lane(tid_int)
            seen_tasks[tid_int] = lane
            plan["method_counts"][m] += 1
            plan["algorithm_counts"][a] += 1
            if lane == "extreme":
                if m in CANARY_METHODS:
                    plan["extreme_tasks_to_execute"].add(tid_int)
                else:
                    plan["non_canary_methods_refused_on_extreme"].add(m)
            elif lane == "standard":
                plan["standard_tasks_already_executed"].add(tid_int)
            elif lane == "heavy":
                plan["heavy_tasks_already_executed"].add(tid_int)
    plan["task_lane_counts_universe"] = Counter(seen_tasks.values())
    plan["extreme_tasks_to_execute"] = sorted(plan["extreme_tasks_to_execute"])
    plan["standard_tasks_already_executed"] = sorted(
        plan["standard_tasks_already_executed"],
    )
    plan["heavy_tasks_already_executed"] = sorted(
        plan["heavy_tasks_already_executed"],
    )
    plan["non_canary_methods_refused_on_extreme"] = sorted(
        plan["non_canary_methods_refused_on_extreme"],
    )
    plan["method_counts"] = dict(plan["method_counts"])
    plan["algorithm_counts"] = dict(plan["algorithm_counts"])
    plan["task_lane_counts_universe"] = dict(plan["task_lane_counts_universe"])
    return plan


# ---------------------------------------------------------------------------
# Top-level planning entry point
# ---------------------------------------------------------------------------


def plan_stage0_extreme_lane(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    heavy_lane_summary: Path = DEFAULT_HEAVY_LANE_SUMMARY,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    max_age_days: float = LANE_SUMMARY_MAX_AGE_DAYS,
    run_id: str = DRY_RUN_ID,
    expected_extreme_canary_cells: int = EXPECTED_EXTREME_CANARY_CELLS,
    enforce_pinned_policy_version: bool = True,
    write_summary: bool = True,
) -> dict:
    """Inventory-only planning entry point. Refuses to mutate any
    committed artifact and refuses to contact OpenML."""
    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )
    live_policy_version = _sha256(policy_csv)
    if (
        enforce_pinned_policy_version
        and live_policy_version != PINNED_POLICY_VERSION
    ):
        raise GateRefusalError(
            f"refusing: live policy CSV hashes "
            f"{live_policy_version} but Commit 42 pins "
            f"{PINNED_POLICY_VERSION} (= Commit 40 / 41 "
            "policy_version). Recalibrating the policy mid-replica "
            "is forbidden."
        )

    std_gate, hvy_gate = verify_prior_stages(
        standard_summary=standard_lane_summary,
        heavy_summary=heavy_lane_summary,
        live_policy_version=live_policy_version,
        max_age_days=max_age_days,
    )

    if SIGNOFF_FILE.exists():
        raise GateRefusalError(
            f"refusing: stage-3 sign-off file already exists at "
            f"{SIGNOFF_FILE}"
        )

    source_shards = sorted(shards_dir.glob("shard_*.sqlite"))
    if len(source_shards) != 10:
        raise GateRefusalError(
            f"expected 10 stage-0 shards under {shards_dir}, "
            f"found {len(source_shards)}"
        )
    plan = build_pre_run_plan(source_shards, guardrails)
    if plan["n_runnable_extreme_canary"] != expected_extreme_canary_cells:
        raise GateRefusalError(
            f"pre-run plan inconsistency: expected "
            f"{expected_extreme_canary_cells} extreme-canary cells "
            f"but found {plan['n_runnable_extreme_canary']}. "
            "Verify heavy_task_policy.csv classification."
        )

    md5_source = {sh.name: _md5(sh) for sh in source_shards}

    # ETA anchored on batch_03 (Commit 37) observed runtimes.
    runtime_anchor = {
        "batch_03_devnagari_doe_rsm_xgboost_seconds": 11090.6,
        "batch_03_devnagari_doe_rsm_catboost_seconds": 10575.0,
        "batch_03_devnagari_tpe_optuna_catboost_seconds": 7944.0,
        "batch_03_devnagari_random_search_catboost_seconds": 7646.9,
        "batch_03_devnagari_total_devnagari_canary_seconds": (
            11090.6 + 10575.0 + 7944.0 + 7646.9
            + 6524.3 + 5435.7 + 1594.8 + 1505.0
            + 1500.0 + 1500.0 + 200.0 + 200.0
        ),  # ~55 ks; the remaining 4 are estimates.
        "extreme_letter_max_observed_seconds_batch_03": 26.3,
    }
    eta = {
        "expected_total_runner_cpu_seconds": float(
            runtime_anchor[
                "batch_03_devnagari_total_devnagari_canary_seconds"
            ]
        )
        + runtime_anchor["extreme_letter_max_observed_seconds_batch_03"] * 12,
        "expected_wall_clock_hours_dedicated_mac": 15.7,
        "expected_wall_clock_hours_local_laptop": "DO NOT RUN — "
            "Devnagari catboost OOM risk under thermal limits",
    }

    summary = {
        "schema_version": 1,
        "batch_id": BATCH_ID_PLAN,
        "lane": LANE,
        "stage": CANARY_STAGE,
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "execution_status": "planned_not_executed",
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "policy_version": live_policy_version,
        "policy_version_pinned": PINNED_POLICY_VERSION,
        "policy_csv_path": _safe_rel(policy_csv),
        "guardrails_yaml_path": _safe_rel(guardrails_yaml),
        "plan_doc": PLAN_DOC,
        "n_jobs_total": plan["n_jobs_total"],
        "expected_extreme_canary_cells": expected_extreme_canary_cells,
        "n_runnable_extreme_canary": plan["n_runnable_extreme_canary"],
        "n_skipped_standard_lane": plan["n_skipped_standard_lane"],
        "n_skipped_heavy_lane": plan["n_skipped_heavy_lane"],
        "n_refused_not_in_canary_set": plan["n_refused_not_in_canary_set"],
        "task_lane_counts_universe": plan["task_lane_counts_universe"],
        "extreme_tasks_to_execute": plan["extreme_tasks_to_execute"],
        "standard_tasks_already_executed": plan[
            "standard_tasks_already_executed"
        ],
        "heavy_tasks_already_executed": plan["heavy_tasks_already_executed"],
        "non_canary_methods_refused_on_extreme": plan[
            "non_canary_methods_refused_on_extreme"
        ],
        "method_counts_universe": plan["method_counts"],
        "algorithm_counts_universe": plan["algorithm_counts"],
        "per_shard_plan": plan["per_shard"],
        "standard_lane_gate": std_gate,
        "heavy_lane_gate": hvy_gate,
        "source_shard_md5": md5_source,
        "openml_payloads_loaded": False,
        "openml_payloads_committed": False,
        "execution_shards_committed": False,
        "execution_shards_created": False,
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "stage3_signoff_path": _safe_rel(SIGNOFF_FILE),
        "runtime_anchor_batch_03": runtime_anchor,
        "eta": eta,
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "capability_audit": _capability_audit_summary(),
        "promotion_criteria": [
            "standard / heavy / extreme stage-run summaries all green",
            "all three summaries share the same policy_version",
            "all three carry source_shards_unchanged=True",
            "all three carry stage3_signoff_present=False",
            "extreme summary execution_status == 'executed' "
            "(planned_not_executed does not count)",
        ],
        "next_step": (
            "After human review of docs/EXTREME_LANE_PLAN.md, Commit 43 "
            "may invoke this same script with --execute-extreme-lane. "
            "Commit 42 must not."
        ),
    }

    if write_summary:
        stage_runs_dir.mkdir(parents=True, exist_ok=True)
        json_p = stage_runs_dir / f"{run_id}_summary.json"
        md_p = stage_runs_dir / f"{run_id}_summary.md"
        json_p.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        md_p.write_text(_render_md(summary), encoding="utf-8")

    return summary


def _render_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"# stage0 extreme-lane plan — {summary['run_id']}\n"
    )
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- batch_id: `{summary['batch_id']}`")
    lines.append(f"- lane: `{summary['lane']}`")
    lines.append(f"- stage: `{summary['stage']}`")
    lines.append(f"- exported_at: `{summary['exported_at']}`")
    lines.append(
        f"- **execution_status: `{summary['execution_status']}`**"
    )
    lines.append(f"- plan_doc: `{summary['plan_doc']}`")
    lines.append(
        f"- policy_version: `{str(summary['policy_version'])[:16]}` "
        f"(pinned: `{str(summary['policy_version_pinned'])[:16]}`)\n"
    )

    lines.append("## Universe of stage-0 rows\n")
    lines.append(f"- total: {summary['n_jobs_total']}")
    lines.append(
        f"- expected extreme-canary cells: "
        f"{summary['expected_extreme_canary_cells']}"
    )
    lines.append(
        f"- runnable extreme canary (this lane would execute): "
        f"**{summary['n_runnable_extreme_canary']}**"
    )
    lines.append(
        f"- skipped standard (already completed in Commit 40): "
        f"{summary['n_skipped_standard_lane']}"
    )
    lines.append(
        f"- skipped heavy (already completed in Commit 41): "
        f"{summary['n_skipped_heavy_lane']}"
    )
    lines.append(
        f"- refused (extreme × non-canary methods): "
        f"{summary['n_refused_not_in_canary_set']}\n"
    )
    lines.append(
        f"- extreme tasks to execute: "
        f"{summary['extreme_tasks_to_execute']}"
    )
    lines.append(
        f"- non_canary_methods_refused_on_extreme: "
        f"{summary['non_canary_methods_refused_on_extreme']}\n"
    )

    g = summary["standard_lane_gate"]
    lines.append("## stage0 standard-lane pre-flight\n")
    lines.append(f"- exported_at: `{g.get('exported_at')}`")
    lines.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
    lines.append(
        f"- n_executed={g.get('n_jobs_executed')}, "
        f"failed={g.get('n_jobs_failed')}, "
        f"pending={g.get('n_jobs_pending_after')}"
    )
    lines.append(
        f"- policy_version: `{str(g.get('policy_version'))[:16]}`\n"
    )
    g = summary["heavy_lane_gate"]
    lines.append("## stage0 heavy-lane pre-flight\n")
    lines.append(f"- exported_at: `{g.get('exported_at')}`")
    lines.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
    lines.append(
        f"- n_executed={g.get('n_jobs_executed')}, "
        f"failed={g.get('n_jobs_failed')}, "
        f"pending={g.get('n_jobs_pending_after')}"
    )
    lines.append(
        f"- policy_version: `{str(g.get('policy_version'))[:16]}`\n"
    )

    lines.append("## Per-shard plan\n")
    lines.append(
        "| shard | total | runnable_extreme | skip_std | skip_heavy | refused |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_plan"]:
        lines.append(
            f"| `{sh['shard']}` | {sh['n_jobs']} | "
            f"{sh['runnable_extreme_canary']} | "
            f"{sh['skipped_standard_lane_already_completed']} | "
            f"{sh['skipped_heavy_lane_already_completed']} | "
            f"{sh['refused_not_in_canary_set']} |"
        )
    lines.append("")

    lines.append("## Source-shard MD5 (read-only)\n")
    lines.append("| shard | md5 |")
    lines.append("|---|---|")
    for k, v in sorted(summary["source_shard_md5"].items()):
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

    eta = summary["eta"]
    lines.append("## Runtime ETA (anchored on batch_03)\n")
    lines.append(
        f"- expected total runner CPU: "
        f"~{eta['expected_total_runner_cpu_seconds']:.0f} s"
    )
    lines.append(
        f"- dedicated Mac wall-clock estimate: "
        f"~{eta['expected_wall_clock_hours_dedicated_mac']:.1f} h"
    )
    lines.append(
        f"- local laptop: {eta['expected_wall_clock_hours_local_laptop']}\n"
    )

    lines.append("## Promotion criteria for stage 0 replica 1\n")
    for c in summary["promotion_criteria"]:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## What happens next\n")
    lines.append(summary["next_step"])
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution stub (Commit 43)
# ---------------------------------------------------------------------------


def execute_stage0_extreme_lane(*args, **kwargs):  # pragma: no cover
    """Execution path. Intentionally a stub in Commit 42 — actual
    execution wiring lands in Commit 43 once the plan in
    ``docs/EXTREME_LANE_PLAN.md`` has been reviewed."""
    raise NotImplementedError(
        "Commit 42 ships only the planning entry point. Real execution "
        "lands in Commit 43; see docs/EXTREME_LANE_PLAN.md first."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument(
        "--stage-runs-dir", type=Path, default=DEFAULT_STAGE_RUNS_DIR,
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
        "--policy-csv", type=Path, default=DEFAULT_POLICY_CSV,
    )
    parser.add_argument(
        "--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS_YAML,
    )
    parser.add_argument(
        "--max-age-days", type=float, default=LANE_SUMMARY_MAX_AGE_DAYS,
        help="Reject prior-stage gates when older than this; default 7d.",
    )
    parser.add_argument(
        "--run-id", default=DRY_RUN_ID,
        help="Run id; defaults to the planning run id so the dry-run "
             "summary lands at "
             "experiments/_stage_runs/<run_id>_summary.{json,md}.",
    )
    parser.add_argument(
        "--include-extreme-tasks", action="store_true",
        help="Accepted for symmetry with other lane runners. The "
             "extreme lane's whole purpose is the extreme tasks, so "
             "this flag is always implicitly true. It does NOT enable "
             "execution; that requires --execute-extreme-lane.",
    )
    parser.add_argument(
        "--execute-extreme-lane", action="store_true",
        help="UNLOCK execution. Commit 42 must NOT pass this flag; "
             "without it the script runs in planning-only mode and "
             "publishes a dry-run summary. Real execution lands in a "
             "later commit after docs/EXTREME_LANE_PLAN.md is reviewed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Same as the default mode (planning-only). Kept for "
             "symmetry with --dry-run on the other lane runners.",
    )
    parser.add_argument(
        "--allow-policy-drift", action="store_true",
        help="Skip the PINNED_POLICY_VERSION check. Off by default.",
    )
    args = parser.parse_args(argv)

    try:
        if not args.execute_extreme_lane:
            # Commit 42 path: planning only.
            t0 = time.perf_counter()
            summary = plan_stage0_extreme_lane(
                shards_dir=args.shards_dir,
                stage_runs_dir=args.stage_runs_dir,
                standard_lane_summary=args.standard_lane_summary,
                heavy_lane_summary=args.heavy_lane_summary,
                policy_csv=args.policy_csv,
                guardrails_yaml=args.guardrails_yaml,
                max_age_days=args.max_age_days,
                run_id=args.run_id,
                enforce_pinned_policy_version=not args.allow_policy_drift,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"PLAN: runnable_extreme_canary="
                f"{summary['n_runnable_extreme_canary']}/"
                f"{summary['expected_extreme_canary_cells']}  "
                f"skip_standard={summary['n_skipped_standard_lane']}  "
                f"skip_heavy={summary['n_skipped_heavy_lane']}  "
                f"refused={summary['n_refused_not_in_canary_set']}  "
                f"(planning elapsed {elapsed:.2f}s)"
            )
            print(
                f"json: {args.stage_runs_dir / (args.run_id + '_summary.json')}"
            )
            print(
                f"md:   {args.stage_runs_dir / (args.run_id + '_summary.md')}"
            )
            print(
                "Execution disabled. Pass --execute-extreme-lane in a "
                "future commit (Commit 43+) AFTER reviewing "
                "docs/EXTREME_LANE_PLAN.md."
            )
            return 0

        # If we get here, the operator has explicitly opted into
        # execution. Commit 42 never reaches this branch; Commit 43
        # will implement it.
        execute_stage0_extreme_lane(args)
        return 0
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        return 2


# Surface the helpers that downstream commits + tests reach for.
__all__ = [
    "BATCH_ID_PLAN",
    "DRY_RUN_ID",
    "EXPECTED_EXTREME_CANARY_CELLS",
    "GateRefusalError",
    "PINNED_POLICY_VERSION",
    "build_pre_run_plan",
    "classify_rows",
    "execute_stage0_extreme_lane",
    "main",
    "plan_stage0_extreme_lane",
    "verify_prior_stages",
]


if __name__ == "__main__":
    sys.exit(main())


# Silence flake about `defaultdict` import since we only use it
# inside the future execution path. Keeping the import out of the
# `if False:` block makes the future Commit 43 edit smaller.
_ = defaultdict
