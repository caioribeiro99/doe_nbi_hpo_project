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
    signoff_file: Path | None = None,
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

    effective_signoff = signoff_file if signoff_file is not None else SIGNOFF_FILE
    if effective_signoff.exists():
        raise GateRefusalError(
            f"refusing: stage-3 sign-off file already exists at "
            f"{effective_signoff}"
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
        "stage3_signoff_present": effective_signoff.exists(),
        "stage3_signoff_path": _safe_rel(effective_signoff),
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
# Execution path (Commit 43)
# ---------------------------------------------------------------------------


PLAN_SUMMARY_PATH = (
    DEFAULT_STAGE_RUNS_DIR
    / "stage0_extreme_lane_plan_latest_summary.json"
)


def _verify_plan_summary(
    plan_summary_path: Path = PLAN_SUMMARY_PATH,
    *, max_age_days: float = LANE_SUMMARY_MAX_AGE_DAYS,
    expected_runnable: int = EXPECTED_EXTREME_CANARY_CELLS,
    expected_tasks: tuple[int, ...] = (6, 167121),
    expected_policy_version: str = PINNED_POLICY_VERSION,
) -> dict:
    """Refuse execution if the Commit 42 dry-run plan is missing,
    stale, or inconsistent with what Commit 43 expects."""
    if not plan_summary_path.exists():
        raise GateRefusalError(
            f"Commit 42 plan summary not found at "
            f"{plan_summary_path}; run scripts/run_stage0_extreme_lane.py "
            "in dry-run mode first."
        )
    payload = json.loads(plan_summary_path.read_text(encoding="utf-8"))
    status = payload.get("execution_status")
    if status != "planned_not_executed":
        raise GateRefusalError(
            f"Commit 42 plan summary has execution_status={status!r}; "
            "expected 'planned_not_executed'. The extreme lane has "
            "already been executed (or a stale summary is on disk)."
        )
    runnable = int(payload.get("n_runnable_extreme_canary", -1))
    if runnable != expected_runnable:
        raise GateRefusalError(
            f"Commit 42 plan summary reports "
            f"n_runnable_extreme_canary={runnable}; expected "
            f"{expected_runnable}. Policy or shard composition "
            "drifted between planning and execution."
        )
    tasks = tuple(payload.get("extreme_tasks_to_execute", ()))
    if tuple(sorted(tasks)) != tuple(sorted(expected_tasks)):
        raise GateRefusalError(
            f"Commit 42 plan summary reports "
            f"extreme_tasks_to_execute={list(tasks)}; expected "
            f"{list(expected_tasks)}."
        )
    pv = payload.get("policy_version")
    if pv != expected_policy_version:
        raise GateRefusalError(
            f"Commit 42 plan summary recorded policy_version={pv}; "
            f"expected pinned {expected_policy_version}."
        )
    age = _summary_age_days(plan_summary_path)
    if age > float(max_age_days):
        raise GateRefusalError(
            f"Commit 42 plan summary is {age:.2f} days old "
            f"(>{max_age_days:.0f}d); re-publish the plan first."
        )
    if payload.get("stage3_signoff_present") is True:
        raise GateRefusalError(
            "Commit 42 plan summary reports stage3_signoff_present="
            "True; refusing to run extreme lane in pre-signoff "
            "territory."
        )
    return {
        "plan_summary_path": str(plan_summary_path),
        "execution_status": status,
        "runnable_extreme_canary": runnable,
        "extreme_tasks_to_execute": list(tasks),
        "policy_version": pv,
        "age_days": age,
        "exported_at": payload.get("exported_at"),
    }


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
            "stdout_tail": (
                (exc.stdout or b"").decode("utf-8", errors="replace")[-2000:]
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")[-2000:]
            ),
            "stderr_tail": (
                (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")[-2000:]
            ),
            "timed_out": True,
            "runtime_seconds": time.perf_counter() - t0,
            "cmd": cmd,
        }


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


def execute_stage0_extreme_lane(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    heavy_lane_summary: Path = DEFAULT_HEAVY_LANE_SUMMARY,
    plan_summary_path: Path = PLAN_SUMMARY_PATH,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    max_age_days: float = LANE_SUMMARY_MAX_AGE_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
    run_id: str = RUN_ID,
    hard_cap_hours_per_shard: float = 16.0,
    skip_train: bool = False,
    force_run_dir: bool = True,
    expected_extreme_canary_cells: int = EXPECTED_EXTREME_CANARY_CELLS,
    enforce_pinned_policy_version: bool = True,
    signoff_file: Path | None = None,
) -> dict:
    """Real execution of the stage-0 extreme lane.

    Mirrors the heavy-lane runner but uses the policy's
    ``extreme.stage0_max_evaluations`` (default 1 per
    ``runtime_guardrails.yaml``) and ``extreme.timeout_seconds_per_cell``
    (default 14,400 s).
    """
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

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
            f"{live_policy_version} but Commit 43 pins "
            f"{PINNED_POLICY_VERSION} (= Commit 40 / 41 / 42 "
            "policy_version). Recalibrating the policy mid-replica "
            "is forbidden."
        )

    std_gate, hvy_gate = verify_prior_stages(
        standard_summary=standard_lane_summary,
        heavy_summary=heavy_lane_summary,
        live_policy_version=live_policy_version,
        max_age_days=max_age_days,
    )
    plan_gate = _verify_plan_summary(
        plan_summary_path=plan_summary_path,
        max_age_days=max_age_days,
        expected_runnable=expected_extreme_canary_cells,
        expected_policy_version=live_policy_version,
    )

    effective_signoff = signoff_file if signoff_file is not None else SIGNOFF_FILE
    if effective_signoff.exists():
        raise GateRefusalError(
            f"refusing: stage-3 sign-off file already exists at "
            f"{effective_signoff}"
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
            f"but found {plan['n_runnable_extreme_canary']}."
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
    #    cc18_runner with the extreme lane's stage-0 budget.
    md5_before = {sh.name: _md5(sh) for sh in source_shards}
    runner_invocations: list[dict] = []
    cells_total_before_run = 0
    cells_runnable_per_shard: dict[str, int] = {}
    lane_spec = guardrails.get_lane_spec(LANE)
    # extreme.stage0_max_evaluations is the YAML default (1) unless the
    # policy CSV overrides per-task. The runner explicitly uses
    # stage0_max_evaluations, NOT gate_max_evaluations.
    eff_max_evals_used: dict[int, int] = {}
    eff_timeout_used: dict[int, float] = {}
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
        cells_runnable_per_shard[exec_p.name] = len(
            buckets["runnable_extreme_canary"],
        )

        cx = sqlite3.connect(exec_p)
        try:
            _set_status_for_job_ids(
                cx,
                [e["job_id"] for e in
                 buckets["skipped_standard_lane_already_completed"]],
                status="skipped", last_error="deferred_standard_lane",
                assigned_worker="stage0_extreme_lane_policy",
            )
            _set_status_for_job_ids(
                cx,
                [e["job_id"] for e in
                 buckets["skipped_heavy_lane_already_completed"]],
                status="skipped", last_error="deferred_heavy_lane",
                assigned_worker="stage0_extreme_lane_policy",
            )
            _set_status_for_job_ids(
                cx,
                [e["job_id"] for e in
                 buckets["refused_not_in_canary_set"]],
                status="skipped", last_error="refused_not_in_canary_set",
                assigned_worker="stage0_extreme_lane_policy",
            )
        finally:
            cx.close()

        if skip_train or not buckets["runnable_extreme_canary"]:
            continue

        # Per-task budgets. The extreme lane has two tasks, and we use
        # the same stage0_max_evaluations / timeout for both since the
        # CSV does not override per-task (the lane defaults apply).
        # We invoke cc18_runner once for the whole shard's runnable
        # set; cc18_runner picks max-evaluations as one value, so we
        # use the lane spec's stage0_max_evaluations.
        eff_max = max(1, int(lane_spec.stage0_max_evaluations))
        eff_timeout = float(lane_spec.timeout_seconds_per_cell)
        for e in buckets["runnable_extreme_canary"]:
            tid = int(e["openml_task_id"])
            eff_max_evals_used[tid] = guardrails.get_effective_max_evaluations(
                tid, requested_max_evaluations=5, context="stage0",
            )
            eff_timeout_used[tid] = guardrails.get_timeout_seconds(tid)
        n_runnable = len(buckets["runnable_extreme_canary"])
        # Subprocess timeout = per-cell timeout * number of cells * 1.5
        # safety, capped at hard_cap_hours_per_shard.
        timeout_s = min(
            eff_timeout * n_runnable * 1.5,
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
            "max_evaluations_used": eff_max,
            "timeout_seconds_per_cell": eff_timeout,
            "subprocess_timeout_seconds": timeout_s,
            **{k: v for k, v in res.items() if k != "cmd"},
        })

        if res["timed_out"]:
            in_flight_ids = [
                e["job_id"] for e in buckets["runnable_extreme_canary"]
            ]
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
    n_deferred_heavy = int(
        last_error_counts.get("deferred_heavy_lane", 0),
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

    # Per-task breakdown (only for extreme tasks).
    by_task: dict[int, list[dict]] = defaultdict(list)
    for c in cells:
        if guardrails.get_task_lane(c["openml_task_id"]) == "extreme":
            by_task[c["openml_task_id"]].append(c)
    per_task_breakdown: list[dict] = []
    for tid in sorted(by_task.keys()):
        cells_t = by_task[tid]
        sc = Counter(c["status"] for c in cells_t)
        lec = Counter(
            (c.get("last_error") or "") for c in cells_t
            if c.get("last_error")
        )
        rts = [
            float(c["runtime_seconds"]) for c in cells_t
            if c.get("runtime_seconds") is not None
        ]
        per_task_breakdown.append({
            "openml_task_id": tid,
            "n_total": len(cells_t),
            "success": int(sc.get("success", 0)),
            "failed": int(sc.get("failed", 0)),
            "failed_timeout": int(lec.get("failed_timeout", 0)),
            "skipped": int(sc.get("skipped", 0)),
            "deferred_extreme_lane": int(
                lec.get("deferred_extreme_lane", 0),
            ),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
            "runtime_seconds_total": float(sum(rts)),
            "runtime_seconds_max": float(max(rts)) if rts else 0.0,
        })

    # Per-algorithm + per-method breakdowns over executed extreme rows.
    def _exec_breakdown(group_key: str) -> list[dict]:
        by_group: dict[str, list[dict]] = defaultdict(list)
        for c in cells:
            if guardrails.get_task_lane(c["openml_task_id"]) != "extreme":
                continue
            if c["method"] not in CANARY_METHODS:
                continue
            by_group[c[group_key]].append(c)
        out: list[dict] = []
        for key in sorted(by_group.keys()):
            sub = by_group[key]
            sc = Counter(c["status"] for c in sub)
            rts = [
                float(c["runtime_seconds"]) for c in sub
                if c.get("runtime_seconds") is not None
            ]
            out.append({
                group_key: key,
                "n_total": len(sub),
                "success": int(sc.get("success", 0)),
                "failed": int(sc.get("failed", 0)),
                "runtime_seconds_total": float(sum(rts)),
                "runtime_seconds_max": float(max(rts)) if rts else 0.0,
            })
        return out

    per_algorithm_breakdown = _exec_breakdown("algorithm")
    per_method_breakdown = _exec_breakdown("method")

    # Per-shard rollup.
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
            "deferred_heavy_lane": int(
                lec.get("deferred_heavy_lane", 0),
            ),
            "refused_not_in_canary_set": int(
                lec.get("refused_not_in_canary_set", 0),
            ),
        })

    extreme_tasks_executed = sorted({
        c["openml_task_id"] for c in cells
        if c["status"] == "success"
        and guardrails.get_task_lane(c["openml_task_id"]) == "extreme"
    })

    # 4. Publish via the protocol exporter.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID_EXEC,
    )

    summary.update({
        "batch_id": BATCH_ID_EXEC,
        "lane": LANE,
        "stage": CANARY_STAGE,
        "execution_status": "executed",
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "execution_shards": [_safe_rel(p) for p in exec_shards],
        "policy_version": live_policy_version,
        "policy_version_pinned": PINNED_POLICY_VERSION,
        "policy_csv_path": _safe_rel(policy_csv),
        "guardrails_yaml_path": _safe_rel(guardrails_yaml),
        "plan_doc": "docs/EXTREME_LANE_PLAN.md",
        "n_jobs_total": int(cells_total_before_run),
        "n_jobs_executed": n_success + n_failed_other,
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
        "extreme_tasks_executed": extreme_tasks_executed,
        "standard_tasks_already_executed": plan[
            "standard_tasks_already_executed"
        ],
        "heavy_tasks_already_executed": plan["heavy_tasks_already_executed"],
        "non_canary_methods_refused_on_extreme": plan[
            "non_canary_methods_refused_on_extreme"
        ],
        "expected_extreme_canary_cells": expected_extreme_canary_cells,
        "per_task_breakdown": per_task_breakdown,
        "per_algorithm_breakdown": per_algorithm_breakdown,
        "per_method_breakdown": per_method_breakdown,
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
        "max_evaluations_used_per_task": {
            str(k): v for k, v in sorted(eff_max_evals_used.items())
        },
        "timeout_seconds_per_cell_per_task": {
            str(k): v for k, v in sorted(eff_timeout_used.items())
        },
        "policy_max_evaluations_note": (
            "extreme lane executed with policy-defined "
            f"stage0_max_evaluations={lane_spec.stage0_max_evaluations}"
        ),
        "policy_timeout_note": (
            "extreme lane executed with policy-defined "
            f"timeout_seconds_per_cell={lane_spec.timeout_seconds_per_cell}"
        ),
        "openml_cache_root": _safe_rel(openml_cache_root),
        "openml_payloads_committed": False,
        "execution_shards_committed": False,
        "standard_lane_gate": std_gate,
        "heavy_lane_gate": hvy_gate,
        "plan_summary_gate": plan_gate,
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
        "stage3_signoff_present": effective_signoff.exists(),
        "stage3_signoff_note": (
            "stage0_replica_001 now has standard, heavy, and "
            "extreme lane summaries. stage3_signoff.json is "
            "intentionally absent until a later signoff commit."
        ),
    })

    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    _augment_exec_md(summary_md, summary)
    return summary


def _augment_exec_md(md_path: Path, summary: dict) -> None:
    extra: list[str] = []
    extra.append("\n---\n")
    extra.append("## stage0 extreme-lane execution summary\n")
    extra.append(f"- batch_id: `{summary['batch_id']}`")
    extra.append(f"- lane: `{summary['lane']}`")
    extra.append(
        f"- **execution_status: `{summary['execution_status']}`**"
    )
    extra.append(f"- n_source_shards: {summary['n_source_shards']}")
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(
        f"- policy_version: `{str(summary['policy_version'])[:16]}` "
        f"(pinned: `{str(summary['policy_version_pinned'])[:16]}`)"
    )
    extra.append(f"- plan_doc: `{summary['plan_doc']}`\n")
    extra.append(f"- n_jobs_total: {summary['n_jobs_total']}")
    extra.append(
        f"- expected extreme-canary cells: "
        f"{summary['expected_extreme_canary_cells']}"
    )
    extra.append(
        f"- executed: **{summary['n_jobs_executed']}**, "
        f"success: **{summary['n_jobs_success']}**, "
        f"deferred_standard: {summary['n_jobs_deferred_standard']}, "
        f"deferred_heavy: {summary['n_jobs_deferred_heavy']}, "
        f"refused: {summary['n_jobs_refused_non_canary']}, "
        f"failed_timeout: **{summary['n_jobs_failed_timeout']}**, "
        f"failed_other: **{summary['n_jobs_failed_other']}**, "
        f"pending_after: {summary['n_jobs_pending_after']}"
    )
    extra.append(
        f"- runtime (runner only): "
        f"{summary['runtime_seconds_runner_total']:.1f} s\n"
    )
    extra.append(f"- {summary['policy_max_evaluations_note']}")
    extra.append(f"- {summary['policy_timeout_note']}\n")

    extra.append(
        f"- task_lane_counts_universe: "
        f"{dict(summary['task_lane_counts_universe'])}"
    )
    extra.append(
        f"- extreme tasks executed: {summary['extreme_tasks_executed']}\n"
    )

    plan_gate = summary["plan_summary_gate"]
    extra.append("### Commit 42 plan pre-flight\n")
    extra.append(f"- plan_summary_path: `{plan_gate['plan_summary_path']}`")
    extra.append(f"- exported_at: `{plan_gate['exported_at']}`")
    extra.append(f"- age_days: {float(plan_gate['age_days']):.2f}")
    extra.append(
        f"- runnable_extreme_canary (planned): "
        f"{plan_gate['runnable_extreme_canary']}"
    )
    extra.append(
        f"- extreme_tasks_to_execute (planned): "
        f"{plan_gate['extreme_tasks_to_execute']}\n"
    )

    for lane_name, gate_key in (
        ("standard", "standard_lane_gate"),
        ("heavy", "heavy_lane_gate"),
    ):
        g = summary[gate_key]
        extra.append(f"### stage0 {lane_name}-lane pre-flight\n")
        extra.append(f"- exported_at: `{g.get('exported_at')}`")
        extra.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
        extra.append(
            f"- n_executed={g.get('n_jobs_executed')}, "
            f"failed={g.get('n_jobs_failed')}, "
            f"pending={g.get('n_jobs_pending_after')}"
        )
        extra.append(
            f"- policy_version: `{str(g.get('policy_version'))[:16]}`\n"
        )

    extra.append("### Per-task breakdown (extreme only)\n")
    extra.append(
        "| task_id | total | success | failed | failed_timeout | "
        "skipped | runtime_total_s | runtime_max_s |"
    )
    extra.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["per_task_breakdown"]:
        extra.append(
            f"| {r['openml_task_id']} | {r['n_total']} | {r['success']} | "
            f"{r['failed']} | {r['failed_timeout']} | "
            f"{r['skipped']} | {r['runtime_seconds_total']:.1f} | "
            f"{r['runtime_seconds_max']:.1f} |"
        )
    extra.append("")

    extra.append("### Per-algorithm breakdown (executed extreme rows)\n")
    extra.append(
        "| algorithm | n_total | success | failed | runtime_total_s | runtime_max_s |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|")
    for r in summary["per_algorithm_breakdown"]:
        extra.append(
            f"| `{r['algorithm']}` | {r['n_total']} | {r['success']} | "
            f"{r['failed']} | {r['runtime_seconds_total']:.1f} | "
            f"{r['runtime_seconds_max']:.1f} |"
        )
    extra.append("")

    extra.append("### Per-method breakdown (executed extreme rows)\n")
    extra.append(
        "| method | n_total | success | failed | runtime_total_s | runtime_max_s |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|")
    for r in summary["per_method_breakdown"]:
        extra.append(
            f"| `{r['method']}` | {r['n_total']} | {r['success']} | "
            f"{r['failed']} | {r['runtime_seconds_total']:.1f} | "
            f"{r['runtime_seconds_max']:.1f} |"
        )
    extra.append("")

    extra.append("### Per-shard status\n")
    extra.append(
        "| shard | total | success | failed | failed_to | pending | "
        "skipped | def_std | def_heavy | refused |"
    )
    extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_status"]:
        extra.append(
            f"| `{sh['shard']}` | {sh['n_total']} | {sh['success']} | "
            f"{sh['failed']} | {sh['failed_timeout']} | "
            f"{sh['pending']} | {sh['skipped']} | "
            f"{sh['deferred_standard_lane']} | "
            f"{sh['deferred_heavy_lane']} | "
            f"{sh['refused_not_in_canary_set']} |"
        )
    extra.append("")

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

    extra.append("### Signoff note\n")
    extra.append(summary["stage3_signoff_note"])
    extra.append("")

    if (
        summary["n_jobs_failed_timeout"] == 0
        and summary["n_jobs_failed_other"] == 0
        and summary["n_jobs_pending_after"] == 0
        and summary["source_shards_unchanged"]
        and not summary["stage3_signoff_present"]
    ):
        extra.append("### stage0 extreme-lane verdict: **GATE PASS**\n")
        extra.append(
            "Stage 0 replica 1 now has standard + heavy + extreme "
            "stage-run summaries pinned to the same policy_version. "
            "Commit 44 may begin the aggregate signoff plan; "
            "stage3_signoff.json should NOT be created until that "
            "planning step ships.\n"
        )
    else:
        extra.append("### stage0 extreme-lane verdict: **GATE FAIL**\n")
        extra.append(
            "Investigate failures / timeouts. The doe_rsm xgboost "
            "cell on Devnagari-Script is the most likely "
            "failed_timeout candidate; consider a per-cell "
            "timeout override.\n"
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
    parser.add_argument(
        "--run-root", type=Path, default=DEFAULT_RUN_ROOT,
        help="Used only by the execute path (Commit 43+).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Used only by the execute path (Commit 43+).",
    )
    parser.add_argument(
        "--openml-cache-root", type=Path, default=DEFAULT_OPENML_CACHE_ROOT,
        help="Used only by the execute path (Commit 43+).",
    )
    parser.add_argument(
        "--plan-summary-path", type=Path, default=PLAN_SUMMARY_PATH,
        help="Used only by the execute path; the runner refuses to "
             "execute if this Commit 42 plan summary is missing or "
             "doesn't carry execution_status=planned_not_executed.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=DEFAULT_N_FOLDS,
        help="Used only by the execute path (Commit 43+).",
    )
    parser.add_argument(
        "--hard-cap-hours-per-shard", type=float, default=16.0,
        help="Per-shard subprocess timeout ceiling for the execute "
             "path; the extreme lane is the longest, so this defaults "
             "to 16 h.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Execute path only: run pre-flight + run-dir + "
             "classification + summary, but do NOT invoke the cc18 "
             "runner. Used by the test suite.",
    )
    parser.add_argument(
        "--signoff-file", type=Path, default=None,
        help="Override the stage-3 sign-off file path. Defaults to "
             "the committed location. Tests pass a tmp path so the "
             "guard sees the file as absent.",
    )
    args = parser.parse_args(argv)

    try:
        if not args.execute_extreme_lane:
            # Planning path (default): publishes
            # ``stage0_extreme_lane_plan_latest_summary.{json,md}``.
            run_id = (
                args.run_id if args.run_id != RUN_ID else DRY_RUN_ID
            )
            t0 = time.perf_counter()
            summary = plan_stage0_extreme_lane(
                shards_dir=args.shards_dir,
                stage_runs_dir=args.stage_runs_dir,
                standard_lane_summary=args.standard_lane_summary,
                heavy_lane_summary=args.heavy_lane_summary,
                policy_csv=args.policy_csv,
                guardrails_yaml=args.guardrails_yaml,
                max_age_days=args.max_age_days,
                run_id=run_id,
                enforce_pinned_policy_version=not args.allow_policy_drift,
                signoff_file=args.signoff_file,
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
                f"json: {args.stage_runs_dir / (run_id + '_summary.json')}"
            )
            print(
                f"md:   {args.stage_runs_dir / (run_id + '_summary.md')}"
            )
            print(
                "Execution disabled. Pass --execute-extreme-lane AND "
                "--include-extreme-tasks to actually run."
            )
            return 0

        # Execution path. Requires --include-extreme-tasks as a
        # second explicit consent — the runner enforces both so a
        # single accidental flag never starts a 10+ hour run.
        if not args.include_extreme_tasks:
            print(
                "GATE REFUSAL: --execute-extreme-lane requires "
                "--include-extreme-tasks (both flags must be set "
                "explicitly so a single typo never launches the "
                "extreme lane).",
                file=sys.stderr,
            )
            return 3

        run_id = (
            args.run_id if args.run_id != DRY_RUN_ID else RUN_ID
        )
        t0 = time.perf_counter()
        summary = execute_stage0_extreme_lane(
            shards_dir=args.shards_dir,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            standard_lane_summary=args.standard_lane_summary,
            heavy_lane_summary=args.heavy_lane_summary,
            plan_summary_path=args.plan_summary_path,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            max_age_days=args.max_age_days,
            n_folds=args.n_folds,
            run_id=run_id,
            hard_cap_hours_per_shard=args.hard_cap_hours_per_shard,
            skip_train=args.skip_train,
            enforce_pinned_policy_version=not args.allow_policy_drift,
            signoff_file=args.signoff_file,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"EXEC: executed={summary['n_jobs_executed']}/"
            f"{summary['expected_extreme_canary_cells']}  "
            f"success={summary['n_jobs_success']}  "
            f"failed={summary['n_jobs_failed']}  "
            f"failed_timeout={summary['n_jobs_failed_timeout']}  "
            f"deferred_std={summary['n_jobs_deferred_standard']}  "
            f"deferred_heavy={summary['n_jobs_deferred_heavy']}  "
            f"refused={summary['n_jobs_refused_non_canary']}  "
            f"pending_after={summary['n_jobs_pending_after']}  "
            f"(elapsed {elapsed:.2f}s)"
        )
        print(
            f"json: {args.stage_runs_dir / (run_id + '_summary.json')}"
        )
        print(
            f"md:   {args.stage_runs_dir / (run_id + '_summary.md')}"
        )
        rc = 0 if (
            summary["n_jobs_failed_timeout"] == 0
            and summary["n_jobs_failed_other"] == 0
            and summary["n_jobs_pending_after"] == 0
            and summary["source_shards_unchanged"]
            and not summary["stage3_signoff_present"]
        ) else 4
        return rc
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3


# Surface the helpers that downstream commits + tests reach for.
__all__ = [
    "BATCH_ID_EXEC",
    "BATCH_ID_PLAN",
    "DRY_RUN_ID",
    "EXPECTED_EXTREME_CANARY_CELLS",
    "GateRefusalError",
    "PINNED_POLICY_VERSION",
    "PLAN_SUMMARY_PATH",
    "RUN_ID",
    "build_pre_run_plan",
    "classify_rows",
    "execute_stage0_extreme_lane",
    "main",
    "plan_stage0_extreme_lane",
    "verify_prior_stages",
]


if __name__ == "__main__":
    sys.exit(main())
