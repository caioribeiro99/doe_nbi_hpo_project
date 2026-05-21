#!/usr/bin/env python
"""Plan the Stage-3 / replica_002 extreme-lane dispatch (Commit 50).

This is a **planning-only** script. It:

- chains four gates on top of Commits 45 / 46 / 48 / 49 (signoff,
  top-up plan, Commit 48 standard summary, Commit 49 heavy
  summary);
- snapshots the live `policy_version` from
  `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv` and
  refuses if it drifts from the pinned signoff value;
- inspects the 10 committed source template shards under
  `jobs/doctoral/openml_cc18/shards/stage0_replica_001/`;
- classifies every row that would land in a future Commit 51 /
  extreme execution into four disjoint buckets:
  ``runnable_extreme_canary``,
  ``refused_extreme_non_canary``,
  ``skipped_standard_lane_already_completed``,
  ``skipped_heavy_lane_already_completed``;
- emits
  ``experiments/_stage_runs/
  stage3_replica_002_extreme_lane_plan_latest_summary.{json,md}``
  with the planning artifact.

What this script does NOT do
----------------------------
- create execution SQLite files under ``runs/``;
- create any ``runs/`` artifact;
- download new OpenML payloads;
- run training (no cell ever leaves planning);
- mutate any committed source shard;
- regenerate ``heavy_task_policy.csv`` or
  ``runtime_guardrails.yaml``;
- change ``policy_version``;
- create or modify ``stage3_signoff.json``;
- start replica_003 / 004 / 005;
- run the standard or heavy lane (already executed in Commits 48
  and 49);
- run the full ``topup_to_5`` tier.

Devnagari-Script runtime caveat (signoff caveat 2)
--------------------------------------------------
- task 167121 / Devnagari-Script previously dominated runtime in
  batch_03 (xgboost / doe_rsm_vrf_true_nbi ≈ 11,090 s; catboost
  / doe_rsm_vrf_true_nbi ≈ 10,575 s); the stage-0 extreme lane
  later ran under the policy-defined
  ``extreme.stage0_max_evaluations = 1`` budget which keeps
  Devnagari-Script tractable. This plan recommends the same
  policy-defined budget for the future Commit 51 execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from collections import Counter
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
DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
DEFAULT_GUARDRAILS_YAML = (
    REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
)
DEFAULT_OUT_JSON = (
    DEFAULT_STAGE_RUNS_DIR
    / "stage3_replica_002_extreme_lane_plan_latest_summary.json"
)
DEFAULT_OUT_MD = (
    DEFAULT_STAGE_RUNS_DIR
    / "stage3_replica_002_extreme_lane_plan_latest_summary.md"
)

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
SOURCE_STAGE = "stage0_replica_001"
TARGET_STAGE_LABEL = "stage1_topup_to_005"   # Commit 47/48/49 convention
TARGET_REPLICA = 2
RUN_ID = "stage3_replica_002_extreme_lane_plan_latest"
BATCH_ID = "stage3_replica_002_extreme_lane_plan"
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
DEVNAGARI_TASK_ID = 167121
LETTER_TASK_ID = 6


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExtremePlanRefusalError(RuntimeError):
    """Raised when a pre-flight gate rejects the planning run."""


# ---------------------------------------------------------------------------
# Hash + platform helpers
# ---------------------------------------------------------------------------


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
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


# ---------------------------------------------------------------------------
# Gate verification: signoff
# ---------------------------------------------------------------------------


def verify_signoff(
    signoff_path: Path, *, expected_policy_version: str,
) -> dict:
    if not signoff_path.exists():
        raise ExtremePlanRefusalError(
            f"signoff file not found at {signoff_path}; this planning run "
            "requires the Commit 45 signoff to exist."
        )
    try:
        record = json.loads(signoff_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExtremePlanRefusalError(
            f"{signoff_path} is not valid JSON: {exc}"
        ) from exc
    status = record.get("signoff_status")
    if status != "signed":
        raise ExtremePlanRefusalError(
            f"{signoff_path} has signoff_status={status!r}; expected 'signed'."
        )
    stype = record.get("signoff_type")
    if stype != SIGNOFF_TYPE_EXPECTED:
        raise ExtremePlanRefusalError(
            f"{signoff_path} has signoff_type={stype!r}; expected "
            f"{SIGNOFF_TYPE_EXPECTED!r}."
        )
    signed_pv = record.get("policy_version")
    if signed_pv != expected_policy_version:
        raise ExtremePlanRefusalError(
            f"{signoff_path} carries policy_version={signed_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing plan "
            "against drifted policy."
        )
    return record


# ---------------------------------------------------------------------------
# Gate verification: top-up plan
# ---------------------------------------------------------------------------


def verify_topup_plan(
    plan_path: Path, *, expected_policy_version: str,
) -> dict:
    if not plan_path.exists():
        raise ExtremePlanRefusalError(
            f"stage3 top-up plan summary not found at {plan_path}; run "
            "scripts/plan_stage3_topup.py first (Commit 46)."
        )
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExtremePlanRefusalError(
            f"{plan_path} is not valid JSON: {exc}"
        ) from exc
    exec_status = record.get("execution_status")
    if exec_status != "planned_not_executed":
        raise ExtremePlanRefusalError(
            f"{plan_path} has execution_status={exec_status!r}; expected "
            "'planned_not_executed'."
        )
    plan_pv = record.get("policy_version")
    if plan_pv != expected_policy_version:
        raise ExtremePlanRefusalError(
            f"{plan_path} carries policy_version={plan_pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing plan "
            "against drifted policy."
        )
    tiers = record.get("tier_plans") or []
    tier_topup_5 = next(
        (t for t in tiers if t.get("tier") == "topup_to_5"), None,
    )
    if tier_topup_5 is None:
        raise ExtremePlanRefusalError(
            f"{plan_path} does not list a 'topup_to_5' tier; refusing."
        )
    rs = int(tier_topup_5.get("replica_start") or 0)
    re_ = int(tier_topup_5.get("replica_end") or 0)
    if not (rs <= TARGET_REPLICA <= re_):
        raise ExtremePlanRefusalError(
            f"topup_to_5 tier covers replicas {rs}..{re_}; replica="
            f"{TARGET_REPLICA} is outside that range."
        )
    return record


# ---------------------------------------------------------------------------
# Gate verification: Commit 48 standard-lane summary
# ---------------------------------------------------------------------------


def verify_standard_lane_summary(
    summary_path: Path, *, expected_policy_version: str,
    expected_success: int = EXPECTED_STANDARD_SUCCESS,
) -> dict:
    return _verify_executed_summary(
        summary_path,
        expected_policy_version=expected_policy_version,
        expected_lane="standard",
        expected_success=expected_success,
        commit_label="Commit 48 standard-lane",
    )


# ---------------------------------------------------------------------------
# Gate verification: Commit 49 heavy-lane summary
# ---------------------------------------------------------------------------


def verify_heavy_lane_summary(
    summary_path: Path, *, expected_policy_version: str,
    expected_success: int = EXPECTED_HEAVY_SUCCESS,
) -> dict:
    return _verify_executed_summary(
        summary_path,
        expected_policy_version=expected_policy_version,
        expected_lane="heavy",
        expected_success=expected_success,
        commit_label="Commit 49 heavy-lane",
    )


def _verify_executed_summary(
    summary_path: Path, *,
    expected_policy_version: str, expected_lane: str,
    expected_success: int, commit_label: str,
) -> dict:
    if not summary_path.exists():
        raise ExtremePlanRefusalError(
            f"{commit_label} summary not found at {summary_path}."
        )
    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExtremePlanRefusalError(
            f"{summary_path} is not valid JSON: {exc}"
        ) from exc
    if record.get("execution_status") != "executed":
        raise ExtremePlanRefusalError(
            f"{summary_path} has execution_status="
            f"{record.get('execution_status')!r}; expected 'executed'."
        )
    if int(record.get("replica") or 0) != TARGET_REPLICA:
        raise ExtremePlanRefusalError(
            f"{summary_path} replica={record.get('replica')!r}; expected "
            f"{TARGET_REPLICA}."
        )
    if record.get("lane") != expected_lane:
        raise ExtremePlanRefusalError(
            f"{summary_path} lane={record.get('lane')!r}; expected "
            f"{expected_lane!r}."
        )
    pv = record.get("policy_version")
    if pv != expected_policy_version:
        raise ExtremePlanRefusalError(
            f"{summary_path} carries policy_version={pv!r}; live "
            f"policy_version={expected_policy_version!r}. Refusing plan "
            "against drifted policy."
        )
    if int(record.get("n_jobs_success") or -1) != int(expected_success):
        raise ExtremePlanRefusalError(
            f"{summary_path} n_jobs_success={record.get('n_jobs_success')}; "
            f"expected {expected_success}."
        )
    for key, expected in (
        ("n_jobs_failed", 0),
        ("n_jobs_failed_timeout", 0),
        ("n_jobs_pending_after", 0),
        ("n_jobs_running_after", 0),
    ):
        actual = int(record.get(key) or 0)
        if actual != expected:
            raise ExtremePlanRefusalError(
                f"{summary_path} {key}={actual}; expected {expected}."
            )
    if not bool(record.get("source_shards_unchanged", False)):
        raise ExtremePlanRefusalError(
            f"{summary_path} reports source_shards_unchanged=False."
        )
    return record


# ---------------------------------------------------------------------------
# Row classification (extreme-lane planning variant)
# ---------------------------------------------------------------------------


def classify_rows(
    rows: list[tuple],
    guardrails: RuntimeGuardrails,
) -> dict[str, list[dict]]:
    """Split shard rows into four disjoint planning buckets.

    rows: (job_id, openml_task_id, method, algorithm).
    Returns dict keys:
      ``runnable_extreme_canary``           — canary methods × extreme tasks;
      ``refused_extreme_non_canary``        — extreme task × non-canary method;
      ``skipped_standard_lane_already_completed`` — any row whose task
                                                  is standard (Commit 48);
      ``skipped_heavy_lane_already_completed``   — any row whose task
                                                  is heavy (Commit 49).
    """
    buckets: dict[str, list[dict]] = {
        "runnable_extreme_canary": [],
        "refused_extreme_non_canary": [],
        "skipped_standard_lane_already_completed": [],
        "skipped_heavy_lane_already_completed": [],
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
        if method in CANARY_METHODS:
            buckets["runnable_extreme_canary"].append(entry)
        else:
            buckets["refused_extreme_non_canary"].append(entry)
    return buckets


# ---------------------------------------------------------------------------
# Pre-run plan (read-only across 10 source shards)
# ---------------------------------------------------------------------------


def build_pre_run_plan(
    shards: list[Path], guardrails: RuntimeGuardrails,
) -> dict:
    """Inventory the committed source shards under the policy,
    read-only, projecting extreme-lane planning buckets."""
    n_jobs_total = 0
    n_runnable_extreme = 0
    n_refused_extreme = 0
    n_skipped_standard = 0
    n_skipped_heavy = 0
    method_counts: Counter = Counter()
    algorithm_counts: Counter = Counter()
    non_canary_methods: set[str] = set()
    extreme_tasks_planned: set[int] = set()
    standard_tasks_completed: set[int] = set()
    heavy_tasks_completed: set[int] = set()
    source_template_stages: set[str] = set()
    source_template_replicas: set[int] = set()
    per_shard: list[dict] = []
    per_task_extreme: Counter = Counter()
    per_method_extreme: Counter = Counter()
    per_algorithm_extreme: Counter = Counter()
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
            "runnable_extreme_canary": len(buckets["runnable_extreme_canary"]),
            "refused_extreme_non_canary": len(
                buckets["refused_extreme_non_canary"],
            ),
            "skipped_standard_lane_already_completed": len(
                buckets["skipped_standard_lane_already_completed"],
            ),
            "skipped_heavy_lane_already_completed": len(
                buckets["skipped_heavy_lane_already_completed"],
            ),
        }
        per_shard.append(per_sh)
        n_jobs_total += len(rows)
        n_runnable_extreme += per_sh["runnable_extreme_canary"]
        n_refused_extreme += per_sh["refused_extreme_non_canary"]
        n_skipped_standard += per_sh["skipped_standard_lane_already_completed"]
        n_skipped_heavy += per_sh["skipped_heavy_lane_already_completed"]
        for tid, m, a, st, rep in rows:
            tid_int = int(tid)
            lane = guardrails.get_task_lane(tid_int)
            seen_tasks[tid_int] = lane
            method_counts[m] += 1
            algorithm_counts[a] += 1
            source_template_stages.add(st)
            source_template_replicas.add(int(rep))
            if lane == "extreme" and m in CANARY_METHODS:
                extreme_tasks_planned.add(tid_int)
                per_task_extreme[tid_int] += 1
                per_method_extreme[m] += 1
                per_algorithm_extreme[a] += 1
            elif lane == "extreme" and m not in CANARY_METHODS:
                non_canary_methods.add(m)
            elif lane == "standard":
                standard_tasks_completed.add(tid_int)
            elif lane == "heavy":
                heavy_tasks_completed.add(tid_int)
    return {
        "n_source_shards": len(shards),
        "n_jobs_total": n_jobs_total,
        "n_runnable_extreme_canary": n_runnable_extreme,
        "n_refused_extreme_non_canary": n_refused_extreme,
        "n_skipped_standard_lane_already_completed": n_skipped_standard,
        "n_skipped_heavy_lane_already_completed": n_skipped_heavy,
        "task_lane_counts_universe": dict(Counter(seen_tasks.values())),
        "method_counts": dict(method_counts),
        "algorithm_counts": dict(algorithm_counts),
        "extreme_non_canary_methods_refused": sorted(non_canary_methods),
        "extreme_tasks_planned": sorted(extreme_tasks_planned),
        "standard_tasks_already_completed": sorted(standard_tasks_completed),
        "heavy_tasks_already_completed": sorted(heavy_tasks_completed),
        "source_template_stages": sorted(source_template_stages),
        "source_template_replicas": sorted(source_template_replicas),
        "per_shard": per_shard,
        "per_extreme_task_planned": dict(per_task_extreme),
        "per_method_planned": dict(per_method_extreme),
        "per_algorithm_planned": dict(per_algorithm_extreme),
    }


# ---------------------------------------------------------------------------
# Committed shard MD5 snapshot (read-only)
# ---------------------------------------------------------------------------


def _committed_shard_md5s(shards: list[Path]) -> dict[str, str]:
    return {sh.name: _md5(sh) for sh in shards}


# ---------------------------------------------------------------------------
# Build planning summary
# ---------------------------------------------------------------------------


def build_extreme_plan(
    *,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    signoff_file: Path = DEFAULT_SIGNOFF_FILE,
    topup_plan_summary: Path = DEFAULT_TOPUP_PLAN_SUMMARY,
    standard_lane_summary: Path = DEFAULT_STANDARD_LANE_SUMMARY,
    heavy_lane_summary: Path = DEFAULT_HEAVY_LANE_SUMMARY,
    policy_csv: Path = DEFAULT_POLICY_CSV,
    guardrails_yaml: Path = DEFAULT_GUARDRAILS_YAML,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    write_summary: bool = True,
    expected_extreme_canary_cells: int = EXPECTED_EXTREME_CANARY_CELLS,
) -> dict:
    """Build the read-only Stage-3 / replica_002 extreme-lane plan."""
    source_shards = sorted(shards_dir.glob("shard_*.sqlite"))
    if len(source_shards) != N_EXPECTED_SHARDS:
        raise ExtremePlanRefusalError(
            f"expected {N_EXPECTED_SHARDS} stage-0 shards under "
            f"{shards_dir}, found {len(source_shards)}"
        )

    # 1. Four gates BEFORE we open anything.
    live_policy_version = _sha256(policy_csv)
    signoff_record = verify_signoff(
        signoff_file, expected_policy_version=live_policy_version,
    )
    topup_record = verify_topup_plan(
        topup_plan_summary, expected_policy_version=live_policy_version,
    )
    std_record = verify_standard_lane_summary(
        standard_lane_summary, expected_policy_version=live_policy_version,
    )
    hvy_record = verify_heavy_lane_summary(
        heavy_lane_summary, expected_policy_version=live_policy_version,
    )

    signoff_sha256 = _sha256(signoff_file)
    topup_plan_sha256 = _sha256(topup_plan_summary)
    std_summary_sha256 = _sha256(standard_lane_summary)
    hvy_summary_sha256 = _sha256(heavy_lane_summary)

    # 2. Load policy and check the live extreme task universe matches
    #    the expected pair (letter, Devnagari-Script).
    guardrails = RuntimeGuardrails.load(
        yaml_path=guardrails_yaml, csv_path=policy_csv,
    )
    live_extreme_tids = tuple(sorted(
        tid for tid, p in guardrails.tasks.items() if p.lane == "extreme"
    ))
    if live_extreme_tids != tuple(sorted(EXPECTED_EXTREME_TASK_IDS)):
        raise ExtremePlanRefusalError(
            f"live extreme task universe is {live_extreme_tids}; expected "
            f"{tuple(sorted(EXPECTED_EXTREME_TASK_IDS))!r}. Refusing plan."
        )

    # 3. Snapshot committed shard MD5s before and after (read-only sanity).
    md5_before = _committed_shard_md5s(source_shards)

    # 4. Pre-run plan across the 10 source shards.
    plan = build_pre_run_plan(source_shards, guardrails)
    if plan["n_runnable_extreme_canary"] != expected_extreme_canary_cells:
        raise ExtremePlanRefusalError(
            f"pre-run plan inconsistency: expected "
            f"{expected_extreme_canary_cells} extreme-lane canary cells "
            f"across all 10 shards but found "
            f"{plan['n_runnable_extreme_canary']}. Verify "
            "heavy_task_policy.csv classification before proceeding."
        )

    # 5. Snapshot committed shard MD5s again to assert no mutation.
    md5_after = _committed_shard_md5s(source_shards)
    source_shards_unchanged = (md5_after == md5_before)

    # 6. Build the planning summary.
    ext_spec = guardrails.get_lane_spec("extreme")
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "batch_id": BATCH_ID,
        "stage": TARGET_STAGE_LABEL,
        "topup_tier": TOPUP_TIER,
        "execution_status": "planned_not_executed",
        "replica": int(TARGET_REPLICA),
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
        "stage3_topup_plan_execution_status": topup_record.get(
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
        "n_source_shards": len(source_shards),
        "source_shards": [_safe_rel(p) for p in source_shards],
        "n_jobs_total": int(plan["n_jobs_total"]),
        "expected_extreme_canary_cells": expected_extreme_canary_cells,
        "n_runnable_extreme_canary": int(plan["n_runnable_extreme_canary"]),
        "n_refused_extreme_non_canary": int(
            plan["n_refused_extreme_non_canary"],
        ),
        "n_skipped_standard_lane_already_completed": int(
            plan["n_skipped_standard_lane_already_completed"],
        ),
        "n_skipped_heavy_lane_already_completed": int(
            plan["n_skipped_heavy_lane_already_completed"],
        ),
        "expected_extreme_task_ids": list(sorted(EXPECTED_EXTREME_TASK_IDS)),
        "extreme_tasks_planned": plan["extreme_tasks_planned"],
        "extreme_task_meta": {
            int(LETTER_TASK_ID): {"dataset": "letter"},
            int(DEVNAGARI_TASK_ID): {"dataset": "Devnagari-Script"},
        },
        "standard_tasks_already_completed": plan[
            "standard_tasks_already_completed"
        ],
        "heavy_tasks_already_completed": plan[
            "heavy_tasks_already_completed"
        ],
        "extreme_non_canary_methods_refused": plan[
            "extreme_non_canary_methods_refused"
        ],
        "task_lane_counts_universe": plan["task_lane_counts_universe"],
        "method_counts_universe": plan["method_counts"],
        "algorithm_counts_universe": plan["algorithm_counts"],
        "per_shard_planned": plan["per_shard"],
        "per_extreme_task_planned": plan["per_extreme_task_planned"],
        "per_method_planned_extreme": plan["per_method_planned"],
        "per_algorithm_planned_extreme": plan["per_algorithm_planned"],
        "source_template_stages": plan["source_template_stages"],
        "source_template_replicas": plan["source_template_replicas"],
        "extreme_lane_timeout_seconds_per_cell": float(
            ext_spec.timeout_seconds_per_cell,
        ),
        "extreme_lane_default_max_evaluations": int(
            ext_spec.default_max_evaluations,
        ),
        "extreme_lane_gate_max_evaluations": int(
            ext_spec.gate_max_evaluations,
        ),
        "extreme_lane_stage0_max_evaluations": int(
            ext_spec.stage0_max_evaluations,
        ),
        "extreme_lane_include_by_default": bool(
            ext_spec.include_by_default,
        ),
        "extreme_lane_requires_manual_review_before_full_stage0": bool(
            ext_spec.requires_manual_review_before_full_stage0,
        ),
        "execution_recommendation_for_commit_51": {
            "require_explicit_include_extreme_tasks_flag": True,
            "require_explicit_execute_extreme_lane_flag": True,
            "max_evaluations_recommended": int(
                ext_spec.stage0_max_evaluations,
            ),
            "timeout_seconds_per_cell_recommended": float(
                ext_spec.timeout_seconds_per_cell,
            ),
            "rationale": (
                "Stage-3 top-up cells stack with the signed-off stage-0 "
                "cells under the same policy_version. The pinned "
                "policy_version sets extreme.stage0_max_evaluations=1 "
                "(see signoff caveat 2 / "
                "devnagari_extreme_budget_non_equivalence). Commit 51 "
                "should reuse this budget so the extreme cells in "
                "replica_002 are directly comparable to the extreme "
                "cells signed off in replica_001."
            ),
        },
        "devnagari_runtime_caveat": (
            "task 167121 / Devnagari-Script previously dominated runtime "
            "in batch_03 (xgboost / doe_rsm_vrf_true_nbi ~ 11,090 s; "
            "catboost / doe_rsm_vrf_true_nbi ~ 10,575 s; catboost / "
            "tpe_optuna ~ 7,944 s; catboost / random_search ~ 7,647 s). "
            "Under the pinned policy_version's "
            "extreme.stage0_max_evaluations=1, those cells run with a "
            "tighter configuration budget; doe_rsm_vrf_true_nbi is "
            "unchanged because it floors at n_doe=max(2*d, "
            "max_evaluations)=8 for d=4. Headline panel-average metrics "
            "must footnote this asymmetry (signoff caveat 2)."
        ),
        "source_shard_md5_before": md5_before,
        "source_shard_md5_after": md5_after,
        "source_shards_unchanged": source_shards_unchanged,
        "no_committed_shard_modified_by_this_script": source_shards_unchanged,
        "no_training_run_by_this_script": True,
        "no_execution_sqlite_created_by_this_script": True,
        "no_runs_directory_artifacts_created_by_this_script": True,
        "no_raw_openml_payloads_staged_by_this_script": True,
        "no_standard_lane_rerun_by_this_script": True,
        "no_heavy_lane_rerun_by_this_script": True,
        "no_extreme_lane_executed_by_this_script": True,
        "no_full_topup_to_5_executed_by_this_script": True,
        "no_other_replica_executed_by_this_script": True,
        "no_new_signoff_file_created_by_this_script": True,
        "no_policy_csv_regenerated_by_this_script": True,
        "no_guardrails_yaml_regenerated_by_this_script": True,
        "operator_review_required_before_execution": True,
        "operator_review_required_before_replica003": True,
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "next_recommended_step": (
            "After Commit 50 is operator-reviewed, Commit 51 should "
            "execute the replica_002 extreme lane under the pinned "
            "policy_version (extreme.stage0_max_evaluations=1, per-cell "
            "timeout=14,400 s). Commit 51 MUST require an explicit "
            "--include-extreme-tasks / --execute-extreme-lane flag pair "
            "and MUST NOT scale to replica_003-005 until replica_002 "
            "standard + heavy + extreme have all been operator-reviewed "
            "end-to-end."
        ),
    }

    if write_summary:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        _write_md(out_md, summary)
    return summary


def _write_md(md_path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append(
        "# Stage 3 / replica_002 extreme-lane plan (Commit 50)\n"
    )
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- batch_id: `{summary['batch_id']}`")
    lines.append(f"- stage: `{summary['stage']}`")
    lines.append(f"- topup_tier: `{summary['topup_tier']}`")
    lines.append(
        f"- **execution_status: `{summary['execution_status']}`**"
    )
    lines.append(
        f"- replica: **{summary['replica']}** (source template replica = "
        f"{summary['source_template_replica']})"
    )
    lines.append(f"- lane: `{summary['lane']}`")
    lines.append(f"- git_sha: `{str(summary['git_sha'])[:12]}`")
    lines.append(f"- host: `{summary['host']}`")
    lines.append(
        f"- policy_version: `{str(summary['policy_version'])[:16]}`"
    )
    lines.append(
        f"- policy_version_pinned: "
        f"`{str(summary['policy_version_pinned'])[:16]}`\n"
    )

    lines.append("## Gate chain\n")
    lines.append(
        f"- signoff: `{summary['signoff_path']}` "
        f"sha256=`{str(summary['signoff_sha256'])[:16]}` "
        f"({summary['signoff_status']} / {summary['signoff_type']} / "
        f"operator {summary['signoff_operator_handle']})"
    )
    lines.append(
        f"- stage3 top-up plan: "
        f"`{summary['stage3_topup_plan_summary_path']}` "
        f"sha256=`{str(summary['stage3_topup_plan_summary_sha256'])[:16]}` "
        f"({summary['stage3_topup_plan_execution_status']})"
    )
    lines.append(
        f"- Commit 48 standard summary: "
        f"`{summary['commit48_standard_lane_summary_path']}` "
        f"sha256=`{str(summary['commit48_standard_lane_summary_sha256'])[:16]}` "
        f"(success={summary['commit48_standard_lane_n_jobs_success']}, "
        f"runtime≈{summary['commit48_standard_lane_runtime_seconds']:.0f}s)"
    )
    lines.append(
        f"- Commit 49 heavy summary: "
        f"`{summary['commit49_heavy_lane_summary_path']}` "
        f"sha256=`{str(summary['commit49_heavy_lane_summary_sha256'])[:16]}` "
        f"(success={summary['commit49_heavy_lane_n_jobs_success']}, "
        f"runtime≈{summary['commit49_heavy_lane_runtime_seconds']:.0f}s)\n"
    )

    lines.append("## Scope of this plan\n")
    lines.append(
        f"- expected runnable extreme canary cells: "
        f"**{summary['expected_extreme_canary_cells']}**"
    )
    lines.append(
        f"- planned runnable extreme canary cells: "
        f"**{summary['n_runnable_extreme_canary']}**"
    )
    lines.append(
        f"- refused extreme non-canary rows: "
        f"{summary['n_refused_extreme_non_canary']}"
    )
    lines.append(
        f"- standard rows already completed (Commit 48): "
        f"{summary['n_skipped_standard_lane_already_completed']}"
    )
    lines.append(
        f"- heavy rows already completed (Commit 49): "
        f"{summary['n_skipped_heavy_lane_already_completed']}"
    )
    lines.append(
        f"- n_jobs_total across 10 source shards: "
        f"{summary['n_jobs_total']}\n"
    )

    lines.append("## Extreme task universe (pinned policy)\n")
    for tid in summary["expected_extreme_task_ids"]:
        meta = summary["extreme_task_meta"].get(str(tid)) or summary[
            "extreme_task_meta"
        ].get(int(tid))
        n = summary["per_extreme_task_planned"].get(
            str(tid),
        ) or summary["per_extreme_task_planned"].get(int(tid), 0)
        dataset = (meta or {}).get("dataset", "?")
        lines.append(
            f"- task **{tid}** / `{dataset}` — {n} planned canary cells"
        )
    lines.append("")

    lines.append("## Execution policy for the future Commit 51\n")
    rec = summary["execution_recommendation_for_commit_51"]
    lines.append(
        f"- require explicit `--include-extreme-tasks` flag: "
        f"{rec['require_explicit_include_extreme_tasks_flag']}"
    )
    lines.append(
        f"- require explicit `--execute-extreme-lane` flag: "
        f"{rec['require_explicit_execute_extreme_lane_flag']}"
    )
    lines.append(
        f"- max_evaluations (recommended): "
        f"**{rec['max_evaluations_recommended']}** "
        f"(policy `extreme.stage0_max_evaluations`)"
    )
    lines.append(
        f"- timeout per cell (recommended): "
        f"**{rec['timeout_seconds_per_cell_recommended']:.0f} s** "
        f"(policy `extreme.timeout_seconds_per_cell`)\n"
    )
    lines.append(rec["rationale"])
    lines.append("")

    lines.append("## Per-shard planned counts\n")
    lines.append(
        "| shard | n_jobs | runnable_ext | refused_ext | std_done | hvy_done |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sh in summary["per_shard_planned"]:
        lines.append(
            f"| `{sh['shard']}` | {sh['n_jobs']} | "
            f"{sh['runnable_extreme_canary']} | "
            f"{sh['refused_extreme_non_canary']} | "
            f"{sh['skipped_standard_lane_already_completed']} | "
            f"{sh['skipped_heavy_lane_already_completed']} |"
        )
    lines.append("")

    lines.append("## Devnagari-Script runtime caveat\n")
    lines.append(summary["devnagari_runtime_caveat"])
    lines.append("")

    lines.append("## What this commit does NOT do\n")
    invariants = [
        ("no_training_run_by_this_script",
         "no training was run"),
        ("no_execution_sqlite_created_by_this_script",
         "no execution SQLite files were created"),
        ("no_runs_directory_artifacts_created_by_this_script",
         "no `runs/` directory artifacts were created"),
        ("no_raw_openml_payloads_staged_by_this_script",
         "no raw OpenML payloads were staged"),
        ("no_standard_lane_rerun_by_this_script",
         "standard lane was not rerun (Commit 48 stands)"),
        ("no_heavy_lane_rerun_by_this_script",
         "heavy lane was not rerun (Commit 49 stands)"),
        ("no_extreme_lane_executed_by_this_script",
         "extreme lane was not executed"),
        ("no_full_topup_to_5_executed_by_this_script",
         "full topup_to_5 was not executed"),
        ("no_other_replica_executed_by_this_script",
         "no replica_003 / 004 / 005 was executed"),
        ("no_new_signoff_file_created_by_this_script",
         "no new signoff file was created"),
        ("no_policy_csv_regenerated_by_this_script",
         "heavy_task_policy.csv was not regenerated"),
        ("no_guardrails_yaml_regenerated_by_this_script",
         "runtime_guardrails.yaml was not regenerated"),
        ("no_committed_shard_modified_by_this_script",
         "no committed SQLite shard was modified"),
    ]
    for key, desc in invariants:
        v = summary[key]
        mark = "✓" if v else "✗"
        lines.append(f"- {mark} {desc} (`{key}` = `{v}`)")
    lines.append("")

    lines.append("## Verdict\n")
    if (
        summary["n_runnable_extreme_canary"]
        == summary["expected_extreme_canary_cells"]
        and summary["source_shards_unchanged"]
        and summary["no_committed_shard_modified_by_this_script"]
        and summary["no_training_run_by_this_script"]
        and summary["no_execution_sqlite_created_by_this_script"]
        and summary["no_new_signoff_file_created_by_this_script"]
    ):
        lines.append(
            "**PLAN PASS — operator review required before Commit 51.** "
            "All gates verified, the planned extreme-lane scope matches "
            "the policy-defined count, and no committed shard was "
            "modified. Commit 51 may execute the replica_002 extreme "
            "lane under the pinned policy_version, but only with an "
            "explicit operator-confirmed scope and the "
            "`--include-extreme-tasks` + `--execute-extreme-lane` "
            "guards.\n"
        )
    else:
        lines.append("**PLAN FAIL** — investigate before Commit 51.\n")

    lines.append("## Next recommended step\n")
    lines.append(summary["next_recommended_step"])
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


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
        "--policy-csv", type=Path, default=DEFAULT_POLICY_CSV,
    )
    parser.add_argument(
        "--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS_YAML,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the plan in memory and print a small JSON summary "
             "without writing the JSON/MD artifacts to disk.",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_extreme_plan(
            shards_dir=args.shards_dir,
            signoff_file=args.signoff_file,
            topup_plan_summary=args.topup_plan_summary,
            standard_lane_summary=args.standard_lane_summary,
            heavy_lane_summary=args.heavy_lane_summary,
            policy_csv=args.policy_csv,
            guardrails_yaml=args.guardrails_yaml,
            out_json=args.out_json,
            out_md=args.out_md,
            write_summary=not args.dry_run,
        )
    except ExtremePlanRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(json.dumps({
        "execution_status": summary["execution_status"],
        "replica": summary["replica"],
        "lane": summary["lane"],
        "topup_tier": summary["topup_tier"],
        "n_source_shards": summary["n_source_shards"],
        "n_jobs_total": summary["n_jobs_total"],
        "expected_extreme_canary_cells": summary[
            "expected_extreme_canary_cells"
        ],
        "n_runnable_extreme_canary": summary["n_runnable_extreme_canary"],
        "n_refused_extreme_non_canary": summary[
            "n_refused_extreme_non_canary"
        ],
        "n_skipped_standard_lane_already_completed": summary[
            "n_skipped_standard_lane_already_completed"
        ],
        "n_skipped_heavy_lane_already_completed": summary[
            "n_skipped_heavy_lane_already_completed"
        ],
        "extreme_tasks_planned": summary["extreme_tasks_planned"],
        "policy_version": summary["policy_version"],
        "policy_version_pinned": summary["policy_version_pinned"],
        "operator_review_required_before_execution": summary[
            "operator_review_required_before_execution"
        ],
        "out_json": str(args.out_json) if not args.dry_run else None,
        "out_md": str(args.out_md) if not args.dry_run else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
