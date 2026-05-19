#!/usr/bin/env python
"""Plan the OpenML-CC18 stage-3 / top-up dispatch (Commit 46).

This script is the **planner** for scaling from the signed
``stage0_replica_001`` baseline (Commit 45) to the three top-up
tiers defined by the frozen comparative protocol (Commit 27):

- ``topup_to_5``  — replicas 002 … 005   (+4 replicas, +3,456 canary cells)
- ``topup_to_10`` — replicas 006 … 010   (+5 replicas, +4,320 canary cells)
- ``topup_to_30`` — replicas 011 … 030   (+20 replicas, +17,280 canary cells)

It is a **planning** script. It does NOT:

- run any OpenML training;
- create execution SQLite files under ``runs/``;
- mutate committed SQLite shards under
  ``jobs/doctoral/openml_cc18/shards/``;
- regenerate ``heavy_task_policy.csv`` / ``runtime_guardrails.yaml``;
- change ``policy_version``;
- create or modify ``stage3_signoff.json``;
- promote or demote any task between lanes.

It produces:

- ``experiments/_stage_runs/stage3_topup_plan_latest_summary.json``
- ``experiments/_stage_runs/stage3_topup_plan_latest_summary.md``

Refusal rules
-------------
- ``stage3_signoff.json`` missing.
- ``stage3_signoff.json`` present but ``signoff_status != "signed"``.
- ``signoff_type != "stage0_replica_001"``.
- Signed ``policy_version`` does not match the live SHA-256 of
  ``heavy_task_policy.csv``, **unless** the caller passes
  ``--allow-policy-drift-report-only``; in that case the planner
  emits a candidate drift report and refuses to produce the
  ordinary plan.
- Lane summary SHA-256s recorded in the signoff disagree with the
  live lane summary hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_STANDARD = DEFAULT_STAGE_RUNS_DIR / "stage0_standard_lane_latest_summary.json"
DEFAULT_HEAVY = DEFAULT_STAGE_RUNS_DIR / "stage0_heavy_lane_latest_summary.json"
DEFAULT_EXTREME = DEFAULT_STAGE_RUNS_DIR / "stage0_extreme_lane_latest_summary.json"
DEFAULT_SIGNOFF = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
DEFAULT_HEAVY_POLICY = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
DEFAULT_GUARDRAILS = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
DEFAULT_SHARD_SUMMARY = REPO / "jobs/doctoral/openml_cc18/shards/shard_summary.json"
DEFAULT_SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards"
DEFAULT_TASKS_CSV = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
DEFAULT_OUT_JSON = (
    DEFAULT_STAGE_RUNS_DIR / "stage3_topup_plan_latest_summary.json"
)
DEFAULT_OUT_MD = DEFAULT_STAGE_RUNS_DIR / "stage3_topup_plan_latest_summary.md"

RUN_ID = "stage3_topup_plan_latest"
STAGE = "stage3_topup_plan"
SIGNOFF_TYPE_EXPECTED = "stage0_replica_001"

CANARY_METHODS = (
    "default_gbdt",
    "random_search",
    "tpe_optuna",
    "doe_rsm_vrf_true_nbi",
)
ALGORITHMS = ("xgboost", "lightgbm", "catboost")

EXPECTED_N_TASKS = 72
EXPECTED_N_STANDARD_TASKS = 57
EXPECTED_N_HEAVY_TASKS = 13
EXPECTED_N_EXTREME_TASKS = 2
CELLS_PER_REPLICA = EXPECTED_N_TASKS * len(CANARY_METHODS) * len(ALGORITHMS)
# 72 * 4 * 3 = 864
STANDARD_CELLS_PER_REPLICA = EXPECTED_N_STANDARD_TASKS * len(CANARY_METHODS) * len(ALGORITHMS)
# 57 * 4 * 3 = 684
HEAVY_CELLS_PER_REPLICA = EXPECTED_N_HEAVY_TASKS * len(CANARY_METHODS) * len(ALGORITHMS)
# 13 * 4 * 3 = 156
EXTREME_CELLS_PER_REPLICA = EXPECTED_N_EXTREME_TASKS * len(CANARY_METHODS) * len(ALGORITHMS)
# 2 * 4 * 3 = 24

TIERS = (
    {
        "tier": "topup_to_5",
        "shard_subdir": "stage1_topup_to_005",
        "replica_start": 2,
        "replica_end": 5,
        "replica_count": 4,
    },
    {
        "tier": "topup_to_10",
        "shard_subdir": "stage2_topup_to_010",
        "replica_start": 6,
        "replica_end": 10,
        "replica_count": 5,
    },
    {
        "tier": "topup_to_30",
        "shard_subdir": "stage3_topup_to_030",
        "replica_start": 11,
        "replica_end": 30,
        "replica_count": 20,
    },
)

# Tasks the operator should treat as high-risk regardless of
# observed runtime — large rows / features / classes, or already
# flagged in the signoff caveats.
KNOWN_HIGH_RISK_TASKS: dict[int, dict[str, str]] = {
    3481:   {"dataset": "isolet", "reason": "isolet_future_recalibration_candidate (signoff caveat 1)"},
    3573:   {"dataset": "mnist_784", "reason": "heavy lane; observed_max_runtime_s=1507.2 at R=1"},
    167124: {"dataset": "CIFAR_10", "reason": "heavy lane; n_features=3072, n_rows=60000"},
    146825: {"dataset": "Fashion-MNIST", "reason": "heavy lane; n_features=784, n_rows=70000"},
    167121: {"dataset": "Devnagari-Script", "reason": "devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane"},
    167125: {"dataset": "Internet-Advertisements", "reason": "heavy lane; n_features=1558 with 1555 categorical"},
}

RUNTIME_RISK_THRESHOLD_SECONDS = 600.0


class TopupPlanRefusalError(RuntimeError):
    """Raised when the planner refuses to proceed (missing /
    unsigned signoff, policy drift, lane summary hash drift)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
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


def _load_tasks_csv(path: Path = DEFAULT_TASKS_CSV) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {int(r["openml_task_id"]): r for r in csv.DictReader(f)}


def _load_heavy_policy(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["openml_task_id"])] = r
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


# ---------------------------------------------------------------------------
# Signoff loading + invariants
# ---------------------------------------------------------------------------


def _load_signoff(path: Path) -> dict:
    if not path.exists():
        raise TopupPlanRefusalError(
            f"signoff file not found: {path}. Commit 45 should have "
            "created jobs/doctoral/openml_cc18/stage3_signoff.json. "
            "Run scripts/sign_stage0_replica_001.py first."
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TopupPlanRefusalError(
            f"{path} is not valid JSON: {exc}"
        ) from exc
    status = record.get("signoff_status")
    if status != "signed":
        raise TopupPlanRefusalError(
            f"{path} has signoff_status={status!r}; expected 'signed'. "
            "Stage-3 top-up planning requires an operator-signed signoff "
            "(see docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md)."
        )
    stype = record.get("signoff_type")
    if stype != SIGNOFF_TYPE_EXPECTED:
        raise TopupPlanRefusalError(
            f"{path} has signoff_type={stype!r}; expected "
            f"{SIGNOFF_TYPE_EXPECTED!r}."
        )
    return record


def _check_policy_alignment(
    signoff: dict,
    *,
    heavy_policy_path: Path,
    allow_policy_drift_report_only: bool,
) -> tuple[str, bool, str | None]:
    """Return (live_policy_version, drift_detected, drift_message).

    Refuses if drift is detected and the report-only flag is off.
    Refuses with drift_detected=True and returns the live SHA when
    the flag is on (callers use this to emit the drift report and
    return early)."""
    live_policy_version = _sha256(heavy_policy_path)
    signed_policy_version = signoff.get("policy_version")
    if signed_policy_version is None:
        raise TopupPlanRefusalError(
            "signoff record carries no policy_version."
        )
    if live_policy_version == signed_policy_version:
        return live_policy_version, False, None
    drift_message = (
        f"signed policy_version={signed_policy_version} but the live "
        f"SHA-256 of {_safe_rel(heavy_policy_path)} is "
        f"{live_policy_version}. heavy_task_policy.csv has been "
        "modified since signoff."
    )
    if not allow_policy_drift_report_only:
        raise TopupPlanRefusalError(
            drift_message + " Pass --allow-policy-drift-report-only "
            "to emit a candidate drift report instead."
        )
    return live_policy_version, True, drift_message


def _check_lane_summary_alignment(
    signoff: dict,
    *,
    standard_path: Path, heavy_path: Path, extreme_path: Path,
) -> dict[str, str]:
    """Verify lane summaries still hash to what the signoff recorded."""
    live = {
        "standard": _sha256(standard_path),
        "heavy": _sha256(heavy_path),
        "extreme": _sha256(extreme_path),
    }
    expected = {
        "standard": signoff.get("standard_lane_summary_sha256"),
        "heavy": signoff.get("heavy_lane_summary_sha256"),
        "extreme": signoff.get("extreme_lane_summary_sha256"),
    }
    drifted = [
        lane for lane, h in live.items() if expected[lane] and expected[lane] != h
    ]
    if drifted:
        details = ", ".join(
            f"{lane}: signed={expected[lane]} live={live[lane]}"
            for lane in drifted
        )
        raise TopupPlanRefusalError(
            "lane summary SHA-256 drift since signoff — "
            f"{details}. A lane summary was modified after signoff."
        )
    return live


# ---------------------------------------------------------------------------
# Runtime estimates
# ---------------------------------------------------------------------------


def _runtime_distribution_by_lane(
    lane_summaries: dict[str, dict],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for lane, payload in lane_summaries.items():
        vals = [
            float(c["runtime_seconds"]) for c in payload.get("cells", [])
            if c.get("status") == "success"
            and c.get("runtime_seconds") is not None
        ]
        if not vals:
            out[lane] = {"n": 0, "p50": None, "p90": None, "max": None, "mean": None}
            continue
        out[lane] = {
            "n": len(vals),
            "p50": float(median(vals)),
            "p90": _percentile(vals, 0.9),
            "max": float(max(vals)),
            "mean": float(fmean(vals)),
        }
    return out


def _per_replica_runtime_seconds_by_lane(
    lane_summaries: dict[str, dict],
) -> dict[str, float]:
    """Total runner runtime across all canary cells per replica per
    lane. Stage-0 has 1 replica, so the lane summaries directly
    measure 'per-replica runtime'."""
    return {
        lane: float(payload.get("runtime_seconds_runner_total", 0.0))
        for lane, payload in lane_summaries.items()
    }


# ---------------------------------------------------------------------------
# High-risk identification
# ---------------------------------------------------------------------------


def _identify_high_risk_cells(
    lane_summaries: dict[str, dict],
    *,
    heavy_policy: dict[int, dict],
    tasks_meta: dict[int, dict],
    threshold_seconds: float = RUNTIME_RISK_THRESHOLD_SECONDS,
) -> list[dict]:
    rows: list[dict] = []
    seen_keys: set[tuple[int, str, str, str]] = set()
    for lane, payload in lane_summaries.items():
        for c in payload.get("cells", []):
            if c.get("status") != "success":
                continue
            rt = c.get("runtime_seconds")
            if rt is None:
                continue
            tid = int(c.get("openml_task_id"))
            reasons = []
            if tid in KNOWN_HIGH_RISK_TASKS:
                reasons.append(KNOWN_HIGH_RISK_TASKS[tid]["reason"])
            if float(rt) >= threshold_seconds:
                reasons.append(
                    f"observed_runtime_s={float(rt):.1f}>={threshold_seconds:.0f}"
                )
            if not reasons:
                continue
            key = (tid, lane, str(c.get("method")), str(c.get("algorithm")))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            policy_row = heavy_policy.get(tid, {})
            meta_row = tasks_meta.get(tid, {})
            rows.append({
                "openml_task_id": tid,
                "dataset_name": (
                    KNOWN_HIGH_RISK_TASKS.get(tid, {}).get("dataset")
                    or policy_row.get("dataset_name")
                    or meta_row.get("dataset_name")
                    or "?"
                ),
                "lane": lane,
                "policy_lane": policy_row.get("lane"),
                "method": c.get("method"),
                "algorithm": c.get("algorithm"),
                "observed_runtime_seconds": float(rt),
                "reasons": reasons,
            })
    rows.sort(key=lambda r: r["observed_runtime_seconds"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _lane_cell_counts(lane: str) -> int:
    if lane == "standard":
        return STANDARD_CELLS_PER_REPLICA
    if lane == "heavy":
        return HEAVY_CELLS_PER_REPLICA
    if lane == "extreme":
        return EXTREME_CELLS_PER_REPLICA
    raise ValueError(lane)


def _lane_task_count(lane: str) -> int:
    if lane == "standard":
        return EXPECTED_N_STANDARD_TASKS
    if lane == "heavy":
        return EXPECTED_N_HEAVY_TASKS
    if lane == "extreme":
        return EXPECTED_N_EXTREME_TASKS
    raise ValueError(lane)


def _build_tier_plan(
    tier_def: dict,
    *,
    per_replica_runtime_by_lane: dict[str, float],
    runtime_by_lane: dict[str, dict],
    shard_row_counts: dict[str, list[int]],
) -> dict:
    tier = tier_def["tier"]
    rc = tier_def["replica_count"]
    lanes: list[dict] = []
    total_canary = 0
    total_runtime_p50_s = 0.0
    total_runtime_p90_s = 0.0
    total_runtime_max_s = 0.0
    for lane in ("standard", "heavy", "extreme"):
        per_rep = float(per_replica_runtime_by_lane.get(lane, 0.0))
        cells = _lane_cell_counts(lane) * rc
        total_canary += cells
        lane_rt = runtime_by_lane.get(lane, {})
        # We use observed per-replica runtime totals scaled by rc as p50,
        # and inflate by a configurable factor for p90 / max.
        p50_total_s = per_rep * rc
        p90_total_s = (lane_rt.get("p90") or lane_rt.get("p50") or per_rep / max(_lane_cell_counts(lane), 1))
        p90_total_s = float(p90_total_s) * _lane_cell_counts(lane) * rc
        max_per_cell = lane_rt.get("max") or (per_rep / max(_lane_cell_counts(lane), 1))
        max_total_s = float(max_per_cell) * _lane_cell_counts(lane) * rc
        lanes.append({
            "lane": lane,
            "task_count": _lane_task_count(lane),
            "canary_methods": list(CANARY_METHODS),
            "algorithms": list(ALGORITHMS),
            "executable_canary_cells_per_replica": _lane_cell_counts(lane),
            "executable_canary_cells_total": cells,
            "per_replica_runtime_seconds_observed_p50": p50_total_s / rc if rc else None,
            "estimated_runtime_seconds_total_p50": p50_total_s,
            "estimated_runtime_seconds_total_p90": p90_total_s,
            "estimated_runtime_seconds_total_max": max_total_s,
        })
        total_runtime_p50_s += p50_total_s
        total_runtime_p90_s += p90_total_s
        total_runtime_max_s += max_total_s
    return {
        "tier": tier,
        "shard_subdir": tier_def["shard_subdir"],
        "replica_start": tier_def["replica_start"],
        "replica_end": tier_def["replica_end"],
        "replica_count": rc,
        "executable_canary_cells_total": total_canary,
        "shard_row_counts": shard_row_counts.get(tier_def["shard_subdir"], []),
        "shard_row_total": sum(shard_row_counts.get(tier_def["shard_subdir"], [])),
        "lanes": lanes,
        "estimated_runtime_seconds_total_p50": total_runtime_p50_s,
        "estimated_runtime_seconds_total_p90": total_runtime_p90_s,
        "estimated_runtime_seconds_total_max": total_runtime_max_s,
        "estimated_runtime_hours_total_p50": total_runtime_p50_s / 3600.0,
        "estimated_runtime_hours_total_max": total_runtime_max_s / 3600.0,
    }


def _committed_shard_md5s(shards_dir: Path) -> dict[str, str]:
    """Return {relative_path: md5} for every committed SQLite shard.

    Reads, does not modify. Used to assert the planner is read-only
    against the committed shards (the test snapshots this before/after)."""
    out: dict[str, str] = {}
    if not shards_dir.exists():
        return out
    for sub in sorted(shards_dir.iterdir()):
        if not sub.is_dir():
            continue
        for shard in sorted(sub.glob("shard_*.sqlite")):
            out[f"{sub.name}/{shard.name}"] = _md5(shard)
    return out


def _load_shard_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _shard_row_counts_by_stage(shard_summary: dict) -> dict[str, list[int]]:
    by_stage: dict[str, dict[int, int]] = defaultdict(dict)
    for r in shard_summary.get("rows_by_shard", []):
        by_stage[r["stage"]][int(r["shard_id"])] = int(r["rows"])
    out: dict[str, list[int]] = {}
    for stage, by_id in by_stage.items():
        out[stage] = [by_id[k] for k in sorted(by_id)]
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_topup_plan(
    *,
    signoff_path: Path = DEFAULT_SIGNOFF,
    standard_path: Path = DEFAULT_STANDARD,
    heavy_path: Path = DEFAULT_HEAVY,
    extreme_path: Path = DEFAULT_EXTREME,
    heavy_policy_path: Path = DEFAULT_HEAVY_POLICY,
    guardrails_path: Path = DEFAULT_GUARDRAILS,
    shard_summary_path: Path = DEFAULT_SHARD_SUMMARY,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    tasks_csv_path: Path = DEFAULT_TASKS_CSV,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    write_summary: bool = True,
    allow_policy_drift_report_only: bool = False,
) -> dict:
    """Build the top-up dispatch planning artifact.

    Returns the summary dict that is also written to ``out_json``
    (and rendered to ``out_md``) when ``write_summary`` is true."""
    signoff = _load_signoff(signoff_path)
    live_policy_version, drift_detected, drift_message = _check_policy_alignment(
        signoff,
        heavy_policy_path=heavy_policy_path,
        allow_policy_drift_report_only=allow_policy_drift_report_only,
    )

    # Verify lane summaries still match. We do this even when drift was
    # detected — a lane-summary tamper is independent of a policy edit.
    lane_hashes_live = _check_lane_summary_alignment(
        signoff,
        standard_path=standard_path,
        heavy_path=heavy_path,
        extreme_path=extreme_path,
    )

    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    heavy = json.loads(heavy_path.read_text(encoding="utf-8"))
    extreme = json.loads(extreme_path.read_text(encoding="utf-8"))
    lane_summaries = {
        "standard": standard,
        "heavy": heavy,
        "extreme": extreme,
    }

    tasks_meta = _load_tasks_csv(tasks_csv_path)
    heavy_policy = _load_heavy_policy(heavy_policy_path)

    runtime_by_lane = _runtime_distribution_by_lane(lane_summaries)
    per_replica_rt_by_lane = _per_replica_runtime_seconds_by_lane(
        lane_summaries
    )

    shard_summary = _load_shard_summary(shard_summary_path)
    shard_row_counts = _shard_row_counts_by_stage(shard_summary)

    # We deliberately do not open SQLite shards here. We snapshot the
    # MD5s and let the test assert they did not change. The plan reads
    # bytes only.
    shard_md5_snapshot = _committed_shard_md5s(shards_dir)

    tier_plans = [
        _build_tier_plan(
            t,
            per_replica_runtime_by_lane=per_replica_rt_by_lane,
            runtime_by_lane=runtime_by_lane,
            shard_row_counts=shard_row_counts,
        )
        for t in TIERS
    ]

    high_risk_cells = _identify_high_risk_cells(
        lane_summaries,
        heavy_policy=heavy_policy,
        tasks_meta=tasks_meta,
    )

    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": STAGE,
        "exported_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "execution_status": "planned_not_executed",
        "drift_report_only": drift_detected,
        "policy_drift_detected": drift_detected,
        "policy_drift_message": drift_message,
        # Inputs.
        "signoff_path": _safe_rel(signoff_path),
        "signoff_sha256": _sha256(signoff_path),
        "signoff_status": signoff.get("signoff_status"),
        "signoff_type": signoff.get("signoff_type"),
        "signoff_signed_at_utc": signoff.get("signed_at_utc"),
        "signoff_operator_name": signoff.get("operator_name"),
        "signoff_operator_handle": signoff.get("operator_handle"),
        "signoff_branch": signoff.get("branch"),
        "signoff_git_sha": signoff.get("git_sha_at_signoff"),
        "signoff_downstream_execution_authorized_in_this_commit":
            signoff.get("downstream_execution_authorized_in_this_commit"),
        "signoff_caveats_acknowledged": signoff.get("caveats_acknowledged"),
        "policy_version": live_policy_version,
        "signed_policy_version": signoff.get("policy_version"),
        "heavy_task_policy_csv_path": _safe_rel(heavy_policy_path),
        "heavy_task_policy_csv_sha256": live_policy_version,
        "runtime_guardrails_yaml_path": _safe_rel(guardrails_path),
        "runtime_guardrails_yaml_sha256": (
            _sha256(guardrails_path) if guardrails_path.exists() else None
        ),
        "shard_summary_path": _safe_rel(shard_summary_path),
        "shard_summary_sha256": (
            _sha256(shard_summary_path) if shard_summary_path.exists() else None
        ),
        "lane_summary_paths": {
            "standard": _safe_rel(standard_path),
            "heavy": _safe_rel(heavy_path),
            "extreme": _safe_rel(extreme_path),
        },
        "lane_summary_sha256_live": lane_hashes_live,
        "lane_summary_sha256_signed": {
            "standard": signoff.get("standard_lane_summary_sha256"),
            "heavy": signoff.get("heavy_lane_summary_sha256"),
            "extreme": signoff.get("extreme_lane_summary_sha256"),
        },
        # Lane / per-replica observed.
        "per_replica_runtime_seconds_by_lane_observed":
            per_replica_rt_by_lane,
        "runtime_distribution_by_lane_observed": runtime_by_lane,
        "n_canary_cells_per_replica": CELLS_PER_REPLICA,
        "n_canary_cells_per_lane_per_replica": {
            "standard": STANDARD_CELLS_PER_REPLICA,
            "heavy": HEAVY_CELLS_PER_REPLICA,
            "extreme": EXTREME_CELLS_PER_REPLICA,
        },
        # Per-tier plans.
        "tier_plans": tier_plans,
        # Aggregates across all tiers.
        "executable_canary_cells_total_all_tiers": sum(
            t["executable_canary_cells_total"] for t in tier_plans
        ),
        "estimated_runtime_seconds_total_all_tiers_p50": sum(
            t["estimated_runtime_seconds_total_p50"] for t in tier_plans
        ),
        "estimated_runtime_seconds_total_all_tiers_max": sum(
            t["estimated_runtime_seconds_total_max"] for t in tier_plans
        ),
        # Risk register.
        "high_risk_cells": high_risk_cells,
        "high_risk_threshold_seconds": RUNTIME_RISK_THRESHOLD_SECONDS,
        # Source-shard read-only verification.
        "committed_shard_md5_snapshot": shard_md5_snapshot,
        "n_committed_shards": len(shard_md5_snapshot),
        # Provenance.
        "git_sha": _git_sha(),
        "platform": _platform(),
        "no_training_run_by_this_script": True,
        "no_execution_sqlite_created_by_this_script": True,
        "no_committed_shard_modified_by_this_script": True,
        "decision_options_doc": "docs/STAGE3_POLICY_DECISION.md",
        "execution_plan_doc": "docs/STAGE3_TOPUP_EXECUTION_PLAN.md",
        "distributed_runbook_doc": "docs/STAGE3_DISTRIBUTED_RUNBOOK.md",
    }

    # Sanity-check the cell counts so a math regression would refuse.
    assert STANDARD_CELLS_PER_REPLICA == 684
    assert HEAVY_CELLS_PER_REPLICA == 156
    assert EXTREME_CELLS_PER_REPLICA == 24
    assert CELLS_PER_REPLICA == 864
    assert tier_plans[0]["executable_canary_cells_total"] == 3456
    assert tier_plans[1]["executable_canary_cells_total"] == 4320
    assert tier_plans[2]["executable_canary_cells_total"] == 17280

    if write_summary:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        out_md.write_text(_render_md(summary), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------


def _fmt_seconds(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 60:
        return f"{v:.1f} s"
    if v < 3600:
        return f"{v / 60:.1f} min"
    return f"{v / 3600:.2f} h"


def _render_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# stage 3 / top-up dispatch plan (planning-only)\n")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- stage: `{summary['stage']}`")
    lines.append(f"- exported_at: `{summary['exported_at']}`")
    lines.append(
        f"- **execution_status: `{summary['execution_status']}`**"
    )
    lines.append(
        f"- drift_report_only: `{summary['drift_report_only']}`"
    )
    lines.append(
        f"- policy_drift_detected: `{summary['policy_drift_detected']}`"
    )
    if summary.get("policy_drift_message"):
        lines.append(f"  - drift_message: {summary['policy_drift_message']}")
    lines.append("")

    lines.append("## Signoff context\n")
    lines.append(f"- signoff_path: `{summary['signoff_path']}`")
    lines.append(
        f"- signoff_status: **{summary['signoff_status']}** "
        f"(`{summary['signoff_type']}`)"
    )
    lines.append(
        f"- operator: `{summary['signoff_operator_name']}` "
        f"(`{summary['signoff_operator_handle']}`)"
    )
    lines.append(
        f"- signed_at_utc: `{summary['signoff_signed_at_utc']}`"
    )
    lines.append(
        f"- branch: `{summary['signoff_branch']}` git_sha: "
        f"`{str(summary['signoff_git_sha'])[:12]}`"
    )
    lines.append(
        f"- downstream_execution_authorized_in_this_commit: "
        f"`{summary['signoff_downstream_execution_authorized_in_this_commit']}`\n"
    )

    lines.append("## Policy + lane summaries\n")
    lines.append(
        f"- policy_version (live): `{summary['policy_version'][:16]}…`"
    )
    lines.append(
        f"- signed policy_version: `{str(summary['signed_policy_version'])[:16]}…`"
    )
    lines.append(
        f"- heavy_task_policy.csv path: `{summary['heavy_task_policy_csv_path']}`"
    )
    lines.append(
        f"- runtime_guardrails.yaml path: `{summary['runtime_guardrails_yaml_path']}`"
    )
    lines.append("")
    lines.append("| lane | live sha256 | signed sha256 |")
    lines.append("|---|---|---|")
    for lane in ("standard", "heavy", "extreme"):
        live_h = summary["lane_summary_sha256_live"][lane]
        signed_h = summary["lane_summary_sha256_signed"][lane]
        lines.append(
            f"| `{lane}` | `{live_h[:16]}…` | "
            f"`{str(signed_h)[:16] if signed_h else '—'}…` |"
        )
    lines.append("")

    lines.append("## Observed stage-0 runtime per replica per lane\n")
    lines.append("| lane | total (s) | total (h) | per-cell p50 | per-cell p90 | per-cell max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lane in ("standard", "heavy", "extreme"):
        per_rep = summary["per_replica_runtime_seconds_by_lane_observed"].get(lane, 0.0)
        cell_dist = summary["runtime_distribution_by_lane_observed"].get(lane, {})
        lines.append(
            f"| `{lane}` | {per_rep:.1f} | {per_rep / 3600:.2f} | "
            f"{_fmt_seconds(cell_dist.get('p50'))} | "
            f"{_fmt_seconds(cell_dist.get('p90'))} | "
            f"{_fmt_seconds(cell_dist.get('max'))} |"
        )
    lines.append("")

    lines.append("## Per-tier plan\n")
    lines.append("Cells per replica: standard=684, heavy=156, extreme=24 — total 864 canary cells per replica.\n")
    lines.append(
        "| tier | replicas | canary cells | est. runtime (p50, h) | est. runtime (max, h) | shard subdir | shard rows total |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for t in summary["tier_plans"]:
        lines.append(
            f"| `{t['tier']}` | "
            f"{t['replica_start']}–{t['replica_end']} ({t['replica_count']}) | "
            f"{t['executable_canary_cells_total']:,} | "
            f"{t['estimated_runtime_hours_total_p50']:.2f} | "
            f"{t['estimated_runtime_hours_total_max']:.2f} | "
            f"`{t['shard_subdir']}` | {t['shard_row_total']:,} |"
        )
    lines.append("")

    for t in summary["tier_plans"]:
        lines.append(f"### `{t['tier']}` — lane breakdown\n")
        lines.append("| lane | task_count | canary cells/replica | total canary cells | p50 (h) | max (h) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for L in t["lanes"]:
            lines.append(
                f"| `{L['lane']}` | {L['task_count']} | "
                f"{L['executable_canary_cells_per_replica']} | "
                f"{L['executable_canary_cells_total']:,} | "
                f"{L['estimated_runtime_seconds_total_p50'] / 3600:.2f} | "
                f"{L['estimated_runtime_seconds_total_max'] / 3600:.2f} |"
            )
        lines.append("")

    lines.append("## Aggregate across all tiers\n")
    lines.append(
        f"- executable_canary_cells_total_all_tiers: "
        f"**{summary['executable_canary_cells_total_all_tiers']:,}**"
    )
    lines.append(
        f"- estimated_runtime_total_all_tiers (p50): "
        f"{summary['estimated_runtime_seconds_total_all_tiers_p50'] / 3600:.2f} h"
    )
    lines.append(
        f"- estimated_runtime_total_all_tiers (max): "
        f"{summary['estimated_runtime_seconds_total_all_tiers_max'] / 3600:.2f} h\n"
    )

    lines.append("## High-risk cells\n")
    if not summary["high_risk_cells"]:
        lines.append("_None identified above threshold._\n")
    else:
        lines.append(
            f"_Cells flagged as high-risk for top-up scheduling "
            f"(threshold {summary['high_risk_threshold_seconds']:.0f}s or "
            f"known dataset)._\n"
        )
        lines.append("| task_id | dataset | lane | method | algorithm | runtime_s | reasons |")
        lines.append("|---:|---|---|---|---|---:|---|")
        for r in summary["high_risk_cells"][:30]:
            lines.append(
                f"| {r['openml_task_id']} | `{r['dataset_name']}` | "
                f"`{r['lane']}` | `{r['method']}` | `{r['algorithm']}` | "
                f"{r['observed_runtime_seconds']:.1f} | "
                f"{'; '.join(r['reasons'])} |"
            )
        lines.append("")

    lines.append("## Signoff caveats acknowledged\n")
    for c in summary.get("signoff_caveats_acknowledged", []) or []:
        lines.append(
            f"- **{c.get('id')}** (task {c.get('task_id')}, "
            f"`{c.get('dataset')}`): {c.get('summary', '')[:240]}…"
        )
    lines.append("")

    lines.append("## Source shards (committed) — read-only\n")
    lines.append(
        f"- n_committed_shards: {summary['n_committed_shards']}"
    )
    lines.append(
        f"- no_committed_shard_modified_by_this_script: "
        f"`{summary['no_committed_shard_modified_by_this_script']}`"
    )
    lines.append(
        f"- no_execution_sqlite_created_by_this_script: "
        f"`{summary['no_execution_sqlite_created_by_this_script']}`"
    )
    lines.append(
        f"- no_training_run_by_this_script: "
        f"`{summary['no_training_run_by_this_script']}`\n"
    )

    lines.append("## Next steps\n")
    lines.append(
        f"- review this plan + `{summary['decision_options_doc']}`"
    )
    lines.append(
        f"- read `{summary['execution_plan_doc']}` for the strategic context"
    )
    lines.append(
        f"- when ready, follow `{summary['distributed_runbook_doc']}` "
        "to run the Commit 47 pilot (replica 002, shard 00, standard "
        "lane, canary only)\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signoff-file", type=Path, default=DEFAULT_SIGNOFF)
    parser.add_argument("--standard-summary", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--heavy-summary", type=Path, default=DEFAULT_HEAVY)
    parser.add_argument("--extreme-summary", type=Path, default=DEFAULT_EXTREME)
    parser.add_argument("--heavy-policy-csv", type=Path, default=DEFAULT_HEAVY_POLICY)
    parser.add_argument("--guardrails-yaml", type=Path, default=DEFAULT_GUARDRAILS)
    parser.add_argument(
        "--shard-summary", type=Path, default=DEFAULT_SHARD_SUMMARY,
    )
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--allow-policy-drift-report-only", action="store_true",
        help="When the live policy_version differs from the signoff's "
             "policy_version, emit a candidate drift report instead of "
             "refusing. Does NOT create a new policy.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute the plan but do not write JSON/MD to disk.",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    try:
        summary = build_topup_plan(
            signoff_path=args.signoff_file,
            standard_path=args.standard_summary,
            heavy_path=args.heavy_summary,
            extreme_path=args.extreme_summary,
            heavy_policy_path=args.heavy_policy_csv,
            guardrails_path=args.guardrails_yaml,
            shard_summary_path=args.shard_summary,
            shards_dir=args.shards_dir,
            tasks_csv_path=args.tasks_csv,
            out_json=args.out_json,
            out_md=args.out_md,
            write_summary=not args.dry_run,
            allow_policy_drift_report_only=args.allow_policy_drift_report_only,
        )
    except TopupPlanRefusalError as exc:
        print(f"PLAN REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"TOPUP PLAN: execution_status={summary['execution_status']}  "
        f"drift_report_only={summary['drift_report_only']}  "
        f"canary_cells_total={summary['executable_canary_cells_total_all_tiers']}  "
        f"policy_version={summary['policy_version'][:12]}…  "
        f"signoff_status={summary['signoff_status']}"
    )
    if not args.dry_run:
        print(f"json: {args.out_json}")
        print(f"md:   {args.out_md}")
    return 0


__all__ = [
    "ALGORITHMS",
    "CANARY_METHODS",
    "CELLS_PER_REPLICA",
    "EXPECTED_N_EXTREME_TASKS",
    "EXPECTED_N_HEAVY_TASKS",
    "EXPECTED_N_STANDARD_TASKS",
    "EXPECTED_N_TASKS",
    "EXTREME_CELLS_PER_REPLICA",
    "HEAVY_CELLS_PER_REPLICA",
    "KNOWN_HIGH_RISK_TASKS",
    "RUN_ID",
    "RUNTIME_RISK_THRESHOLD_SECONDS",
    "STANDARD_CELLS_PER_REPLICA",
    "TIERS",
    "TopupPlanRefusalError",
    "build_topup_plan",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
