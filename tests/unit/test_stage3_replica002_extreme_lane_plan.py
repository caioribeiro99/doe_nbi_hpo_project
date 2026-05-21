"""Tests for the Stage-3 / replica_002 extreme-lane planner (Commit 50).

``scripts/plan_stage3_replica002_extreme_lane.py`` is a
**planning-only** script. It reads the four upstream artifacts
(signoff + top-up plan + Commit 48 standard summary + Commit 49
heavy summary), inventories the 10 committed source shards, and
emits an extreme-lane plan summary. It must NOT:

- run training;
- create execution SQLite under ``runs/``;
- mutate committed SQLite shards;
- create or modify ``stage3_signoff.json``;
- change ``policy_version``;
- promote any task between lanes.

This module covers the contract that protects those invariants.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/plan_stage3_replica002_extreme_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
TOPUP_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs/stage3_topup_plan_latest_summary.json"
)
STANDARD_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_standard_lane_latest_summary.json"
)
HEAVY_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_heavy_lane_latest_summary.json"
)
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
PLAN_DOC = REPO / "docs/STAGE3_REPLICA002_EXTREME_PLAN.md"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------


def _write_signoff(
    path: Path, *,
    signoff_status: str = "signed",
    signoff_type: str = "stage0_replica_001",
    policy_version: str = PINNED_POLICY_VERSION,
) -> Path:
    record = {
        "schema_version": 1,
        "signoff_type": signoff_type,
        "signoff_status": signoff_status,
        "operator_name": "Caio Tertuliano Ribeiro",
        "operator_handle": "caioribeiro99",
        "branch": "repo-publication-readiness",
        "policy_version": policy_version,
        "signed_at_utc": "2026-05-18T18:17:24Z",
        "git_sha_at_signoff": "0" * 40,
        "downstream_execution_authorized_in_this_commit": False,
        "standard_lane_summary_sha256": "0" * 64,
        "heavy_lane_summary_sha256": "0" * 64,
        "extreme_lane_summary_sha256": "0" * 64,
        "n_canary_success_total": 864,
        "lane_success_counts": {"standard": 684, "heavy": 156, "extreme": 24},
        "caveats_acknowledged": [
            {
                "id": "isolet_future_recalibration_candidate",
                "task_id": 3481, "dataset": "isolet",
                "summary": "future recalibration candidate",
            },
            {
                "id": "devnagari_extreme_budget_non_equivalence",
                "task_id": 167121, "dataset": "Devnagari-Script",
                "summary": "extreme stage0_max_evaluations=1",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _write_topup_plan(
    path: Path, *,
    execution_status: str = "planned_not_executed",
    policy_version: str = PINNED_POLICY_VERSION,
    include_replica2_in_topup_5: bool = True,
) -> Path:
    record = {
        "schema_version": 1,
        "run_id": "stage3_topup_plan_latest",
        "stage": "stage3_topup_plan",
        "execution_status": execution_status,
        "policy_version": policy_version,
        "no_training_run_by_this_script": True,
        "no_execution_sqlite_created_by_this_script": True,
        "no_committed_shard_modified_by_this_script": True,
        "tier_plans": [
            {
                "tier": "topup_to_5",
                "shard_subdir": "stage1_topup_to_005",
                "replica_start": 2 if include_replica2_in_topup_5 else 3,
                "replica_end": 5,
                "replica_count": 4 if include_replica2_in_topup_5 else 3,
            },
            {
                "tier": "topup_to_10",
                "shard_subdir": "stage2_topup_to_010",
                "replica_start": 6, "replica_end": 10, "replica_count": 5,
            },
            {
                "tier": "topup_to_30",
                "shard_subdir": "stage3_topup_to_030",
                "replica_start": 11, "replica_end": 30, "replica_count": 20,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _write_lane_summary(
    path: Path, *,
    lane: str,
    n_jobs_success: int,
    execution_status: str = "executed",
    replica: int = 2,
    policy_version: str = PINNED_POLICY_VERSION,
    n_jobs_failed: int = 0,
    n_jobs_failed_timeout: int = 0,
    n_jobs_pending_after: int = 0,
    n_jobs_running_after: int = 0,
    source_shards_unchanged: bool = True,
    runtime_seconds_runner_total: float = 1000.0,
) -> Path:
    record = {
        "schema_version": 1,
        "execution_status": execution_status,
        "replica": replica,
        "source_template_replica": 1,
        "lane": lane,
        "policy_version": policy_version,
        "n_jobs_success": n_jobs_success,
        "n_jobs_failed": n_jobs_failed,
        "n_jobs_failed_timeout": n_jobs_failed_timeout,
        "n_jobs_pending_after": n_jobs_pending_after,
        "n_jobs_running_after": n_jobs_running_after,
        "source_shards_unchanged": source_shards_unchanged,
        "runtime_seconds_runner_total": runtime_seconds_runner_total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _write_standard(path: Path, **kw) -> Path:
    kw.setdefault("lane", "standard")
    kw.setdefault("n_jobs_success", 684)
    return _write_lane_summary(path, **kw)


def _write_heavy(path: Path, **kw) -> Path:
    kw.setdefault("lane", "heavy")
    kw.setdefault("n_jobs_success", 156)
    return _write_lane_summary(path, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_run_script_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.lower()
    assert "plan_stage3_replica002_extreme_lane.py" in out
    assert "--signoff-file" in out
    assert "--topup-plan-summary" in out
    assert "--standard-lane-summary" in out
    assert "--heavy-lane-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--dry-run" in out


def test_run_script_dry_run_reports_planned_not_executed() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["replica"] == 2
    assert payload["lane"] == "extreme"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["n_source_shards"] == 10
    assert payload["n_jobs_total"] == 2304
    assert payload["expected_extreme_canary_cells"] == 24
    assert payload["n_runnable_extreme_canary"] == 24
    assert payload["n_refused_extreme_non_canary"] == 42
    assert payload["n_skipped_standard_lane_already_completed"] == 1815
    assert payload["n_skipped_heavy_lane_already_completed"] == 423
    assert payload["extreme_tasks_planned"] == [6, 167121]
    assert payload["policy_version"] == PINNED_POLICY_VERSION
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert payload["operator_review_required_before_execution"] is True
    # Dry-run must not write files
    assert payload["out_json"] is None
    assert payload["out_md"] is None


# ---------------------------------------------------------------------------
# Signoff refusal
# ---------------------------------------------------------------------------


def test_refuses_when_signoff_missing(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_signoff,
    )

    with pytest.raises(ExtremePlanRefusalError, match="signoff file not found"):
        verify_signoff(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_signoff_not_signed(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", signoff_status="planned_not_signed",
    )
    with pytest.raises(ExtremePlanRefusalError, match="signoff_status"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_signoff_policy_drift(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", policy_version="f" * 64,
    )
    with pytest.raises(ExtremePlanRefusalError, match="policy_version"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Top-up plan refusal
# ---------------------------------------------------------------------------


def test_refuses_when_topup_plan_missing(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_topup_plan,
    )

    with pytest.raises(ExtremePlanRefusalError, match="stage3 top-up plan"):
        verify_topup_plan(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_replica2_not_in_topup_5(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_topup_plan,
    )

    p = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
        include_replica2_in_topup_5=False,
    )
    with pytest.raises(ExtremePlanRefusalError, match="replica=2"):
        verify_topup_plan(p, expected_policy_version=PINNED_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Commit 48 standard-lane refusal
# ---------------------------------------------------------------------------


def test_refuses_when_standard_summary_missing(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_standard_lane_summary,
    )

    with pytest.raises(ExtremePlanRefusalError, match="Commit 48 standard-lane"):
        verify_standard_lane_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_failed(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard(
        tmp_path / "std.json", n_jobs_failed=2, n_jobs_success=684,
    )
    with pytest.raises(ExtremePlanRefusalError, match="n_jobs_failed"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_policy_drift(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard(tmp_path / "std.json", policy_version="f" * 64)
    with pytest.raises(ExtremePlanRefusalError, match="policy_version"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


# ---------------------------------------------------------------------------
# Commit 49 heavy-lane refusal
# ---------------------------------------------------------------------------


def test_refuses_when_heavy_summary_missing(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_heavy_lane_summary,
    )

    with pytest.raises(ExtremePlanRefusalError, match="Commit 49 heavy-lane"):
        verify_heavy_lane_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_heavy_summary_failed(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_heavy_lane_summary,
    )

    p = _write_heavy(
        tmp_path / "hvy.json", n_jobs_failed=1, n_jobs_success=156,
    )
    with pytest.raises(ExtremePlanRefusalError, match="n_jobs_failed"):
        verify_heavy_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_heavy_summary_policy_drift(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
        verify_heavy_lane_summary,
    )

    p = _write_heavy(tmp_path / "hvy.json", policy_version="f" * 64)
    with pytest.raises(ExtremePlanRefusalError, match="policy_version"):
        verify_heavy_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.plan_stage3_replica002_extreme_lane import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # extreme task × canary -> runnable_extreme_canary
        ("j_ex_can_1", 6, "default_gbdt", "xgboost"),
        ("j_ex_can_2", 167121, "tpe_optuna", "lightgbm"),
        ("j_ex_can_3", 167121, "doe_rsm_vrf_true_nbi", "catboost"),
        # extreme task × non-canary -> refused_extreme_non_canary
        ("j_ex_nc_1", 6, "smac3", "xgboost"),
        ("j_ex_nc_2", 167121, "asha", "catboost"),
        # standard task -> skipped_standard_lane_already_completed
        ("j_s1", 11, "default_gbdt", "xgboost"),
        ("j_s2", 3481, "tpe_optuna", "catboost"),  # isolet stays standard
        # heavy task -> skipped_heavy_lane_already_completed
        ("j_h1", 3573, "default_gbdt", "xgboost"),
        ("j_h2", 167124, "tpe_optuna", "lightgbm"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(e["job_id"] for e in buckets["runnable_extreme_canary"]) == [
        "j_ex_can_1", "j_ex_can_2", "j_ex_can_3",
    ]
    assert sorted(e["job_id"] for e in buckets["refused_extreme_non_canary"]) == [
        "j_ex_nc_1", "j_ex_nc_2",
    ]
    assert sorted(
        e["job_id"]
        for e in buckets["skipped_standard_lane_already_completed"]
    ) == ["j_s1", "j_s2"]
    assert sorted(
        e["job_id"] for e in buckets["skipped_heavy_lane_already_completed"]
    ) == ["j_h1", "j_h2"]


def test_pre_run_plan_across_all_10_shards() -> None:
    """Plan over the 10 committed stage-0 shards reports 24 runnable
    extreme-lane canary cells split evenly across two tasks."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.plan_stage3_replica002_extreme_lane import build_pre_run_plan

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_extreme_canary"] == 24
    assert plan["n_refused_extreme_non_canary"] == 42
    assert plan["n_skipped_standard_lane_already_completed"] == 1815
    assert plan["n_skipped_heavy_lane_already_completed"] == 423
    assert plan["extreme_tasks_planned"] == [6, 167121]
    assert plan["per_extreme_task_planned"] == {6: 12, 167121: 12}
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }
    assert plan["extreme_non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    # 24 = 4 canary methods × 3 algorithms × 2 tasks distributed equally
    assert plan["per_method_planned"] == {
        "default_gbdt": 6, "random_search": 6,
        "tpe_optuna": 6, "doe_rsm_vrf_true_nbi": 6,
    }
    assert plan["per_algorithm_planned"] == {
        "xgboost": 8, "lightgbm": 8, "catboost": 8,
    }


def test_extreme_task_universe_under_pinned_policy() -> None:
    """The live policy must classify exactly tasks 6 and 167121 as
    extreme — the planner refuses otherwise."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    extreme_tids = sorted(
        tid for tid, p in g.tasks.items() if p.lane == "extreme"
    )
    assert extreme_tids == [6, 167121]


# ---------------------------------------------------------------------------
# Build-plan refusal paths
# ---------------------------------------------------------------------------


def _build(
    tmp_path: Path, *,
    signoff: Path | None = None,
    topup: Path | None = None,
    std: Path | None = None,
    hvy: Path | None = None,
    write_summary: bool = False,
) -> dict:
    from scripts.plan_stage3_replica002_extreme_lane import build_extreme_plan

    return build_extreme_plan(
        shards_dir=SHARDS_DIR,
        signoff_file=signoff or _write_signoff(tmp_path / "stage3_signoff.json"),
        topup_plan_summary=topup or _write_topup_plan(
            tmp_path / "stage3_topup_plan_latest_summary.json",
        ),
        standard_lane_summary=std or _write_standard(tmp_path / "std.json"),
        heavy_lane_summary=hvy or _write_heavy(tmp_path / "hvy.json"),
        policy_csv=POLICY_CSV,
        guardrails_yaml=GUARDRAILS_YAML,
        out_json=tmp_path / "out.json",
        out_md=tmp_path / "out.md",
        write_summary=write_summary,
    )


def test_build_refuses_on_missing_signoff(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
    )

    with pytest.raises(ExtremePlanRefusalError, match="signoff file not found"):
        _build(tmp_path, signoff=tmp_path / "absent.json")


def test_build_refuses_on_missing_topup_plan(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
    )

    with pytest.raises(ExtremePlanRefusalError, match="stage3 top-up plan"):
        _build(tmp_path, topup=tmp_path / "absent.json")


def test_build_refuses_on_missing_standard_summary(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
    )

    with pytest.raises(ExtremePlanRefusalError, match="Commit 48 standard-lane"):
        _build(tmp_path, std=tmp_path / "absent.json")


def test_build_refuses_on_missing_heavy_summary(tmp_path: Path) -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        ExtremePlanRefusalError,
    )

    with pytest.raises(ExtremePlanRefusalError, match="Commit 49 heavy-lane"):
        _build(tmp_path, hvy=tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# End-to-end build (no execution) — schema + invariants
# ---------------------------------------------------------------------------


SUMMARY_REQUIRED_KEYS = (
    "schema_version", "run_id", "batch_id", "stage", "topup_tier",
    "execution_status", "replica", "source_template_replica", "lane",
    "policy_version", "policy_version_pinned",
    "policy_csv_path", "guardrails_yaml_path",
    "signoff_path", "signoff_sha256",
    "signoff_signed_at_utc", "signoff_operator_handle",
    "signoff_operator_name", "signoff_type", "signoff_status",
    "stage3_topup_plan_summary_path",
    "stage3_topup_plan_summary_sha256",
    "stage3_topup_plan_execution_status",
    "commit48_standard_lane_summary_path",
    "commit48_standard_lane_summary_sha256",
    "commit48_standard_lane_n_jobs_success",
    "commit48_standard_lane_runtime_seconds",
    "commit49_heavy_lane_summary_path",
    "commit49_heavy_lane_summary_sha256",
    "commit49_heavy_lane_n_jobs_success",
    "commit49_heavy_lane_runtime_seconds",
    "n_source_shards", "source_shards", "n_jobs_total",
    "expected_extreme_canary_cells", "n_runnable_extreme_canary",
    "n_refused_extreme_non_canary",
    "n_skipped_standard_lane_already_completed",
    "n_skipped_heavy_lane_already_completed",
    "expected_extreme_task_ids", "extreme_tasks_planned",
    "extreme_task_meta",
    "standard_tasks_already_completed", "heavy_tasks_already_completed",
    "extreme_non_canary_methods_refused",
    "task_lane_counts_universe", "method_counts_universe",
    "algorithm_counts_universe",
    "per_shard_planned", "per_extreme_task_planned",
    "per_method_planned_extreme", "per_algorithm_planned_extreme",
    "source_template_stages", "source_template_replicas",
    "extreme_lane_timeout_seconds_per_cell",
    "extreme_lane_default_max_evaluations",
    "extreme_lane_gate_max_evaluations",
    "extreme_lane_stage0_max_evaluations",
    "extreme_lane_include_by_default",
    "extreme_lane_requires_manual_review_before_full_stage0",
    "execution_recommendation_for_commit_51",
    "devnagari_runtime_caveat",
    "source_shard_md5_before", "source_shard_md5_after",
    "source_shards_unchanged",
    "no_committed_shard_modified_by_this_script",
    "no_training_run_by_this_script",
    "no_execution_sqlite_created_by_this_script",
    "no_runs_directory_artifacts_created_by_this_script",
    "no_raw_openml_payloads_staged_by_this_script",
    "no_standard_lane_rerun_by_this_script",
    "no_heavy_lane_rerun_by_this_script",
    "no_extreme_lane_executed_by_this_script",
    "no_full_topup_to_5_executed_by_this_script",
    "no_other_replica_executed_by_this_script",
    "no_new_signoff_file_created_by_this_script",
    "no_policy_csv_regenerated_by_this_script",
    "no_guardrails_yaml_regenerated_by_this_script",
    "operator_review_required_before_execution",
    "operator_review_required_before_replica003",
    "package_versions", "platform", "git_sha", "host", "python",
    "next_recommended_step",
)


def test_build_produces_planned_summary_with_all_keys(tmp_path: Path) -> None:
    summary = _build(tmp_path, write_summary=True)
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    # core invariants
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["replica"] == 2
    assert payload["lane"] == "extreme"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["expected_extreme_canary_cells"] == 24
    assert payload["n_runnable_extreme_canary"] == 24
    assert payload["n_refused_extreme_non_canary"] == 42
    assert payload["n_skipped_standard_lane_already_completed"] == 1815
    assert payload["n_skipped_heavy_lane_already_completed"] == 423
    assert payload["extreme_tasks_planned"] == [6, 167121]
    assert payload["expected_extreme_task_ids"] == [6, 167121]
    assert payload["policy_version"] == PINNED_POLICY_VERSION
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    # All SHA-256 fields present and 64 chars
    for key in (
        "signoff_sha256",
        "stage3_topup_plan_summary_sha256",
        "commit48_standard_lane_summary_sha256",
        "commit49_heavy_lane_summary_sha256",
    ):
        assert isinstance(payload[key], str), key
        assert len(payload[key]) == 64, (key, payload[key])
    # Execution policy recommendations
    rec = payload["execution_recommendation_for_commit_51"]
    assert rec["require_explicit_include_extreme_tasks_flag"] is True
    assert rec["require_explicit_execute_extreme_lane_flag"] is True
    assert rec["max_evaluations_recommended"] == 1
    assert float(rec["timeout_seconds_per_cell_recommended"]) == 14400.0
    # extreme lane spec mirrored
    assert payload["extreme_lane_stage0_max_evaluations"] == 1
    assert payload["extreme_lane_timeout_seconds_per_cell"] == 14400.0
    assert payload["extreme_lane_include_by_default"] is False
    # invariants
    assert payload["no_training_run_by_this_script"] is True
    assert payload["no_execution_sqlite_created_by_this_script"] is True
    assert payload["no_runs_directory_artifacts_created_by_this_script"] is True
    assert payload["no_standard_lane_rerun_by_this_script"] is True
    assert payload["no_heavy_lane_rerun_by_this_script"] is True
    assert payload["no_extreme_lane_executed_by_this_script"] is True
    assert payload["no_full_topup_to_5_executed_by_this_script"] is True
    assert payload["no_other_replica_executed_by_this_script"] is True
    assert payload["no_new_signoff_file_created_by_this_script"] is True
    assert payload["no_policy_csv_regenerated_by_this_script"] is True
    assert payload["no_guardrails_yaml_regenerated_by_this_script"] is True
    assert payload["no_committed_shard_modified_by_this_script"] is True
    assert payload["operator_review_required_before_execution"] is True
    assert payload["operator_review_required_before_replica003"] is True
    # Devnagari caveat is present in the summary text
    caveat = payload["devnagari_runtime_caveat"]
    assert "Devnagari-Script" in caveat
    assert "policy_version" in caveat or "stage0_max_evaluations" in caveat
    # Make sure the returned summary dict matches the file
    assert summary["execution_status"] == "planned_not_executed"


def test_build_does_not_touch_committed_shards(tmp_path: Path) -> None:
    """Committed source shards must be byte-identical before and after
    building the plan."""
    md5_before = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    _build(tmp_path, write_summary=True)
    md5_after = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after


def test_build_does_not_create_runs_or_execution_sqlite(
    tmp_path: Path,
) -> None:
    """The planner never writes anything under runs/cc18/ for the
    replica_002 extreme run id."""
    _build(tmp_path, write_summary=True)
    runs_dir = (
        REPO / "runs/cc18/stage3_replica_002_extreme_lane_plan_latest"
    )
    assert not runs_dir.exists(), runs_dir
    # No execution SQLite anywhere created by this build
    leftover = list(tmp_path.rglob("*.execution.sqlite"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Gitignore / commit hygiene
# ---------------------------------------------------------------------------


def test_summary_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_plan_latest_summary.json",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_plan_latest_summary.md",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_plan_latest/extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout


def test_runs_extreme_plan_dir_is_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage3_replica_002_extreme_lane_plan_latest/"
         "shards/stage0_replica_001/shard_00.execution.sqlite"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    # Even though we never expect to create this directory, the
    # gitignore must catch it if a buggy future script ever did.
    assert res.returncode == 0, (res.stdout, res.stderr)


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def test_extreme_plan_doc_mentions_devnagari_caveat() -> None:
    assert PLAN_DOC.exists()
    text = PLAN_DOC.read_text(encoding="utf-8")
    assert "Devnagari-Script" in text
    assert "devnagari" in text.lower() or "Devnagari" in text
    # Must clearly explain why this is planning-only
    assert "planning-only" in text.lower() or "planning only" in text.lower()
    # Must document the execution policy for Commit 51
    assert "Commit 51" in text
    assert "stage0_max_evaluations" in text
    assert "14,400" in text or "14400" in text


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.plan_stage3_replica002_extreme_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_HEAVY_LANE_SUMMARY,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
        DEFAULT_SIGNOFF_FILE,
        DEFAULT_STANDARD_LANE_SUMMARY,
        DEFAULT_TOPUP_PLAN_SUMMARY,
        EXPECTED_EXTREME_CANARY_CELLS,
        EXPECTED_EXTREME_TASK_IDS,
    )
    from scripts.plan_stage3_replica002_extreme_lane import (
        PINNED_POLICY_VERSION as PV,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
    assert DEFAULT_SIGNOFF_FILE == SIGNOFF_FILE
    assert DEFAULT_TOPUP_PLAN_SUMMARY == TOPUP_PLAN_SUMMARY
    assert DEFAULT_STANDARD_LANE_SUMMARY == STANDARD_LANE_SUMMARY
    assert DEFAULT_HEAVY_LANE_SUMMARY == HEAVY_LANE_SUMMARY
    assert EXPECTED_EXTREME_CANARY_CELLS == 24
    assert tuple(sorted(EXPECTED_EXTREME_TASK_IDS)) == (6, 167121)
    assert PV == PINNED_POLICY_VERSION


def test_live_signoff_plan_std_and_hvy_match_pinned_policy() -> None:
    """All four upstream artifacts must pin the same policy_version
    as the live heavy_task_policy.csv at the moment Commit 50 lands."""
    live_pv = hashlib.sha256(POLICY_CSV.read_bytes()).hexdigest()
    signoff = json.loads(SIGNOFF_FILE.read_text(encoding="utf-8"))
    plan = json.loads(TOPUP_PLAN_SUMMARY.read_text(encoding="utf-8"))
    std = json.loads(STANDARD_LANE_SUMMARY.read_text(encoding="utf-8"))
    hvy = json.loads(HEAVY_LANE_SUMMARY.read_text(encoding="utf-8"))
    assert signoff["policy_version"] == live_pv
    assert plan["policy_version"] == live_pv
    assert std["policy_version"] == live_pv
    assert hvy["policy_version"] == live_pv
    assert std["execution_status"] == "executed"
    assert hvy["execution_status"] == "executed"
    assert int(std["n_jobs_success"]) == 684
    assert int(hvy["n_jobs_success"]) == 156
