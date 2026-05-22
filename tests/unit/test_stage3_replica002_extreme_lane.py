"""Tests for the Stage-3 / replica_002 extreme-lane execution (Commit 51).

``scripts/run_stage3_replica002_extreme_lane.py`` executes the
24-cell extreme lane that completes replica_002. Scope:

- all 10 source template shards;
- ``replica = 2`` only;
- ``extreme`` lane only;
- four canary methods only;
- 2 extreme tasks: 6 (`letter`) and 167121 (`Devnagari-Script`).

This module covers the contract that protects those invariants:

- ``--help`` / ``--dry-run`` exit zero;
- real execution refuses unless both
  ``--include-extreme-tasks`` and ``--execute-extreme-lane`` are
  passed (CLI without both flags falls back to a planning-only
  report);
- the runner refuses when any of the five upstream gates is
  missing, drifted on ``policy_version``, or out of scope;
- the classifier puts extreme-canary rows in
  ``runnable_extreme``, standard rows in
  ``deferred_standard_lane``, heavy rows in
  ``deferred_heavy_lane``, and non-canary in
  ``refused_not_in_canary_set``;
- the pre-run plan reports **24** runnable extreme-lane canary
  cells across the 10 committed shards, split evenly between
  task 6 and task 167121;
- ``--skip-train`` end-to-end produces 10 execution SQLite files
  under ``runs/cc18/<run_id>/`` carrying ``replica = 2`` and the
  Stage-3 / top-up stage label, leaves every committed source
  shard byte-identical, and publishes a summary with all keys
  the prompt anchors on.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/run_stage3_replica002_extreme_lane.py"
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
EXTREME_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_extreme_lane_plan_latest_summary.json"
)
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)

EXPECTED_EXTREME_TASK_IDS = (6, 167121)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


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
        "execution_status": execution_status,
        "policy_version": policy_version,
        "tier_plans": [
            {
                "tier": "topup_to_5",
                "replica_start": 2 if include_replica2_in_topup_5 else 3,
                "replica_end": 5,
                "replica_count": 4 if include_replica2_in_topup_5 else 3,
            },
            {
                "tier": "topup_to_10",
                "replica_start": 6, "replica_end": 10, "replica_count": 5,
            },
            {
                "tier": "topup_to_30",
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


def _write_extreme_plan(
    path: Path, *,
    execution_status: str = "planned_not_executed",
    policy_version: str = PINNED_POLICY_VERSION,
    n_runnable_extreme_canary: int = 24,
    extreme_tasks_planned: list[int] | None = None,
    replica: int = 2,
    lane: str = "extreme",
    no_training_run: bool = True,
) -> Path:
    record = {
        "schema_version": 1,
        "execution_status": execution_status,
        "replica": replica,
        "source_template_replica": 1,
        "lane": lane,
        "policy_version": policy_version,
        "n_runnable_extreme_canary": n_runnable_extreme_canary,
        "extreme_tasks_planned": (
            extreme_tasks_planned
            if extreme_tasks_planned is not None
            else [6, 167121]
        ),
        "no_training_run_by_this_script": no_training_run,
        "no_execution_sqlite_created_by_this_script": True,
        "no_runs_directory_artifacts_created_by_this_script": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


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
    assert "run_stage3_replica002_extreme_lane.py" in out
    assert "--signoff-file" in out
    assert "--topup-plan-summary" in out
    assert "--standard-lane-summary" in out
    assert "--heavy-lane-summary" in out
    assert "--extreme-plan-summary" in out
    assert "--include-extreme-tasks" in out
    assert "--execute-extreme-lane" in out
    assert "--dry-run" in out


def test_run_script_dry_run_reports_planned_not_executed() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["target_replica"] == 2
    assert payload["target_stage_label"] == "stage1_topup_to_005"
    assert payload["lane"] == "extreme"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["expected_extreme_canary_cells"] == 24
    assert payload["expected_extreme_task_ids"] == [6, 167121]
    assert payload["policy_version"] == PINNED_POLICY_VERSION
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert payload["signoff_status"] == "signed"
    assert payload["topup_plan_execution_status"] == "planned_not_executed"
    assert payload["commit48_standard_lane_execution_status"] == "executed"
    assert payload["commit49_heavy_lane_execution_status"] == "executed"
    assert payload["commit50_extreme_plan_execution_status"] == (
        "planned_not_executed"
    )
    assert payload["commit50_extreme_plan_n_runnable_extreme_canary"] == 24
    assert payload["extreme_lane_max_evaluations_recommended"] == 1
    assert payload["extreme_lane_timeout_seconds_per_cell_recommended"] == 14400.0
    plan = payload["pre_run_plan"]
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_extreme"] == 24
    assert plan["n_deferred_standard_lane"] == 1815
    assert plan["n_deferred_heavy_lane"] == 423
    assert plan["n_refused_not_in_canary_set"] == 42
    assert plan["extreme_tasks_executed"] == [6, 167121]


def test_run_script_without_flags_does_not_execute() -> None:
    """Running with NO flags (no --dry-run, no --skip-train, neither
    extreme flag) MUST NOT execute — the script falls back to a
    planning-only report identical to --dry-run."""
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert "execute-extreme-lane" in payload["reason_not_executed"]
    assert payload["include_extreme_tasks_flag"] is False
    assert payload["execute_extreme_lane_flag"] is False


def test_run_script_only_include_flag_does_not_execute() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--include-extreme-tasks"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["include_extreme_tasks_flag"] is True
    assert payload["execute_extreme_lane_flag"] is False


def test_run_script_only_execute_flag_does_not_execute() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--execute-extreme-lane"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["include_extreme_tasks_flag"] is False
    assert payload["execute_extreme_lane_flag"] is True


# ---------------------------------------------------------------------------
# Gate refusals
# ---------------------------------------------------------------------------


def test_refuses_when_signoff_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_signoff,
    )

    with pytest.raises(GateRefusalError, match="signoff file not found"):
        verify_signoff(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_signoff_not_signed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", signoff_status="planned_not_signed",
    )
    with pytest.raises(GateRefusalError, match="signoff_status"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_signoff_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", policy_version="f" * 64,
    )
    with pytest.raises(GateRefusalError, match="policy_version"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_topup_plan_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_topup_plan,
    )

    with pytest.raises(GateRefusalError, match="stage3 top-up plan"):
        verify_topup_plan(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    with pytest.raises(GateRefusalError, match="Commit 48 standard-lane"):
        verify_standard_lane_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_failed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard(
        tmp_path / "std.json", n_jobs_failed=2, n_jobs_success=684,
    )
    with pytest.raises(GateRefusalError, match="n_jobs_failed"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_heavy_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_heavy_lane_summary,
    )

    with pytest.raises(GateRefusalError, match="Commit 49 heavy-lane"):
        verify_heavy_lane_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_heavy_summary_failed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_heavy_lane_summary,
    )

    p = _write_heavy(
        tmp_path / "hvy.json", n_jobs_failed=1, n_jobs_success=156,
    )
    with pytest.raises(GateRefusalError, match="n_jobs_failed"):
        verify_heavy_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_extreme_plan_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_extreme_plan_summary,
    )

    with pytest.raises(GateRefusalError, match="Commit 50 extreme-lane"):
        verify_extreme_plan_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_extreme_plan_not_planning_only(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_extreme_plan_summary,
    )

    p = _write_extreme_plan(
        tmp_path / "plan.json", execution_status="executed",
    )
    with pytest.raises(GateRefusalError, match="execution_status"):
        verify_extreme_plan_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_extreme_plan_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_extreme_plan_summary,
    )

    p = _write_extreme_plan(tmp_path / "plan.json", policy_version="f" * 64)
    with pytest.raises(GateRefusalError, match="policy_version"):
        verify_extreme_plan_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_extreme_plan_wrong_count(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_extreme_plan_summary,
    )

    p = _write_extreme_plan(
        tmp_path / "plan.json", n_runnable_extreme_canary=23,
    )
    with pytest.raises(GateRefusalError, match="n_runnable_extreme_canary"):
        verify_extreme_plan_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_extreme_plan_wrong_tasks(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        GateRefusalError,
        verify_extreme_plan_summary,
    )

    p = _write_extreme_plan(
        tmp_path / "plan.json", extreme_tasks_planned=[6, 999],
    )
    with pytest.raises(GateRefusalError, match="extreme_tasks_planned"):
        verify_extreme_plan_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_replica002_extreme_lane import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # extreme × canary -> runnable_extreme
        ("j_ex_can_1", 6, "default_gbdt", "xgboost"),
        ("j_ex_can_2", 167121, "tpe_optuna", "lightgbm"),
        ("j_ex_can_3", 167121, "doe_rsm_vrf_true_nbi", "catboost"),
        # extreme × non-canary -> refused_not_in_canary_set
        ("j_ex_nc_1", 6, "smac3", "xgboost"),
        ("j_ex_nc_2", 167121, "asha", "catboost"),
        # standard -> deferred_standard_lane (incl. isolet)
        ("j_s1", 11, "default_gbdt", "xgboost"),
        ("j_s2", 3481, "tpe_optuna", "catboost"),
        # heavy -> deferred_heavy_lane
        ("j_h1", 3573, "default_gbdt", "xgboost"),
        ("j_h2", 167124, "tpe_optuna", "lightgbm"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(e["job_id"] for e in buckets["runnable_extreme"]) == [
        "j_ex_can_1", "j_ex_can_2", "j_ex_can_3",
    ]
    assert sorted(e["job_id"] for e in buckets["refused_not_in_canary_set"]) == [
        "j_ex_nc_1", "j_ex_nc_2",
    ]
    assert sorted(
        e["job_id"] for e in buckets["deferred_standard_lane"]
    ) == ["j_s1", "j_s2"]
    assert sorted(
        e["job_id"] for e in buckets["deferred_heavy_lane"]
    ) == ["j_h1", "j_h2"]


def test_pre_run_plan_across_all_10_shards() -> None:
    """Plan over the 10 committed stage-0 shards reports 24 runnable
    extreme-lane canary cells across exactly tasks 6 and 167121."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_replica002_extreme_lane import (
        build_pre_run_plan,
    )

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_extreme"] == 24
    assert plan["n_deferred_standard_lane"] == 1815
    assert plan["n_deferred_heavy_lane"] == 423
    assert plan["n_refused_not_in_canary_set"] == 42
    assert plan["extreme_tasks_executed"] == [6, 167121]
    assert plan["non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }
    assert plan["source_template_replicas"] == [1]


def test_extreme_task_universe_under_pinned_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    extreme_tids = sorted(
        tid for tid, p in g.tasks.items() if p.lane == "extreme"
    )
    assert extreme_tids == [6, 167121]


# ---------------------------------------------------------------------------
# Gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_stage3_replica002_extreme_lane import DEFAULT_RUN_ROOT

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage3_replica_002_extreme_lane_latest/run_manifest.json",
         "runs/cc18/stage3_replica_002_extreme_lane_latest/"
         "shards/stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/stage3_replica_002_extreme_lane_latest/"
         "shards/stage0_replica_001/shard_09.execution.sqlite"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 3


def test_stage_runs_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_latest_summary.json",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_latest_summary.md",
         "experiments/_stage_runs/"
         "stage3_replica_002_extreme_lane_latest/extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout


# ---------------------------------------------------------------------------
# Skip-train end-to-end
# ---------------------------------------------------------------------------


SUMMARY_REQUIRED_KEYS = (
    # protocol-level keys
    "schema_version", "run_id", "stage", "exported_at",
    "source_git_sha", "host", "python", "package_versions",
    "n_total", "n_pending", "n_claimed", "n_running",
    "n_success", "n_failed", "n_skipped", "status_counts",
    "shards", "n_shards", "run_dir", "run_manifest_path",
    "execution_suffix", "source_shards_unchanged",
    "source_md5_recorded", "source_md5_now", "source_drift",
    "stage3_signoff_present", "stage3_signoff_path",
    "protocol_doc",
    # commit 51 augmentation
    "batch_id", "topup_tier", "execution_status", "replica",
    "source_template_replica", "lane",
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
    "commit50_extreme_plan_summary_path",
    "commit50_extreme_plan_summary_sha256",
    "commit50_extreme_plan_execution_status",
    "commit50_extreme_plan_n_runnable_extreme_canary",
    "n_source_shards", "source_shards", "execution_shards",
    "execution_sqlite_sha256",
    "n_jobs_total", "n_jobs_executed", "n_jobs_success",
    "n_jobs_deferred_standard", "n_jobs_deferred_heavy",
    "n_jobs_refused_non_canary", "n_jobs_failed",
    "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after", "n_jobs_running_after",
    "status_counts_extended", "task_lane_counts_universe",
    "extreme_tasks_executed", "extreme_tasks_in_universe",
    "extreme_task_meta", "standard_tasks_deferred",
    "heavy_tasks_deferred", "non_canary_methods_refused",
    "expected_extreme_canary_cells",
    "per_task_status_breakdown", "per_method_status_breakdown",
    "per_algorithm_status_breakdown",
    "per_shard_status", "per_shard_planned",
    "cells_runnable_per_shard",
    "method_counts_universe", "algorithm_counts_universe",
    "metric_keys", "slowest_cells", "cells",
    "runtime_seconds_runner_total", "runner_invocations",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed",
    "source_shard_md5_before", "source_shard_md5_after",
    "source_shard_md5_after_copy", "source_shard_md5_after_rewrite",
    "execution_copy_rewrite",
    "platform", "git_sha", "capability_audit",
    "extreme_lane_max_evaluations_used",
    "extreme_lane_timeout_seconds_per_cell_used",
    "extreme_lane_max_evaluations_note",
    "no_other_replica_executed_by_this_script",
    "no_full_topup_to_5_executed_by_this_script",
    "no_standard_lane_rerun_by_this_script",
    "no_heavy_lane_rerun_by_this_script",
    "no_committed_shard_modified_by_this_script",
    "no_raw_openml_payloads_staged_by_this_script",
    "no_execution_sqlite_staged_by_this_script",
    "only_replica_002_extreme_lane_executed",
    "operator_review_required_before_replica002_signoff",
    "next_recommended_step",
)


def _run_skip_train(
    tmp_path: Path, *,
    run_id: str = "test_c51_skip_train",
    signoff: Path | None = None,
    topup: Path | None = None,
    standard: Path | None = None,
    heavy: Path | None = None,
    plan: Path | None = None,
) -> dict:
    from scripts.run_stage3_replica002_extreme_lane import (
        run_replica002_extreme_lane,
    )

    return run_replica002_extreme_lane(
        shards_dir=SHARDS_DIR,
        signoff_file=signoff or _write_signoff(tmp_path / "stage3_signoff.json"),
        topup_plan_summary=topup or _write_topup_plan(
            tmp_path / "stage3_topup_plan_latest_summary.json",
        ),
        standard_lane_summary=standard or _write_standard(tmp_path / "std.json"),
        heavy_lane_summary=heavy or _write_heavy(tmp_path / "hvy.json"),
        extreme_plan_summary=plan or _write_extreme_plan(tmp_path / "plan.json"),
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        policy_csv=POLICY_CSV,
        guardrails_yaml=GUARDRAILS_YAML,
        run_id=run_id,
        skip_train=True,
    )


def test_skip_train_copies_all_10_shards_and_preserves_md5(
    tmp_path: Path,
) -> None:
    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    summary = _run_skip_train(tmp_path, run_id="t_c51_copy_md5")
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["no_committed_shard_modified_by_this_script"] is True
    exec_dir = (
        tmp_path / "runs" / "t_c51_copy_md5"
        / "shards" / "stage0_replica_001"
    )
    exec_files = sorted(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 10


def test_skip_train_all_10_copies_have_replica_2_and_topup_stage(
    tmp_path: Path,
) -> None:
    summary = _run_skip_train(tmp_path, run_id="t_c51_replica2")
    exec_dir = (
        tmp_path / "runs" / "t_c51_replica2"
        / "shards" / "stage0_replica_001"
    )
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(f"file:{exec_p}?mode=ro", uri=True)
        try:
            replicas = sorted({
                r[0] for r in cx.execute(
                    "SELECT DISTINCT replica FROM cc18_jobs",
                )
            })
            stages = sorted({
                r[0] for r in cx.execute(
                    "SELECT DISTINCT stage FROM cc18_jobs",
                )
            })
        finally:
            cx.close()
        assert replicas == [2]
        assert stages == ["stage1_topup_to_005"]
    assert summary["replica"] == 2
    assert summary["stage"] == "stage1_topup_to_005"


def test_skip_train_lane_isolation(tmp_path: Path) -> None:
    """In skip-train mode no extreme cell actually runs, but standard
    / heavy / non-canary rows must be pre-marked as 'skipped' with
    the right last_error."""
    _run_skip_train(tmp_path, run_id="t_c51_lane_isolation")
    exec_dir = (
        tmp_path / "runs" / "t_c51_lane_isolation"
        / "shards" / "stage0_replica_001"
    )
    # standard tasks (sample)
    std_tids = (3481, 11, 53, 16, 14970)
    # heavy tasks (sample)
    hvy_tids = (3573, 167124, 146825)
    # extreme tasks
    ext_tids = (6, 167121)
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(f"file:{exec_p}?mode=ro", uri=True)
        try:
            std_rows = list(cx.execute(
                "SELECT openml_task_id, status, last_error FROM cc18_jobs "
                "WHERE openml_task_id IN "
                f"({','.join('?' * len(std_tids))})",
                std_tids,
            ))
            hvy_rows = list(cx.execute(
                "SELECT openml_task_id, status, last_error FROM cc18_jobs "
                "WHERE openml_task_id IN "
                f"({','.join('?' * len(hvy_tids))})",
                hvy_tids,
            ))
            ext_rows = list(cx.execute(
                "SELECT openml_task_id, status, last_error, method "
                "FROM cc18_jobs "
                "WHERE openml_task_id IN "
                f"({','.join('?' * len(ext_tids))})",
                ext_tids,
            ))
        finally:
            cx.close()
        for tid, status, err in std_rows:
            assert status == "skipped", (tid, status, err)
            assert err == "deferred_standard_lane", (tid, status, err)
        for tid, status, err in hvy_rows:
            assert status == "skipped", (tid, status, err)
            assert err == "deferred_heavy_lane", (tid, status, err)
        # Extreme canary rows stay 'pending' in skip-train (they
        # would have been executed); extreme non-canary rows are
        # pre-marked refused.
        canary = ("default_gbdt", "random_search", "tpe_optuna",
                  "doe_rsm_vrf_true_nbi")
        for tid, status, err, method in ext_rows:
            if method in canary:
                assert status == "pending", (tid, status, err, method)
            else:
                assert status == "skipped", (tid, status, err, method)
                assert err == "refused_not_in_canary_set", (
                    tid, status, err, method,
                )


def test_skip_train_summary_has_all_required_keys(tmp_path: Path) -> None:
    _run_skip_train(tmp_path, run_id="t_c51_schema")
    json_p = tmp_path / "stage_runs" / "t_c51_schema_summary.json"
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    assert len(payload["policy_version"]) == 64
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert len(payload["signoff_sha256"]) == 64
    assert len(payload["stage3_topup_plan_summary_sha256"]) == 64
    assert len(payload["commit48_standard_lane_summary_sha256"]) == 64
    assert len(payload["commit49_heavy_lane_summary_sha256"]) == 64
    assert len(payload["commit50_extreme_plan_summary_sha256"]) == 64
    assert payload["execution_status"] == "executed"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["lane"] == "extreme"
    assert payload["expected_extreme_canary_cells"] == 24
    assert payload["n_jobs_total"] == 2304
    assert payload["n_jobs_deferred_standard"] == 1815
    assert payload["n_jobs_deferred_heavy"] == 423
    assert payload["n_jobs_refused_non_canary"] == 42
    assert payload["extreme_lane_max_evaluations_used"] == 1
    assert payload["extreme_lane_timeout_seconds_per_cell_used"] == 14400.0
    assert "max_evaluations=1" in payload["extreme_lane_max_evaluations_note"]
    assert payload["no_full_topup_to_5_executed_by_this_script"] is True
    assert payload["no_standard_lane_rerun_by_this_script"] is True
    assert payload["no_heavy_lane_rerun_by_this_script"] is True
    assert payload["no_other_replica_executed_by_this_script"] is True
    assert payload["no_committed_shard_modified_by_this_script"] is True
    assert payload["only_replica_002_extreme_lane_executed"] is True
    assert payload[
        "operator_review_required_before_replica002_signoff"
    ] is True
    assert len(payload["execution_sqlite_sha256"]) == 10
    for h in payload["execution_sqlite_sha256"].values():
        assert isinstance(h, str) and len(h) == 64


def test_skip_train_refuses_on_missing_signoff(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="signoff file not found"):
        _run_skip_train(
            tmp_path,
            signoff=tmp_path / "absent_signoff.json",
            run_id="t_c51_no_signoff",
        )


def test_skip_train_refuses_on_missing_topup_plan(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="stage3 top-up plan"):
        _run_skip_train(
            tmp_path,
            topup=tmp_path / "absent_plan.json",
            run_id="t_c51_no_plan",
        )


def test_skip_train_refuses_on_missing_standard_summary(
    tmp_path: Path,
) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="Commit 48 standard-lane"):
        _run_skip_train(
            tmp_path,
            standard=tmp_path / "absent_std.json",
            run_id="t_c51_no_std",
        )


def test_skip_train_refuses_on_missing_heavy_summary(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="Commit 49 heavy-lane"):
        _run_skip_train(
            tmp_path,
            heavy=tmp_path / "absent_hvy.json",
            run_id="t_c51_no_hvy",
        )


def test_skip_train_refuses_on_missing_extreme_plan(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="Commit 50 extreme-lane"):
        _run_skip_train(
            tmp_path,
            plan=tmp_path / "absent_plan.json",
            run_id="t_c51_no_extreme_plan",
        )


def test_skip_train_refuses_on_extreme_plan_already_executed(
    tmp_path: Path,
) -> None:
    from scripts.run_stage3_replica002_extreme_lane import GateRefusalError

    p = _write_extreme_plan(
        tmp_path / "executed_plan.json", execution_status="executed",
    )
    with pytest.raises(GateRefusalError, match="execution_status"):
        _run_skip_train(
            tmp_path,
            plan=p,
            run_id="t_c51_executed_plan",
        )


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage3_replica002_extreme_lane import (
        DEFAULT_EXTREME_PLAN_SUMMARY,
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_HEAVY_LANE_SUMMARY,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
        DEFAULT_SIGNOFF_FILE,
        DEFAULT_STANDARD_LANE_SUMMARY,
        DEFAULT_TOPUP_PLAN_SUMMARY,
        EXPECTED_EXTREME_CANARY_CELLS,
        TARGET_REPLICA,
        TARGET_STAGE_LABEL,
    )
    from scripts.run_stage3_replica002_extreme_lane import (
        EXPECTED_EXTREME_TASK_IDS as EET,
    )
    from scripts.run_stage3_replica002_extreme_lane import (
        PINNED_POLICY_VERSION as PV,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
    assert DEFAULT_SIGNOFF_FILE == SIGNOFF_FILE
    assert DEFAULT_TOPUP_PLAN_SUMMARY == TOPUP_PLAN_SUMMARY
    assert DEFAULT_STANDARD_LANE_SUMMARY == STANDARD_LANE_SUMMARY
    assert DEFAULT_HEAVY_LANE_SUMMARY == HEAVY_LANE_SUMMARY
    assert DEFAULT_EXTREME_PLAN_SUMMARY == EXTREME_PLAN_SUMMARY
    assert TARGET_REPLICA == 2
    assert TARGET_STAGE_LABEL == "stage1_topup_to_005"
    assert EXPECTED_EXTREME_CANARY_CELLS == 24
    assert tuple(sorted(EET)) == EXPECTED_EXTREME_TASK_IDS
    assert PV == PINNED_POLICY_VERSION


def test_live_all_five_gates_match_pinned_policy() -> None:
    """All five upstream artifacts must pin the same policy_version
    as the live heavy_task_policy.csv at the moment Commit 51 lands."""
    live_pv = hashlib.sha256(POLICY_CSV.read_bytes()).hexdigest()
    signoff = json.loads(SIGNOFF_FILE.read_text(encoding="utf-8"))
    plan = json.loads(TOPUP_PLAN_SUMMARY.read_text(encoding="utf-8"))
    std = json.loads(STANDARD_LANE_SUMMARY.read_text(encoding="utf-8"))
    hvy = json.loads(HEAVY_LANE_SUMMARY.read_text(encoding="utf-8"))
    plan50 = json.loads(EXTREME_PLAN_SUMMARY.read_text(encoding="utf-8"))
    assert signoff["policy_version"] == live_pv
    assert plan["policy_version"] == live_pv
    assert std["policy_version"] == live_pv
    assert hvy["policy_version"] == live_pv
    assert plan50["policy_version"] == live_pv
    assert std["execution_status"] == "executed"
    assert hvy["execution_status"] == "executed"
    assert plan50["execution_status"] == "planned_not_executed"
    assert int(std["n_jobs_success"]) == 684
    assert int(hvy["n_jobs_success"]) == 156
    assert plan50["n_runnable_extreme_canary"] == 24
    assert plan50["extreme_tasks_planned"] == [6, 167121]
