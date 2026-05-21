"""Tests for the Stage-3 / replica_002 heavy-lane runner (Commit 49).

``scripts/run_stage3_replica002_heavy_lane.py`` is the heavy-lane
companion to Commit 48's standard-lane pass. Scope:

- all 10 source template shards;
- ``replica = 2`` only;
- ``heavy`` lane only;
- four canary methods only.

This module covers the contract that protects those invariants:

- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses when the signoff is missing, not signed, or
  drifted on ``policy_version``;
- the runner refuses when the Stage-3 top-up plan is missing,
  drifted, or does not list ``replica = 2`` under ``topup_to_5``;
- the runner refuses when the Commit 48 standard-lane summary is
  missing, failed, of the wrong scope, or drifted on
  ``policy_version``;
- the classifier puts heavy-canary rows in ``runnable_heavy``,
  standard rows in ``deferred_standard_lane``, extreme rows in
  ``deferred_extreme_lane``, and non-canary in
  ``refused_not_in_canary_set``;
- the pre-run plan reports **156** runnable heavy-lane canary
  cells across the 10 committed shards;
- isolet (task 3481) is NOT promoted to heavy by this commit;
- a ``--skip-train`` end-to-end run produces 10 execution SQLite
  files under ``runs/cc18/<run_id>/`` carrying ``replica = 2`` and
  the Stage-3 / top-up stage label, leaves every committed source
  shard byte-identical, and publishes a summary with all keys the
  prompt anchors on;
- ``runs/`` + execution SQLite files are gitignored; only the
  summary JSON / MD pair under ``experiments/_stage_runs/`` is
  allowlisted.
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
RUN_SCRIPT = REPO / "scripts/run_stage3_replica002_heavy_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
TOPUP_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs/stage3_topup_plan_latest_summary.json"
)
STANDARD_LANE_SUMMARY = (
    REPO / "experiments/_stage_runs"
    / "stage3_replica_002_standard_lane_latest_summary.json"
)
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)

EXPECTED_HEAVY_TASKS = (
    32, 219, 3573, 7592, 9910, 9981, 14965,
    146195, 146825, 167119, 167120, 167124, 167125,
)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


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
                "task_id": 3481,
                "dataset": "isolet",
                "summary": "future recalibration candidate",
            },
            {
                "id": "devnagari_extreme_budget_non_equivalence",
                "task_id": 167121,
                "dataset": "Devnagari-Script",
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
        "policy_drift_detected": False,
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


def _write_standard_lane(
    path: Path, *,
    execution_status: str = "executed",
    replica: int = 2,
    lane: str = "standard",
    policy_version: str = PINNED_POLICY_VERSION,
    n_jobs_success: int = 684,
    n_jobs_failed: int = 0,
    n_jobs_failed_timeout: int = 0,
    n_jobs_pending_after: int = 0,
    n_jobs_running_after: int = 0,
    source_shards_unchanged: bool = True,
    no_full_topup_to_5_executed: bool = True,
    no_heavy_lane_executed: bool = True,
) -> Path:
    record = {
        "schema_version": 1,
        "run_id": "stage3_replica_002_standard_lane_latest",
        "batch_id": "stage3_replica_002_standard_lane",
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
        "no_full_topup_to_5_executed_by_this_script":
            no_full_topup_to_5_executed,
        "no_heavy_lane_executed_by_this_script": no_heavy_lane_executed,
        "no_extreme_lane_executed_by_this_script": True,
        "runtime_seconds_runner_total": 7287.9,
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
    assert "run_stage3_replica002_heavy_lane.py" in out
    assert "--signoff-file" in out
    assert "--topup-plan-summary" in out
    assert "--standard-lane-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--target-stage-label" in out
    assert "--target-replica" in out
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
    assert payload["lane"] == "heavy"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["expected_heavy_canary_cells"] == 156
    assert payload["policy_version"] == PINNED_POLICY_VERSION
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert payload["signoff_status"] == "signed"
    assert payload["signoff_type"] == "stage0_replica_001"
    assert payload["topup_plan_execution_status"] == "planned_not_executed"
    assert payload["commit48_standard_lane_execution_status"] == "executed"
    assert payload["commit48_standard_lane_n_jobs_success"] == 684
    # isolet must not be promoted
    assert payload["isolet_task_id"] == 3481
    assert payload["isolet_lane_under_pinned_policy"] == "standard"
    assert payload["isolet_promoted_to_heavy_in_this_commit"] is False
    plan = payload["pre_run_plan"]
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_heavy"] == 156
    assert len(payload["per_shard"]) == 10


# ---------------------------------------------------------------------------
# Signoff refusal
# ---------------------------------------------------------------------------


def test_refuses_when_signoff_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_signoff,
    )

    with pytest.raises(GateRefusalError, match="signoff file not found"):
        verify_signoff(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_signoff_not_signed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", signoff_status="planned_not_signed",
    )
    with pytest.raises(GateRefusalError, match="signoff_status"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_signoff_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", policy_version="f" * 64,
    )
    with pytest.raises(GateRefusalError, match="policy_version"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Top-up plan refusal
# ---------------------------------------------------------------------------


def test_refuses_when_topup_plan_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_topup_plan,
    )

    with pytest.raises(GateRefusalError, match="stage3 top-up plan"):
        verify_topup_plan(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_replica2_not_in_topup_5(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_topup_plan,
    )

    p = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
        include_replica2_in_topup_5=False,
    )
    with pytest.raises(GateRefusalError, match="replica=2"):
        verify_topup_plan(p, expected_policy_version=PINNED_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Commit 48 standard-lane summary refusal
# ---------------------------------------------------------------------------


def test_refuses_when_standard_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    with pytest.raises(GateRefusalError, match="Commit 48 standard-lane"):
        verify_standard_lane_summary(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_not_executed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(
        tmp_path / "std.json", execution_status="planned_not_executed",
    )
    with pytest.raises(GateRefusalError, match="execution_status"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_failed(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(
        tmp_path / "std.json", n_jobs_failed=5, n_jobs_success=684,
    )
    with pytest.raises(GateRefusalError, match="n_jobs_failed"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_wrong_success(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(tmp_path / "std.json", n_jobs_success=680)
    with pytest.raises(GateRefusalError, match="n_jobs_success"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_wrong_replica(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(tmp_path / "std.json", replica=3)
    with pytest.raises(GateRefusalError, match="replica"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_wrong_lane(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(tmp_path / "std.json", lane="heavy")
    with pytest.raises(GateRefusalError, match="lane"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(tmp_path / "std.json", policy_version="f" * 64)
    with pytest.raises(GateRefusalError, match="policy_version"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_source_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(
        tmp_path / "std.json", source_shards_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_standard_summary_ran_heavy(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    p = _write_standard_lane(
        tmp_path / "std.json", no_heavy_lane_executed=False,
    )
    with pytest.raises(GateRefusalError, match="no_heavy_lane"):
        verify_standard_lane_summary(
            p, expected_policy_version=PINNED_POLICY_VERSION,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_replica002_heavy_lane import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # extreme task -> deferred_extreme_lane
        ("j_ex1", 6, "default_gbdt", "xgboost"),
        ("j_ex2", 167121, "tpe_optuna", "lightgbm"),
        # standard task (incl. isolet) -> deferred_standard_lane
        ("j_s1", 11, "default_gbdt", "xgboost"),
        ("j_s2", 3481, "tpe_optuna", "catboost"),  # isolet stays standard
        # heavy task × non-canary -> refused
        ("j_r1", 3573, "smac3", "xgboost"),
        ("j_r2", 167124, "asha", "catboost"),
        # heavy task × canary -> runnable_heavy
        ("j_run1", 3573, "default_gbdt", "xgboost"),
        ("j_run2", 167124, "tpe_optuna", "lightgbm"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(e["job_id"] for e in buckets["deferred_extreme_lane"]) == [
        "j_ex1", "j_ex2",
    ]
    assert sorted(e["job_id"] for e in buckets["deferred_standard_lane"]) == [
        "j_s1", "j_s2",
    ]
    assert sorted(e["job_id"] for e in buckets["refused_not_in_canary_set"]) == [
        "j_r1", "j_r2",
    ]
    assert sorted(e["job_id"] for e in buckets["runnable_heavy"]) == [
        "j_run1", "j_run2",
    ]


def test_pre_run_plan_across_all_10_shards() -> None:
    """Pre-run plan over the 10 committed stage-0 shards reports 156
    runnable heavy-lane canary cells across 2,304 total."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_replica002_heavy_lane import build_pre_run_plan

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_heavy"] == 156
    assert plan["n_deferred_extreme_lane"] == 66
    # Standard-lane rows = 684 canary + 1131 non-canary = 1815
    assert plan["n_deferred_standard_lane"] == 684 + 1131
    # Non-canary refusals only count heavy-lane × non-canary methods
    assert plan["n_refused_not_in_canary_set"] == 267
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }
    assert plan["non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    assert sorted(plan["heavy_tasks_executed"]) == list(EXPECTED_HEAVY_TASKS)
    assert plan["source_template_replicas"] == [1]
    assert plan["source_template_stages"] == ["stage0_replica_001"]
    # isolet stays standard, not heavy
    assert plan["isolet_task_id"] == 3481
    assert plan["isolet_lane_under_pinned_policy"] == "standard"
    assert plan["isolet_promoted_to_heavy_in_this_commit"] is False


def test_isolet_stays_standard_under_pinned_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    assert g.get_task_lane(3481) == "standard"


# ---------------------------------------------------------------------------
# Gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_stage3_replica002_heavy_lane import DEFAULT_RUN_ROOT

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage3_replica_002_heavy_lane_latest/run_manifest.json",
         "runs/cc18/stage3_replica_002_heavy_lane_latest/"
         "shards/stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/stage3_replica_002_heavy_lane_latest/"
         "shards/stage0_replica_001/shard_09.execution.sqlite"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 3


def test_stage_runs_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/"
         "stage3_replica_002_heavy_lane_latest_summary.json",
         "experiments/_stage_runs/"
         "stage3_replica_002_heavy_lane_latest_summary.md",
         "experiments/_stage_runs/"
         "stage3_replica_002_heavy_lane_latest/extras.bin"],
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
    # commit 49 augmentation
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
    "n_source_shards", "source_shards", "execution_shards",
    "execution_sqlite_sha256",
    "n_jobs_total", "n_jobs_executed", "n_jobs_success",
    "n_jobs_deferred_standard", "n_jobs_deferred_extreme",
    "n_jobs_refused_non_canary", "n_jobs_failed",
    "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after", "n_jobs_running_after",
    "status_counts_extended", "task_lane_counts_universe",
    "heavy_tasks_executed", "heavy_tasks_in_universe",
    "standard_tasks_deferred", "extreme_tasks_deferred",
    "non_canary_methods_refused", "expected_heavy_canary_cells",
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
    "isolet_task_id", "isolet_lane_under_pinned_policy",
    "isolet_promoted_to_heavy_in_this_commit", "isolet_note",
    "no_other_replica_executed_by_this_script",
    "no_full_topup_to_5_executed_by_this_script",
    "no_standard_lane_rerun_by_this_script",
    "no_extreme_lane_executed_by_this_script",
    "no_committed_shard_modified_by_this_script",
    "no_raw_openml_payloads_staged_by_this_script",
    "no_execution_sqlite_staged_by_this_script",
    "only_replica_002_heavy_lane_executed",
    "operator_review_required_before_replica002_extreme",
    "next_recommended_step",
)


def _run_skip_train(
    tmp_path: Path, *, run_id: str = "test_c49_skip_train",
    signoff: Path | None = None, topup: Path | None = None,
    standard: Path | None = None,
) -> dict:
    from scripts.run_stage3_replica002_heavy_lane import (
        run_replica002_heavy_lane,
    )

    return run_replica002_heavy_lane(
        shards_dir=SHARDS_DIR,
        signoff_file=signoff or _write_signoff(tmp_path / "stage3_signoff.json"),
        topup_plan_summary=topup or _write_topup_plan(
            tmp_path / "stage3_topup_plan_latest_summary.json",
        ),
        standard_lane_summary=standard or _write_standard_lane(
            tmp_path / "std.json",
        ),
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
    summary = _run_skip_train(tmp_path, run_id="t_c49_copy_md5")
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["no_committed_shard_modified_by_this_script"] is True
    exec_dir = (
        tmp_path / "runs" / "t_c49_copy_md5"
        / "shards" / "stage0_replica_001"
    )
    exec_files = sorted(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 10
    for p in exec_files:
        assert "runs" in p.resolve().parts


def test_skip_train_all_10_copies_have_replica_2_and_topup_stage(
    tmp_path: Path,
) -> None:
    summary = _run_skip_train(tmp_path, run_id="t_c49_replica2")
    exec_dir = (
        tmp_path / "runs" / "t_c49_replica2"
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
        assert replicas == [2], (exec_p, replicas)
        assert stages == ["stage1_topup_to_005"], (exec_p, stages)
    assert summary["replica"] == 2
    assert summary["source_template_replica"] == 1
    assert summary["stage"] == "stage1_topup_to_005"


def test_skip_train_lane_isolation_no_standard_or_extreme_run(
    tmp_path: Path,
) -> None:
    """In skip-train mode no heavy cell actually runs, but standard /
    extreme rows must be pre-marked as 'skipped' with the right
    last_error. No standard row should be flagged 'success'."""
    _run_skip_train(tmp_path, run_id="t_c49_lane_isolation")
    exec_dir = (
        tmp_path / "runs" / "t_c49_lane_isolation"
        / "shards" / "stage0_replica_001"
    )
    # standard tasks (sample: isolet 3481 + a few others)
    standard_tids = (3481, 11, 53, 16, 14970)
    extreme_tids = (6, 167121)
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(f"file:{exec_p}?mode=ro", uri=True)
        try:
            std_rows = list(cx.execute(
                "SELECT openml_task_id, status, last_error FROM cc18_jobs "
                "WHERE openml_task_id IN "
                f"({','.join('?' * len(standard_tids))})",
                standard_tids,
            ))
            ex_rows = list(cx.execute(
                "SELECT openml_task_id, status, last_error FROM cc18_jobs "
                "WHERE openml_task_id IN "
                f"({','.join('?' * len(extreme_tids))})",
                extreme_tids,
            ))
        finally:
            cx.close()
        for tid, status, err in std_rows:
            assert status == "skipped", (tid, status, err)
            assert err in {
                "deferred_standard_lane", "refused_not_in_canary_set",
            }, (tid, status, err)
            # In particular isolet is deferred_standard_lane (not promoted).
            if tid == 3481:
                assert err == "deferred_standard_lane"
        for tid, status, err in ex_rows:
            assert status == "skipped", (tid, status, err)
            assert err == "deferred_extreme_lane", (tid, status, err)


def test_skip_train_summary_has_all_required_keys(tmp_path: Path) -> None:
    _run_skip_train(tmp_path, run_id="t_c49_schema")
    json_p = tmp_path / "stage_runs" / "t_c49_schema_summary.json"
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    assert isinstance(payload["policy_version"], str)
    assert len(payload["policy_version"]) == 64
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert len(payload["signoff_sha256"]) == 64
    assert len(payload["stage3_topup_plan_summary_sha256"]) == 64
    assert len(payload["commit48_standard_lane_summary_sha256"]) == 64
    assert payload["execution_status"] == "executed"
    assert payload["topup_tier"] == "topup_to_5_partial"
    assert payload["lane"] == "heavy"
    assert payload["expected_heavy_canary_cells"] == 156
    assert payload["n_jobs_total"] == 2304
    assert payload["n_jobs_deferred_standard"] == 684 + 1131
    assert payload["n_jobs_deferred_extreme"] == 66
    assert payload["n_jobs_refused_non_canary"] == 267
    assert payload["no_full_topup_to_5_executed_by_this_script"] is True
    assert payload["no_standard_lane_rerun_by_this_script"] is True
    assert payload["no_extreme_lane_executed_by_this_script"] is True
    assert payload["no_other_replica_executed_by_this_script"] is True
    assert payload["no_committed_shard_modified_by_this_script"] is True
    assert payload["only_replica_002_heavy_lane_executed"] is True
    assert payload["operator_review_required_before_replica002_extreme"] is True
    # isolet stays standard
    assert payload["isolet_task_id"] == 3481
    assert payload["isolet_lane_under_pinned_policy"] == "standard"
    assert payload["isolet_promoted_to_heavy_in_this_commit"] is False
    assert "future" in payload["isolet_note"].lower() or "candidate" in payload[
        "isolet_note"
    ].lower()
    # 10 execution shards, each with a 64-char sha
    assert len(payload["execution_sqlite_sha256"]) == 10
    for h in payload["execution_sqlite_sha256"].values():
        assert isinstance(h, str)
        assert len(h) == 64


def test_skip_train_refuses_on_missing_signoff(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="signoff file not found"):
        _run_skip_train(
            tmp_path,
            signoff=tmp_path / "absent_signoff.json",
            run_id="t_c49_no_signoff",
        )


def test_skip_train_refuses_on_missing_topup_plan(tmp_path: Path) -> None:
    from scripts.run_stage3_replica002_heavy_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="stage3 top-up plan"):
        _run_skip_train(
            tmp_path,
            topup=tmp_path / "absent_plan.json",
            run_id="t_c49_no_plan",
        )


def test_skip_train_refuses_on_missing_standard_summary(
    tmp_path: Path,
) -> None:
    from scripts.run_stage3_replica002_heavy_lane import GateRefusalError

    with pytest.raises(GateRefusalError, match="Commit 48 standard-lane"):
        _run_skip_train(
            tmp_path,
            standard=tmp_path / "absent_std.json",
            run_id="t_c49_no_std",
        )


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage3_replica002_heavy_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
        DEFAULT_SIGNOFF_FILE,
        DEFAULT_STANDARD_LANE_SUMMARY,
        DEFAULT_TOPUP_PLAN_SUMMARY,
        EXPECTED_HEAVY_CANARY_CELLS,
        ISOLET_TASK_ID,
        TARGET_REPLICA,
        TARGET_STAGE_LABEL,
    )
    from scripts.run_stage3_replica002_heavy_lane import (
        PINNED_POLICY_VERSION as PV,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
    assert DEFAULT_SIGNOFF_FILE == SIGNOFF_FILE
    assert DEFAULT_TOPUP_PLAN_SUMMARY == TOPUP_PLAN_SUMMARY
    assert DEFAULT_STANDARD_LANE_SUMMARY == STANDARD_LANE_SUMMARY
    assert TARGET_REPLICA == 2
    assert TARGET_STAGE_LABEL == "stage1_topup_to_005"
    assert EXPECTED_HEAVY_CANARY_CELLS == 156
    assert ISOLET_TASK_ID == 3481
    assert PV == PINNED_POLICY_VERSION


def test_live_signoff_plan_and_standard_match_pinned_policy() -> None:
    """All three gates and the live heavy_task_policy.csv must pin the
    same policy_version. Commit 48 standard-lane summary must be
    green with 684 success."""
    live_pv = hashlib.sha256(POLICY_CSV.read_bytes()).hexdigest()
    signoff = json.loads(SIGNOFF_FILE.read_text(encoding="utf-8"))
    plan = json.loads(TOPUP_PLAN_SUMMARY.read_text(encoding="utf-8"))
    std = json.loads(STANDARD_LANE_SUMMARY.read_text(encoding="utf-8"))
    assert signoff["policy_version"] == live_pv
    assert plan["policy_version"] == live_pv
    assert std["policy_version"] == live_pv
    assert plan["execution_status"] == "planned_not_executed"
    assert std["execution_status"] == "executed"
    assert int(std["replica"]) == 2
    assert std["lane"] == "standard"
    assert int(std["n_jobs_success"]) == 684
    assert int(std["n_jobs_failed"]) == 0
    assert int(std["n_jobs_pending_after"]) == 0
    tier_topup_5 = next(
        t for t in plan["tier_plans"] if t["tier"] == "topup_to_5"
    )
    assert tier_topup_5["replica_start"] <= 2 <= tier_topup_5["replica_end"]
