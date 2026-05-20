"""Tests for the Stage-3 pilot (Commit 47).

``scripts/run_stage3_pilot_replica002_shard00_standard_lane.py`` is the
first real Stage-3 / top-up execution on top of the Commit 45 sign-off
and the Commit 46 plan. It is deliberately tiny:

- ``shard_00`` only;
- ``replica = 2`` (the first replica of the ``topup_to_5`` tier);
- ``standard`` lane only;
- four canary methods only.

This module covers the contract that protects those invariants:

- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses when the signoff is missing / not signed /
  carries a different policy_version;
- the runner refuses when the Stage-3 top-up plan is missing,
  not in ``planned_not_executed`` status, drifts on
  ``policy_version``, or excludes ``replica = 2`` from
  ``topup_to_5``;
- the classifier puts standard-canary rows in
  ``runnable_standard``, heavy in ``deferred_heavy_lane``,
  extreme in ``deferred_extreme_lane``, and non-canary in
  ``refused_not_in_canary_set``;
- the expected runnable standard-lane canary count on shard_00 is 68;
- a ``--skip-train`` end-to-end run produces a single execution
  SQLite under ``runs/cc18/<run_id>/`` carrying ``replica = 2`` and
  the Stage-3 / top-up stage label, leaves the committed source
  shard byte-identical, and publishes a summary with all keys the
  prompt anchors on;
- runs/ + execution SQLite files are gitignored; only the small
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
RUN_SCRIPT = REPO / "scripts/run_stage3_pilot_replica002_shard00_standard_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SHARD_00 = SHARDS_DIR / "shard_00.sqlite"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
TOPUP_PLAN_SUMMARY = (
    REPO / "experiments/_stage_runs/stage3_topup_plan_latest_summary.json"
)
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
        "exported_at": "2026-05-19T14:53:19Z",
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
                "executable_canary_cells_total": 3456,
                "lanes": [],
            },
            {
                "tier": "topup_to_10",
                "shard_subdir": "stage2_topup_to_010",
                "replica_start": 6, "replica_end": 10, "replica_count": 5,
                "executable_canary_cells_total": 4320,
                "lanes": [],
            },
            {
                "tier": "topup_to_30",
                "shard_subdir": "stage3_topup_to_030",
                "replica_start": 11, "replica_end": 30, "replica_count": 20,
                "executable_canary_cells_total": 17280,
                "lanes": [],
            },
        ],
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
    assert "run_stage3_pilot_replica002_shard00_standard_lane.py" in out
    assert "--signoff-file" in out
    assert "--topup-plan-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--target-stage-label" in out
    assert "--target-replica" in out
    assert "--dry-run" in out


def test_run_script_dry_run_reports_planned_not_executed() -> None:
    """--dry-run prints a JSON plan with execution_status set to
    'planned_not_executed' and the 68-cell expectation for shard_00."""
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["target_replica"] == 2
    assert payload["target_stage_label"] == "stage1_topup_to_005"
    assert payload["lane"] == "standard"
    assert payload["topup_tier"] == "topup_to_5_pilot"
    assert payload["expected_standard_canary_cells"] == 68
    assert payload["policy_version"] == PINNED_POLICY_VERSION
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert payload["signoff_status"] == "signed"
    assert payload["signoff_type"] == "stage0_replica_001"
    assert payload["topup_plan_execution_status"] == "planned_not_executed"
    plan = payload["pre_run_plan"]
    assert plan["n_jobs_total"] == 219
    assert plan["n_runnable_standard"] == 68
    assert plan["n_deferred_heavy_lane"] == 31
    assert plan["n_deferred_extreme_lane"] == 11
    assert plan["n_refused_not_in_canary_set"] == 109


# ---------------------------------------------------------------------------
# Signoff refusal
# ---------------------------------------------------------------------------


def test_refuses_when_signoff_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_signoff,
    )

    with pytest.raises(PilotRefusalError, match="signoff file not found"):
        verify_signoff(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_signoff_not_signed(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", signoff_status="planned_not_signed",
    )
    with pytest.raises(PilotRefusalError, match="signoff_status"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_signoff_type_wrong(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", signoff_type="some_other_signoff",
    )
    with pytest.raises(PilotRefusalError, match="signoff_type"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_signoff_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_signoff,
    )

    p = _write_signoff(
        tmp_path / "stage3_signoff.json", policy_version="f" * 64,
    )
    with pytest.raises(PilotRefusalError, match="policy_version"):
        verify_signoff(p, expected_policy_version=PINNED_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Top-up plan refusal
# ---------------------------------------------------------------------------


def test_refuses_when_topup_plan_missing(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_topup_plan,
    )

    with pytest.raises(PilotRefusalError, match="stage3 top-up plan"):
        verify_topup_plan(
            tmp_path / "absent.json",
            expected_policy_version=PINNED_POLICY_VERSION,
        )


def test_refuses_when_topup_plan_already_executed(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_topup_plan,
    )

    p = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
        execution_status="executed",
    )
    with pytest.raises(PilotRefusalError, match="execution_status"):
        verify_topup_plan(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_topup_plan_policy_drift(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_topup_plan,
    )

    p = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
        policy_version="f" * 64,
    )
    with pytest.raises(PilotRefusalError, match="policy_version"):
        verify_topup_plan(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_replica2_not_in_topup_5(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_topup_plan,
    )

    p = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
        include_replica2_in_topup_5=False,
    )
    with pytest.raises(PilotRefusalError, match="replica=2"):
        verify_topup_plan(p, expected_policy_version=PINNED_POLICY_VERSION)


def test_refuses_when_topup_plan_missing_topup_5_tier(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
        verify_topup_plan,
    )

    plan_path = tmp_path / "stage3_topup_plan_latest_summary.json"
    plan_path.write_text(json.dumps({
        "execution_status": "planned_not_executed",
        "policy_version": PINNED_POLICY_VERSION,
        "tier_plans": [
            {
                "tier": "topup_to_10",
                "replica_start": 6, "replica_end": 10, "replica_count": 5,
            },
        ],
    }), encoding="utf-8")
    with pytest.raises(PilotRefusalError, match="topup_to_5"):
        verify_topup_plan(
            plan_path, expected_policy_version=PINNED_POLICY_VERSION,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        classify_rows,
    )

    g = RuntimeGuardrails.load()
    rows = [
        # extreme task → deferred_extreme_lane
        ("j_ex1", 6, "default_gbdt", "xgboost"),
        ("j_ex2", 167121, "tpe_optuna", "lightgbm"),
        # heavy task → deferred_heavy_lane regardless of method
        ("j_h_c", 3573, "default_gbdt", "xgboost"),
        ("j_h_n", 3573, "smac3", "lightgbm"),
        # standard task × non-canary → refused
        ("j_r1", 11, "smac3", "xgboost"),
        ("j_r2", 53, "asha", "catboost"),
        # standard task × canary → runnable
        ("j_run1", 11, "default_gbdt", "xgboost"),
        ("j_run2", 53, "tpe_optuna", "lightgbm"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(e["job_id"] for e in buckets["deferred_extreme_lane"]) == [
        "j_ex1", "j_ex2",
    ]
    assert sorted(e["job_id"] for e in buckets["deferred_heavy_lane"]) == [
        "j_h_c", "j_h_n",
    ]
    assert sorted(e["job_id"] for e in buckets["refused_not_in_canary_set"]) == [
        "j_r1", "j_r2",
    ]
    assert sorted(e["job_id"] for e in buckets["runnable_standard"]) == [
        "j_run1", "j_run2",
    ]


def test_pre_run_plan_matches_shard_00_inventory() -> None:
    """The pre-run plan over the committed shard_00 must match the
    Commit 47 expectation: 68 runnable standard cells, 31 deferred
    heavy, 11 deferred extreme, 109 refused non-canary, 219 total."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        build_pre_run_plan,
    )

    g = RuntimeGuardrails.load()
    plan = build_pre_run_plan(SHARD_00, g)
    assert plan["n_jobs_total"] == 219
    assert plan["n_runnable_standard"] == 68
    assert plan["n_deferred_heavy_lane"] == 31
    assert plan["n_deferred_extreme_lane"] == 11
    assert plan["n_refused_not_in_canary_set"] == 109
    assert plan["task_lane_counts_universe"] == {
        "standard": 17, "heavy": 3, "extreme": 1,
    }
    # The 7 non-canary methods built into the committed schedule.
    assert plan["non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    # Source template is replica 1 / stage0_replica_001 in the
    # committed shard.
    assert plan["source_template_replicas"] == [1]
    assert plan["source_template_stages"] == ["stage0_replica_001"]


# ---------------------------------------------------------------------------
# Gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        DEFAULT_RUN_ROOT,
    )

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage3_pilot_replica_002_shard00_standard_lane_latest/"
         "run_manifest.json",
         "runs/cc18/stage3_pilot_replica_002_shard00_standard_lane_latest/"
         "shards/stage0_replica_001/shard_00.execution.sqlite"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 2


def test_stage_runs_pilot_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/"
         "stage3_pilot_replica_002_shard00_standard_lane_latest_summary.json",
         "experiments/_stage_runs/"
         "stage3_pilot_replica_002_shard00_standard_lane_latest_summary.md",
         "experiments/_stage_runs/"
         "stage3_pilot_replica_002_shard00_standard_lane_latest/extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout
    assert "!experiments/_stage_runs/*.json" in res.stdout
    assert "!experiments/_stage_runs/*.md" in res.stdout


# ---------------------------------------------------------------------------
# Skip-train end-to-end: rewrite + classification + summary, no training
# ---------------------------------------------------------------------------


SUMMARY_REQUIRED_KEYS = (
    # protocol-level keys from export_cc18_run_summary
    "schema_version", "run_id", "stage", "exported_at",
    "source_git_sha", "host", "python", "package_versions",
    "n_total", "n_pending", "n_claimed", "n_running",
    "n_success", "n_failed", "n_skipped", "status_counts",
    "shards", "n_shards", "run_dir", "run_manifest_path",
    "execution_suffix", "source_shards_unchanged",
    "source_md5_recorded", "source_md5_now", "source_drift",
    "stage3_signoff_present", "stage3_signoff_path",
    "protocol_doc",
    # stage 3 pilot augmentation
    "batch_id", "topup_tier", "execution_status", "replica",
    "source_template_replica", "lane",
    "policy_version", "policy_version_pinned",
    "policy_csv_path", "guardrails_yaml_path",
    "signoff_path", "signoff_sha256", "signoff_signed_at_utc",
    "signoff_operator_handle", "signoff_operator_name",
    "signoff_type", "signoff_status",
    "stage3_topup_plan_summary_path",
    "stage3_topup_plan_summary_sha256",
    "stage3_topup_plan_execution_status",
    "n_source_shards", "source_shards", "execution_shards",
    "execution_sqlite_sha256_after_rewrite",
    "n_jobs_total", "n_jobs_executed", "n_jobs_success",
    "n_jobs_deferred_heavy", "n_jobs_deferred_extreme",
    "n_jobs_refused_non_canary", "n_jobs_failed",
    "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after", "n_jobs_running_after",
    "status_counts_extended", "task_lane_counts_universe",
    "standard_tasks_executed", "standard_tasks_in_shard",
    "heavy_tasks_deferred", "extreme_tasks_deferred",
    "non_canary_methods_refused", "expected_standard_canary_cells",
    "per_shard_status", "method_counts_universe",
    "algorithm_counts_universe", "metric_keys",
    "slowest_cells", "cells",
    "runtime_seconds_runner_total", "runner_invocations",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed",
    "source_shard_md5_before", "source_shard_md5_after",
    "source_shard_md5_after_copy", "source_shard_md5_after_rewrite",
    "execution_copy_rewrite",
    "platform", "git_sha", "capability_audit",
    "no_full_topup_to_5_executed_by_this_script",
    "no_heavy_lane_executed_by_this_script",
    "no_extreme_lane_executed_by_this_script",
    "no_committed_shard_modified_by_this_script",
    "no_raw_openml_payloads_staged_by_this_script",
    "no_execution_sqlite_staged_by_this_script",
    "operator_review_required_before_scaling",
    "next_recommended_step",
)


def _run_skip_train(
    tmp_path: Path, *, signoff_path: Path | None = None,
    topup_plan_path: Path | None = None,
    run_id: str = "test_stage3_pilot_skip_train",
) -> dict:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        run_pilot,
    )

    signoff = signoff_path or _write_signoff(tmp_path / "stage3_signoff.json")
    topup = topup_plan_path or _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
    )
    return run_pilot(
        shards_dir=SHARDS_DIR,
        shard_name="shard_00.sqlite",
        signoff_file=signoff,
        topup_plan_summary=topup,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        policy_csv=POLICY_CSV,
        guardrails_yaml=GUARDRAILS_YAML,
        run_id=run_id,
        skip_train=True,
    )


def test_skip_train_copies_only_shard_00_and_preserves_md5(
    tmp_path: Path,
) -> None:
    md5_before = _md5(SHARD_00)
    md5_before_others = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    summary = _run_skip_train(tmp_path, run_id="t_copy_only_shard_00")
    md5_after_others = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    # 1. committed source shards (all of them, not just shard_00)
    #    are byte-identical pre / post pilot.
    assert md5_before_others == md5_after_others
    assert md5_before == md5_after_others["shard_00.sqlite"]
    # 2. one (and only one) execution copy was created.
    exec_dir = (
        tmp_path / "runs" / "t_copy_only_shard_00"
        / "shards" / "stage0_replica_001"
    )
    exec_files = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 1
    assert exec_files[0].name == "shard_00.execution.sqlite"
    # 3. the execution copy lives under runs/ (gitignored).
    assert "runs" in exec_files[0].resolve().parts
    # 4. summary records the source-shard MD5 equality both ways.
    assert summary["source_shard_md5_before"]["shard_00.sqlite"] == md5_before
    assert summary["source_shard_md5_after"]["shard_00.sqlite"] == md5_before
    assert summary["source_shard_md5_after_copy"]["shard_00.sqlite"] == md5_before
    assert summary["source_shard_md5_after_rewrite"]["shard_00.sqlite"] == md5_before
    assert summary["no_committed_shard_modified_by_this_script"] is True


def test_skip_train_execution_copy_has_replica_2_and_topup_stage(
    tmp_path: Path,
) -> None:
    summary = _run_skip_train(
        tmp_path, run_id="t_replica2_topup_stage",
    )
    exec_dir = (
        tmp_path / "runs" / "t_replica2_topup_stage"
        / "shards" / "stage0_replica_001"
    )
    exec_p = exec_dir / "shard_00.execution.sqlite"
    cx = sqlite3.connect(f"file:{exec_p}?mode=ro", uri=True)
    try:
        replicas = sorted({
            r[0] for r in cx.execute("SELECT DISTINCT replica FROM cc18_jobs")
        })
        stages = sorted({
            r[0] for r in cx.execute("SELECT DISTINCT stage FROM cc18_jobs")
        })
    finally:
        cx.close()
    assert replicas == [2]
    assert stages == ["stage1_topup_to_005"]
    assert summary["replica"] == 2
    assert summary["source_template_replica"] == 1
    assert summary["stage"] == "stage1_topup_to_005"
    assert summary["execution_copy_rewrite"]["target_replica"] == 2
    assert (
        summary["execution_copy_rewrite"]["target_stage"]
        == "stage1_topup_to_005"
    )


def test_skip_train_classifies_lanes_correctly(tmp_path: Path) -> None:
    """No heavy / extreme row should end up as 'success'; non-canary
    rows must end as 'skipped' with the right last_error."""
    _run_skip_train(tmp_path, run_id="t_lane_isolation")
    exec_p = (
        tmp_path / "runs" / "t_lane_isolation"
        / "shards" / "stage0_replica_001"
        / "shard_00.execution.sqlite"
    )
    cx = sqlite3.connect(f"file:{exec_p}?mode=ro", uri=True)
    try:
        heavy_or_extreme_tids = (
            6, 167121,  # extreme tasks in shard_00
            3573, 167124, 167125,  # known heavy tasks in shard_00
        )
        rows = list(cx.execute(
            "SELECT openml_task_id, status, last_error FROM cc18_jobs "
            "WHERE openml_task_id IN "
            f"({','.join('?' * len(heavy_or_extreme_tids))})",
            heavy_or_extreme_tids,
        ))
    finally:
        cx.close()
    assert rows, "expected at least one heavy/extreme row in shard_00"
    for tid, status, err in rows:
        assert status == "skipped", (tid, status, err)
        assert err in {
            "deferred_heavy_lane", "deferred_extreme_lane",
            "refused_not_in_canary_set",
        }, (tid, status, err)


def test_skip_train_summary_has_all_required_keys(tmp_path: Path) -> None:
    _run_skip_train(tmp_path, run_id="t_summary_schema")
    json_p = (
        tmp_path / "stage_runs" / "t_summary_schema_summary.json"
    )
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    assert isinstance(payload["policy_version"], str)
    assert len(payload["policy_version"]) == 64
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert isinstance(payload["signoff_sha256"], str)
    assert len(payload["signoff_sha256"]) == 64
    assert isinstance(payload["stage3_topup_plan_summary_sha256"], str)
    assert len(payload["stage3_topup_plan_summary_sha256"]) == 64
    # Pilot-specific invariants.
    assert payload["execution_status"] == "executed"
    assert payload["topup_tier"] == "topup_to_5_pilot"
    assert payload["lane"] == "standard"
    assert payload["expected_standard_canary_cells"] == 68
    assert payload["n_jobs_total"] == 219
    # skip_train mode: nothing executed, but classification and
    # pre-marking still produced the expected deferred / refused counts.
    assert payload["n_jobs_deferred_heavy"] == 31
    assert payload["n_jobs_deferred_extreme"] == 11
    assert payload["n_jobs_refused_non_canary"] == 109
    assert payload["no_full_topup_to_5_executed_by_this_script"] is True
    assert payload["no_heavy_lane_executed_by_this_script"] is True
    assert payload["no_extreme_lane_executed_by_this_script"] is True
    assert payload["no_committed_shard_modified_by_this_script"] is True
    assert payload["operator_review_required_before_scaling"] is True


def test_skip_train_refuses_on_missing_signoff(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
    )

    topup = _write_topup_plan(
        tmp_path / "stage3_topup_plan_latest_summary.json",
    )
    with pytest.raises(PilotRefusalError, match="signoff file not found"):
        _run_skip_train(
            tmp_path,
            signoff_path=tmp_path / "absent_signoff.json",
            topup_plan_path=topup,
            run_id="t_no_signoff",
        )


def test_skip_train_refuses_on_missing_topup_plan(tmp_path: Path) -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PilotRefusalError,
    )

    signoff = _write_signoff(tmp_path / "stage3_signoff.json")
    with pytest.raises(PilotRefusalError, match="stage3 top-up plan"):
        _run_skip_train(
            tmp_path,
            signoff_path=signoff,
            topup_plan_path=tmp_path / "absent_plan.json",
            run_id="t_no_plan",
        )


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
        DEFAULT_SIGNOFF_FILE,
        DEFAULT_TOPUP_PLAN_SUMMARY,
        EXPECTED_STANDARD_CANARY_CELLS_SHARD_00,
        TARGET_REPLICA,
        TARGET_STAGE_LABEL,
    )
    from scripts.run_stage3_pilot_replica002_shard00_standard_lane import (
        PINNED_POLICY_VERSION as PV,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
    assert DEFAULT_SIGNOFF_FILE == SIGNOFF_FILE
    assert DEFAULT_TOPUP_PLAN_SUMMARY == TOPUP_PLAN_SUMMARY
    assert TARGET_REPLICA == 2
    assert TARGET_STAGE_LABEL == "stage1_topup_to_005"
    assert EXPECTED_STANDARD_CANARY_CELLS_SHARD_00 == 68
    assert PV == PINNED_POLICY_VERSION


def test_live_signoff_and_topup_plan_match_pinned_policy() -> None:
    """The committed signoff and Stage-3 top-up plan must both pin the
    same policy_version as the live heavy_task_policy.csv. If this
    test fails, do not run the pilot: investigate the drift first."""
    live_pv = _sha256(POLICY_CSV)
    signoff = json.loads(SIGNOFF_FILE.read_text(encoding="utf-8"))
    plan = json.loads(TOPUP_PLAN_SUMMARY.read_text(encoding="utf-8"))
    assert signoff["policy_version"] == live_pv
    assert plan["policy_version"] == live_pv
    assert plan["execution_status"] == "planned_not_executed"
    tier_topup_5 = next(
        t for t in plan["tier_plans"] if t["tier"] == "topup_to_5"
    )
    assert tier_topup_5["replica_start"] <= 2 <= tier_topup_5["replica_end"]
