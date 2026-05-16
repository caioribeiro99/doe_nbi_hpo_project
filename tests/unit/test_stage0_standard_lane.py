"""Tests for the stage0 standard-lane runner (Commit 40).

Covers:
- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses to proceed when the batch_04 summary is
  missing, failed, has unfinished work, mutated source shards, or
  is stale;
- the runner refuses if a stage-3 sign-off file exists;
- the classifier puts standard-canary rows in
  ``runnable_standard``, heavy in ``deferred_heavy_lane``,
  extreme in ``deferred_extreme_lane``, and non-canary in
  ``refused_not_in_canary_set``;
- the pre-run plan finds the expected counts across all 10
  committed stage-0 shards;
- a ``--skip-train`` end-to-end run materializes 10 execution
  copies under ``runs/cc18/<run_id>/shards/stage0_replica_001/``,
  leaves every committed source shard byte-identical, and
  publishes a summary with the full augmentation block;
- the run dir + execution SQLite files are gitignored;
- the stage-run summary JSON / MD are the only allowlisted
  artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/run_stage0_standard_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_batch04_summary(
    path: Path, *, exported_at: str | None = None,
    n_jobs_failed_timeout: int = 0, n_jobs_failed_other: int = 0,
    n_jobs_pending_after: int = 0,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": "batch_04_stage0_shard00_only",
        "run_id": "batch_04_stage0_shard00_only_latest",
        "exported_at": exported_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "source_git_sha": "0" * 40,
        "n_jobs_failed_timeout": n_jobs_failed_timeout,
        "n_jobs_failed_other": n_jobs_failed_other,
        "n_jobs_pending_after": n_jobs_pending_after,
        "n_failed": n_jobs_failed_timeout + n_jobs_failed_other,
        "n_pending": n_jobs_pending_after,
        "n_running": 0,
        "source_shards_unchanged": source_shards_unchanged,
        "stage3_signoff_present": stage3_signoff_present,
        "policy_version": "f" * 64,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    assert "run_stage0_standard_lane.py" in out
    assert "--shards-dir" in out
    assert "--batch04-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--max-age-days" in out
    assert "--max-evaluations" in out


def test_run_script_dry_run_emits_pre_run_plan() -> None:
    """--dry-run prints a JSON plan including the standard-canary
    count expected from the committed policy CSV."""
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert "pre_run_plan" in payload
    plan = payload["pre_run_plan"]
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_standard"] == 684


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_batch_04_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    with pytest.raises(GateRefusalError, match="not found"):
        verify_batch04_summary(tmp_path / "absent.json")


def test_refuses_when_batch_04_summary_stale(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    stale = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    summary = _write_fake_batch04_summary(
        tmp_path / "stale.json", exported_at=stale,
    )
    with pytest.raises(GateRefusalError, match="days old"):
        verify_batch04_summary(summary, max_age_days=7)


def test_refuses_when_batch_04_failed(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    summary = _write_fake_batch04_summary(
        tmp_path / "failed.json", n_jobs_failed_other=3,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch04_summary(summary)


def test_refuses_when_batch_04_timeout(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    summary = _write_fake_batch04_summary(
        tmp_path / "timeout.json", n_jobs_failed_timeout=1,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch04_summary(summary)


def test_refuses_when_batch_04_pending(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    summary = _write_fake_batch04_summary(
        tmp_path / "pending.json", n_jobs_pending_after=4,
    )
    with pytest.raises(GateRefusalError, match="unfinished work"):
        verify_batch04_summary(summary)


def test_refuses_when_batch_04_shards_mutated(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    summary = _write_fake_batch04_summary(
        tmp_path / "mut.json", source_shards_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_batch04_summary(summary)


def test_refuses_when_batch_04_signed_off(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import (
        GateRefusalError,
        verify_batch04_summary,
    )

    summary = _write_fake_batch04_summary(
        tmp_path / "signed.json", stage3_signoff_present=True,
    )
    with pytest.raises(GateRefusalError, match="stage3_signoff_present"):
        verify_batch04_summary(summary)


def test_refuses_when_stage3_signoff_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_stage0_standard_lane as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    fake_gate = _write_fake_batch04_summary(tmp_path / "ok.json")
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.run_stage0_standard_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            batch04_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_standard_lane import classify_rows

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


def test_pre_run_plan_matches_committed_policy() -> None:
    """The pre-run plan over the 10 committed shards must match the
    counts the prompt anchors on (684 runnable, 57 standard tasks,
    13 heavy + 2 extreme deferred, 7 non-canary methods)."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_standard_lane import build_pre_run_plan

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_standard"] == 684
    # heavy + extreme deferral universes
    assert plan["n_deferred_extreme_lane"] == 66
    # Heavy deferral covers heavy-canary + heavy-non-canary rows.
    assert plan["n_deferred_heavy_lane"] == 156 + 267
    assert plan["n_refused_not_in_canary_set"] == 1131
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }
    assert plan["non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    assert len(plan["standard_tasks_executed"]) == 57
    assert sorted(plan["heavy_tasks_deferred"]) == [
        32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825,
        167119, 167120, 167124, 167125,
    ]
    assert sorted(plan["extreme_tasks_deferred"]) == [6, 167121]


# ---------------------------------------------------------------------------
# Gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_stage0_standard_lane import DEFAULT_RUN_ROOT

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage0_standard_lane_latest/run_manifest.json",
         "runs/cc18/stage0_standard_lane_latest/shards/"
         "stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/stage0_standard_lane_latest/outputs/abc/manifest.json",
         "runs/cc18/stage0_standard_lane_latest/outputs/abc/"
         "catboost_info/learn_error.tsv"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 4


def test_stage_runs_summary_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/stage0_standard_lane_latest_summary.json",
         "experiments/_stage_runs/stage0_standard_lane_latest_summary.md",
         "experiments/_stage_runs/stage0_standard_lane_latest/"
         "extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout
    assert "!experiments/_stage_runs/*.json" in res.stdout
    assert "!experiments/_stage_runs/*.md" in res.stdout


# ---------------------------------------------------------------------------
# Skip-train end-to-end: 10 execution copies + shards unchanged
# ---------------------------------------------------------------------------


HAS_OPENML = importlib.util.find_spec("openml") is not None


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
    # stage0 standard-lane augmentation
    "batch_id", "lane", "n_source_shards", "source_shards",
    "execution_shards", "policy_version", "policy_csv_path",
    "guardrails_yaml_path",
    "n_jobs_total", "n_jobs_executed", "n_jobs_deferred_heavy",
    "n_jobs_deferred_extreme", "n_jobs_refused_non_canary",
    "n_jobs_failed", "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after", "n_jobs_running_after",
    "status_counts_extended", "task_lane_counts_universe",
    "standard_tasks_executed", "heavy_tasks_deferred",
    "extreme_tasks_deferred", "non_canary_methods_refused",
    "expected_standard_canary_cells", "per_shard_status",
    "cells_runnable_per_shard", "method_counts_universe",
    "algorithm_counts_universe",
    "slowest_cells", "cells",
    "runtime_seconds_runner_total", "runner_invocations",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed", "batch_04_gate",
    "source_shard_md5_before", "source_shard_md5_after",
    "platform", "git_sha", "capability_audit",
)


def test_skip_train_copies_all_10_shards_and_preserves_md5(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_standard_lane import (
        BATCH_ID,
        run_stage0_standard_lane,
    )

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    fake_gate = _write_fake_batch04_summary(tmp_path / "good_gate.json")
    summary = run_stage0_standard_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch04_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_skip_train",
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["batch_id"] == BATCH_ID
    assert summary["lane"] == "standard"
    assert summary["n_source_shards"] == 10
    assert summary["n_jobs_total"] == 2304
    # In skip-train mode no cells executed; runnable standard are
    # still classified but stay 'pending' on disk.
    assert summary["n_jobs_executed"] == 0
    assert summary["n_jobs_deferred_heavy"] == 156 + 267
    assert summary["n_jobs_deferred_extreme"] == 66
    assert summary["n_jobs_refused_non_canary"] == 1131
    assert summary["source_shards_unchanged"] is True
    assert summary["stage3_signoff_present"] is False
    assert not SIGNOFF_FILE.exists()
    assert summary["expected_standard_canary_cells"] == 684

    # Execution copies exist (10 of them) under runs/.
    exec_dir = (
        tmp_path / "runs" / "test_stage0_skip_train"
        / "shards" / "stage0_replica_001"
    )
    exec_files = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 10
    for p in exec_files:
        assert "runs" in p.resolve().parts


def test_skip_train_summary_has_all_required_keys(tmp_path: Path) -> None:
    from scripts.run_stage0_standard_lane import run_stage0_standard_lane

    fake_gate = _write_fake_batch04_summary(tmp_path / "good_gate.json")
    run_stage0_standard_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch04_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_schema",
    )
    json_p = (
        tmp_path / "stage_runs" / "test_stage0_schema_summary.json"
    )
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    # policy_version is a 64-char SHA-256 hex string.
    assert isinstance(payload["policy_version"], str)
    assert len(payload["policy_version"]) == 64


def test_skip_train_does_not_execute_heavy_or_extreme(tmp_path: Path) -> None:
    """No row whose task lane is heavy or extreme should end up as
    'success' on disk; they must remain skipped with the right
    last_error."""
    import sqlite3

    from scripts.run_stage0_standard_lane import run_stage0_standard_lane

    fake_gate = _write_fake_batch04_summary(tmp_path / "good_gate.json")
    run_stage0_standard_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch04_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_lane_isolation",
    )
    exec_dir = (
        tmp_path / "runs" / "test_stage0_lane_isolation"
        / "shards" / "stage0_replica_001"
    )
    extreme_or_heavy_task_ids = (6, 167121, 32, 219, 3573, 7592, 9910,
                                 9981, 14965, 146195, 146825, 167119,
                                 167120, 167124, 167125)
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(exec_p)
        rows = list(cx.execute(
            "SELECT openml_task_id, status, last_error FROM cc18_jobs "
            "WHERE openml_task_id IN "
            + "(" + ",".join("?" * len(extreme_or_heavy_task_ids)) + ")",
            extreme_or_heavy_task_ids,
        ))
        cx.close()
        for tid, status, err in rows:
            assert status in {"skipped"}, (tid, status, err)
            assert err in {
                "deferred_heavy_lane", "deferred_extreme_lane",
                "refused_not_in_canary_set",
            }, (tid, status, err)


def test_signoff_file_still_absent_on_disk() -> None:
    assert not SIGNOFF_FILE.exists()


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage0_standard_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
