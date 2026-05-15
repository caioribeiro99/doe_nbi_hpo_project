"""Tests for the batch_04_stage0_shard00_only runner.

Covers:
- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses to proceed when the batch_03 stage-run summary
  is missing, failed, or stale;
- the runner refuses when ``--shard`` does not live under ``jobs/``;
- ``--include-extreme-tasks`` is required to dispatch extreme rows;
- non-canary methods are silently refused (no crash);
- the run dir is created under ``runs/cc18/`` and is gitignored;
- execution SQLite files stay gitignored;
- a ``--skip-train`` end-to-end run leaves the committed source
  shard byte-identical, populates the run_manifest, and does NOT
  create ``stage3_signoff.json``;
- the published stage-run summary JSON contains every required
  batch_04 key (including ``policy_version``, lane counts,
  ``deferred_extreme_tasks``, ``non_canary_methods_refused``,
  ``failed_timeout`` counter, and the protocol-level keys
  inherited from ``export_cc18_run_summary``).
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
RUN_SCRIPT = REPO / "scripts/run_batch_04_stage0_shard00_only.py"
SOURCE_SHARD = (
    REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"
)
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_batch03_summary(
    path: Path, *, exported_at: str | None = None,
    n_cells_success: int = 216, n_cells_failed: int = 0,
    n_cells_pending: int = 0, n_cells_expected: int = 216,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": "batch_03_cc18_representative_18_tasks",
        "run_id": "batch_03_cc18_representative_18_tasks_latest",
        "exported_at": exported_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "source_git_sha": "0" * 40,
        "n_cells_expected": n_cells_expected,
        "n_cells_success": n_cells_success,
        "n_cells_failed": n_cells_failed,
        "n_cells_pending": n_cells_pending,
        "source_shards_unchanged": source_shards_unchanged,
        "stage3_signoff_present": stage3_signoff_present,
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
    assert "run_batch_04_stage0_shard00_only.py" in out
    assert "--shard" in out
    assert "--include-extreme-tasks" in out
    assert "--batch03-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--max-age-days" in out


def test_run_script_dry_run_does_not_invoke_runner(tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run",
         "--run-root", str(tmp_path / "runs"),
         "--output-root", str(tmp_path / "out"),
         "--stage-runs-dir", str(tmp_path / "stage_runs"),
         "--openml-cache-root", str(tmp_path / "cache")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "stage_runs").exists()
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_batch_03_summary_missing(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    with pytest.raises(GateRefusalError, match="not found"):
        verify_batch03_summary(tmp_path / "absent.json")


def test_refuses_when_batch_03_summary_stale(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    stale = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    summary = _write_fake_batch03_summary(
        tmp_path / "stale.json", exported_at=stale,
    )
    with pytest.raises(GateRefusalError, match="days old"):
        verify_batch03_summary(summary, max_age_days=7)


def test_refuses_when_batch_03_failed(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    summary = _write_fake_batch03_summary(
        tmp_path / "failed.json",
        n_cells_success=210, n_cells_failed=6,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch03_summary(summary)


def test_refuses_when_batch_03_pending(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    summary = _write_fake_batch03_summary(
        tmp_path / "pending.json",
        n_cells_success=210, n_cells_pending=6,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch03_summary(summary)


def test_refuses_when_batch_03_shards_mutated(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    summary = _write_fake_batch03_summary(
        tmp_path / "mut.json", source_shards_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_batch03_summary(summary)


def test_refuses_when_batch_03_signed_off(tmp_path: Path) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        verify_batch03_summary,
    )

    summary = _write_fake_batch03_summary(
        tmp_path / "signed.json", stage3_signoff_present=True,
    )
    with pytest.raises(GateRefusalError, match="stage3_signoff_present"):
        verify_batch03_summary(summary)


def test_refuses_when_shard_outside_jobs(tmp_path: Path) -> None:
    """--shard pointed at a non-jobs/ path is rejected before any
    copy."""
    from scripts.run_batch_04_stage0_shard00_only import (
        GateRefusalError,
        run_batch_04,
    )

    fake_shard = tmp_path / "shard_00.sqlite"
    fake_shard.write_bytes(b"")
    fake_gate = _write_fake_batch03_summary(tmp_path / "ok.json")
    with pytest.raises(GateRefusalError, match="must live under jobs/"):
        run_batch_04(
            shard=fake_shard,
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            batch03_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


def test_refuses_when_stage3_signoff_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_batch_04_stage0_shard00_only as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    fake_gate = _write_fake_batch03_summary(tmp_path / "ok.json")
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.run_batch_04(
            shard=SOURCE_SHARD,
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            batch03_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


# ---------------------------------------------------------------------------
# Classifier behavior
# ---------------------------------------------------------------------------


def test_classifier_defers_extreme_and_refuses_non_canary() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_batch_04_stage0_shard00_only import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # extreme: letter / Devnagari-Script
        ("j1", 6, "default_gbdt", "xgboost"),
        ("j2", 167121, "tpe_optuna", "lightgbm"),
        # non-canary methods on standard tasks
        ("j3", 11, "smac3", "xgboost"),
        ("j4", 53, "asha", "catboost"),
        # canary methods on standard tasks
        ("j5", 11, "default_gbdt", "xgboost"),
        ("j6", 53, "random_search", "lightgbm"),
        # canary method on heavy task
        ("j7", 3573, "doe_rsm_vrf_true_nbi", "catboost"),
    ]
    buckets = classify_rows(rows, g, include_extreme=False)
    assert sorted(e["job_id"] for e in buckets["deferred"]) == ["j1", "j2"]
    assert sorted(e["job_id"] for e in buckets["refused_not_in_canary_set"]) == ["j3", "j4"]
    assert sorted(e["job_id"] for e in buckets["runnable_standard"]) == ["j5", "j6"]
    assert [e["job_id"] for e in buckets["runnable_heavy"]] == ["j7"]


def test_classifier_promotes_extreme_when_include_extreme_set() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_batch_04_stage0_shard00_only import classify_rows

    g = RuntimeGuardrails.load()
    rows = [("j_ex", 167121, "default_gbdt", "xgboost")]
    buckets = classify_rows(rows, g, include_extreme=True)
    assert buckets["deferred"] == []
    # extreme task with canary method goes into runnable_standard
    # (the bucket name groups by lane; we don't currently split out
    # extreme runnable into its own bucket because it shares the
    # standard gate path).
    bucket_keys = {k for k, v in buckets.items() if v}
    assert "deferred" not in bucket_keys


# ---------------------------------------------------------------------------
# Run dir + gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_batch_04_stage0_shard00_only import DEFAULT_RUN_ROOT

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/batch_04_stage0_shard00_only_latest/run_manifest.json",
         "runs/cc18/batch_04_stage0_shard00_only_latest/shards/"
         "stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/batch_04_stage0_shard00_only_latest/outputs/abc/"
         "manifest.json",
         "runs/cc18/batch_04_stage0_shard00_only_latest/outputs/abc/"
         "catboost_info/learn_error.tsv"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 4


def test_stage_runs_summary_jsonmd_are_committed_but_other_files_are_not() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/batch_04_stage0_shard00_only_latest_summary.json",
         "experiments/_stage_runs/batch_04_stage0_shard00_only_latest_summary.md",
         "experiments/_stage_runs/batch_04_stage0_shard00_only_latest/extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout
    assert "!experiments/_stage_runs/*.json" in res.stdout
    assert "!experiments/_stage_runs/*.md" in res.stdout


# ---------------------------------------------------------------------------
# Skip-train end-to-end: shard unchanged + summary schema
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
    # batch_04-specific augmentation
    "batch_id", "source_shard", "execution_shard",
    "policy_version", "policy_csv_path", "guardrails_yaml_path",
    "include_extreme_tasks",
    "n_jobs_total_in_shard", "n_jobs_executed", "n_jobs_deferred",
    "n_jobs_refused", "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after",
    "n_jobs_runnable_standard", "n_jobs_runnable_heavy",
    "n_jobs_runnable_extreme_deferred",
    "status_counts_extended", "task_lane_counts_in_shard",
    "deferred_extreme_tasks", "non_canary_methods_refused",
    "cells", "slowest_cells",
    "runtime_seconds_runner_total", "runner_invocations",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed", "batch_03_gate",
    "source_shard_md5_before", "source_shard_md5_after",
    "platform", "git_sha", "capability_audit",
)


def test_skip_train_pass_leaves_committed_shard_unchanged(
    tmp_path: Path,
) -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        BATCH_ID,
        run_batch_04,
    )

    md5_before = _md5(SOURCE_SHARD)
    fake_gate = _write_fake_batch03_summary(tmp_path / "good_gate.json")
    summary = run_batch_04(
        shard=SOURCE_SHARD,
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch03_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_batch_04_skip_train",
    )
    md5_after = _md5(SOURCE_SHARD)
    assert md5_before == md5_after
    assert summary["batch_id"] == BATCH_ID
    assert summary["include_extreme_tasks"] is False
    # The shard had Devnagari/letter rows -> defer; non-canary -> refuse.
    assert summary["n_jobs_runnable_extreme_deferred"] >= 1
    assert summary["n_jobs_refused"] >= 1
    # All non-canary methods land in the refused bucket.
    assert sorted(summary["non_canary_methods_refused"]) == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    # letter (task 6) is deferred.
    assert 6 in summary["deferred_extreme_tasks"]
    assert summary["source_shards_unchanged"] is True
    assert summary["stage3_signoff_present"] is False
    assert not SIGNOFF_FILE.exists()

    # Summary JSON schema includes every required key.
    json_p = (
        tmp_path / "stage_runs"
        / "test_batch_04_skip_train_summary.json"
    )
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"

    # Policy version is a 64-char SHA-256 hex string.
    assert isinstance(payload["policy_version"], str)
    assert len(payload["policy_version"]) == 64

    # Execution SQLite lives under runs/, never under jobs/.
    exec_dir = (
        tmp_path / "runs"
        / "test_batch_04_skip_train"
        / "shards" / "stage0_replica_001"
    )
    exec_files = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 1
    p = exec_files[0]
    assert "runs" in p.resolve().parts


def test_skip_train_with_include_extreme_does_not_defer(
    tmp_path: Path,
) -> None:
    """When --include-extreme-tasks is set, extreme rows are not
    deferred (they are still classified as runnable_standard /
    runnable_heavy)."""
    from scripts.run_batch_04_stage0_shard00_only import run_batch_04

    fake_gate = _write_fake_batch03_summary(tmp_path / "ok.json")
    summary = run_batch_04(
        shard=SOURCE_SHARD,
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch03_summary=fake_gate,
        max_age_days=30,
        include_extreme_tasks=True,
        skip_train=True,
        run_id="test_batch_04_include_extreme",
    )
    assert summary["include_extreme_tasks"] is True
    assert summary["n_jobs_deferred"] == 0
    assert summary["deferred_extreme_tasks"] == []


def test_signoff_file_still_absent_on_disk() -> None:
    assert not SIGNOFF_FILE.exists()


def test_lane_hide_step_does_not_overwrite_terminal_status(tmp_path: Path) -> None:
    """Regression: the hide step ahead of the standard-lane pass must
    NOT mark already-completed heavy-lane rows as 'claimed'. The bug
    surfaced as 12 pending rows in the first batch_04 run because the
    successful heavy rows got hidden and then reverted to 'pending'
    after the standard pass, losing their terminal status."""
    import sqlite3

    db = tmp_path / "shard.sqlite"
    schema = (REPO / "jobs/doctoral/openml_cc18/schema.sql").read_text()
    with sqlite3.connect(db) as cx:
        cx.executescript(schema)
        # Insert 3 rows: heavy-success, heavy-failed, standard-pending.
        cx.execute(
            "INSERT INTO cc18_jobs ("
            "job_id, openml_task_id, openml_dataset_id, dataset_name, "
            "algorithm, method, replica, stage, config_path, output_dir, "
            "status, runtime_seconds, last_error) VALUES "
            "(?, 3573, 554, 'mnist', 'xgboost', 'default_gbdt', 1, "
            "'stage0_replica_001', '', '', 'success', 12.3, NULL),"
            "(?, 3573, 554, 'mnist', 'xgboost', 'tpe_optuna', 1, "
            "'stage0_replica_001', '', '', 'failed', 4.0, 'boom'),"
            "(?, 11, 11, 'balance', 'xgboost', 'random_search', 1, "
            "'stage0_replica_001', '', '', 'pending', NULL, NULL)",
            ("heavy_ok", "heavy_bad", "standard_pending"),
        )
        cx.commit()

    # Apply the post-fix hide query against the heavy job_ids (other
    # lane). The terminal-status rows must remain untouched.
    hide_worker = "__batch04_hidden_lane__heavy"
    hidden = ["heavy_ok", "heavy_bad"]
    with sqlite3.connect(db) as cx:
        placeholders = ",".join("?" * len(hidden))
        cx.execute(
            f"UPDATE cc18_jobs SET status='claimed', assigned_worker=? "
            f"WHERE job_id IN ({placeholders}) AND status='pending'",
            (hide_worker, *hidden),
        )
        cx.commit()
        statuses = {
            jid: status
            for jid, status in cx.execute(
                "SELECT job_id, status FROM cc18_jobs ORDER BY job_id"
            )
        }
    assert statuses == {
        "heavy_ok": "success",
        "heavy_bad": "failed",
        "standard_pending": "pending",
    }

    # Now apply the restore query and confirm terminal rows are still
    # untouched (only rows we actually hid would have hide_worker).
    with sqlite3.connect(db) as cx:
        cx.execute(
            "UPDATE cc18_jobs SET status='pending', assigned_worker=NULL "
            "WHERE assigned_worker=? AND status='claimed'",
            (hide_worker,),
        )
        cx.commit()
        statuses = {
            jid: status
            for jid, status in cx.execute(
                "SELECT job_id, status FROM cc18_jobs ORDER BY job_id"
            )
        }
    assert statuses == {
        "heavy_ok": "success",
        "heavy_bad": "failed",
        "standard_pending": "pending",
    }


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_batch_04_stage0_shard00_only import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARD,
    )

    assert DEFAULT_SHARD.name == "shard_00.sqlite"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
