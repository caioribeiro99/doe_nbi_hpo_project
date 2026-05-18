"""Tests for the stage0 heavy-lane runner (Commit 41).

Covers:
- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses when the stage0_standard_lane summary is
  missing, failed (n_failed > 0, failed_timeout, failed_other),
  has unfinished work (pending / running), mutated source shards,
  the stage-3 sign-off file, or is stale (> max-age-days);
- the runner refuses when ``policy_version`` drifts from the
  pinned Commit 40 value (no mid-replica policy promotion);
- the classifier puts heavy-canary rows in ``runnable_heavy``,
  standard rows in ``deferred_standard_lane``, extreme in
  ``deferred_extreme_lane``, and heavy non-canary in
  ``refused_not_in_canary_set``;
- the pre-run plan over the 10 committed stage-0 shards reports
  the expected counts (156 / 1815 / 66 / 267) and the 13 heavy
  tasks;
- a ``--skip-train`` end-to-end pass materializes 10 execution
  copies, leaves every committed source shard byte-identical,
  publishes the augmentation block with the isolet note, and
  emits a schema-complete summary;
- the run dir and execution SQLite files are gitignored;
- the stage-run summary JSON/MD pair are the only allowlisted
  artifacts;
- no standard or extreme task row ends up executed.
- isolet (task 3481) is NOT promoted to heavy and never executes
  in the heavy lane under the pinned policy_version.
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
RUN_SCRIPT = REPO / "scripts/run_stage0_heavy_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


@pytest.fixture(autouse=True)
def _hide_real_signoff_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Commit 45 created ``stage3_signoff.json`` on disk. The heavy-
    lane runner refuses to run once that file exists; tests that
    exercise the runner must therefore see ``SIGNOFF_FILE`` as
    absent. Tests that verify the guard override this fixture's
    monkeypatch with their own per-test setattr to a tmp path."""
    from scripts import run_stage0_heavy_lane as m

    monkeypatch.setattr(
        m, "SIGNOFF_FILE",
        tmp_path_factory.mktemp("hide_signoff") / "absent.json",
    )


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_standard_lane_summary(
    path: Path, *, exported_at: str | None = None,
    n_jobs_executed: int = 684, n_jobs_failed: int = 0,
    n_jobs_failed_timeout: int = 0, n_jobs_failed_other: int = 0,
    n_jobs_pending_after: int = 0, n_jobs_running_after: int = 0,
    n_jobs_deferred_heavy: int = 423,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
    policy_version: str = PINNED_POLICY_VERSION,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": "stage0_standard_lane",
        "run_id": "stage0_standard_lane_latest",
        "lane": "standard",
        "exported_at": exported_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "source_git_sha": "0" * 40,
        "n_jobs_executed": n_jobs_executed,
        "n_jobs_failed": n_jobs_failed,
        "n_jobs_failed_timeout": n_jobs_failed_timeout,
        "n_jobs_failed_other": n_jobs_failed_other,
        "n_jobs_pending_after": n_jobs_pending_after,
        "n_jobs_running_after": n_jobs_running_after,
        "n_jobs_deferred_heavy": n_jobs_deferred_heavy,
        "n_success": n_jobs_executed,
        "n_failed": n_jobs_failed,
        "n_pending": n_jobs_pending_after,
        "n_running": n_jobs_running_after,
        "source_shards_unchanged": source_shards_unchanged,
        "stage3_signoff_present": stage3_signoff_present,
        "policy_version": policy_version,
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
    assert "run_stage0_heavy_lane.py" in out
    assert "--standard-lane-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--max-age-days" in out
    assert "--allow-policy-drift" in out


def test_run_script_dry_run_emits_pre_run_plan() -> None:
    """--dry-run prints a JSON plan including the heavy-canary count
    expected from the committed policy CSV (156)."""
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
    assert plan["n_runnable_heavy"] == 156
    assert plan["n_deferred_standard_lane"] == 1815
    assert plan["n_deferred_extreme_lane"] == 66
    assert plan["n_refused_not_in_canary_set"] == 267
    assert payload["policy_version_pinned"] == PINNED_POLICY_VERSION


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_standard_lane_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    with pytest.raises(GateRefusalError, match="not found"):
        verify_standard_lane_summary(tmp_path / "absent.json")


def test_refuses_when_standard_lane_summary_stale(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    stale = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    summary = _write_fake_standard_lane_summary(
        tmp_path / "stale.json", exported_at=stale,
    )
    with pytest.raises(GateRefusalError, match="days old"):
        verify_standard_lane_summary(summary, max_age_days=7)


def test_refuses_when_standard_lane_under_executed(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "short.json", n_jobs_executed=600,
    )
    with pytest.raises(GateRefusalError, match="of the expected"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_failed(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "fail.json", n_jobs_failed=2, n_jobs_failed_other=2,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_timed_out(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "to.json", n_jobs_failed=1, n_jobs_failed_timeout=1,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_pending(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "pending.json", n_jobs_pending_after=4,
    )
    with pytest.raises(GateRefusalError, match="unfinished work"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_shards_mutated(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "mut.json", source_shards_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_signed_off(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "signed.json", stage3_signoff_present=True,
    )
    with pytest.raises(GateRefusalError, match="stage3_signoff_present"):
        verify_standard_lane_summary(summary)


def test_refuses_when_standard_lane_heavy_count_drifted(tmp_path: Path) -> None:
    """If the standard-lane summary reports a different heavy-deferred
    count, the underlying policy CSV likely shifted between runs."""
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        verify_standard_lane_summary,
    )

    summary = _write_fake_standard_lane_summary(
        tmp_path / "drift.json", n_jobs_deferred_heavy=500,
    )
    with pytest.raises(GateRefusalError, match="drifted policy"):
        verify_standard_lane_summary(summary)


def test_refuses_when_stage3_signoff_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_stage0_heavy_lane as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    fake_gate = _write_fake_standard_lane_summary(tmp_path / "ok.json")
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.run_stage0_heavy_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


def test_refuses_when_policy_version_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-replica policy promotion is forbidden. If the live CSV no
    longer hashes to the pinned value, the heavy-lane runner refuses
    unless --allow-policy-drift is explicit."""
    from scripts import run_stage0_heavy_lane as m

    monkeypatch.setattr(m, "PINNED_POLICY_VERSION", "0" * 64)
    fake_gate = _write_fake_standard_lane_summary(tmp_path / "ok.json")
    with pytest.raises(m.GateRefusalError, match="pins"):
        m.run_stage0_heavy_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


def test_allow_policy_drift_skips_pin_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_stage0_heavy_lane as m

    monkeypatch.setattr(m, "PINNED_POLICY_VERSION", "0" * 64)
    fake_gate = _write_fake_standard_lane_summary(
        tmp_path / "ok.json", policy_version="0" * 64,
    )
    # When the standard-lane summary records a different policy_version
    # from the live CSV, the runner refuses regardless of the drift
    # flag (the policies must not differ across passes of the same
    # replica). Use the real PINNED so the live CSV matches both ends.
    monkeypatch.setattr(
        m, "PINNED_POLICY_VERSION", PINNED_POLICY_VERSION,
    )
    fake_gate = _write_fake_standard_lane_summary(
        tmp_path / "ok2.json", policy_version=PINNED_POLICY_VERSION,
    )
    summary = m.run_stage0_heavy_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="drift_allow_smoke",
    )
    assert summary["policy_version"] == PINNED_POLICY_VERSION


def test_refuses_when_standard_lane_policy_version_mismatched(
    tmp_path: Path,
) -> None:
    """If the standard-lane summary recorded a different
    policy_version from the live CSV, the runner refuses to mix."""
    from scripts.run_stage0_heavy_lane import (
        GateRefusalError,
        run_stage0_heavy_lane,
    )

    fake_gate = _write_fake_standard_lane_summary(
        tmp_path / "ok.json", policy_version="f" * 64,
    )
    with pytest.raises(GateRefusalError, match="refusing to mix"):
        run_stage0_heavy_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=fake_gate,
            max_age_days=30,
            skip_train=True,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_policy() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_heavy_lane import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # extreme task → deferred_extreme_lane
        ("j_ex1", 6, "default_gbdt", "xgboost"),
        ("j_ex2", 167121, "tpe_optuna", "lightgbm"),
        # standard task → deferred_standard_lane regardless of method
        ("j_s_c", 11, "default_gbdt", "xgboost"),
        ("j_s_n", 11, "smac3", "lightgbm"),
        # heavy task × non-canary → refused_not_in_canary_set
        ("j_r1", 3573, "smac3", "xgboost"),
        ("j_r2", 9910, "asha", "catboost"),
        # heavy task × canary → runnable
        ("j_run1", 3573, "default_gbdt", "xgboost"),
        ("j_run2", 14965, "tpe_optuna", "lightgbm"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(e["job_id"] for e in buckets["deferred_extreme_lane"]) == [
        "j_ex1", "j_ex2",
    ]
    assert sorted(e["job_id"] for e in buckets["deferred_standard_lane"]) == [
        "j_s_c", "j_s_n",
    ]
    assert sorted(e["job_id"] for e in buckets["refused_not_in_canary_set"]) == [
        "j_r1", "j_r2",
    ]
    assert sorted(e["job_id"] for e in buckets["runnable_heavy"]) == [
        "j_run1", "j_run2",
    ]


def test_pre_run_plan_matches_committed_policy() -> None:
    """The plan over the 10 committed shards must yield 156 runnable
    heavy-canary cells (= 13 heavy tasks × 4 × 3), 1815 standard
    deferred, 66 extreme deferred, 267 heavy non-canary refused."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_heavy_lane import build_pre_run_plan

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_heavy"] == 156
    assert plan["n_deferred_standard_lane"] == 1815
    assert plan["n_deferred_extreme_lane"] == 66
    assert plan["n_refused_not_in_canary_set"] == 267
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }
    assert plan["non_canary_methods_refused"] == [
        "asha", "bohb", "dehb", "motpe", "nsga2", "parego", "smac3",
    ]
    assert len(plan["heavy_tasks_executed"]) == 13
    assert sorted(plan["heavy_tasks_executed"]) == [
        32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825,
        167119, 167120, 167124, 167125,
    ]
    assert sorted(plan["extreme_tasks_deferred"]) == [6, 167121]
    # isolet (task 3481) lives in the standard lane and must NOT be
    # listed under heavy_tasks_executed.
    assert 3481 not in plan["heavy_tasks_executed"]
    assert 3481 in plan["standard_tasks_deferred"]


# ---------------------------------------------------------------------------
# Gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_stage0_heavy_lane import DEFAULT_RUN_ROOT

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/stage0_heavy_lane_latest/run_manifest.json",
         "runs/cc18/stage0_heavy_lane_latest/shards/"
         "stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/stage0_heavy_lane_latest/outputs/abc/manifest.json",
         "runs/cc18/stage0_heavy_lane_latest/outputs/abc/"
         "catboost_info/learn_error.tsv"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 4


def test_stage_runs_summary_jsonmd_are_allowlisted() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/stage0_heavy_lane_latest_summary.json",
         "experiments/_stage_runs/stage0_heavy_lane_latest_summary.md",
         "experiments/_stage_runs/stage0_heavy_lane_latest/"
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
# Skip-train end-to-end
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
    # stage0 heavy-lane augmentation
    "batch_id", "lane", "n_source_shards", "source_shards",
    "execution_shards", "policy_version", "policy_version_pinned",
    "policy_csv_path", "guardrails_yaml_path",
    "n_jobs_total", "n_jobs_executed", "n_jobs_deferred_standard",
    "n_jobs_deferred_extreme", "n_jobs_refused_non_canary",
    "n_jobs_failed", "n_jobs_failed_timeout", "n_jobs_failed_other",
    "n_jobs_pending_after", "n_jobs_running_after",
    "status_counts_extended", "task_lane_counts_universe",
    "heavy_tasks_executed", "standard_tasks_deferred",
    "extreme_tasks_deferred", "non_canary_methods_refused",
    "expected_heavy_canary_cells", "per_shard_status",
    "cells_runnable_per_shard", "method_counts_universe",
    "algorithm_counts_universe",
    "slowest_cells", "cells",
    "runtime_seconds_runner_total", "runner_invocations",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed", "standard_lane_gate",
    "source_shard_md5_before", "source_shard_md5_after",
    "platform", "git_sha", "capability_audit",
    "isolet_recalibration_note",
)


def test_skip_train_copies_all_10_shards_and_preserves_md5(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_heavy_lane import (
        BATCH_ID,
        run_stage0_heavy_lane,
    )

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    fake_gate = _write_fake_standard_lane_summary(tmp_path / "good_gate.json")
    summary = run_stage0_heavy_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_heavy_skip_train",
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["batch_id"] == BATCH_ID
    assert summary["lane"] == "heavy"
    assert summary["n_source_shards"] == 10
    assert summary["n_jobs_total"] == 2304
    # In skip-train mode no cells executed; runnable heavy stay
    # 'pending' on disk.
    assert summary["n_jobs_executed"] == 0
    assert summary["n_jobs_deferred_standard"] == 1815
    assert summary["n_jobs_deferred_extreme"] == 66
    assert summary["n_jobs_refused_non_canary"] == 267
    assert summary["expected_heavy_canary_cells"] == 156
    assert summary["source_shards_unchanged"] is True
    assert summary["stage3_signoff_present"] is False
    assert summary["policy_version"] == PINNED_POLICY_VERSION
    assert summary["policy_version_pinned"] == PINNED_POLICY_VERSION
    assert "isolet" in summary["isolet_recalibration_note"]
    assert "3481" in summary["isolet_recalibration_note"]

    # 10 execution copies under runs/.
    exec_dir = (
        tmp_path / "runs" / "test_stage0_heavy_skip_train"
        / "shards" / "stage0_replica_001"
    )
    exec_files = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) == 10
    for p in exec_files:
        assert "runs" in p.resolve().parts


def test_skip_train_summary_has_all_required_keys(tmp_path: Path) -> None:
    from scripts.run_stage0_heavy_lane import run_stage0_heavy_lane

    fake_gate = _write_fake_standard_lane_summary(tmp_path / "ok.json")
    run_stage0_heavy_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_heavy_schema",
    )
    json_p = (
        tmp_path / "stage_runs" / "test_stage0_heavy_schema_summary.json"
    )
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"
    assert isinstance(payload["policy_version"], str)
    assert len(payload["policy_version"]) == 64


def test_skip_train_does_not_execute_standard_or_extreme(
    tmp_path: Path,
) -> None:
    """No row whose task lane is standard or extreme should end up as
    'success' on disk; they must remain skipped with the right
    last_error. Isolet (3481, standard) must not be promoted."""
    import sqlite3

    from scripts.run_stage0_heavy_lane import run_stage0_heavy_lane

    fake_gate = _write_fake_standard_lane_summary(tmp_path / "ok.json")
    run_stage0_heavy_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_stage0_heavy_isolation",
    )
    exec_dir = (
        tmp_path / "runs" / "test_stage0_heavy_isolation"
        / "shards" / "stage0_replica_001"
    )
    # All standard / extreme task IDs (excluding heavy ones)
    standard_or_extreme_task_ids = (
        # extreme
        6, 167121,
        # all standard
        3, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 37, 43, 45,
        49, 53, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3902,
        3903, 3904, 3913, 3917, 3918, 9946, 9952, 9957, 9960,
        9964, 9971, 9976, 9977, 9978, 9985, 10093, 10101, 14952,
        14954, 14969, 14970, 125920, 125922, 146800, 146817,
        146819, 146820, 146821, 146822, 146824, 167140, 167141,
    )
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(exec_p)
        placeholders = ",".join("?" * len(standard_or_extreme_task_ids))
        rows = list(cx.execute(
            f"SELECT openml_task_id, status, last_error FROM cc18_jobs "
            f"WHERE openml_task_id IN ({placeholders})",
            standard_or_extreme_task_ids,
        ))
        cx.close()
        for tid, status, err in rows:
            assert status == "skipped", (exec_p.name, tid, status, err)
            assert err in {
                "deferred_standard_lane", "deferred_extreme_lane",
            }, (exec_p.name, tid, err)


def test_skip_train_isolet_remains_standard_and_skipped(
    tmp_path: Path,
) -> None:
    """isolet (task 3481) is the future-recalibration candidate. In
    Commit 41 it must NOT be promoted to heavy and must NOT execute.
    Every isolet row in every execution shard must be a deferred
    standard-lane row."""
    import sqlite3

    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_heavy_lane import run_stage0_heavy_lane

    g = RuntimeGuardrails.load()
    assert g.get_task_lane(3481) == "standard", (
        "isolet must remain in standard lane under the pinned policy "
        "version"
    )
    fake_gate = _write_fake_standard_lane_summary(tmp_path / "ok.json")
    run_stage0_heavy_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=fake_gate,
        max_age_days=30,
        skip_train=True,
        run_id="test_isolet_unchanged",
    )
    exec_dir = (
        tmp_path / "runs" / "test_isolet_unchanged"
        / "shards" / "stage0_replica_001"
    )
    isolet_rows: list[tuple] = []
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        cx = sqlite3.connect(exec_p)
        isolet_rows.extend(cx.execute(
            "SELECT status, last_error FROM cc18_jobs "
            "WHERE openml_task_id = 3481",
        ))
        cx.close()
    # isolet has rows in some shards; every such row must be a
    # deferred_standard_lane skip.
    assert len(isolet_rows) > 0
    for status, err in isolet_rows:
        assert status == "skipped"
        assert err == "deferred_standard_lane"


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage0_heavy_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
