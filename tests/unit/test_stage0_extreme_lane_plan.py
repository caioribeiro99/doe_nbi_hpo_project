"""Tests for the stage0 extreme-lane planning script (Commit 42).

The script ships in PLANNING-ONLY mode for this commit. Execution
is locked behind ``--execute-extreme-lane``, which Commit 42 must
NOT pass.

Covers:
- ``--help`` / ``--dry-run`` exit zero;
- Without ``--execute-extreme-lane`` the script runs in planning
  mode and publishes a dry-run summary;
- ``execute_stage0_extreme_lane()`` is a stub that raises
  ``NotImplementedError`` (Commit 43 will fill it in);
- the planning entry point refuses when either prior-stage summary
  (standard / heavy) is missing, failed, has unfinished work, has
  mutated source shards, has the stage-3 sign-off file, or is
  stale;
- the planning entry point refuses when the two prior-stage
  summaries disagree on ``policy_version`` or when the live policy
  CSV drifted from the pinned Commit 40 value;
- the classifier puts canary × extreme rows in
  ``runnable_extreme_canary``, any standard row in
  ``skipped_standard_lane_already_completed``, any heavy row in
  ``skipped_heavy_lane_already_completed``, and extreme × non-canary
  in ``refused_not_in_canary_set``;
- the pre-run plan finds exactly 24 extreme canary cells across
  tasks 6 and 167121;
- the planning run does NOT mutate any committed source shard,
  does NOT create execution SQLite files under ``runs/cc18/``, does
  NOT contact OpenML, and does NOT create
  ``stage3_signoff.json``;
- the dry-run summary records
  ``execution_status = "planned_not_executed"``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/run_stage0_extreme_lane.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
OPENML_CACHE = REPO / "data/source/openml_cc18"
RUNS_ROOT = REPO / "runs/cc18"
PLAN_DOC = REPO / "docs/EXTREME_LANE_PLAN.md"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


@pytest.fixture(autouse=True)
def _hide_real_signoff_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Commit 45 created ``stage3_signoff.json`` on disk. The extreme-
    lane runner refuses to run (in both planning and execute paths)
    once that file exists; tests that exercise the runner must
    therefore see ``SIGNOFF_FILE`` as absent. Tests that verify the
    guard override this with their own per-test setattr to a tmp
    path."""
    from scripts import run_stage0_extreme_lane as m

    monkeypatch.setattr(
        m, "SIGNOFF_FILE",
        tmp_path_factory.mktemp("hide_signoff") / "absent.json",
    )


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_lane_summary(
    path: Path, *, lane_name: str, expected_executed: int,
    exported_at: str | None = None, n_jobs_executed: int | None = None,
    n_jobs_failed: int = 0, n_jobs_failed_timeout: int = 0,
    n_jobs_failed_other: int = 0, n_jobs_pending_after: int = 0,
    n_jobs_running_after: int = 0,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
    policy_version: str = PINNED_POLICY_VERSION,
) -> Path:
    n_executed = expected_executed if n_jobs_executed is None else n_jobs_executed
    payload = {
        "schema_version": 1,
        "batch_id": f"stage0_{lane_name}_lane",
        "run_id": f"stage0_{lane_name}_lane_latest",
        "lane": lane_name,
        "exported_at": exported_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "source_git_sha": "0" * 40,
        "n_jobs_executed": n_executed,
        "n_jobs_failed": n_jobs_failed,
        "n_jobs_failed_timeout": n_jobs_failed_timeout,
        "n_jobs_failed_other": n_jobs_failed_other,
        "n_jobs_pending_after": n_jobs_pending_after,
        "n_jobs_running_after": n_jobs_running_after,
        "n_success": n_executed,
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


def _write_pair(
    tmp_path: Path, *,
    standard_policy_version: str = PINNED_POLICY_VERSION,
    heavy_policy_version: str = PINNED_POLICY_VERSION,
    std_kwargs: dict | None = None, hvy_kwargs: dict | None = None,
) -> tuple[Path, Path]:
    std = _write_fake_lane_summary(
        tmp_path / "std.json", lane_name="standard",
        expected_executed=684,
        policy_version=standard_policy_version,
        **(std_kwargs or {}),
    )
    hvy = _write_fake_lane_summary(
        tmp_path / "hvy.json", lane_name="heavy",
        expected_executed=156,
        policy_version=heavy_policy_version,
        **(hvy_kwargs or {}),
    )
    return std, hvy


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
    assert "run_stage0_extreme_lane.py" in out
    assert "--standard-lane-summary" in out
    assert "--heavy-lane-summary" in out
    assert "--policy-csv" in out
    assert "--guardrails-yaml" in out
    assert "--execute-extreme-lane" in out
    assert "--max-age-days" in out
    assert "--include-extreme-tasks" in out
    assert "--dry-run" in out


def test_dry_run_exits_zero_and_publishes_planning_summary(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "stage_runs"
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run",
         "--stage-runs-dir", str(out_dir),
         "--signoff-file", str(tmp_path / "absent_signoff.json")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    # The script always defaults to planning mode regardless of
    # --dry-run, so the summary lands at the default run_id.
    json_p = out_dir / "stage0_extreme_lane_plan_latest_summary.json"
    md_p = out_dir / "stage0_extreme_lane_plan_latest_summary.md"
    assert json_p.exists()
    assert md_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    assert payload["execution_status"] == "planned_not_executed"
    assert payload["expected_extreme_canary_cells"] == 24
    assert payload["n_runnable_extreme_canary"] == 24
    assert payload["extreme_tasks_to_execute"] == [6, 167121]
    assert payload["openml_payloads_loaded"] is False
    assert payload["execution_shards_created"] is False
    assert payload["execution_shards_committed"] is False


def test_default_invocation_runs_planning_mode_not_execution(
    tmp_path: Path,
) -> None:
    """Without --execute-extreme-lane the script prints the
    planning verdict and exits 0 — never reaches execution code."""
    out_dir = tmp_path / "stage_runs"
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--stage-runs-dir", str(out_dir),
         "--signoff-file", str(tmp_path / "absent_signoff.json")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "PLAN:" in res.stdout
    assert "Execution disabled" in res.stdout
    assert "--execute-extreme-lane" in res.stdout


def test_execute_path_skip_train_smoke(tmp_path: Path) -> None:
    """Commit 43 added the real execute_stage0_extreme_lane entry
    point. The skip_train smoke test runs the whole pre-flight,
    run-dir, classification, and summary path without invoking
    cc18_runner. Verifies the executed summary carries
    execution_status='executed' and the per-task / per-algorithm /
    per-method breakdowns."""
    from scripts.run_stage0_extreme_lane import execute_stage0_extreme_lane

    md5_before = _committed_shard_md5s()
    summary = execute_stage0_extreme_lane(
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        standard_lane_summary=(
            REPO / "experiments/_stage_runs"
            / "stage0_standard_lane_latest_summary.json"
        ),
        heavy_lane_summary=(
            REPO / "experiments/_stage_runs"
            / "stage0_heavy_lane_latest_summary.json"
        ),
        plan_summary_path=(
            REPO / "experiments/_stage_runs"
            / "stage0_extreme_lane_plan_latest_summary.json"
        ),
        max_age_days=30,
        skip_train=True,
        run_id="test_exec_skip_train",
    )
    assert summary["execution_status"] == "executed"
    assert summary["batch_id"] == "stage0_extreme_lane"
    assert summary["lane"] == "extreme"
    assert summary["n_source_shards"] == 10
    assert summary["expected_extreme_canary_cells"] == 24
    assert summary["n_jobs_executed"] == 0  # skip_train suppresses cc18_runner
    assert summary["n_jobs_deferred_standard"] == 1815
    assert summary["n_jobs_deferred_heavy"] == 423
    assert summary["n_jobs_refused_non_canary"] == 42
    assert summary["source_shards_unchanged"] is True
    assert summary["stage3_signoff_present"] is False
    md5_after = _committed_shard_md5s()
    assert md5_before == md5_after
    # Per-task / per-algorithm / per-method breakdowns are present.
    assert "per_task_breakdown" in summary
    assert "per_algorithm_breakdown" in summary
    assert "per_method_breakdown" in summary
    assert "max_evaluations_used_per_task" in summary
    assert "timeout_seconds_per_cell_per_task" in summary
    # policy_max_evaluations_note reports stage0_max_evaluations=1.
    assert "stage0_max_evaluations=1" in summary["policy_max_evaluations_note"]
    assert "timeout_seconds_per_cell=14400" in summary["policy_timeout_note"]


def test_execute_path_refuses_when_plan_summary_missing(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        execute_stage0_extreme_lane,
    )

    std, hvy = _write_pair(tmp_path)
    with pytest.raises(GateRefusalError, match="Commit 42 plan summary not found"):
        execute_stage0_extreme_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=std,
            heavy_lane_summary=hvy,
            plan_summary_path=tmp_path / "absent.json",
            max_age_days=30,
            skip_train=True,
            run_id="test_exec_missing_plan",
        )


def test_execute_path_refuses_when_plan_already_executed(
    tmp_path: Path,
) -> None:
    """If the plan summary on disk reports execution_status='executed'
    (i.e. the lane was already run), Commit 43's executor refuses to
    re-run it."""
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        execute_stage0_extreme_lane,
    )

    fake_plan = tmp_path / "plan.json"
    fake_plan.write_text(
        json.dumps({
            "execution_status": "executed",  # not planned_not_executed
            "n_runnable_extreme_canary": 24,
            "extreme_tasks_to_execute": [6, 167121],
            "policy_version": PINNED_POLICY_VERSION,
            "exported_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            "stage3_signoff_present": False,
        }),
        encoding="utf-8",
    )
    std, hvy = _write_pair(tmp_path)
    with pytest.raises(GateRefusalError, match="planned_not_executed"):
        execute_stage0_extreme_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=std,
            heavy_lane_summary=hvy,
            plan_summary_path=fake_plan,
            max_age_days=30,
            skip_train=True,
        )


def test_execute_path_refuses_when_plan_has_policy_drift(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        execute_stage0_extreme_lane,
    )

    fake_plan = tmp_path / "plan.json"
    fake_plan.write_text(
        json.dumps({
            "execution_status": "planned_not_executed",
            "n_runnable_extreme_canary": 24,
            "extreme_tasks_to_execute": [6, 167121],
            "policy_version": "f" * 64,  # drift
            "exported_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            "stage3_signoff_present": False,
        }),
        encoding="utf-8",
    )
    std, hvy = _write_pair(tmp_path)
    with pytest.raises(GateRefusalError, match="policy_version"):
        execute_stage0_extreme_lane(
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            standard_lane_summary=std,
            heavy_lane_summary=hvy,
            plan_summary_path=fake_plan,
            max_age_days=30,
            skip_train=True,
        )


def test_cli_execute_refuses_without_include_extreme_tasks(
    tmp_path: Path,
) -> None:
    """The CLI requires BOTH --execute-extreme-lane AND
    --include-extreme-tasks. A single flag is not enough."""
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--execute-extreme-lane",
         "--stage-runs-dir", str(tmp_path / "stage_runs"),
         "--run-root", str(tmp_path / "runs"),
         "--output-root", str(tmp_path / "out"),
         "--openml-cache-root", str(tmp_path / "cache")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 3, res.stdout + res.stderr
    assert "--include-extreme-tasks" in res.stderr


def test_cli_include_extreme_tasks_alone_does_not_execute(
    tmp_path: Path,
) -> None:
    """--include-extreme-tasks WITHOUT --execute-extreme-lane stays
    in planning mode."""
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--include-extreme-tasks",
         "--stage-runs-dir", str(tmp_path / "stage_runs"),
         "--signoff-file", str(tmp_path / "absent_signoff.json")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    # The planning summary was published — NOT the execute summary.
    assert (
        tmp_path / "stage_runs"
        / "stage0_extreme_lane_plan_latest_summary.json"
    ).exists()
    assert not (
        tmp_path / "stage_runs"
        / "stage0_extreme_lane_latest_summary.json"
    ).exists()


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_standard_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    hvy = _write_fake_lane_summary(
        tmp_path / "hvy.json", lane_name="heavy", expected_executed=156,
    )
    with pytest.raises(GateRefusalError, match="standard-lane summary not found"):
        verify_prior_stages(
            standard_summary=tmp_path / "absent.json",
            heavy_summary=hvy,
        )


def test_refuses_when_heavy_summary_missing(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std = _write_fake_lane_summary(
        tmp_path / "std.json", lane_name="standard", expected_executed=684,
    )
    with pytest.raises(GateRefusalError, match="heavy-lane summary not found"):
        verify_prior_stages(
            standard_summary=std,
            heavy_summary=tmp_path / "absent.json",
        )


def test_refuses_when_standard_under_executed(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path,
        std_kwargs={"n_jobs_executed": 600},
    )
    with pytest.raises(GateRefusalError, match="executed only"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_heavy_under_executed(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path, hvy_kwargs={"n_jobs_executed": 100},
    )
    with pytest.raises(GateRefusalError, match="executed only"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_standard_failed(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path,
        std_kwargs={"n_jobs_failed": 3, "n_jobs_failed_other": 3},
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_heavy_failed(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path,
        hvy_kwargs={"n_jobs_failed": 1, "n_jobs_failed_timeout": 1},
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_standard_pending(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path, std_kwargs={"n_jobs_pending_after": 1},
    )
    with pytest.raises(GateRefusalError, match="unfinished work"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_heavy_running(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path, hvy_kwargs={"n_jobs_running_after": 2},
    )
    with pytest.raises(GateRefusalError, match="unfinished work"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_standard_shards_mutated(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path, std_kwargs={"source_shards_unchanged": False},
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_heavy_signed_off(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path, hvy_kwargs={"stage3_signoff_present": True},
    )
    with pytest.raises(GateRefusalError, match="stage3_signoff_present"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_lanes_disagree_on_policy_version(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path,
        standard_policy_version="a" * 64,
        heavy_policy_version="b" * 64,
    )
    with pytest.raises(GateRefusalError, match="do not share a policy"):
        verify_prior_stages(standard_summary=std, heavy_summary=hvy)


def test_refuses_when_live_policy_drifts_from_prior_lanes(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    std, hvy = _write_pair(
        tmp_path,
        standard_policy_version="c" * 64,
        heavy_policy_version="c" * 64,
    )
    with pytest.raises(GateRefusalError, match="mid-replica"):
        verify_prior_stages(
            standard_summary=std, heavy_summary=hvy,
            live_policy_version="d" * 64,
        )


def test_refuses_when_summary_stale(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import (
        GateRefusalError,
        verify_prior_stages,
    )

    stale = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    std, hvy = _write_pair(
        tmp_path, std_kwargs={"exported_at": stale},
    )
    with pytest.raises(GateRefusalError, match="days old"):
        verify_prior_stages(
            standard_summary=std, heavy_summary=hvy, max_age_days=7,
        )


def test_refuses_when_live_policy_drifts_from_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_stage0_extreme_lane as m

    monkeypatch.setattr(m, "PINNED_POLICY_VERSION", "0" * 64)
    std, hvy = _write_pair(
        tmp_path,
        standard_policy_version="0" * 64,
        heavy_policy_version="0" * 64,
    )
    with pytest.raises(m.GateRefusalError, match="Commit 42 pins"):
        m.plan_stage0_extreme_lane(
            shards_dir=SHARDS_DIR,
            stage_runs_dir=tmp_path / "stage_runs",
            standard_lane_summary=std,
            heavy_lane_summary=hvy,
            max_age_days=30,
        )


def test_refuses_when_stage3_signoff_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_stage0_extreme_lane as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    std, hvy = _write_pair(tmp_path)
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.plan_stage0_extreme_lane(
            shards_dir=SHARDS_DIR,
            stage_runs_dir=tmp_path / "stage_runs",
            standard_lane_summary=std,
            heavy_lane_summary=hvy,
            max_age_days=30,
        )


# ---------------------------------------------------------------------------
# Classifier + pre-run plan
# ---------------------------------------------------------------------------


def test_classifier_buckets_match_extreme_lane_intent() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_extreme_lane import classify_rows

    g = RuntimeGuardrails.load()
    rows = [
        # standard task → skipped_standard_lane_already_completed
        ("j_std_c", 11, "default_gbdt", "xgboost"),
        ("j_std_n", 11, "smac3", "lightgbm"),
        # heavy task → skipped_heavy_lane_already_completed
        ("j_h_c", 3573, "default_gbdt", "xgboost"),
        ("j_h_n", 3573, "asha", "lightgbm"),
        # extreme task × non-canary → refused
        ("j_e_r1", 6, "smac3", "xgboost"),
        ("j_e_r2", 167121, "parego", "catboost"),
        # extreme task × canary → runnable_extreme_canary
        ("j_e_run1", 6, "default_gbdt", "xgboost"),
        ("j_e_run2", 167121, "doe_rsm_vrf_true_nbi", "catboost"),
    ]
    buckets = classify_rows(rows, g)
    assert sorted(
        e["job_id"] for e in
        buckets["skipped_standard_lane_already_completed"]
    ) == ["j_std_c", "j_std_n"]
    assert sorted(
        e["job_id"] for e in
        buckets["skipped_heavy_lane_already_completed"]
    ) == ["j_h_c", "j_h_n"]
    assert sorted(
        e["job_id"] for e in buckets["refused_not_in_canary_set"]
    ) == ["j_e_r1", "j_e_r2"]
    assert sorted(
        e["job_id"] for e in buckets["runnable_extreme_canary"]
    ) == ["j_e_run1", "j_e_run2"]


def test_pre_run_plan_finds_24_extreme_canary_cells() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails
    from scripts.run_stage0_extreme_lane import build_pre_run_plan

    g = RuntimeGuardrails.load()
    shards = sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    plan = build_pre_run_plan(shards, g)
    assert plan["n_source_shards"] == 10
    assert plan["n_jobs_total"] == 2304
    assert plan["n_runnable_extreme_canary"] == 24
    assert plan["n_skipped_standard_lane"] == 1815
    assert plan["n_skipped_heavy_lane"] == 423
    assert plan["n_refused_not_in_canary_set"] == 42
    assert plan["extreme_tasks_to_execute"] == [6, 167121]
    assert plan["task_lane_counts_universe"] == {
        "standard": 57, "heavy": 13, "extreme": 2,
    }


# ---------------------------------------------------------------------------
# Planning entry point: does not mutate state
# ---------------------------------------------------------------------------


def _committed_shard_md5s() -> dict[str, str]:
    return {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }


def _openml_cache_fingerprint() -> tuple[str, ...]:
    if not OPENML_CACHE.exists():
        return ()
    return tuple(sorted(p.name for p in OPENML_CACHE.iterdir() if p.is_dir()))


def test_plan_run_does_not_mutate_committed_shards(tmp_path: Path) -> None:
    from scripts.run_stage0_extreme_lane import plan_stage0_extreme_lane

    before = _committed_shard_md5s()
    std, hvy = _write_pair(tmp_path)
    plan_stage0_extreme_lane(
        shards_dir=SHARDS_DIR,
        stage_runs_dir=tmp_path / "stage_runs",
        standard_lane_summary=std,
        heavy_lane_summary=hvy,
        max_age_days=30,
        run_id="test_plan_no_mutation",
    )
    after = _committed_shard_md5s()
    assert before == after


def test_plan_run_does_not_create_execution_sqlite(tmp_path: Path) -> None:
    """The planning entry point may NOT create execution copies
    under runs/cc18/. Verify by snapshotting runs/cc18/ before and
    after."""
    from scripts.run_stage0_extreme_lane import plan_stage0_extreme_lane

    def snapshot() -> set[Path]:
        if not RUNS_ROOT.exists():
            return set()
        return {p for p in RUNS_ROOT.rglob("*") if p.is_file()}

    before = snapshot()
    std, hvy = _write_pair(tmp_path)
    plan_stage0_extreme_lane(
        shards_dir=SHARDS_DIR,
        stage_runs_dir=tmp_path / "stage_runs",
        standard_lane_summary=std,
        heavy_lane_summary=hvy,
        max_age_days=30,
        run_id="test_plan_no_runs_writes",
    )
    after = snapshot()
    assert before == after


def test_plan_run_does_not_download_new_openml_payloads(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import plan_stage0_extreme_lane

    before = _openml_cache_fingerprint()
    std, hvy = _write_pair(tmp_path)
    plan_stage0_extreme_lane(
        shards_dir=SHARDS_DIR,
        stage_runs_dir=tmp_path / "stage_runs",
        standard_lane_summary=std,
        heavy_lane_summary=hvy,
        max_age_days=30,
        run_id="test_plan_no_openml_writes",
    )
    after = _openml_cache_fingerprint()
    assert before == after


def test_plan_run_marks_execution_status_planned_not_executed(
    tmp_path: Path,
) -> None:
    from scripts.run_stage0_extreme_lane import plan_stage0_extreme_lane

    std, hvy = _write_pair(tmp_path)
    summary = plan_stage0_extreme_lane(
        shards_dir=SHARDS_DIR,
        stage_runs_dir=tmp_path / "stage_runs",
        standard_lane_summary=std,
        heavy_lane_summary=hvy,
        max_age_days=30,
        run_id="test_plan_execution_status",
    )
    assert summary["execution_status"] == "planned_not_executed"
    assert summary["openml_payloads_loaded"] is False
    assert summary["execution_shards_created"] is False
    assert summary["expected_extreme_canary_cells"] == 24
    assert summary["extreme_tasks_to_execute"] == [6, 167121]
    json_p = tmp_path / "stage_runs" / "test_plan_execution_status_summary.json"
    md_p = tmp_path / "stage_runs" / "test_plan_execution_status_summary.md"
    assert json_p.exists()
    assert md_p.exists()
    md_text = md_p.read_text(encoding="utf-8")
    assert "planned_not_executed" in md_text
    assert "EXTREME_LANE_PLAN.md" in md_text


# ---------------------------------------------------------------------------
# Misc invariants
# ---------------------------------------------------------------------------


def test_plan_doc_exists_and_mentions_key_concepts() -> None:
    text = PLAN_DOC.read_text(encoding="utf-8")
    text_lower = text.lower()
    for token in (
        "Devnagari-Script", "letter", "167121",
        "extreme.stage0_max_evaluations",
        "policy_version",
        "execution_status",
        "planned_not_executed",
        "executed",
        "14,400", "4 h",
        "max_evaluations",
        "extreme-lane plan", "Commit 42", "Commit 43",
    ):
        assert token in text, f"plan doc missing token: {token}"
    for lower_token in ("promotion criteria",):
        assert lower_token in text_lower, (
            f"plan doc missing case-insensitive token: {lower_token}"
        )


def test_default_paths_resolve_to_committed_artifacts() -> None:
    from scripts.run_stage0_extreme_lane import (
        DEFAULT_GUARDRAILS_YAML,
        DEFAULT_HEAVY_LANE_SUMMARY,
        DEFAULT_POLICY_CSV,
        DEFAULT_SHARDS_DIR,
        DEFAULT_STANDARD_LANE_SUMMARY,
    )

    assert DEFAULT_SHARDS_DIR.name == "stage0_replica_001"
    assert DEFAULT_POLICY_CSV == POLICY_CSV
    assert DEFAULT_GUARDRAILS_YAML == GUARDRAILS_YAML
    assert DEFAULT_STANDARD_LANE_SUMMARY.name == (
        "stage0_standard_lane_latest_summary.json"
    )
    assert DEFAULT_HEAVY_LANE_SUMMARY.name == (
        "stage0_heavy_lane_latest_summary.json"
    )


def test_module_exports_match_expected_surface() -> None:
    from scripts import run_stage0_extreme_lane as m

    for name in (
        "plan_stage0_extreme_lane", "execute_stage0_extreme_lane",
        "classify_rows", "build_pre_run_plan",
        "verify_prior_stages", "GateRefusalError",
        "PINNED_POLICY_VERSION", "EXPECTED_EXTREME_CANARY_CELLS",
        "DRY_RUN_ID", "BATCH_ID_PLAN", "main",
    ):
        assert name in m.__all__
        assert hasattr(m, name)


def test_dry_run_does_not_touch_data_source_openml_cc18(
    tmp_path: Path,
) -> None:
    """End-to-end CLI dry-run smoke: no writes to data/source/."""
    before = _openml_cache_fingerprint()
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--stage-runs-dir", str(tmp_path / "stage_runs"),
         "--signoff-file", str(tmp_path / "absent_signoff.json")],
        capture_output=True, text=True, check=False,
        env={**os.environ},
    )
    assert res.returncode == 0, res.stderr
    after = _openml_cache_fingerprint()
    assert before == after
