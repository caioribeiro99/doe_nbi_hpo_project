"""Tests for the stage 0 replica 1 aggregate signoff planner
(Commit 44).

The builder reads the three lane summaries
(``stage0_standard_lane_latest_summary.json``,
``stage0_heavy_lane_latest_summary.json``,
``stage0_extreme_lane_latest_summary.json``), verifies cross-lane
invariants, and publishes a small planning artifact with
``signoff_status = "planned_not_signed"``. It is **not** a
sign-off file; creating
``jobs/doctoral/openml_cc18/stage3_signoff.json`` is a separate
operator-reviewed commit.

Covers:
- ``--help`` exits zero;
- refuses when any lane summary is missing, failed, has
  failed_timeout, pending, or running cells, has mutated source
  shards, or reports stage3_signoff_present=True;
- refuses when the three lane summaries disagree on
  ``policy_version``;
- refuses when ``stage3_signoff.json`` already exists on disk;
- emits ``signoff_status = "planned_not_signed"``;
- does NOT create ``stage3_signoff.json``;
- reports the canonical lane counts (684 / 156 / 24);
- reports ``n_canary_success_total = 864``;
- reports ``n_tasks_total_expected = 72``;
- records SHA-256 hashes of the three lane summary JSONs;
- summary JSON includes every required schema key;
- Markdown mentions ``isolet`` and ``Devnagari-Script`` caveats;
- no raw payload / execution SQLite / fold CSV / fitted model /
  catboost_info file is staged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/build_stage0_replica_signoff.py"
STANDARD_PATH = (
    REPO / "experiments/_stage_runs/stage0_standard_lane_latest_summary.json"
)
HEAVY_PATH = (
    REPO / "experiments/_stage_runs/stage0_heavy_lane_latest_summary.json"
)
EXTREME_PATH = (
    REPO / "experiments/_stage_runs/stage0_extreme_lane_latest_summary.json"
)
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
SIGNOFF_DOC = REPO / "docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _committed_shard_md5s() -> dict[str, str]:
    return {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }


def _write_lane_summary(
    path: Path, *, lane_name: str, n_executed: int,
    n_failed: int = 0, n_failed_timeout: int = 0,
    n_failed_other: int = 0, n_pending_after: int = 0,
    n_running_after: int = 0,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
    policy_version: str = PINNED_POLICY_VERSION,
    cells: list[dict] | None = None,
    runtime_seconds_runner_total: float = 0.0,
    extras: dict | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": f"stage0_{lane_name}_lane",
        "run_id": f"stage0_{lane_name}_lane_latest",
        "lane": lane_name,
        "exported_at": "2026-05-18T00:00:00Z",
        "source_git_sha": "0" * 40,
        "n_jobs_executed": n_executed,
        "n_jobs_failed": n_failed,
        "n_jobs_failed_timeout": n_failed_timeout,
        "n_jobs_failed_other": n_failed_other,
        "n_jobs_pending_after": n_pending_after,
        "n_jobs_running_after": n_running_after,
        "n_success": n_executed,
        "n_failed": n_failed,
        "n_pending": n_pending_after,
        "n_running": n_running_after,
        "source_shards_unchanged": source_shards_unchanged,
        "stage3_signoff_present": stage3_signoff_present,
        "policy_version": policy_version,
        "cells": cells or [],
        "runtime_seconds_runner_total": runtime_seconds_runner_total,
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_canonical_triple(
    tmp_path: Path, *,
    std_kwargs: dict | None = None,
    hvy_kwargs: dict | None = None,
    ext_kwargs: dict | None = None,
) -> tuple[Path, Path, Path]:
    std = _write_lane_summary(
        tmp_path / "stage0_standard_lane_latest_summary.json",
        lane_name="standard", n_executed=684, **(std_kwargs or {}),
    )
    hvy = _write_lane_summary(
        tmp_path / "stage0_heavy_lane_latest_summary.json",
        lane_name="heavy", n_executed=156, **(hvy_kwargs or {}),
    )
    ext = _write_lane_summary(
        tmp_path / "stage0_extreme_lane_latest_summary.json",
        lane_name="extreme", n_executed=24, **(ext_kwargs or {}),
    )
    return std, hvy, ext


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
    assert "build_stage0_replica_signoff.py" in out
    assert "--standard-summary" in out
    assert "--heavy-summary" in out
    assert "--extreme-summary" in out
    assert "--out-json" in out
    assert "--out-md" in out
    assert "--dry-run" in out


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_refuses_when_standard_summary_missing(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    _, hvy, ext = _write_canonical_triple(tmp_path)
    with pytest.raises(SignoffRefusalError, match="standard-lane summary not found"):
        build_signoff_plan(
            standard_path=tmp_path / "absent.json",
            heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json",
            out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_heavy_summary_missing(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, _, ext = _write_canonical_triple(tmp_path)
    with pytest.raises(SignoffRefusalError, match="heavy-lane summary not found"):
        build_signoff_plan(
            standard_path=std,
            heavy_path=tmp_path / "absent.json",
            extreme_path=ext,
            out_json=tmp_path / "x.json",
            out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_extreme_summary_missing(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, _ = _write_canonical_triple(tmp_path)
    with pytest.raises(SignoffRefusalError, match="extreme-lane summary not found"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy,
            extreme_path=tmp_path / "absent.json",
            out_json=tmp_path / "x.json",
            out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_any_lane_failed(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        hvy_kwargs={"n_failed": 1, "n_failed_other": 1},
    )
    with pytest.raises(SignoffRefusalError, match="not green"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_any_lane_timed_out(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        ext_kwargs={"n_failed": 1, "n_failed_timeout": 1},
    )
    with pytest.raises(SignoffRefusalError, match="not green"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_any_lane_has_pending(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        std_kwargs={"n_pending_after": 3},
    )
    with pytest.raises(SignoffRefusalError, match="unfinished work"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_any_lane_has_running(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        hvy_kwargs={"n_running_after": 1},
    )
    with pytest.raises(SignoffRefusalError, match="unfinished work"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_lanes_disagree_on_policy_version(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        ext_kwargs={"policy_version": "0" * 64},
    )
    with pytest.raises(SignoffRefusalError, match="policy_versions differ"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_source_shards_mutated(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        hvy_kwargs={"source_shards_unchanged": False},
    )
    with pytest.raises(SignoffRefusalError, match="source_shards_unchanged"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_stage3_signoff_present_in_lane_summary(
    tmp_path: Path,
) -> None:
    from scripts.build_stage0_replica_signoff import (
        SignoffRefusalError,
        build_signoff_plan,
    )

    std, hvy, ext = _write_canonical_triple(
        tmp_path,
        ext_kwargs={"stage3_signoff_present": True},
    )
    with pytest.raises(SignoffRefusalError, match="stage3_signoff_present"):
        build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_stage3_signoff_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_stage0_replica_signoff as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    std, hvy, ext = _write_canonical_triple(tmp_path)
    with pytest.raises(m.SignoffRefusalError, match="already exists"):
        m.build_signoff_plan(
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


# ---------------------------------------------------------------------------
# Successful aggregate
# ---------------------------------------------------------------------------


def test_aggregate_records_canonical_counts_and_hashes(
    tmp_path: Path,
) -> None:
    """End-to-end happy path: feed in a clean triple, check the
    emitted JSON carries the canonical counts + lane SHA-256s."""
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    out_json = tmp_path / "agg.json"
    out_md = tmp_path / "agg.md"
    summary = build_signoff_plan(
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_json=out_json, out_md=out_md,
    )
    assert summary["signoff_status"] == "planned_not_signed"
    assert summary["stage3_signoff_present"] is False
    assert summary["policy_version"] == PINNED_POLICY_VERSION
    assert summary["policy_version_consistent"] is True
    assert summary["all_lane_summaries_green"] is True
    assert summary["source_shards_unchanged_all_lanes"] is True
    assert summary["no_pending_running_failed_all_lanes"] is True
    assert summary["n_standard_success"] == 684
    assert summary["n_heavy_success"] == 156
    assert summary["n_extreme_success"] == 24
    assert summary["n_canary_success_total"] == 864
    assert summary["n_failed_total"] == 0
    assert summary["n_failed_timeout_total"] == 0
    assert summary["n_pending_total"] == 0
    assert summary["n_running_total"] == 0
    assert summary["task_coverage"]["n_tasks_total_expected"] == 72
    assert summary["task_coverage"]["n_standard_tasks_expected"] == 57
    assert summary["task_coverage"]["n_heavy_tasks_expected"] == 13
    assert summary["task_coverage"]["n_extreme_tasks_expected"] == 2
    assert summary["final_recommendation"] == "ready_for_operator_review"
    # SHA-256 hashes of each lane summary recorded.
    assert summary["standard_sha256"] == _sha256(std)
    assert summary["heavy_sha256"] == _sha256(hvy)
    assert summary["extreme_sha256"] == _sha256(ext)
    # JSON / MD written to disk.
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    # Required schema keys.
    for key in (
        "schema_version", "run_id", "stage", "lane",
        "exported_at", "signoff_status", "stage3_signoff_present",
        "stage3_signoff_path", "signoff_plan_doc",
        "standard_summary_path", "heavy_summary_path",
        "extreme_summary_path", "extreme_lane_plan_summary_path",
        "standard_sha256", "heavy_sha256", "extreme_sha256",
        "extreme_lane_plan_sha256",
        "policy_version", "policy_version_consistent",
        "all_lane_summaries_green",
        "source_shards_unchanged_all_lanes",
        "no_pending_running_failed_all_lanes",
        "n_jobs_total_expected", "n_standard_success",
        "n_heavy_success", "n_extreme_success",
        "n_canary_success_total", "n_failed_total",
        "n_failed_timeout_total", "n_pending_total",
        "n_running_total", "non_canary_refused_total",
        "deferred_skip_by_lane", "task_coverage", "method_coverage",
        "algorithm_coverage", "runtime_seconds_by_lane",
        "runtime_seconds_total", "slowest_cells_overall",
        "slowest_tasks_overall", "metric_aggregates_by_task_type",
        "metric_aggregates_per_lane", "metric_aggregates_per_method",
        "metric_aggregates_per_algorithm", "metric_aggregates_per_task",
        "caveats", "final_recommendation",
        "what_a_later_signoff_commit_should_do",
        "package_versions", "platform", "git_sha",
        "extreme_lane_plan",
    ):
        assert key in payload, f"missing summary key: {key}"


def test_does_not_create_stage3_signoff_file(tmp_path: Path) -> None:
    """The script's sole purpose is to publish a plan summary; it
    must never create stage3_signoff.json (whose presence would
    unlock the stage-3 top-up machinery)."""
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    build_signoff_plan(
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        out_json=tmp_path / "agg.json",
        out_md=tmp_path / "agg.md",
    )
    assert not SIGNOFF_FILE.exists()


def test_md_mentions_isolet_and_devnagari_caveats(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    out_md = tmp_path / "agg.md"
    build_signoff_plan(
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        out_json=tmp_path / "agg.json",
        out_md=out_md,
    )
    text = out_md.read_text(encoding="utf-8")
    assert "isolet" in text
    assert "3481" in text
    assert "Devnagari-Script" in text
    assert "167121" in text or "Devnagari-Script" in text
    assert "stage0_max_evaluations" in text or "max_evaluations" in text
    assert "planned_not_signed" in text


def test_committed_shards_unchanged_after_run(tmp_path: Path) -> None:
    """The aggregator only reads JSON files; it must not touch
    committed SQLite shards. We snapshot the live committed shard
    MD5s before and after invoking the script."""
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    before = _committed_shard_md5s()
    std, hvy, ext = _write_canonical_triple(tmp_path)
    build_signoff_plan(
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        out_json=tmp_path / "agg.json",
        out_md=tmp_path / "agg.md",
    )
    after = _committed_shard_md5s()
    assert before == after


# ---------------------------------------------------------------------------
# Cross-references + invariants
# ---------------------------------------------------------------------------


def test_signoff_doc_exists_and_mentions_key_concepts() -> None:
    text = SIGNOFF_DOC.read_text(encoding="utf-8")
    for token in (
        "stage0_replica_001",
        "planned_not_signed",
        "stage3_signoff.json",
        "policy_version",
        "isolet",
        "Devnagari-Script",
        "max_evaluations",
        "Commit 40",
        "Commit 41",
        "Commit 43",
        "Commit 44",
    ):
        assert token in text, f"signoff doc missing token: {token}"


def test_signoff_file_still_absent_on_disk() -> None:
    """Sanity: the actual sign-off file must not exist under jobs/
    anywhere in the test suite's lifetime."""
    assert not SIGNOFF_FILE.exists()


def test_committed_lane_summaries_round_trip_through_real_aggregator(
    tmp_path: Path,
) -> None:
    """Smoke: feed the real committed lane summaries to the
    aggregator. This is the same path the CLI uses by default."""
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    out_json = tmp_path / "real_agg.json"
    out_md = tmp_path / "real_agg.md"
    summary = build_signoff_plan(
        standard_path=STANDARD_PATH,
        heavy_path=HEAVY_PATH,
        extreme_path=EXTREME_PATH,
        out_json=out_json, out_md=out_md,
    )
    assert summary["n_canary_success_total"] == 864
    assert summary["n_standard_success"] == 684
    assert summary["n_heavy_success"] == 156
    assert summary["n_extreme_success"] == 24
    assert summary["policy_version_consistent"] is True
    assert summary["policy_version"] == PINNED_POLICY_VERSION
    assert summary["all_lane_summaries_green"] is True
    assert summary["source_shards_unchanged_all_lanes"] is True
    assert summary["final_recommendation"] == "ready_for_operator_review"


def test_no_raw_payload_or_execution_artifact_paths_referenced() -> None:
    """The published aggregate plan must reference only the
    committed stage-run summaries — no per-cell manifests, fold CSVs,
    catboost_info, or execution SQLite files."""
    plan_path = (
        REPO / "experiments/_stage_runs"
        / "stage0_replica_001_signoff_plan_latest_summary.json"
    )
    if not plan_path.exists():
        pytest.skip("plan summary not generated yet; run the script first")
    text = plan_path.read_text(encoding="utf-8")
    forbidden = (
        ".execution.sqlite",
        "fold_metrics.json",
        "fold_metrics.csv",
        "catboost_info",
        "data/source/openml_cc18/",
        "/payload.pkl",
    )
    for tok in forbidden:
        assert tok not in text, f"plan summary leaks forbidden token: {tok}"
