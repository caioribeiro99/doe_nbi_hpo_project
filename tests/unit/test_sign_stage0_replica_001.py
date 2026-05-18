"""Tests for ``scripts/sign_stage0_replica_001.py`` (Commit 45).

The sign-off script reads the aggregate signoff plan published by
``scripts/build_stage0_replica_signoff.py`` (Commit 44), validates
the aggregate gates one more time, then writes the operator
sign-off JSON at ``jobs/doctoral/openml_cc18/stage3_signoff.json``
and (by default) re-invokes the aggregator so the published plan
summary flips ``signoff_status`` to ``"signed"``.

This commit is the **only** commit allowed to create that file
on the repo; the runner refuses to claim stage-3 top-up rows
until it exists. Creating it is a deliberate, operator-reviewed
capacity decision, so every gate the aggregator advertises must
be re-checked here, the operator metadata must be present and
explicit, and the file must not authorize downstream execution by
its mere presence.

Covers:
- ``--help`` exits zero;
- happy path writes a well-formed sign-off JSON with all required
  fields, the operator defaults from the Commit 45 prompt, both
  required caveat acknowledgements, the same SHA-256s the
  aggregate plan recorded, and
  ``downstream_execution_authorized_in_this_commit = false``;
- after writing, optionally re-runs the aggregator and the
  republished plan summary reports ``signoff_status = "signed"``
  with hashes that match the on-disk sign-off file;
- refuses if the aggregate plan summary is missing;
- refuses if the aggregate plan summary's
  ``final_recommendation`` is not
  ``"ready_for_operator_review"``;
- refuses if a sign-off file already exists (unless ``--force``);
- refuses if the aggregate plan's policy_version diverges from
  the pinned Commit 40 SHA-256;
- emits no paths to mutable run dirs, per-cell payloads, fold
  CSVs, fitted models, or catboost_info traces;
- ``--no-refresh-aggregate`` skips the aggregator re-run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/sign_stage0_replica_001.py"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_lane_summary(
    path: Path, *, lane_name: str, n_executed: int,
    policy_version: str = PINNED_POLICY_VERSION,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": f"stage0_{lane_name}_lane",
        "run_id": f"stage0_{lane_name}_lane_latest",
        "lane": lane_name,
        "exported_at": "2026-05-18T00:00:00Z",
        "source_git_sha": "0" * 40,
        "n_jobs_executed": n_executed,
        "n_jobs_failed": 0,
        "n_jobs_failed_timeout": 0,
        "n_jobs_failed_other": 0,
        "n_jobs_pending_after": 0,
        "n_jobs_running_after": 0,
        "n_success": n_executed,
        "n_failed": 0,
        "n_pending": 0,
        "n_running": 0,
        "source_shards_unchanged": True,
        "stage3_signoff_present": False,
        "policy_version": policy_version,
        "cells": [],
        "runtime_seconds_runner_total": 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_canonical_triple(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        _write_lane_summary(
            tmp_path / "stage0_standard_lane_latest_summary.json",
            lane_name="standard", n_executed=684,
        ),
        _write_lane_summary(
            tmp_path / "stage0_heavy_lane_latest_summary.json",
            lane_name="heavy", n_executed=156,
        ),
        _write_lane_summary(
            tmp_path / "stage0_extreme_lane_latest_summary.json",
            lane_name="extreme", n_executed=24,
        ),
    )


def _build_plan(
    tmp_path: Path,
    std: Path, hvy: Path, ext: Path,
    *, write: bool = True,
) -> tuple[Path, Path, dict]:
    """Run the Commit 44 aggregator against the test triple and
    return ``(plan_json_path, plan_md_path, summary_dict)``."""
    from scripts.build_stage0_replica_signoff import build_signoff_plan

    plan_json = tmp_path / "stage0_replica_001_signoff_plan_latest_summary.json"
    plan_md = tmp_path / "stage0_replica_001_signoff_plan_latest_summary.md"
    summary = build_signoff_plan(
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_json=plan_json, out_md=plan_md,
        write_summary=write,
        signoff_file=tmp_path / "absent_signoff.json",
    )
    return plan_json, plan_md, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_run_script_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "sign_stage0_replica_001.py" in out
    assert "--operator-name" in out
    assert "--operator-handle" in out
    assert "--justification" in out
    assert "--force" in out
    assert "--no-refresh-aggregate" in out


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_refuses_when_aggregate_plan_missing(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    with pytest.raises(SignoffRefusalError, match="aggregate plan summary not found"):
        sign_stage0_replica_001(
            aggregate_plan_path=tmp_path / "absent.json",
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=tmp_path / "stage3_signoff.json",
            refresh_aggregate=False,
        )


def test_refuses_when_signoff_already_exists(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, _ = _build_plan(tmp_path, std, hvy, ext)
    out = tmp_path / "stage3_signoff.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SignoffRefusalError, match="already exists"):
        sign_stage0_replica_001(
            aggregate_plan_path=plan_json,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=out,
            refresh_aggregate=False,
        )


def test_force_allows_overwrite(tmp_path: Path) -> None:
    """``--force`` must be honoured but Commit 45 itself never uses
    it. The test exists to keep that override behaviour pinned."""
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, _ = _build_plan(tmp_path, std, hvy, ext)
    out = tmp_path / "stage3_signoff.json"
    out.write_text("{}", encoding="utf-8")
    signoff_path, record, _ = sign_stage0_replica_001(
        aggregate_plan_path=plan_json,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_signoff=out,
        force=True,
        refresh_aggregate=False,
    )
    assert signoff_path == out
    assert record["signoff_status"] == "signed"


def test_refuses_when_aggregate_status_not_ready(tmp_path: Path) -> None:
    """If a previous run already flipped the aggregate plan to
    ``signoff_status='signed'``, calling the signer again should
    refuse rather than re-write the file."""
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, summary = _build_plan(tmp_path, std, hvy, ext)
    # Forge a plan that claims to already be signed.
    summary["signoff_status"] = "signed"
    summary["final_recommendation"] = "signed_ready_for_next_stage_planning"
    plan_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pytest.raises(SignoffRefusalError, match="planned_not_signed"):
        sign_stage0_replica_001(
            aggregate_plan_path=plan_json,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=tmp_path / "stage3_signoff.json",
            refresh_aggregate=False,
        )


def test_refuses_on_policy_version_drift(tmp_path: Path) -> None:
    """The signer pins the Commit 40 policy_version constant; if the
    aggregate plan reports a different one, refuse."""
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, summary = _build_plan(tmp_path, std, hvy, ext)
    summary["policy_version"] = "0" * 64
    plan_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pytest.raises(SignoffRefusalError, match="pinned"):
        sign_stage0_replica_001(
            aggregate_plan_path=plan_json,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=tmp_path / "stage3_signoff.json",
            refresh_aggregate=False,
        )


def test_refuses_on_lane_count_drift(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, summary = _build_plan(tmp_path, std, hvy, ext)
    summary["n_heavy_success"] = 999
    plan_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pytest.raises(SignoffRefusalError, match="n_heavy_success"):
        sign_stage0_replica_001(
            aggregate_plan_path=plan_json,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=tmp_path / "stage3_signoff.json",
            refresh_aggregate=False,
        )


def test_refuses_on_residual_failure_count(tmp_path: Path) -> None:
    from scripts.build_stage0_replica_signoff import SignoffRefusalError
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, summary = _build_plan(tmp_path, std, hvy, ext)
    summary["n_failed_total"] = 1
    plan_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pytest.raises(SignoffRefusalError, match="n_failed_total"):
        sign_stage0_replica_001(
            aggregate_plan_path=plan_json,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            plan_path=tmp_path / "absent_plan.json",
            out_signoff=tmp_path / "stage3_signoff.json",
            refresh_aggregate=False,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_writes_full_signoff_record(tmp_path: Path) -> None:
    from scripts.sign_stage0_replica_001 import (
        CAVEATS_ACKNOWLEDGED,
        DEFAULT_OPERATOR_HANDLE,
        DEFAULT_OPERATOR_NAME,
        SCHEMA_VERSION,
        SIGNOFF_TYPE,
        sign_stage0_replica_001,
    )
    from scripts.sign_stage0_replica_001 import (
        PINNED_POLICY_VERSION as PV_CONST,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, _ = _build_plan(tmp_path, std, hvy, ext)
    out = tmp_path / "stage3_signoff.json"
    signoff_path, record, refreshed = sign_stage0_replica_001(
        aggregate_plan_path=plan_json,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_signoff=out,
        refresh_aggregate=False,
    )
    assert signoff_path == out
    assert out.exists()
    assert refreshed is None

    # Required schema keys.
    for key in (
        "schema_version", "signoff_type", "signoff_status",
        "signed_at_utc", "operator_name", "operator_handle",
        "branch", "git_sha_at_signoff", "policy_version",
        "declared_scope", "justification", "caveats_acknowledged",
        "aggregate_plan_summary_path", "aggregate_plan_summary_sha256",
        "signoff_plan_summary_path", "signoff_plan_summary_sha256",
        "standard_lane_summary_path", "standard_lane_summary_sha256",
        "heavy_lane_summary_path", "heavy_lane_summary_sha256",
        "extreme_lane_summary_path", "extreme_lane_summary_sha256",
        "n_jobs_total_expected", "n_canary_success_total",
        "lane_success_counts", "failure_counts",
        "source_shards_unchanged_all_lanes",
        "no_pending_running_failed_all_lanes",
        "downstream_execution_authorized_in_this_commit",
        "notes",
    ):
        assert key in record, f"signoff record missing {key}"

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["signoff_type"] == SIGNOFF_TYPE
    assert record["signoff_status"] == "signed"
    assert record["operator_name"] == DEFAULT_OPERATOR_NAME
    assert record["operator_handle"] == DEFAULT_OPERATOR_HANDLE
    assert record["policy_version"] == PV_CONST
    assert record["n_canary_success_total"] == 864
    assert record["n_jobs_total_expected"] == 2304
    assert record["lane_success_counts"] == {
        "standard": 684, "heavy": 156, "extreme": 24,
    }
    assert record["failure_counts"] == {
        "failed_total": 0, "failed_timeout_total": 0,
        "pending_total": 0, "running_total": 0,
    }
    assert record["source_shards_unchanged_all_lanes"] is True
    assert record["no_pending_running_failed_all_lanes"] is True
    assert record["downstream_execution_authorized_in_this_commit"] is False

    # Lane SHA-256s match the files on disk.
    assert record["standard_lane_summary_sha256"] == _sha256(std)
    assert record["heavy_lane_summary_sha256"] == _sha256(hvy)
    assert record["extreme_lane_summary_sha256"] == _sha256(ext)
    assert record["aggregate_plan_summary_sha256"] == _sha256(plan_json)

    # Both required caveats acknowledged.
    caveat_ids = {c["id"] for c in record["caveats_acknowledged"]}
    assert "isolet_future_recalibration_candidate" in caveat_ids
    assert "devnagari_extreme_budget_non_equivalence" in caveat_ids
    assert record["caveats_acknowledged"] == CAVEATS_ACKNOWLEDGED

    # The justification names both caveats explicitly.
    j = record["justification"]
    assert "isolet" in j
    assert "Devnagari-Script" in j
    assert "stage0_max_evaluations" in j

    # Declared scope is non-trivial and mentions the no-downstream gate.
    assert len(record["declared_scope"]) >= 2
    assert any(
        "downstream" in item.lower() or "execution" in item.lower()
        for item in record["declared_scope"]
    )


def test_happy_path_refreshes_aggregate_to_signed(tmp_path: Path) -> None:
    """When ``refresh_aggregate=True`` (the default), the signer
    re-runs the aggregator and the republished plan summary should
    report ``signoff_status='signed'`` and carry the on-disk
    sign-off file's SHA-256."""
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json = tmp_path / "stage0_replica_001_signoff_plan_latest_summary.json"
    plan_md = tmp_path / "stage0_replica_001_signoff_plan_latest_summary.md"
    out = tmp_path / "stage3_signoff.json"
    _build_plan(tmp_path, std, hvy, ext)
    _signoff_path, _record, refreshed = sign_stage0_replica_001(
        aggregate_plan_path=plan_json,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_signoff=out,
        aggregate_out_json=plan_json,
        aggregate_out_md=plan_md,
        refresh_aggregate=True,
    )
    assert refreshed is not None
    assert refreshed["signoff_status"] == "signed"
    assert refreshed["final_recommendation"] == "signed_ready_for_next_stage_planning"
    assert refreshed["stage3_signoff_present"] is True
    assert refreshed["stage3_signoff_sha256"] == _sha256(out)
    # The aggregate file on disk reflects the new state too.
    disk = json.loads(plan_json.read_text(encoding="utf-8"))
    assert disk["signoff_status"] == "signed"
    assert disk["stage3_signoff_sha256"] == _sha256(out)


def test_signoff_record_contains_no_mutable_run_paths(tmp_path: Path) -> None:
    """The sign-off JSON must reference only immutable lane summary
    JSONs, not per-cell payloads / fold metrics / fitted models /
    catboost_info / execution SQLite files / OpenML cache paths."""
    from scripts.sign_stage0_replica_001 import sign_stage0_replica_001

    std, hvy, ext = _write_canonical_triple(tmp_path)
    plan_json, _, _ = _build_plan(tmp_path, std, hvy, ext)
    out = tmp_path / "stage3_signoff.json"
    sign_stage0_replica_001(
        aggregate_plan_path=plan_json,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        plan_path=tmp_path / "absent_plan.json",
        out_signoff=out,
        refresh_aggregate=False,
    )
    text = out.read_text(encoding="utf-8")
    for forbidden in (
        ".execution.sqlite",
        "fold_metrics.json", "fold_metrics.csv",
        "catboost_info", "/payload.pkl",
        "data/source/openml_cc18/",
        "runs/cc18/",
    ):
        assert forbidden not in text, (
            f"sign-off record leaks forbidden token: {forbidden}"
        )


def test_signoff_file_still_absent_on_disk() -> None:
    """The actual repo-committed sign-off file must not exist while
    the test suite runs in CI (Commit 45 creates it; until then, the
    file is absent). This guards against accidental commits during
    iteration of Commit 45."""
    on_disk = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
    if on_disk.exists():
        # Once Commit 45 is committed, the file exists; it must
        # carry the operator fields, the pinned policy_version, and
        # downstream_execution_authorized=false.
        record = json.loads(on_disk.read_text(encoding="utf-8"))
        assert record["signoff_status"] == "signed"
        assert record["policy_version"] == PINNED_POLICY_VERSION
        assert (
            record["downstream_execution_authorized_in_this_commit"] is False
        )
        caveat_ids = {c["id"] for c in record["caveats_acknowledged"]}
        assert "isolet_future_recalibration_candidate" in caveat_ids
        assert "devnagari_extreme_budget_non_equivalence" in caveat_ids
