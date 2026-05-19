"""Tests for the stage-3 / top-up planner (Commit 46).

``scripts/plan_stage3_topup.py`` is a planning script. It reads
``jobs/doctoral/openml_cc18/stage3_signoff.json`` and the three
stage-0 lane summaries, computes the three top-up tiers
(``topup_to_5`` / ``topup_to_10`` / ``topup_to_30``), and emits a
JSON / MD pair under ``experiments/_stage_runs/``. It must NOT:

- run training;
- create execution SQLite under ``runs/``;
- mutate committed SQLite shards;
- create or modify ``stage3_signoff.json``;
- change ``policy_version``.

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
RUN_SCRIPT = REPO / "scripts/plan_stage3_topup.py"
STANDARD_PATH = (
    REPO / "experiments/_stage_runs/stage0_standard_lane_latest_summary.json"
)
HEAVY_PATH = REPO / "experiments/_stage_runs/stage0_heavy_lane_latest_summary.json"
EXTREME_PATH = REPO / "experiments/_stage_runs/stage0_extreme_lane_latest_summary.json"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
HEAVY_POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards"
SHARD_SUMMARY = REPO / "jobs/doctoral/openml_cc18/shards/shard_summary.json"
RUNS_ROOT = REPO / "runs/cc18"
DATA_SOURCE = REPO / "data/source/openml_cc18"
EXECUTION_PLAN_DOC = REPO / "docs/STAGE3_TOPUP_EXECUTION_PLAN.md"
POLICY_DECISION_DOC = REPO / "docs/STAGE3_POLICY_DECISION.md"
RUNBOOK_DOC = REPO / "docs/STAGE3_DISTRIBUTED_RUNBOOK.md"
MANIFEST_CSV = REPO / "benchmarks/doctoral/openml_cc18/stage3_topup_manifest.csv"
MANIFEST_MD = REPO / "benchmarks/doctoral/openml_cc18/stage3_topup_manifest.md"
WORKER_CSV = REPO / "benchmarks/doctoral/openml_cc18/stage3_worker_plan.csv"
WORKER_MD = REPO / "benchmarks/doctoral/openml_cc18/stage3_worker_plan.md"

PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _committed_shard_md5s() -> dict[str, str]:
    out: dict[str, str] = {}
    if not SHARDS_DIR.exists():
        return out
    for sub in sorted(SHARDS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for shard in sorted(sub.glob("shard_*.sqlite")):
            out[f"{sub.name}/{shard.name}"] = _md5(shard)
    return out


def _write_lane_summary(
    path: Path, *, lane_name: str, n_executed: int,
    policy_version: str = PINNED_POLICY_VERSION,
    cells: list[dict] | None = None,
    runtime_seconds_runner_total: float = 0.0,
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
        "source_shards_unchanged": True,
        "stage3_signoff_present": True,
        "policy_version": policy_version,
        "cells": cells or [],
        "runtime_seconds_runner_total": runtime_seconds_runner_total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_canonical_triple(tmp_path: Path) -> tuple[Path, Path, Path]:
    std = _write_lane_summary(
        tmp_path / "stage0_standard_lane_latest_summary.json",
        lane_name="standard", n_executed=684,
        runtime_seconds_runner_total=6801.3,
    )
    hvy = _write_lane_summary(
        tmp_path / "stage0_heavy_lane_latest_summary.json",
        lane_name="heavy", n_executed=156,
        runtime_seconds_runner_total=34889.3,
    )
    ext = _write_lane_summary(
        tmp_path / "stage0_extreme_lane_latest_summary.json",
        lane_name="extreme", n_executed=24,
        runtime_seconds_runner_total=30844.5,
    )
    return std, hvy, ext


def _write_signoff(
    path: Path, *,
    std: Path, hvy: Path, ext: Path,
    signoff_status: str = "signed",
    signoff_type: str = "stage0_replica_001",
    policy_version: str = PINNED_POLICY_VERSION,
    downstream_execution_authorized_in_this_commit: bool = False,
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
        "downstream_execution_authorized_in_this_commit":
            downstream_execution_authorized_in_this_commit,
        "standard_lane_summary_sha256": _sha256(std),
        "heavy_lane_summary_sha256": _sha256(hvy),
        "extreme_lane_summary_sha256": _sha256(ext),
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
    assert "plan_stage3_topup.py" in out
    assert "--signoff-file" in out
    assert "--standard-summary" in out
    assert "--heavy-summary" in out
    assert "--extreme-summary" in out
    assert "--heavy-policy-csv" in out
    assert "--allow-policy-drift-report-only" in out
    assert "--dry-run" in out


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_refuses_when_signoff_missing(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    with pytest.raises(TopupPlanRefusalError, match="signoff file not found"):
        build_topup_plan(
            signoff_path=tmp_path / "absent_signoff.json",
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_signoff_not_signed(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
        signoff_status="planned_not_signed",
    )
    with pytest.raises(TopupPlanRefusalError, match="planned_not_signed"):
        build_topup_plan(
            signoff_path=signoff,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_signoff_type_wrong(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
        signoff_type="some_other_signoff",
    )
    with pytest.raises(TopupPlanRefusalError, match="signoff_type"):
        build_topup_plan(
            signoff_path=signoff,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_when_signoff_file_is_invalid_json(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = tmp_path / "stage3_signoff.json"
    signoff.write_text("not json!", encoding="utf-8")
    with pytest.raises(TopupPlanRefusalError, match="not valid JSON"):
        build_topup_plan(
            signoff_path=signoff,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


def test_refuses_on_policy_drift_without_report_only(tmp_path: Path) -> None:
    """Live heavy_task_policy.csv hashes differently from the
    signed policy_version → refuse unless
    --allow-policy-drift-report-only."""
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
        policy_version="0" * 64,  # deliberately wrong
    )
    with pytest.raises(TopupPlanRefusalError, match="policy_version"):
        build_topup_plan(
            signoff_path=signoff,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
            allow_policy_drift_report_only=False,
        )


def test_allows_policy_drift_report_only(tmp_path: Path) -> None:
    """With ``--allow-policy-drift-report-only`` the planner emits a
    summary with ``drift_report_only = True`` and
    ``policy_drift_detected = True`` instead of refusing."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
        policy_version="0" * 64,
    )
    summary = build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
        write_summary=False,
        allow_policy_drift_report_only=True,
    )
    assert summary["drift_report_only"] is True
    assert summary["policy_drift_detected"] is True
    assert summary["policy_drift_message"]
    assert summary["execution_status"] == "planned_not_executed"


def test_refuses_when_lane_summary_hashes_drift(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import (
        TopupPlanRefusalError,
        build_topup_plan,
    )

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
    )
    # Bump heavy lane summary so its hash no longer matches signoff.
    hvy.write_text(hvy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TopupPlanRefusalError, match="lane summary SHA-256 drift"):
        build_topup_plan(
            signoff_path=signoff,
            standard_path=std, heavy_path=hvy, extreme_path=ext,
            heavy_policy_path=HEAVY_POLICY_CSV,
            guardrails_path=GUARDRAILS_YAML,
            shard_summary_path=SHARD_SUMMARY,
            shards_dir=SHARDS_DIR,
            out_json=tmp_path / "x.json", out_md=tmp_path / "x.md",
            write_summary=False,
        )


# ---------------------------------------------------------------------------
# Successful planning paths
# ---------------------------------------------------------------------------


def test_planner_emits_canonical_tier_counts(tmp_path: Path) -> None:
    """Feed the planner a synthetic clean signoff + lane summaries
    and verify the three tier cell counts are exactly
    864 × {4, 5, 20} = {3,456, 4,320, 17,280}."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json",
        std=std, hvy=hvy, ext=ext,
    )
    summary = build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json",
        out_md=tmp_path / "plan.md",
    )
    assert summary["execution_status"] == "planned_not_executed"
    assert summary["signoff_status"] == "signed"
    assert summary["policy_drift_detected"] is False
    assert summary["n_canary_cells_per_replica"] == 864
    counts = {t["tier"]: t for t in summary["tier_plans"]}
    assert counts["topup_to_5"]["executable_canary_cells_total"] == 3456
    assert counts["topup_to_5"]["replica_count"] == 4
    assert counts["topup_to_5"]["replica_start"] == 2
    assert counts["topup_to_5"]["replica_end"] == 5
    assert counts["topup_to_10"]["executable_canary_cells_total"] == 4320
    assert counts["topup_to_10"]["replica_count"] == 5
    assert counts["topup_to_30"]["executable_canary_cells_total"] == 17280
    assert counts["topup_to_30"]["replica_count"] == 20
    # Lane breakdown sanity: 684 + 156 + 24 = 864 per replica.
    for plan in summary["tier_plans"]:
        lane_sum = sum(L["executable_canary_cells_total"] for L in plan["lanes"])
        assert lane_sum == plan["executable_canary_cells_total"]
    assert summary["executable_canary_cells_total_all_tiers"] == 25056


def test_planner_writes_json_and_md_with_required_keys(tmp_path: Path) -> None:
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json", std=std, hvy=hvy, ext=ext,
    )
    out_json = tmp_path / "plan.json"
    out_md = tmp_path / "plan.md"
    build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=out_json, out_md=out_md,
    )
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    for key in (
        "schema_version", "run_id", "stage", "exported_at",
        "execution_status", "drift_report_only",
        "policy_drift_detected", "policy_drift_message",
        "signoff_path", "signoff_sha256", "signoff_status",
        "signoff_type", "signoff_signed_at_utc",
        "signoff_operator_name", "signoff_operator_handle",
        "signoff_branch", "signoff_git_sha",
        "signoff_downstream_execution_authorized_in_this_commit",
        "signoff_caveats_acknowledged",
        "policy_version", "signed_policy_version",
        "heavy_task_policy_csv_path", "heavy_task_policy_csv_sha256",
        "runtime_guardrails_yaml_path",
        "lane_summary_paths", "lane_summary_sha256_live",
        "lane_summary_sha256_signed",
        "per_replica_runtime_seconds_by_lane_observed",
        "runtime_distribution_by_lane_observed",
        "n_canary_cells_per_replica",
        "n_canary_cells_per_lane_per_replica",
        "tier_plans",
        "executable_canary_cells_total_all_tiers",
        "estimated_runtime_seconds_total_all_tiers_p50",
        "estimated_runtime_seconds_total_all_tiers_max",
        "high_risk_cells", "high_risk_threshold_seconds",
        "committed_shard_md5_snapshot", "n_committed_shards",
        "git_sha", "platform",
        "no_training_run_by_this_script",
        "no_execution_sqlite_created_by_this_script",
        "no_committed_shard_modified_by_this_script",
        "decision_options_doc", "execution_plan_doc",
        "distributed_runbook_doc",
    ):
        assert key in payload, f"missing summary key: {key}"
    assert payload["execution_status"] == "planned_not_executed"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_planner_does_not_create_execution_sqlite_or_runs(tmp_path: Path) -> None:
    """The planner is a read-only artifact producer. It must not
    write anything outside out_json / out_md (and certainly never
    create runs/ or execution SQLite files)."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json", std=std, hvy=hvy, ext=ext,
    )

    runs_before = set(p.name for p in RUNS_ROOT.glob("*")) if RUNS_ROOT.exists() else set()
    data_before = (
        set(p.name for p in DATA_SOURCE.glob("*"))
        if DATA_SOURCE.exists() else set()
    )
    sqlite_before = (
        set(str(p) for p in REPO.glob("**/*.execution.sqlite"))
    )

    build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )

    runs_after = set(p.name for p in RUNS_ROOT.glob("*")) if RUNS_ROOT.exists() else set()
    data_after = (
        set(p.name for p in DATA_SOURCE.glob("*"))
        if DATA_SOURCE.exists() else set()
    )
    sqlite_after = (
        set(str(p) for p in REPO.glob("**/*.execution.sqlite"))
    )

    assert runs_before == runs_after
    assert data_before == data_after
    assert sqlite_before == sqlite_after


def test_planner_does_not_modify_committed_shards(tmp_path: Path) -> None:
    """Snapshot the committed shard MD5s before/after a planner run
    and assert they are byte-identical."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json", std=std, hvy=hvy, ext=ext,
    )
    before = _committed_shard_md5s()
    build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )
    after = _committed_shard_md5s()
    assert before == after
    assert before  # sanity: shards exist


def test_planner_does_not_create_or_modify_signoff_file(tmp_path: Path) -> None:
    """The planner must never create / modify ``stage3_signoff.json``."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json", std=std, hvy=hvy, ext=ext,
    )
    signoff_md5_before = _md5(signoff)
    build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )
    assert _md5(signoff) == signoff_md5_before


def test_planner_does_not_regenerate_heavy_policy(tmp_path: Path) -> None:
    """The planner must never rewrite ``heavy_task_policy.csv`` or
    ``runtime_guardrails.yaml``."""
    from scripts.plan_stage3_topup import build_topup_plan

    std, hvy, ext = _write_canonical_triple(tmp_path)
    signoff = _write_signoff(
        tmp_path / "stage3_signoff.json", std=std, hvy=hvy, ext=ext,
    )
    policy_md5_before = _md5(HEAVY_POLICY_CSV)
    guardrails_md5_before = _md5(GUARDRAILS_YAML)
    build_topup_plan(
        signoff_path=signoff,
        standard_path=std, heavy_path=hvy, extreme_path=ext,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )
    assert _md5(HEAVY_POLICY_CSV) == policy_md5_before
    assert _md5(GUARDRAILS_YAML) == guardrails_md5_before


# ---------------------------------------------------------------------------
# Manifest + worker plan validation
# ---------------------------------------------------------------------------


def test_manifest_csv_validates() -> None:
    import csv
    assert MANIFEST_CSV.exists()
    with MANIFEST_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_cols = {
        "topup_tier", "replica_start", "replica_end", "replica_count",
        "lane", "source_stage", "task_count",
        "executable_canary_cell_count",
        "deferred_or_refused_count_estimate", "recommended_worker_type",
        "estimated_runtime_seconds_p50",
        "estimated_runtime_seconds_p90",
        "estimated_runtime_seconds_max",
        "can_run_on_personal_mac", "can_run_on_dedicated_mac",
        "requires_manual_review", "notes",
    }
    assert rows
    assert expected_cols.issubset(set(rows[0].keys()))
    by_tier_lane: dict[tuple[str, str], dict] = {
        (r["topup_tier"], r["lane"]): r for r in rows
    }
    for tier in ("topup_to_5", "topup_to_10", "topup_to_30"):
        for lane in ("standard", "heavy", "extreme"):
            assert (tier, lane) in by_tier_lane, (
                f"missing row for {tier} / {lane}"
            )
    # Cell counts must add up to 864 × replica_count.
    for r in rows:
        rc = int(r["replica_count"])
        cells = int(r["executable_canary_cell_count"])
        per_replica = {"standard": 684, "heavy": 156, "extreme": 24}[r["lane"]]
        assert cells == per_replica * rc, (
            f"{r['topup_tier']} / {r['lane']}: "
            f"{cells} != {per_replica} × {rc}"
        )
    assert MANIFEST_MD.exists()


def test_worker_plan_csv_validates() -> None:
    import csv
    assert WORKER_CSV.exists()
    with WORKER_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_cols = {
        "worker_id", "hardware", "role", "dedicated",
        "recommended_lanes", "recommended_tiers", "recommended_shards",
        "wall_clock_budget_hours_per_week", "prerequisites",
        "not_suitable_for", "notes",
    }
    assert rows
    assert expected_cols.issubset(set(rows[0].keys()))
    worker_ids = {r["worker_id"] for r in rows}
    assert "personal_mac_m4_pro" in worker_ids
    assert "dedicated_mac" in worker_ids
    # Cloud is documented but optional.
    assert "optional_cloud" in worker_ids
    assert WORKER_MD.exists()


# ---------------------------------------------------------------------------
# Doc invariants
# ---------------------------------------------------------------------------


def test_execution_plan_doc_mentions_caveats_and_tiers() -> None:
    text = EXECUTION_PLAN_DOC.read_text(encoding="utf-8")
    for tok in (
        "topup_to_5", "topup_to_10", "topup_to_30",
        "stage0_replica_001", "policy_version",
        "isolet", "Devnagari-Script",
        "standard", "heavy", "extreme",
        "864", "3,456", "4,320", "17,280",
    ):
        assert tok in text, f"execution plan missing: {tok}"


def test_policy_decision_doc_mentions_options_and_caveats() -> None:
    text = POLICY_DECISION_DOC.read_text(encoding="utf-8")
    for tok in (
        "Option A", "Option B", "Option C",
        "isolet", "Devnagari-Script",
        "policy_version", "47b6b50c",
        "stage0_max_evaluations",
        "isolet_future_recalibration_candidate",
        "devnagari_extreme_budget_non_equivalence",
    ):
        assert tok in text, f"policy decision doc missing: {tok}"


def test_runbook_doc_mentions_setup_and_caveats() -> None:
    text = RUNBOOK_DOC.read_text(encoding="utf-8")
    for tok in (
        "git fetch origin",
        "stage3_signoff.json",
        "caffeinate",
        "cc18_runner",
        "--canary-only",
        "failed_timeout", "failed_other",
        "isolet", "Devnagari-Script",
        "runs/", "catboost_info",
    ):
        assert tok in text, f"runbook doc missing: {tok}"


# ---------------------------------------------------------------------------
# Round-trip against real committed artifacts
# ---------------------------------------------------------------------------


def test_planner_round_trip_against_committed_artifacts(tmp_path: Path) -> None:
    """Run the planner against the real Commit-45 signoff and lane
    summaries. The output must report 25,056 executable canary cells
    across the three tiers and no drift."""
    if not SIGNOFF_FILE.exists():
        pytest.skip("stage3_signoff.json absent (pre-Commit-45 state)")
    from scripts.plan_stage3_topup import build_topup_plan

    summary = build_topup_plan(
        signoff_path=SIGNOFF_FILE,
        standard_path=STANDARD_PATH,
        heavy_path=HEAVY_PATH,
        extreme_path=EXTREME_PATH,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )
    assert summary["signoff_status"] == "signed"
    assert summary["signoff_type"] == "stage0_replica_001"
    assert summary["policy_version"] == PINNED_POLICY_VERSION
    assert summary["policy_drift_detected"] is False
    assert summary["executable_canary_cells_total_all_tiers"] == 25056
    # All three tiers reported.
    tier_names = {t["tier"] for t in summary["tier_plans"]}
    assert tier_names == {"topup_to_5", "topup_to_10", "topup_to_30"}
    # MD mentions the canonical concepts.
    md_text = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "isolet" in md_text
    assert "Devnagari-Script" in md_text
    assert "planned_not_executed" in md_text
    assert "topup_to_5" in md_text
    assert "topup_to_10" in md_text
    assert "topup_to_30" in md_text


def test_committed_shards_unchanged_after_real_round_trip(tmp_path: Path) -> None:
    """Run the planner against the real artifacts and snapshot the
    shard MD5s before/after — must be identical."""
    if not SIGNOFF_FILE.exists():
        pytest.skip("stage3_signoff.json absent (pre-Commit-45 state)")
    from scripts.plan_stage3_topup import build_topup_plan

    before = _committed_shard_md5s()
    build_topup_plan(
        signoff_path=SIGNOFF_FILE,
        standard_path=STANDARD_PATH,
        heavy_path=HEAVY_PATH,
        extreme_path=EXTREME_PATH,
        heavy_policy_path=HEAVY_POLICY_CSV,
        guardrails_path=GUARDRAILS_YAML,
        shard_summary_path=SHARD_SUMMARY,
        shards_dir=SHARDS_DIR,
        out_json=tmp_path / "plan.json", out_md=tmp_path / "plan.md",
    )
    after = _committed_shard_md5s()
    assert before == after
    assert len(before) == 40  # 4 stages × 10 shards
