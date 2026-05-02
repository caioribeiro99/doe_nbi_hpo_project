"""Tests for the CC18 method adapters, capability audit, and runner
skeleton (Commit 29).

Covers:
- every non-literature method_id from method_matrix.csv has an adapter;
- every adapter imports cleanly and exposes a CapabilityStatus;
- audit_method_capabilities.py writes JSON+MD reports;
- literature-only methods are skipped by the audit;
- missing optional packages do not crash the audit;
- runner --dry-run does not mutate the shard;
- runner non-dry-run skeleton mode reverts every job back to pending;
- stage-3 sign-off guardrail refuses jobs without the sign-off file;
- adapter.run() is never invoked end-to-end.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from doe_xgb.methods import (
    ADAPTERS,
    ALL_METHOD_IDS,
    AdapterBase,
    CapabilityStatus,
    get_adapter,
)
from doe_xgb.methods.registry import UnknownMethodError

REPO = Path(__file__).resolve().parents[2]
MATRIX_CSV = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards"
AUDIT_SCRIPT = REPO / "scripts/audit_method_capabilities.py"
RUNNER_SCRIPT = REPO / "scripts/cc18_runner.py"


# ---------------------------------------------------------------------------
# Adapter completeness
# ---------------------------------------------------------------------------


def test_adapter_exists_for_every_non_literature_method() -> None:
    with MATRIX_CSV.open() as f:
        non_lit = [
            r["method_id"]
            for r in csv.DictReader(f)
            if r["primary_or_ablation"] != "literature_only"
        ]
    missing = [m for m in non_lit if m not in ADAPTERS]
    assert not missing, f"adapters missing: {missing}"
    extra = [m for m in ADAPTERS if m not in non_lit]
    assert not extra, f"adapters with no matrix row: {extra}"


def test_every_adapter_can_be_imported_and_instantiated() -> None:
    for mid in ALL_METHOD_IDS:
        a = get_adapter(mid)
        assert isinstance(a, AdapterBase)
        assert a.method_id == mid


def test_get_adapter_unknown_method_raises() -> None:
    with pytest.raises(UnknownMethodError):
        get_adapter("flaml_optional")  # literature-only by design
    with pytest.raises(UnknownMethodError):
        get_adapter("not_a_real_method")


def test_every_adapter_import_check_returns_capability_status() -> None:
    for mid in ALL_METHOD_IDS:
        status = get_adapter(mid).import_check()
        assert isinstance(status, CapabilityStatus)
        assert status.method_id == mid
        assert status.adapter_import_ok is True
        assert status.run_status in {
            "stub_only", "dispatch_only", "smoke_ready", "full_ready"
        }


def test_stub_or_dispatch_only_adapters_run_is_blocked() -> None:
    """Adapters that have not yet been promoted to smoke_ready must keep
    their run() guarded behind NotImplementedError. Smoke-ready adapters
    (Commit 30 promoted default_gbdt, random_search, tpe_optuna,
    doe_rsm_vrf_true_nbi) have a real implementation and are exercised
    by tests/unit/test_cc18_canary_adapters.py."""
    for mid in ALL_METHOD_IDS:
        adapter = get_adapter(mid)
        if adapter.run_status in ("smoke_ready", "full_ready"):
            continue
        with pytest.raises(NotImplementedError):
            adapter.run()


def test_adapter_supports_filters_by_task_type_and_algorithm() -> None:
    a = get_adapter("doe_rsm_vrf_true_nbi")
    assert a.supports({"task_type": "binary"}, "xgboost") is True
    assert a.supports({"task_type": "multiclass"}, "lightgbm") is True
    assert a.supports({"task_type": "binary"}, "not_a_real_alg") is False


# ---------------------------------------------------------------------------
# Capability audit script
# ---------------------------------------------------------------------------


def test_audit_script_runs_and_writes_reports(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    res = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT),
         "--out-dir", str(out_dir), "--quiet"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    json_p = out_dir / "cc18_capability_report.json"
    md_p = out_dir / "cc18_capability_report.md"
    assert json_p.exists() and md_p.exists()
    payload = json.loads(json_p.read_text())
    assert payload["n_benchmarked"] == len(ALL_METHOD_IDS)
    assert payload["n_literature_only"] == 3
    # Literature-only ids must not appear in adapters_found.
    for lit in ("flaml_optional", "auto_sklearn_context", "autogluon_context"):
        assert lit not in payload["adapters_found"]
        assert lit in payload["literature_only"]


def test_audit_classifies_run_status_into_known_buckets() -> None:
    from scripts.audit_method_capabilities import audit
    report = audit(MATRIX_CSV)
    buckets = (
        report["stub_only"] + report["dispatch_only"]
        + report["smoke_ready"] + report["full_ready"]
    )
    assert sorted(buckets) == sorted(report["adapters_found"])


def test_audit_does_not_crash_on_missing_optional_packages() -> None:
    from scripts.audit_method_capabilities import audit
    report = audit(MATRIX_CSV)
    # The audit MUST collect the missing-package list rather than raising.
    assert isinstance(report["missing_packages_overall"], list)


# ---------------------------------------------------------------------------
# Runner skeleton — guardrails
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_stage0_shard(tmp_path: Path) -> Path:
    src = SHARDS_DIR / "stage0_replica_001" / "shard_00.sqlite"
    dst = tmp_path / "shard_00.sqlite"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def temp_stage3_shard(tmp_path: Path) -> Path:
    src = SHARDS_DIR / "stage3_topup_to_030" / "shard_00.sqlite"
    dst = tmp_path / "shard_00.sqlite"
    shutil.copy(src, dst)
    return dst


def _row_state_snapshot(shard: Path) -> dict[str, tuple[str, str | None]]:
    cx = sqlite3.connect(shard)
    out = {
        jid: (status, worker)
        for jid, status, worker in cx.execute(
            "SELECT job_id, status, assigned_worker FROM cc18_jobs"
        )
    }
    cx.close()
    return out


def test_runner_dry_run_does_not_mutate_shard(temp_stage0_shard: Path) -> None:
    before = _row_state_snapshot(temp_stage0_shard)
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(temp_stage0_shard),
         "--max-jobs", "5", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    after = _row_state_snapshot(temp_stage0_shard)
    assert before == after


def test_runner_no_train_reverts_status_to_pending(temp_stage0_shard: Path) -> None:
    before = _row_state_snapshot(temp_stage0_shard)
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(temp_stage0_shard),
         "--max-jobs", "5", "--no-train"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    after = _row_state_snapshot(temp_stage0_shard)
    # Logical state of every (status, assigned_worker) tuple must match.
    assert before == after
    # Every row must still be pending with no assigned worker.
    for status, worker in after.values():
        assert status == "pending"
        assert worker is None


def test_runner_refuses_stage3_without_signoff_file(temp_stage3_shard: Path,
                                                    tmp_path: Path) -> None:
    """Without the sign-off file, every stage-3 row carrying the note must
    be marked refused, not claimed."""
    # Use a non-existent sign-off path explicitly.
    fake_signoff = tmp_path / "no_signoff_here.json"
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(temp_stage3_shard),
         "--max-jobs", "10", "--no-train",
         "--signoff-file", str(fake_signoff)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "refused_stage3_signoff_missing" in res.stdout
    # Status must still be pending; no row promoted.
    after = _row_state_snapshot(temp_stage3_shard)
    for status, _ in after.values():
        assert status == "pending"


def test_runner_with_signoff_file_does_not_refuse_stage3(temp_stage3_shard: Path,
                                                         tmp_path: Path) -> None:
    signoff = tmp_path / "stage3_signoff.json"
    signoff.write_text(json.dumps({
        "approved_by": "test", "approved_at": "2026-05-02T00:00:00Z"
    }))
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(temp_stage3_shard),
         "--max-jobs", "5", "--no-train",
         "--signoff-file", str(signoff)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "refused_stage3_signoff_missing" not in res.stdout


def test_runner_help_runs() -> None:
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "shard" in res.stdout.lower()


def test_runner_dispatch_decision_does_not_invoke_run() -> None:
    """Spot-check the in-process API: dispatch_decision() never calls
    adapter.run(). Use a stub_only method so the decision is
    deterministic across Commit 29 (all adapters stub) and Commit 30
    (four adapters smoke_ready)."""
    from scripts.cc18_runner import dispatch_decision

    class FakeRow(dict):
        def __getitem__(self, k):
            return super().__getitem__(k)

    row = FakeRow(
        job_id="x", method="bohb", algorithm="xgboost",
        openml_task_id=3, stage="stage0_replica_001", notes=None,
    )
    decision = dispatch_decision(row, signoff_ok=True, train=False)
    # bohb is stub_only. With train=False the decision is stub_only.
    # Crucially, dispatch_decision never invoked NotImplementedError.
    assert decision["decision"] == "stub_only"
    assert decision["would_run"] is False
    # And with train=True under canary_only it is refused as not in
    # the canary set (Commit 30 guardrail) — still no run() invocation.
    decision_canary = dispatch_decision(row, signoff_ok=True, train=True,
                                         canary_only=True)
    assert decision_canary["decision"] == "refused_not_in_canary_set"
    assert decision_canary["would_run"] is False
