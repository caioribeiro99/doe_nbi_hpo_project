"""Tests for the executable canary adapters (Commit 30).

Covers:
- the four canary adapters declare run_status='smoke_ready';
- adapter.run() executes end-to-end on a synthetic binary task;
- tpe_optuna raises a clear ImportError when optuna is missing;
- runner --canary-only --train --synthetic-task succeeds on a temp
  shard with at least one canary cell;
- runner refuses to run a non-canary method even with --train;
- runner refuses --train without --canary-only;
- stage-3 sign-off guardrail still applies under --canary-only;
- committed shards are not mutated by tests.

External GBDT libraries are required only by the algorithms they
support. The canary uses xgboost as the package present in the
project's default dev environment; tests that need lightgbm /
catboost / optuna are decorated with importorskip.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from doe_xgb.methods import get_adapter
from doe_xgb.methods.canary import (
    MethodRunContext,
    make_synthetic_binary_task,
)

REPO = Path(__file__).resolve().parents[2]
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards"
RUNNER_SCRIPT = REPO / "scripts/cc18_runner.py"
MATRIX_CSV = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"

CANARY_METHODS = ("default_gbdt", "random_search",
                  "tpe_optuna", "doe_rsm_vrf_true_nbi")


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


HAS_XGBOOST = _have("xgboost")
HAS_LIGHTGBM = _have("lightgbm")
HAS_CATBOOST = _have("catboost")
HAS_OPTUNA = _have("optuna")


# ---------------------------------------------------------------------------
# run_status
# ---------------------------------------------------------------------------


def test_canary_adapters_are_smoke_ready() -> None:
    for mid in CANARY_METHODS:
        assert get_adapter(mid).run_status == "smoke_ready", mid


def test_non_canary_smoke_ready_count_is_zero() -> None:
    """Only the four canary adapters should advertise smoke_ready in
    Commit 30. Anything else is either dispatch_only or stub_only."""
    from doe_xgb.methods import ALL_METHOD_IDS
    smoke = [mid for mid in ALL_METHOD_IDS
             if get_adapter(mid).run_status == "smoke_ready"]
    assert sorted(smoke) == sorted(CANARY_METHODS)


# ---------------------------------------------------------------------------
# In-process adapter.run() smokes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_default_gbdt_runs_on_synthetic_binary() -> None:
    task = make_synthetic_binary_task(n_samples=120, n_features=4, seed=0)
    ctx = MethodRunContext(method_id="default_gbdt", algorithm="xgboost",
                            replica=1, seed=42, n_folds=2,
                            max_evaluations=1)
    res = get_adapter("default_gbdt").run(task=task, ctx=ctx)
    assert res.method_id == "default_gbdt"
    assert res.n_configurations_tried == 1
    assert "accuracy" in res.aggregate_metrics
    assert 0.0 <= res.aggregate_metrics["accuracy"] <= 1.0
    assert len(res.fold_metrics) == 2


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_random_search_runs_and_records_all_configs() -> None:
    task = make_synthetic_binary_task(n_samples=120, n_features=4, seed=0)
    ctx = MethodRunContext(method_id="random_search", algorithm="xgboost",
                            replica=1, seed=42, n_folds=2,
                            max_evaluations=4)
    res = get_adapter("random_search").run(task=task, ctx=ctx)
    assert res.n_configurations_tried == 4
    assert res.best_config is not None
    assert "all_configs" in res.extra
    assert len(res.extra["all_configs"]) == 4


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_doe_rsm_vrf_true_nbi_runs_on_synthetic_binary() -> None:
    task = make_synthetic_binary_task(n_samples=160, n_features=4, seed=0)
    ctx = MethodRunContext(
        method_id="doe_rsm_vrf_true_nbi", algorithm="xgboost",
        replica=1, seed=42, n_folds=2, max_evaluations=8,
    )
    res = get_adapter("doe_rsm_vrf_true_nbi").run(task=task, ctx=ctx)
    assert res.n_configurations_tried >= 8
    assert "post_optimization_diagnostics" in res.extra
    assert "nbi_residual_max" in res.extra
    assert isinstance(res.extra["mbpa_fired"], bool)


def test_tpe_optuna_raises_clear_import_error_when_optuna_missing() -> None:
    if HAS_OPTUNA:
        pytest.skip("optuna is installed; the missing-package path "
                    "cannot be exercised on this environment")
    task = make_synthetic_binary_task(n_samples=80, n_features=3, seed=0)
    ctx = MethodRunContext(method_id="tpe_optuna", algorithm="xgboost",
                            replica=1, seed=42, n_folds=2,
                            max_evaluations=2)
    with pytest.raises(ImportError, match="optuna is required"):
        get_adapter("tpe_optuna").run(task=task, ctx=ctx)


@pytest.mark.skipif(not (HAS_OPTUNA and HAS_XGBOOST),
                    reason="optuna or xgboost missing")
def test_tpe_optuna_runs_when_optuna_present() -> None:
    task = make_synthetic_binary_task(n_samples=120, n_features=4, seed=0)
    ctx = MethodRunContext(method_id="tpe_optuna", algorithm="xgboost",
                            replica=1, seed=42, n_folds=2,
                            max_evaluations=3)
    res = get_adapter("tpe_optuna").run(task=task, ctx=ctx)
    assert res.n_configurations_tried == 3
    assert "optuna_best_value" in res.extra


# ---------------------------------------------------------------------------
# Runner --canary-only --train --synthetic-task on a temp shard
# ---------------------------------------------------------------------------


@pytest.fixture
def canary_shard(tmp_path: Path) -> Path:
    """Build a tiny shard containing exactly one xgboost cell per canary
    method, by copying a real shard and pruning rows. Skips this fixture
    if xgboost isn't installed."""
    src = SHARDS_DIR / "stage0_replica_001" / "shard_00.sqlite"
    dst = tmp_path / "shard.sqlite"
    shutil.copy(src, dst)
    cx = sqlite3.connect(dst)
    cx.execute(
        "DELETE FROM cc18_jobs WHERE method NOT IN "
        "('default_gbdt','random_search','tpe_optuna','doe_rsm_vrf_true_nbi') "
        "OR algorithm != 'xgboost'"
    )
    # Keep only one row per (method, algorithm) for speed.
    rows = cx.execute(
        "SELECT method, algorithm, MIN(job_id) FROM cc18_jobs "
        "GROUP BY method, algorithm"
    ).fetchall()
    keep = {jid for _, _, jid in rows}
    cx.executemany(
        "DELETE FROM cc18_jobs WHERE job_id NOT IN "
        "(SELECT job_id FROM cc18_jobs WHERE job_id = ?)",
        [(j,) for j in keep],
    )
    cx.execute(
        "DELETE FROM cc18_jobs WHERE job_id NOT IN ("
        + ",".join("?" for _ in keep) + ")",
        list(keep),
    )
    cx.commit()
    cx.close()
    return dst


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_runner_canary_train_succeeds_on_synthetic_task(canary_shard: Path,
                                                        tmp_path: Path) -> None:
    out = tmp_path / "out"
    expected_methods = {"default_gbdt", "random_search",
                        "doe_rsm_vrf_true_nbi"}
    if HAS_OPTUNA:
        expected_methods.add("tpe_optuna")
    cx = sqlite3.connect(canary_shard)
    methods_in_shard = {
        m for (m,) in cx.execute("SELECT DISTINCT method FROM cc18_jobs")
    }
    cx.close()

    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(canary_shard),
         "--max-jobs", "8",
         "--canary-only", "--train", "--synthetic-task",
         "--max-evaluations", "3", "--n-folds", "2",
         "--output-root", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)

    # At minimum the methods present in the shard for which their
    # required package is installed should succeed.
    cx = sqlite3.connect(canary_shard)
    by_method = dict(cx.execute(
        "SELECT method, status FROM cc18_jobs"
    ).fetchall())
    cx.close()
    successful = [m for m, s in by_method.items() if s == "success"]
    expected_runnable = methods_in_shard & expected_methods
    assert set(successful) == expected_runnable, (successful, expected_runnable)

    # Manifests + fold metrics exist for every successful run.
    manifests = list(out.rglob("manifest.json"))
    fold_files = list(out.rglob("fold_metrics.json"))
    assert len(manifests) == len(successful)
    assert len(fold_files) == len(successful)
    for m in manifests:
        payload = json.loads(m.read_text())
        assert "aggregate_metrics" in payload
        assert "method_id" in payload


# ---------------------------------------------------------------------------
# Runner safety guardrails
# ---------------------------------------------------------------------------


def test_runner_train_without_canary_only_is_refused(tmp_path: Path) -> None:
    src = SHARDS_DIR / "stage0_replica_001" / "shard_00.sqlite"
    dst = tmp_path / "shard.sqlite"
    shutil.copy(src, dst)
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(dst), "--max-jobs", "1", "--train"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode != 0
    assert "canary-only" in res.stderr.lower() or "canary-only" in res.stdout.lower()


def test_runner_canary_only_refuses_non_canary_methods(tmp_path: Path) -> None:
    """Build a shard containing only a non-canary method (smac3) and
    verify that --canary-only --train refuses to claim or train it."""
    src = SHARDS_DIR / "stage0_replica_001" / "shard_00.sqlite"
    dst = tmp_path / "shard.sqlite"
    shutil.copy(src, dst)
    cx = sqlite3.connect(dst)
    cx.execute("DELETE FROM cc18_jobs WHERE method != 'smac3'")
    cx.commit()
    n = cx.execute("SELECT count(*) FROM cc18_jobs").fetchone()[0]
    cx.close()
    if n == 0:
        pytest.skip("shard has no smac3 rows")

    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(dst), "--max-jobs", "5",
         "--canary-only", "--train", "--synthetic-task"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "refused_not_in_canary_set" in res.stdout
    # Status must remain pending; no row promoted to claimed/running.
    cx = sqlite3.connect(dst)
    distinct = {s for (s,) in cx.execute(
        "SELECT DISTINCT status FROM cc18_jobs"
    )}
    cx.close()
    assert distinct == {"pending"}


def test_stage3_signoff_guardrail_active_under_canary_train(tmp_path: Path) -> None:
    """Every stage-3 row carrying the sign-off note must stay pending
    when the sign-off file is missing, even with --canary-only --train.
    Tier-0 controls (default_gbdt / random_search) don't carry the note
    and are allowed to flow through; this test isolates the guardrail
    rows."""
    src = SHARDS_DIR / "stage3_topup_to_030" / "shard_00.sqlite"
    dst = tmp_path / "shard.sqlite"
    shutil.copy(src, dst)
    fake_signoff = tmp_path / "no_signoff.json"
    cx = sqlite3.connect(dst)
    # Snapshot the job_ids of rows that carry the sign-off marker.
    signoff_jobs = [
        jid for (jid,) in cx.execute(
            "SELECT job_id FROM cc18_jobs "
            "WHERE notes LIKE '%requires_manual_signoff_before_stage3%'"
        )
    ]
    cx.close()
    assert signoff_jobs, "expected sign-off-marked jobs in stage 3"

    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(dst), "--max-jobs", "20",
         "--canary-only", "--train", "--synthetic-task",
         "--signoff-file", str(fake_signoff)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "refused_stage3_signoff_missing" in res.stdout
    cx = sqlite3.connect(dst)
    placeholders = ",".join("?" * len(signoff_jobs))
    inspected = cx.execute(
        f"SELECT job_id, status FROM cc18_jobs WHERE job_id IN ({placeholders})",
        signoff_jobs,
    ).fetchall()
    cx.close()
    # Every sign-off-marked row must still be pending.
    for jid, status in inspected:
        assert status == "pending", (jid, status)


def test_committed_shard_unchanged_by_dry_run() -> None:
    """Dry-run on a committed shard must not modify the file at all."""
    import hashlib
    shard = SHARDS_DIR / "stage0_replica_001" / "shard_00.sqlite"
    before = hashlib.md5(shard.read_bytes()).hexdigest()
    res = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT),
         "--shard", str(shard), "--max-jobs", "5", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    after = hashlib.md5(shard.read_bytes()).hexdigest()
    assert before == after


# ---------------------------------------------------------------------------
# Capability audit reflects four smoke_ready adapters
# ---------------------------------------------------------------------------


def test_capability_audit_lists_four_smoke_ready(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts/audit_method_capabilities.py"),
         "--out-dir", str(out_dir), "--quiet"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    payload = json.loads((out_dir / "cc18_capability_report.json").read_text())
    assert sorted(payload["smoke_ready"]) == sorted(CANARY_METHODS)


def test_method_matrix_canary_methods_are_present() -> None:
    with MATRIX_CSV.open() as f:
        ids = {r["method_id"] for r in csv.DictReader(f)}
    for mid in CANARY_METHODS:
        assert mid in ids
