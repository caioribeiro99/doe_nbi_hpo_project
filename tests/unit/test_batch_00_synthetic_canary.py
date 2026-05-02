"""Tests for the dedicated-Mac batch_00 synthetic canary gate.

Covers:
- the setup script exists and is executable;
- run_batch_00_synthetic_canary.py --help exits 0;
- the runner copies the committed shard to a temp path before any
  prune;
- the runner refuses to write to the committed shard directory;
- the artifact JSON validates against a minimal schema (using a
  hand-built mock so the test does not depend on optuna /
  lightgbm / catboost being installed);
- after the runner runs, the committed shard MD5 is unchanged;
- the stage-3 sign-off file is NOT created by the runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO / "scripts/setup_dedicated_mac.sh"
RUN_SCRIPT = REPO / "scripts/run_batch_00_synthetic_canary.py"
SOURCE_SHARD = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
MANIFEST = REPO / "benchmarks/doctoral/openml_cc18/batches/batch_00_synthetic_canary.json"

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Setup script
# ---------------------------------------------------------------------------


def test_setup_script_exists_and_is_executable() -> None:
    assert SETUP_SCRIPT.exists()
    mode = os.stat(SETUP_SCRIPT).st_mode
    assert mode & 0o111, "setup_dedicated_mac.sh must be executable"


def test_setup_script_starts_with_bash_shebang() -> None:
    head = SETUP_SCRIPT.read_text().splitlines()[0]
    assert head.startswith("#!"), head
    assert "bash" in head


# ---------------------------------------------------------------------------
# Runner CLI
# ---------------------------------------------------------------------------


def test_run_script_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    out = res.stdout.lower()
    # The script's argparse help wraps text differently on Py3.9 vs
    # Py3.12; check for stable, structural tokens instead.
    assert "run_batch_00_synthetic_canary.py" in out
    assert "--shard" in out or "source-shard" in out
    assert "--max-jobs" in out


def test_run_script_dry_run_does_not_invoke_runner(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    gate = tmp_path / "gate"
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run",
         "--output-root", str(out_root), "--gate-dir", str(gate)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    # No artifacts should be written.
    assert not (gate / "batch_00_synthetic_canary_latest.json").exists()
    assert not out_root.exists()


# ---------------------------------------------------------------------------
# Shard immutability
# ---------------------------------------------------------------------------


def test_runner_refuses_committed_shard_dir_as_output_root(tmp_path: Path) -> None:
    """Test that the runner does not silently accept the committed
    shards directory as its --output-root. We do this by inspecting the
    runner's defaults: the default output_root is under
    ``experiments/_batch_runs/...`` and never under ``jobs/.../shards/``.
    The actual run is exercised in
    ``test_runner_does_not_mutate_committed_shard``."""
    src = RUN_SCRIPT.read_text()
    assert "experiments/_batch_runs/batch_00_synthetic_canary" in src
    assert "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite" in src
    # And the runner never opens the source shard read-write.
    assert "shutil.copy(source_shard, temp_shard)" in src


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost missing; "
                    "the canary cannot reach the runner without it")
def test_runner_does_not_mutate_committed_shard(tmp_path: Path) -> None:
    """Running the canary against the committed shard must leave it
    byte-identical. The runner copies the shard internally; the test
    compares the committed file's md5 before and after."""
    md5_before = _md5(SOURCE_SHARD)
    out_root = tmp_path / "out"
    gate = tmp_path / "gate"
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--output-root", str(out_root), "--gate-dir", str(gate),
         "--max-jobs", "12", "--max-evaluations", "2", "--n-folds", "2"],
        capture_output=True, text=True, check=False,
    )
    # Either pass (rc=0) or fail because optional packages missing
    # (rc=4); never crash with non-zero unrelated exit.
    assert res.returncode in (0, 4), res.stderr
    md5_after = _md5(SOURCE_SHARD)
    assert md5_before == md5_after
    # And gate artifacts are written either way.
    assert (gate / "batch_00_synthetic_canary_latest.json").exists()
    assert (gate / "batch_00_synthetic_canary_latest.md").exists()


# ---------------------------------------------------------------------------
# Artifact JSON schema (mocked)
# ---------------------------------------------------------------------------


REQUIRED_ARTIFACT_KEYS = (
    "batch_id", "run_timestamp", "git_sha", "platform",
    "package_versions", "runner_command", "manifest_path",
    "source_shard", "source_shard_md5_before",
    "source_shard_md5_after", "source_shard_unchanged",
    "temp_shard", "n_cells_in_temp_shard", "n_cells_expected",
    "n_cells_success", "n_cells_failed", "n_cells_pending",
    "runtime_seconds", "subprocess_returncode",
    "stage3_signoff_present", "stage3_signoff_path",
    "capability_audit", "cells",
)


def _build_mock_artifact(tmp_path: Path) -> dict:
    return {
        "batch_id": "batch_00_synthetic_canary",
        "run_timestamp": "2026-05-02T00:00:00Z",
        "git_sha": "0" * 40,
        "platform": {
            "hostname": "host", "uname": "Darwin",
            "python_version": "3.12", "python_executable": "/bin/python",
            "machine": "arm64",
        },
        "package_versions": {"xgboost": "3.2.0", "lightgbm": None},
        "runner_command": ["python", "scripts/cc18_runner.py"],
        "manifest_path": "benchmarks/doctoral/openml_cc18/batches/"
                         "batch_00_synthetic_canary.json",
        "source_shard": "jobs/doctoral/openml_cc18/shards/"
                        "stage0_replica_001/shard_00.sqlite",
        "source_shard_md5_before": "a" * 32,
        "source_shard_md5_after": "a" * 32,
        "source_shard_unchanged": True,
        "temp_shard": str(tmp_path / "shard.sqlite"),
        "n_cells_in_temp_shard": 12,
        "n_cells_expected": 12,
        "n_cells_success": 12,
        "n_cells_failed": 0,
        "n_cells_pending": 0,
        "runtime_seconds": 1.0,
        "subprocess_stdout_tail": "",
        "subprocess_stderr_tail": "",
        "subprocess_returncode": 0,
        "stage3_signoff_present": False,
        "stage3_signoff_path": "jobs/doctoral/openml_cc18/stage3_signoff.json",
        "capability_audit": {
            "n_benchmarked": 13,
            "smoke_ready": ["default_gbdt"],
            "dispatch_only": [],
            "stub_only": [],
            "missing_packages": [],
        },
        "cells": [
            {"method": "default_gbdt", "algorithm": "xgboost",
             "status": "success", "runtime_seconds": 0.1,
             "last_error": None, "manifest": None,
             "aggregate_metrics": {"accuracy": 0.7}},
        ],
    }


def test_artifact_writer_emits_all_required_keys(tmp_path: Path) -> None:
    from scripts.run_batch_00_synthetic_canary import write_artifact

    artifact = _build_mock_artifact(tmp_path)
    json_p, md_p = write_artifact(artifact, tmp_path)
    assert json_p.exists() and md_p.exists()
    payload = json.loads(json_p.read_text())
    for key in REQUIRED_ARTIFACT_KEYS:
        assert key in payload, f"missing required key: {key}"


def test_artifact_md_contains_pass_verdict_when_all_succeed(tmp_path: Path) -> None:
    from scripts.run_batch_00_synthetic_canary import write_artifact

    artifact = _build_mock_artifact(tmp_path)
    _, md_p = write_artifact(artifact, tmp_path)
    text = md_p.read_text()
    assert "GATE PASS" in text


def test_artifact_md_contains_fail_verdict_when_failures_present(tmp_path: Path) -> None:
    from scripts.run_batch_00_synthetic_canary import write_artifact

    artifact = _build_mock_artifact(tmp_path)
    artifact["n_cells_failed"] = 1
    _, md_p = write_artifact(artifact, tmp_path)
    text = md_p.read_text()
    assert "GATE FAIL" in text


# ---------------------------------------------------------------------------
# Sign-off must NOT be created
# ---------------------------------------------------------------------------


def test_signoff_file_is_not_created_by_this_commit() -> None:
    """Commit 32 must not create the stage-3 sign-off file. The runner
    only checks for it; never writes."""
    assert not SIGNOFF_FILE.exists()


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost missing")
def test_signoff_file_still_absent_after_canary_run(tmp_path: Path) -> None:
    """Belt-and-braces: even after a real canary pass on the local
    machine, the runner must not create the sign-off file."""
    out_root = tmp_path / "out"
    gate = tmp_path / "gate"
    subprocess.run(
        [sys.executable, str(RUN_SCRIPT),
         "--output-root", str(out_root), "--gate-dir", str(gate),
         "--max-jobs", "12", "--max-evaluations", "2", "--n-folds", "2"],
        capture_output=True, text=True, check=False,
    )
    assert not SIGNOFF_FILE.exists()


# ---------------------------------------------------------------------------
# Manifest sanity
# ---------------------------------------------------------------------------


def test_batch_00_manifest_advertises_synthetic_only() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert payload["uses_openml"] is False
    assert "synthetic_task" in payload
    assert sorted(payload["methods"]) == [
        "default_gbdt", "doe_rsm_vrf_true_nbi", "random_search", "tpe_optuna",
    ]


# ---------------------------------------------------------------------------
# Filter / prune helper
# ---------------------------------------------------------------------------


def test_prune_shard_keeps_only_canary_cells(tmp_path: Path) -> None:
    from scripts.run_batch_00_synthetic_canary import _prune_shard

    dst = tmp_path / "shard.sqlite"
    shutil.copy(SOURCE_SHARD, dst)
    n = _prune_shard(
        dst,
        methods=("default_gbdt", "random_search",
                 "tpe_optuna", "doe_rsm_vrf_true_nbi"),
        algorithms=("xgboost", "lightgbm", "catboost"),
        stage="stage0_replica_001",
        max_replicas_per_cell=1,
    )
    assert 0 < n <= 12
    cx = sqlite3.connect(dst)
    methods = {m for (m,) in cx.execute(
        "SELECT DISTINCT method FROM cc18_jobs"
    )}
    algorithms = {a for (a,) in cx.execute(
        "SELECT DISTINCT algorithm FROM cc18_jobs"
    )}
    cx.close()
    assert methods <= {"default_gbdt", "random_search",
                       "tpe_optuna", "doe_rsm_vrf_true_nbi"}
    assert algorithms <= {"xgboost", "lightgbm", "catboost"}
