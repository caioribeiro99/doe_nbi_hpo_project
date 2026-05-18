"""Tests for the batch_01_cc18_tiny_3_tasks runner and the OpenML loader.

Covers:
- script ``--help`` and ``--dry-run`` exit zero;
- the runner refuses to proceed when the batch_00 latest artifact
  is missing, fails its green check, was generated with a mutated
  source shard, or is older than the staleness window;
- the runner refuses to proceed when ``stage3_signoff.json`` exists;
- the batch CSV is restricted to exactly the 3 documented task IDs;
- the OpenML cache path resolves under ``data/source/openml_cc18``
  and is matched by the repo ``.gitignore``;
- the OpenML loader reads a pre-populated cache without contacting
  the network and refuses cache misses when ``allow_download=False``;
- the artifact JSON schema includes every required gate key;
- a ``--skip-train`` end-to-end pass leaves every committed source
  shard byte-identical and does not create the stage-3 sign-off
  file;
- ``scikit-learn`` version reporting works (importlib.metadata
  rather than importlib.import_module on the distribution name).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "scripts/run_batch_01_cc18_tiny_3_tasks.py"
BATCH_CSV = (
    REPO / "benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv"
)
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"

BATCH_TASK_IDS = (9946, 125920, 11)


@pytest.fixture(autouse=True)
def _hide_real_signoff_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Commit 45 created ``stage3_signoff.json`` on disk. The
    batch_01 runner refuses to run once that file exists; tests
    that exercise the runner therefore see ``SIGNOFF_FILE`` as
    absent via this monkeypatch. Tests that verify the guard's
    behavior override with their own per-test setattr."""
    from scripts import run_batch_01_cc18_tiny_3_tasks as m

    monkeypatch.setattr(
        m, "SIGNOFF_FILE",
        tmp_path_factory.mktemp("hide_signoff") / "absent.json",
    )


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_batch00_gate(
    path: Path, *, timestamp: str | None = None,
    n_cells_success: int = 12, n_cells_failed: int = 0,
    n_cells_expected: int = 12, source_shard_unchanged: bool = True,
) -> Path:
    payload = {
        "batch_id": "batch_00_synthetic_canary",
        "run_timestamp": timestamp or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "git_sha": "0" * 40,
        "n_cells_expected": n_cells_expected,
        "n_cells_success": n_cells_success,
        "n_cells_failed": n_cells_failed,
        "source_shard_unchanged": source_shard_unchanged,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_run_script_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.lower()
    assert "run_batch_01_cc18_tiny_3_tasks.py" in out
    assert "--batch-csv" in out
    assert "--openml-cache-root" in out
    assert "--max-age-days" in out


def test_run_script_dry_run_does_not_invoke_runner(tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--dry-run",
         "--output-root", str(tmp_path / "out"),
         "--gate-dir", str(tmp_path / "gate")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "gate" / "batch_01_cc18_tiny_3_tasks_latest.json").exists()


# ---------------------------------------------------------------------------
# Batch CSV: exactly the 3 documented task IDs
# ---------------------------------------------------------------------------


def test_batch_csv_contains_exactly_the_three_task_ids() -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import load_batch_task_ids

    ids = load_batch_task_ids(BATCH_CSV)
    assert ids == list(BATCH_TASK_IDS)


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_batch_00_artifact_missing(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import (
        GateRefusalError,
        verify_batch00_gate,
    )

    missing = tmp_path / "absent.json"
    with pytest.raises(GateRefusalError, match="not found"):
        verify_batch00_gate(missing)


def test_refuses_when_batch_00_artifact_stale(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import (
        GateRefusalError,
        verify_batch00_gate,
    )

    stale_ts = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    gate = _write_fake_batch00_gate(tmp_path / "stale.json", timestamp=stale_ts)
    with pytest.raises(GateRefusalError, match="days old"):
        verify_batch00_gate(gate, max_age_days=7)


def test_refuses_when_batch_00_not_green(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import (
        GateRefusalError,
        verify_batch00_gate,
    )

    gate = _write_fake_batch00_gate(
        tmp_path / "bad.json",
        n_cells_success=10, n_cells_failed=2, n_cells_expected=12,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch00_gate(gate)


def test_refuses_when_source_shard_was_mutated_during_batch_00(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import (
        GateRefusalError,
        verify_batch00_gate,
    )

    gate = _write_fake_batch00_gate(
        tmp_path / "tampered.json", source_shard_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shard_unchanged"):
        verify_batch00_gate(gate)


def test_refuses_when_stage3_signoff_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a future commit accidentally promotes a sign-off file, batch_01
    should refuse to silently re-run pre-stage-0 logic over it."""
    from scripts import run_batch_01_cc18_tiny_3_tasks as m

    fake_signoff = tmp_path / "stage3_signoff.json"
    fake_signoff.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake_signoff)

    fake_gate = _write_fake_batch00_gate(tmp_path / "good_gate.json")
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.run_batch_01(
            batch_csv=BATCH_CSV,
            shards_dir=SHARDS_DIR,
            out_root=tmp_path / "out",
            gate_dir=tmp_path / "gate",
            openml_cache_root=tmp_path / "cache",
            batch00_gate=fake_gate,
            max_age_days=7,
            skip_train=True,
        )


# ---------------------------------------------------------------------------
# Cache layout: gitignored under data/source/openml_cc18
# ---------------------------------------------------------------------------


def test_cache_path_resolves_under_data_source_openml_cc18() -> None:
    from doe_xgb.datasets.openml_cc18_loader import (
        DEFAULT_CACHE_ROOT,
        cache_dir_for_task,
    )

    assert DEFAULT_CACHE_ROOT.resolve() == (
        REPO / "data" / "source" / "openml_cc18"
    ).resolve()
    p = cache_dir_for_task(9946)
    assert p.resolve() == (DEFAULT_CACHE_ROOT / "9946").resolve()


def test_default_openml_cache_root_is_gitignored() -> None:
    """The OpenML-CC18 raw-payload cache must stay gitignored on every
    machine. The dedicated rule covers per-task pickle payloads,
    per-task manifests, and the OpenML library cache subdirectory.

    We additionally verify via ``git check-ignore`` that real paths
    under the cache root are recognized as ignored, so the rule is not
    just textually present but also effective."""
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "data/source/*" in text
    assert "data/source/*/raw/" in text
    assert "data/source/*/processed/" in text
    assert "data/source/openml_cc18/" in text

    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "data/source/openml_cc18/9946/payload.pkl",
         "data/source/openml_cc18/9946/manifest.json",
         "data/source/openml_cc18/_openml_cache/anything.txt"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    # rc=0 means every path is ignored.
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "data/source/openml_cc18/" in res.stdout


def test_loader_uses_cache_when_payload_and_manifest_match(tmp_path: Path) -> None:
    """Pre-populate a cache directory; the loader must NOT contact the
    network. We verify by passing ``allow_download=False`` and checking
    we get back the same arrays we wrote."""
    import numpy as np

    from doe_xgb.datasets.openml_cc18_loader import load_cc18_task

    cache_dir = tmp_path / "openml_cc18" / "9946"
    cache_dir.mkdir(parents=True)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0], dtype=np.int64)
    payload_p = cache_dir / "payload.pkl"
    with payload_p.open("wb") as f:
        pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    sha = hashlib.sha256(payload_p.read_bytes()).hexdigest()

    meta = {
        "task_id": 9946, "dataset_id": 1510, "dataset_name": "wdbc",
        "target_name": "Class", "task_type": "binary",
        "n_classes": 2, "n_rows": 3, "n_features": 2,
        "feature_names": ["a", "b"],
        "categorical_columns": [],
        "class_distribution": {"0": 2, "1": 1},
        "payload_filename": "payload.pkl", "payload_sha256": sha,
        "openml_url": "https://www.openml.org/t/9946",
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
    )

    payload = load_cc18_task(
        9946, cache_root=tmp_path / "openml_cc18", allow_download=False,
    )
    assert payload.task_id == 9946
    assert payload.dataset_name == "wdbc"
    assert payload.task_type == "binary"
    assert payload.X.shape == (3, 2)
    assert payload.y.tolist() == [0, 1, 0]
    assert payload.payload_sha256 == sha


def test_loader_refuses_cache_miss_with_allow_download_false(tmp_path: Path) -> None:
    from doe_xgb.datasets.openml_cc18_loader import load_cc18_task

    with pytest.raises(RuntimeError, match="cache miss"):
        load_cc18_task(
            9946, cache_root=tmp_path / "empty", allow_download=False,
        )


# ---------------------------------------------------------------------------
# Shard merge: read-only + 36 cells
# ---------------------------------------------------------------------------


def test_assemble_canary_shard_yields_36_cells_from_unmutated_sources(
    tmp_path: Path,
) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import (
        CANARY_ALGORITHMS,
        CANARY_METHODS,
        assemble_canary_shard,
    )

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    out_path = tmp_path / "merged.sqlite"
    discovery = assemble_canary_shard(
        shards_dir=SHARDS_DIR,
        task_ids=BATCH_TASK_IDS,
        out_path=out_path,
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert discovery["shards_unchanged"] is True
    expected = (
        len(BATCH_TASK_IDS) * len(CANARY_METHODS) * len(CANARY_ALGORITHMS)
    )
    assert discovery["n_rows_in_temp_shard"] == expected
    # Inspect the merged shard.
    import sqlite3

    cx = sqlite3.connect(out_path)
    n = cx.execute("SELECT COUNT(*) FROM cc18_jobs").fetchone()[0]
    methods = {m for (m,) in cx.execute(
        "SELECT DISTINCT method FROM cc18_jobs"
    )}
    algos = {a for (a,) in cx.execute(
        "SELECT DISTINCT algorithm FROM cc18_jobs"
    )}
    tids = {int(t) for (t,) in cx.execute(
        "SELECT DISTINCT openml_task_id FROM cc18_jobs"
    )}
    cx.close()
    assert n == expected
    assert methods == set(CANARY_METHODS)
    assert algos == set(CANARY_ALGORITHMS)
    assert tids == set(BATCH_TASK_IDS)


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


REQUIRED_ARTIFACT_KEYS = (
    "batch_id", "run_timestamp", "git_sha", "platform",
    "package_versions", "runner_command",
    "batch_csv", "shards_dir", "shard_contributions",
    "source_shard_md5_before", "source_shard_md5_after",
    "source_shards_unchanged", "shards_unchanged_after_download",
    "temp_shard", "n_cells_in_temp_shard", "n_cells_expected",
    "n_cells_success", "n_cells_failed", "n_cells_pending",
    "task_metadata", "task_ids", "openml_cache_root",
    "openml_payloads_committed",
    "stage3_signoff_present", "stage3_signoff_path",
    "batch_00_gate", "capability_audit",
    "runtime_seconds", "subprocess_returncode",
    "subprocess_stdout_tail", "subprocess_stderr_tail",
    "cells",
)


def _build_mock_artifact(tmp_path: Path) -> dict:
    return {
        "batch_id": "batch_01_cc18_tiny_3_tasks",
        "run_timestamp": "2026-05-02T00:00:00Z",
        "git_sha": "0" * 40,
        "platform": {
            "hostname": "host", "uname": "Darwin",
            "python_version": "3.12", "python_executable": "/bin/python",
            "machine": "arm64",
        },
        "package_versions": {
            "scikit-learn": "1.8.0", "xgboost": "3.2.0",
            "lightgbm": None,
        },
        "runner_command": ["python", "scripts/cc18_runner.py"],
        "batch_csv": "benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv",
        "shards_dir": "jobs/doctoral/openml_cc18/shards/stage0_replica_001",
        "shard_contributions": {"shard_00.sqlite": 8},
        "source_shard_md5_before": {"shard_00.sqlite": "a" * 32},
        "source_shard_md5_after": {"shard_00.sqlite": "a" * 32},
        "source_shards_unchanged": True,
        "shards_unchanged_after_download": True,
        "temp_shard": str(tmp_path / "merged.sqlite"),
        "n_cells_in_temp_shard": 36,
        "n_cells_expected": 36,
        "n_cells_success": 36,
        "n_cells_failed": 0,
        "n_cells_pending": 0,
        "task_metadata": [
            {"task_id": 9946, "dataset_id": 1510, "dataset_name": "wdbc",
             "target_name": "Class", "task_type": "binary",
             "n_classes": 2, "n_rows": 569, "n_features": 30,
             "n_categorical_columns": 0, "categorical_columns": [],
             "class_distribution": {"0": 357, "1": 212},
             "payload_sha256": "deadbeef" * 8,
             "cache_dir": "data/source/openml_cc18/9946"},
        ],
        "task_ids": [9946, 125920, 11],
        "openml_cache_root": "data/source/openml_cc18",
        "openml_payloads_committed": False,
        "stage3_signoff_present": False,
        "stage3_signoff_path": "jobs/doctoral/openml_cc18/stage3_signoff.json",
        "batch_00_gate": {
            "n_cells_expected": 12, "n_cells_success": 12,
            "n_cells_failed": 0, "source_shard_unchanged": True,
            "run_timestamp": "2026-05-02T00:00:00Z", "age_days": 0.5,
            "git_sha": "fd36601",
        },
        "capability_audit": {
            "n_benchmarked": 13, "smoke_ready": ["default_gbdt"],
            "dispatch_only": [], "stub_only": [],
            "missing_packages": [],
        },
        "runtime_seconds": 1.0,
        "subprocess_returncode": 0,
        "subprocess_stdout_tail": "",
        "subprocess_stderr_tail": "",
        "cells": [
            {"openml_task_id": 9946, "method": "default_gbdt",
             "algorithm": "xgboost", "status": "success",
             "runtime_seconds": 0.1, "last_error": None,
             "manifest": None, "aggregate_metrics": {"accuracy": 0.95},
             "metric_keys": ["accuracy"]},
        ],
    }


def test_artifact_writer_emits_all_required_keys(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import write_artifact

    artifact = _build_mock_artifact(tmp_path)
    json_p, md_p = write_artifact(artifact, tmp_path)
    assert json_p.exists() and md_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in REQUIRED_ARTIFACT_KEYS:
        assert key in payload, f"missing required key: {key}"


def test_artifact_md_pass_when_all_clear(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import write_artifact

    _, md_p = write_artifact(_build_mock_artifact(tmp_path), tmp_path)
    assert "GATE PASS" in md_p.read_text(encoding="utf-8")


def test_artifact_md_fail_when_any_failure(tmp_path: Path) -> None:
    from scripts.run_batch_01_cc18_tiny_3_tasks import write_artifact

    artifact = _build_mock_artifact(tmp_path)
    artifact["n_cells_failed"] = 1
    _, md_p = write_artifact(artifact, tmp_path)
    assert "GATE FAIL" in md_p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# End-to-end --skip-train: shards unchanged + no sign-off file
# ---------------------------------------------------------------------------


HAS_OPENML = importlib.util.find_spec("openml") is not None


@pytest.mark.skipif(not HAS_OPENML, reason="openml not installed")
def test_skip_train_pass_leaves_shards_unchanged_and_no_signoff(
    tmp_path: Path,
) -> None:
    """Run the batch_01 pre-flight + shard merge, but do NOT actually
    invoke cc18_runner. After the call:
    - every committed source shard has its byte-identical MD5;
    - the stage-3 sign-off file is still absent."""
    from scripts.run_batch_01_cc18_tiny_3_tasks import run_batch_01

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    fake_gate = _write_fake_batch00_gate(tmp_path / "gate.json")
    cache_root = tmp_path / "openml_cc18"
    artifact = run_batch_01(
        batch_csv=BATCH_CSV,
        shards_dir=SHARDS_DIR,
        out_root=tmp_path / "out",
        gate_dir=tmp_path / "gate",
        openml_cache_root=cache_root,
        batch00_gate=fake_gate,
        max_age_days=7,
        skip_train=True,
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert artifact["source_shards_unchanged"] is True
    assert artifact["shards_unchanged_after_download"] is True
    assert artifact["n_cells_in_temp_shard"] == 36
    assert artifact["n_cells_expected"] == 36
    assert artifact["stage3_signoff_present"] is False


# ---------------------------------------------------------------------------
# scikit-learn version reporting (the bug fixed in this commit)
# ---------------------------------------------------------------------------


def test_scikit_learn_version_resolves_via_distribution_name() -> None:
    """The dissertation-era reporter used importlib.import_module on the
    distribution name and silently mapped scikit-learn to None. The fix
    routes through importlib.metadata first."""
    from doe_xgb._versions import package_version

    ver = package_version("scikit-learn")
    assert ver is not None
    assert ver.count(".") >= 1
    # Numeric-ish prefix.
    assert ver[0].isdigit()


def test_collect_package_versions_reports_known_packages() -> None:
    from doe_xgb._versions import collect_package_versions

    versions = collect_package_versions((
        "scikit-learn", "numpy", "definitely-not-a-package",
    ))
    assert versions["scikit-learn"] is not None
    assert versions["numpy"] is not None
    assert versions["definitely-not-a-package"] is None
