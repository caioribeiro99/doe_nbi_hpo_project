"""Tests for the batch_03_cc18_representative_18_tasks runner.

Covers:
- ``--help`` / ``--dry-run`` exit zero;
- the runner refuses to proceed when the batch_02 stage-run summary is
  missing, has failures, was generated with mutated source shards,
  or is older than the staleness window;
- the runner refuses if a stage-3 sign-off file already exists;
- the batch CSV resolves to exactly the 18 documented task IDs;
- the run dir is created under ``runs/cc18/`` and is gitignored;
- execution SQLite files are not staged (covered by gitignore);
- a ``--skip-train`` end-to-end run leaves every committed shard
  byte-identical and does not create the stage-3 sign-off file;
- the published stage-run summary JSON contains every required key
  (including the batch_03-specific augmentation block).
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
RUN_SCRIPT = REPO / "scripts/run_batch_03_cc18_representative_18_tasks.py"
BATCH_CSV = (
    REPO
    / "benchmarks/doctoral/openml_cc18/batches/batch_03_cc18_representative_18_tasks.csv"
)
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
GITIGNORE = REPO / ".gitignore"

BATCH_TASK_IDS = (
    6, 11, 53, 219, 2074, 2079, 3022, 3917, 9946, 9978,
    10093, 14965, 125920, 146817, 146819, 146821, 167121, 167125,
)


@pytest.fixture(autouse=True)
def _hide_real_signoff_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Commit 45 created ``stage3_signoff.json`` on disk. The
    batch_03 runner refuses to run once that file exists; tests
    therefore see ``SIGNOFF_FILE`` as absent via this monkeypatch.
    Tests verifying the guard override with per-test setattr."""
    from scripts import run_batch_03_cc18_representative_18_tasks as m

    monkeypatch.setattr(
        m, "SIGNOFF_FILE",
        tmp_path_factory.mktemp("hide_signoff") / "absent.json",
    )


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_fake_batch02_summary(
    path: Path, *, exported_at: str | None = None,
    n_cells_success: int = 144, n_cells_failed: int = 0,
    n_cells_pending: int = 0, n_cells_expected: int = 144,
    source_shards_unchanged: bool = True,
    stage3_signoff_present: bool = False,
) -> Path:
    payload = {
        "schema_version": 1,
        "batch_id": "batch_02_cc18_small_12_tasks",
        "run_id": "batch_02_cc18_small_12_tasks_latest",
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
    assert "run_batch_03_cc18_representative_18_tasks.py" in out
    assert "--batch-csv" in out
    assert "--openml-cache-root" in out
    assert "--max-age-days" in out
    assert "--run-root" in out
    assert "--stage-runs-dir" in out
    assert "--batch02-summary" in out


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
# Batch CSV
# ---------------------------------------------------------------------------


def test_batch_csv_contains_exactly_the_eighteen_task_ids() -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        load_batch_task_ids,
    )

    ids = load_batch_task_ids(BATCH_CSV)
    assert ids == list(BATCH_TASK_IDS)
    assert len(ids) == 18


# ---------------------------------------------------------------------------
# Pre-flight refusals
# ---------------------------------------------------------------------------


def test_refuses_when_batch_02_summary_missing(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    with pytest.raises(GateRefusalError, match="not found"):
        verify_batch02_summary(tmp_path / "absent.json")


def test_refuses_when_batch_02_summary_stale(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    stale = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    summary = _write_fake_batch02_summary(
        tmp_path / "stale.json", exported_at=stale,
    )
    with pytest.raises(GateRefusalError, match="days old"):
        verify_batch02_summary(summary, max_age_days=7)


def test_refuses_when_batch_02_failed(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    summary = _write_fake_batch02_summary(
        tmp_path / "failed.json",
        n_cells_success=130, n_cells_failed=14,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch02_summary(summary)


def test_refuses_when_batch_02_pending(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    summary = _write_fake_batch02_summary(
        tmp_path / "pending.json",
        n_cells_success=140, n_cells_pending=4,
    )
    with pytest.raises(GateRefusalError, match="not green"):
        verify_batch02_summary(summary)


def test_refuses_when_batch_02_source_shards_mutated(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    summary = _write_fake_batch02_summary(
        tmp_path / "mut.json", source_shards_unchanged=False,
    )
    with pytest.raises(GateRefusalError, match="source_shards_unchanged"):
        verify_batch02_summary(summary)


def test_refuses_when_batch_02_signed_off(tmp_path: Path) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        GateRefusalError,
        verify_batch02_summary,
    )

    summary = _write_fake_batch02_summary(
        tmp_path / "signed.json", stage3_signoff_present=True,
    )
    with pytest.raises(GateRefusalError, match="stage3_signoff_present"):
        verify_batch02_summary(summary)


def test_refuses_when_stage3_signoff_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_batch_03_cc18_representative_18_tasks as m

    fake = tmp_path / "stage3_signoff.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "SIGNOFF_FILE", fake)
    fake_summary = _write_fake_batch02_summary(tmp_path / "ok.json")
    with pytest.raises(m.GateRefusalError, match="sign-off"):
        m.run_batch_03(
            batch_csv=BATCH_CSV,
            shards_dir=SHARDS_DIR,
            run_root=tmp_path / "runs",
            out_root=tmp_path / "out",
            stage_runs_dir=tmp_path / "stage_runs",
            openml_cache_root=tmp_path / "cache",
            batch02_summary=fake_summary,
            max_age_days=7,
            skip_train=True,
        )


# ---------------------------------------------------------------------------
# Run dir + gitignore
# ---------------------------------------------------------------------------


def test_run_root_default_lives_under_runs_and_is_gitignored() -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        DEFAULT_RUN_ROOT,
    )

    rel = DEFAULT_RUN_ROOT.resolve().relative_to(REPO.resolve())
    assert rel.parts[0] == "runs", rel
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text


def test_execution_sqlite_files_are_gitignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/cc18/batch_03_cc18_representative_18_tasks_latest/run_manifest.json",
         "runs/cc18/batch_03_cc18_representative_18_tasks_latest/shards/"
         "stage0_replica_001/shard_00.execution.sqlite",
         "runs/cc18/batch_03_cc18_representative_18_tasks_latest/outputs/abc/"
         "manifest.json",
         "runs/cc18/batch_03_cc18_representative_18_tasks_latest/outputs/abc/"
         "catboost_info/learn_error.tsv"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert res.stdout.count("runs/") >= 4


def test_stage_runs_summary_jsonmd_are_committed_but_other_files_are_not() -> None:
    """Use --no-index so the test works whether the summary files are
    untracked (pre-commit) or tracked (post-commit)."""
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "experiments/_stage_runs/batch_03_cc18_representative_18_tasks_"
         "latest_summary.json",
         "experiments/_stage_runs/batch_03_cc18_representative_18_tasks_"
         "latest_summary.md",
         "experiments/_stage_runs/batch_03_cc18_representative_18_tasks_"
         "latest/extras.bin"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "summary.json" in res.stdout
    assert "summary.md" in res.stdout
    assert "extras.bin" in res.stdout
    assert "!experiments/_stage_runs/*.json" in res.stdout
    assert "!experiments/_stage_runs/*.md" in res.stdout
    assert "experiments/_stage_runs/*\t" in res.stdout


# ---------------------------------------------------------------------------
# Skip-train end-to-end: shards unchanged + summary JSON schema
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
    # batch_03-specific augmentation
    "batch_id", "batch_csv", "task_ids", "task_metadata",
    "n_cells_expected", "n_cells_in_temp_shard",
    "n_cells_success", "n_cells_failed", "n_cells_pending",
    "cells", "slowest_cells",
    "runtime_seconds_runner_total", "runner_returncodes",
    "openml_cache_root", "openml_payloads_committed",
    "execution_shards_committed", "batch_02_gate",
    "shards_unchanged_after_download",
    "source_shard_md5_before", "source_shard_md5_after",
    "platform", "git_sha", "capability_audit",
)


@pytest.mark.skipif(not HAS_OPENML, reason="openml not installed")
def test_skip_train_pass_leaves_committed_shards_unchanged(
    tmp_path: Path,
) -> None:
    from scripts.run_batch_03_cc18_representative_18_tasks import (
        BATCH_ID,
        run_batch_03,
    )

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    fake_summary = _write_fake_batch02_summary(tmp_path / "good_gate.json")
    summary = run_batch_03(
        batch_csv=BATCH_CSV,
        shards_dir=SHARDS_DIR,
        run_root=tmp_path / "runs",
        out_root=tmp_path / "out",
        stage_runs_dir=tmp_path / "stage_runs",
        openml_cache_root=tmp_path / "cache",
        batch02_summary=fake_summary,
        max_age_days=7,
        skip_train=True,
        run_id="test_batch_03_skip_train",
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["batch_id"] == BATCH_ID
    assert summary["task_ids"] == list(BATCH_TASK_IDS)
    assert summary["n_cells_expected"] == 216
    assert summary["n_cells_in_temp_shard"] == 216
    assert summary["source_shards_unchanged"] is True
    assert summary["shards_unchanged_after_download"] is True
    assert summary["stage3_signoff_present"] is False

    json_p = (
        tmp_path / "stage_runs"
        / "test_batch_03_skip_train_summary.json"
    )
    assert json_p.exists()
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    for key in SUMMARY_REQUIRED_KEYS:
        assert key in payload, f"missing summary key: {key}"

    exec_dir = (
        tmp_path / "runs"
        / "test_batch_03_skip_train"
        / "shards" / "stage0_replica_001"
    )
    exec_files = list(exec_dir.glob("*.execution.sqlite"))
    assert len(exec_files) >= 1
    for p in exec_files:
        assert "runs" in p.resolve().parts


