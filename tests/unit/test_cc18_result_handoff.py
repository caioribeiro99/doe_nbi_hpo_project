"""Tests for the CC18 result handoff protocol (Commit 35).

Covers:
- ``scripts/create_cc18_run_dir.py``
  - copies committed shards into ``runs/cc18/<run_id>/shards/<stage>/``
    with an ``.execution.sqlite`` suffix;
  - leaves every committed source shard byte-identical;
  - refuses to materialize under ``jobs/`` and to overwrite an
    existing run dir without ``--force``;
  - records source-shard MD5s in ``run_manifest.json``.
- ``scripts/export_cc18_run_summary.py``
  - produces a JSON + Markdown summary for a freshly-mutated
    execution shard;
  - aggregates status counts across multiple shards;
  - re-checks committed source shards and surfaces drift;
  - refuses to summarize a run dir that does not live under
    ``runs/``;
  - the JSON validates against a minimum schema;
  - large artifacts under the run dir (e.g. fitted models) are
    not staged because the path lives under ``runs/``.
- ``.gitignore`` invariants for ``runs/`` and ``experiments/_stage_runs/``.
- ``docs/RESULT_HANDOFF_PROTOCOL.md`` exists and documents the flow.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CREATE_SCRIPT = REPO / "scripts/create_cc18_run_dir.py"
EXPORT_SCRIPT = REPO / "scripts/export_cc18_run_summary.py"
SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
PROTOCOL_DOC = REPO / "docs/RESULT_HANDOFF_PROTOCOL.md"
GITIGNORE = REPO / ".gitignore"

EXECUTION_SUFFIX = ".execution.sqlite"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_create_run_dir_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.lower()
    assert "create_cc18_run_dir.py" in out
    assert "--run-id" in out
    assert "--stage" in out
    assert "--shard" in out


def test_export_summary_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.lower()
    assert "export_cc18_run_summary.py" in out
    assert "--run-dir" in out
    assert "--out-json" in out
    assert "--out-md" in out


# ---------------------------------------------------------------------------
# Run-dir creation does not mutate sources
# ---------------------------------------------------------------------------


def test_create_run_dir_copies_shards_without_mutating_source(tmp_path: Path) -> None:
    from scripts.create_cc18_run_dir import create_run_dir

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    manifest = create_run_dir(
        run_id="test_run_create_basic",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite", "shard_01.sqlite"],
        run_root=tmp_path / "runs" / "cc18",
        shards_root=SHARDS_DIR.parent,
        force=False,
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert manifest["source_shards_unchanged"] is True
    assert manifest["n_shards"] == 2
    # Execution copies exist with the suffix.
    run_dir = tmp_path / "runs" / "cc18" / "test_run_create_basic"
    exec_files = sorted((run_dir / "shards" / "stage0_replica_001").iterdir())
    assert [p.name for p in exec_files] == [
        "shard_00.execution.sqlite",
        "shard_01.execution.sqlite",
    ]
    # Manifest records both source MD5 and execution MD5.
    rec = json.loads((run_dir / "run_manifest.json").read_text())
    assert rec["run_id"] == "test_run_create_basic"
    assert rec["stage"] == "stage0_replica_001"
    src_md5s = {c["source_md5_before"] for c in rec["shard_copies"]}
    assert all(len(h) == 32 for h in src_md5s)


def test_create_run_dir_refuses_destination_under_jobs(tmp_path: Path) -> None:
    from scripts.create_cc18_run_dir import create_run_dir

    bad_root = REPO / "jobs" / "doctoral" / "openml_cc18" / "fake_runs"
    with pytest.raises(ValueError, match="under jobs/"):
        create_run_dir(
            run_id="should_not_exist",
            stage="stage0_replica_001",
            shard_files=["shard_00.sqlite"],
            run_root=bad_root,
            shards_root=SHARDS_DIR.parent,
            force=False,
        )
    assert not bad_root.exists()


def test_create_run_dir_refuses_existing_run_id_without_force(tmp_path: Path) -> None:
    from scripts.create_cc18_run_dir import create_run_dir

    run_root = tmp_path / "runs" / "cc18"
    create_run_dir(
        run_id="rerun_collision",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite"],
        run_root=run_root,
        shards_root=SHARDS_DIR.parent,
    )
    with pytest.raises(FileExistsError, match="run_dir already exists"):
        create_run_dir(
            run_id="rerun_collision",
            stage="stage0_replica_001",
            shard_files=["shard_00.sqlite"],
            run_root=run_root,
            shards_root=SHARDS_DIR.parent,
        )
    # --force should overwrite.
    manifest = create_run_dir(
        run_id="rerun_collision",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite"],
        run_root=run_root,
        shards_root=SHARDS_DIR.parent,
        force=True,
    )
    assert manifest["n_shards"] == 1


def test_create_run_dir_rejects_invalid_run_id(tmp_path: Path) -> None:
    from scripts.create_cc18_run_dir import create_run_dir

    for bad in ("", "has spaces", "has/slash", "has\\back"):
        with pytest.raises(ValueError, match="invalid run_id"):
            create_run_dir(
                run_id=bad,
                stage="stage0_replica_001",
                shard_files=["shard_00.sqlite"],
                run_root=tmp_path / "runs" / "cc18",
                shards_root=SHARDS_DIR.parent,
            )


# ---------------------------------------------------------------------------
# Run dir is gitignored
# ---------------------------------------------------------------------------


def test_runs_tree_is_gitignored() -> None:
    """Anything under runs/ — execution SQLite, fitted models, fold
    metrics, archives — must be gitignored on every machine."""
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "runs/" in text

    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "runs/foo",
         "runs/cc18/bar/run_manifest.json",
         "runs/cc18/bar/shards/stage0/shard_00.execution.sqlite",
         "runs/cc18/bar/outputs/abc/manifest.json",
         "runs/cc18/bar/outputs/abc/catboost_info/learn_error.tsv",
         "runs/cc18/bar/archives/run.tar.zst"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    # All six paths must show up as ignored by the runs/ rule.
    assert res.stdout.count("runs/") >= 6


def test_stage_runs_jsonmd_allowlist_is_correct() -> None:
    """experiments/_stage_runs/ is gitignored except for *.json/*.md."""
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "!experiments/_stage_runs/" in text
    assert "experiments/_stage_runs/*" in text
    assert "!experiments/_stage_runs/*.json" in text
    assert "!experiments/_stage_runs/*.md" in text

    res = subprocess.run(
        ["git", "check-ignore", "-v",
         "experiments/_stage_runs/runX_summary.json",
         "experiments/_stage_runs/runX_summary.md",
         "experiments/_stage_runs/runX/big_model.bin",
         "experiments/_stage_runs/runX/fold_metrics.csv"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    # Only the .json / .md should be NOT ignored; the others should
    # be ignored. ``git check-ignore -v`` returns 0 if at least one
    # path matched a rule (positive or negative).
    assert res.returncode == 0
    # Non-allowlisted nested files must be matched by the
    # experiments/* rule (not by the negation).
    assert "big_model.bin" in res.stdout
    assert "fold_metrics.csv" in res.stdout


# ---------------------------------------------------------------------------
# Export summary: status counts + JSON schema + source verification
# ---------------------------------------------------------------------------


def _build_run_dir_with_mutated_shard(tmp_path: Path) -> tuple[Path, Path]:
    """Create a run dir, then mutate the execution SQLite so it carries a
    realistic mix of statuses. Returns (run_dir, execution_path)."""
    from scripts.create_cc18_run_dir import create_run_dir

    create_run_dir(
        run_id="mock_export_run",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite"],
        run_root=tmp_path / "runs" / "cc18",
        shards_root=SHARDS_DIR.parent,
    )
    run_dir = tmp_path / "runs" / "cc18" / "mock_export_run"
    exec_p = run_dir / "shards" / "stage0_replica_001" / "shard_00.execution.sqlite"
    cx = sqlite3.connect(exec_p)
    cx.execute("UPDATE cc18_jobs SET status='success', runtime_seconds=0.05 "
               "WHERE rowid IN (SELECT rowid FROM cc18_jobs LIMIT 5)")
    cx.execute("UPDATE cc18_jobs SET status='failed', runtime_seconds=0.10, "
               "last_error='boom' "
               "WHERE rowid IN ("
               "SELECT rowid FROM cc18_jobs WHERE status='pending' LIMIT 2)")
    cx.commit()
    cx.close()
    return run_dir, exec_p


def test_export_summary_writes_json_md_with_required_keys(tmp_path: Path) -> None:
    from scripts.export_cc18_run_summary import export_summary

    run_dir, _ = _build_run_dir_with_mutated_shard(tmp_path)
    out_json = tmp_path / "stage_runs" / "mock_export_run_summary.json"
    out_md = tmp_path / "stage_runs" / "mock_export_run_summary.md"
    summary = export_summary(
        run_dir=run_dir, out_json=out_json, out_md=out_md,
        include_shard_hashes=True, batch_id="mock_batch_id",
    )
    assert out_json.exists() and out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    for key in (
        "schema_version", "run_id", "stage", "batch_id", "exported_at",
        "source_git_sha", "host", "python", "package_versions",
        "n_total", "n_pending", "n_claimed", "n_running",
        "n_success", "n_failed", "n_skipped", "status_counts",
        "runtime_seconds_total", "runtime_seconds_max",
        "runtime_n_recorded", "started_at_min", "finished_at_max",
        "failures_grouped", "n_failures_grouped",
        "shards", "n_shards", "run_dir", "run_manifest_path",
        "execution_suffix", "source_shards_unchanged",
        "source_md5_recorded", "source_md5_now", "source_drift",
        "archive_path", "archive_sha256", "archive_size_bytes",
        "stage3_signoff_present", "stage3_signoff_path",
        "protocol_doc",
    ):
        assert key in payload, f"missing required key: {key}"
    # Sanity on counts.
    assert payload["n_success"] == 5
    assert payload["n_failed"] == 2
    assert payload["n_failures_grouped"] >= 1
    assert payload["batch_id"] == "mock_batch_id"
    # Per-shard SHA-256 recorded.
    assert all(len(sh.get("sha256", "")) == 64 for sh in payload["shards"])
    # Source shards still unchanged.
    assert payload["source_shards_unchanged"] is True
    assert summary["n_total"] >= 7


def test_export_summary_md_pass_when_terminal_and_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'GREEN' verdict requires ``stage3_signoff_present`` to
    be false. Post-Commit-45 the real signoff file exists on disk,
    which would correctly flip the verdict to 'NOT GREEN' (a signed
    replica's per-shard run summaries should not advertise as
    'green-ready-to-promote' since they have already been
    promoted). For this test, monkeypatch the exporter's
    ``SIGNOFF_FILE`` constant to a tmp path so the GREEN code path
    is exercised."""
    from scripts import export_cc18_run_summary as exp
    from scripts.create_cc18_run_dir import create_run_dir
    from scripts.export_cc18_run_summary import export_summary

    monkeypatch.setattr(
        exp, "SIGNOFF_FILE", tmp_path / "absent_signoff.json",
    )
    create_run_dir(
        run_id="md_green_run",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite"],
        run_root=tmp_path / "runs" / "cc18",
        shards_root=SHARDS_DIR.parent,
    )
    run_dir = tmp_path / "runs" / "cc18" / "md_green_run"
    exec_p = run_dir / "shards" / "stage0_replica_001" / "shard_00.execution.sqlite"
    cx = sqlite3.connect(exec_p)
    cx.execute("UPDATE cc18_jobs SET status='success', runtime_seconds=0.01")
    cx.commit()
    cx.close()
    out_json = tmp_path / "stage_runs" / "md_green_run_summary.json"
    out_md = tmp_path / "stage_runs" / "md_green_run_summary.md"
    export_summary(run_dir=run_dir, out_json=out_json, out_md=out_md)
    assert "**GREEN**" in out_md.read_text(encoding="utf-8")


def test_export_summary_marks_drift_when_recorded_md5_lies(tmp_path: Path) -> None:
    """Stage a run dir, then forge run_manifest.json so the recorded
    source MD5 no longer matches the live committed shard. The summary
    must mark ``source_shards_unchanged: false`` and list the drift."""
    from scripts.create_cc18_run_dir import create_run_dir
    from scripts.export_cc18_run_summary import export_summary

    create_run_dir(
        run_id="drift_detector",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite"],
        run_root=tmp_path / "runs" / "cc18",
        shards_root=SHARDS_DIR.parent,
    )
    run_dir = tmp_path / "runs" / "cc18" / "drift_detector"
    manifest_p = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    forged = "0" * 32
    for c in manifest["shard_copies"]:
        c["source_md5_before"] = forged
        c["source_md5_after"] = forged
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_json = tmp_path / "stage_runs" / "drift_detector_summary.json"
    out_md = tmp_path / "stage_runs" / "drift_detector_summary.md"
    summary = export_summary(
        run_dir=run_dir, out_json=out_json, out_md=out_md,
    )
    assert summary["source_shards_unchanged"] is False
    assert any(d["issue"] == "source_shard_mutated"
               for d in summary["source_drift"])
    assert "NOT GREEN" in out_md.read_text(encoding="utf-8")


def test_export_refuses_run_dir_outside_runs_tree(tmp_path: Path) -> None:
    """Summarizing the committed shard tree directly is forbidden."""
    from scripts.export_cc18_run_summary import export_summary

    bad_dir = SHARDS_DIR.parent  # under jobs/, not runs/
    out_json = tmp_path / "x_summary.json"
    out_md = tmp_path / "x_summary.md"
    with pytest.raises(ValueError, match="runs/"):
        export_summary(run_dir=bad_dir, out_json=out_json, out_md=out_md)


def test_aggregate_failures_sorts_and_groups() -> None:
    from scripts.export_cc18_run_summary import aggregate_failures

    per_shard = [
        {"failures": [
            {"method": "tpe_optuna", "algorithm": "lightgbm",
             "openml_task_id": 9946, "last_error": "oom"},
            {"method": "default_gbdt", "algorithm": "xgboost",
             "openml_task_id": 11, "last_error": "shape"},
        ]},
        {"failures": [
            {"method": "tpe_optuna", "algorithm": "lightgbm",
             "openml_task_id": 9946, "last_error": "oom2"},
        ]},
    ]
    rolled = aggregate_failures(per_shard)
    assert len(rolled) == 2
    # Sorted by (method, algorithm, task_id).
    assert rolled[0]["method"] == "default_gbdt"
    assert rolled[1]["method"] == "tpe_optuna"
    assert rolled[1]["count"] == 2


# ---------------------------------------------------------------------------
# Source-shard immutability after a full create -> export pass
# ---------------------------------------------------------------------------


def test_full_create_export_cycle_leaves_committed_shards_unchanged(
    tmp_path: Path,
) -> None:
    """End-to-end: copy several committed shards, mutate the execution
    copies, export the summary. The committed shard MD5s must be
    byte-identical before and after."""
    from scripts.create_cc18_run_dir import create_run_dir
    from scripts.export_cc18_run_summary import export_summary

    md5_before = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    create_run_dir(
        run_id="full_cycle",
        stage="stage0_replica_001",
        shard_files=["shard_00.sqlite", "shard_01.sqlite", "shard_02.sqlite"],
        run_root=tmp_path / "runs" / "cc18",
        shards_root=SHARDS_DIR.parent,
    )
    run_dir = tmp_path / "runs" / "cc18" / "full_cycle"
    for exec_p in run_dir.rglob("*.execution.sqlite"):
        cx = sqlite3.connect(exec_p)
        cx.execute(
            "UPDATE cc18_jobs SET status='success', runtime_seconds=0.02 "
            "WHERE rowid IN (SELECT rowid FROM cc18_jobs LIMIT 3)"
        )
        cx.execute(
            "UPDATE cc18_jobs SET status='failed', last_error='x' "
            "WHERE rowid IN ("
            "SELECT rowid FROM cc18_jobs WHERE status='pending' LIMIT 1)"
        )
        cx.commit()
        cx.close()

    out_json = tmp_path / "stage_runs" / "full_cycle_summary.json"
    out_md = tmp_path / "stage_runs" / "full_cycle_summary.md"
    summary = export_summary(
        run_dir=run_dir, out_json=out_json, out_md=out_md,
        include_shard_hashes=True,
    )
    md5_after = {
        p.name: _md5(p) for p in sorted(SHARDS_DIR.glob("shard_*.sqlite"))
    }
    assert md5_before == md5_after
    assert summary["source_shards_unchanged"] is True
    assert summary["n_shards"] == 3
    assert summary["n_success"] >= 1
    assert summary["n_failed"] >= 1


# ---------------------------------------------------------------------------
# Doc invariants
# ---------------------------------------------------------------------------


def test_protocol_doc_exists_and_mentions_key_paths() -> None:
    text = PROTOCOL_DOC.read_text(encoding="utf-8")
    for token in (
        "runs/cc18/",
        "experiments/_stage_runs/",
        "jobs/doctoral/openml_cc18/shards/",
        "create_cc18_run_dir.py",
        "export_cc18_run_summary.py",
        "stage3_signoff.json",
        ".execution.sqlite",
    ):
        assert token in text, f"protocol doc missing reference: {token}"
