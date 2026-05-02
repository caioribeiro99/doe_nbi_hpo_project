"""Tests for the CC18 reduced-execution batch manifests and the
shard-filter utility (Commit 31).

Covers:
- generator output is deterministic for a fixed BATCH_SEED;
- every selected task_id is present in tasks.csv;
- tiny / small / representative batches have exactly 3 / 12 / 18
  tasks;
- batch_03_representative meets the documented coverage minima;
- shard filtering never mutates the committed source shard;
- filtered shard contains only rows for the requested task_ids
  and is strictly smaller than the source;
- filtered shard preserves the cc18_jobs schema and ships a
  shard_meta row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS_CSV = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
BATCHES_DIR = REPO / "benchmarks/doctoral/openml_cc18/batches"
SHARD = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"
GEN = REPO / "scripts/create_cc18_batches.py"
FILTER = REPO / "scripts/filter_cc18_shard_for_batch.py"

TINY_FILE = BATCHES_DIR / "batch_01_cc18_tiny_3_tasks.csv"
SMALL_FILE = BATCHES_DIR / "batch_02_cc18_small_12_tasks.csv"
REP_FILE = BATCHES_DIR / "batch_03_cc18_representative_18_tasks.csv"


def _ti(s: str) -> int:
    return int(s) if s and s.strip() else 0


def _tf(s: str) -> float:
    return float(s) if s and s.strip() else 0.0


def _load_csv_task_ids(path: Path) -> list[int]:
    with path.open() as f:
        return [int(r["openml_task_id"]) for r in csv.DictReader(f)]


def _load_tasks_csv() -> dict[int, dict]:
    with TASKS_CSV.open() as f:
        return {int(r["openml_task_id"]): r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Generator determinism + counts
# ---------------------------------------------------------------------------


def test_batch_dir_contains_expected_files() -> None:
    expected = {
        "batch_00_synthetic_canary.json",
        "batch_01_cc18_tiny_3_tasks.csv",
        "batch_01_cc18_tiny_3_tasks.meta.json",
        "batch_02_cc18_small_12_tasks.csv",
        "batch_02_cc18_small_12_tasks.meta.json",
        "batch_03_cc18_representative_18_tasks.csv",
        "batch_03_cc18_representative_18_tasks.meta.json",
        "batch_04_stage0_shard00_only.json",
        "README.md",
    }
    actual = {p.name for p in BATCHES_DIR.iterdir()}
    assert expected.issubset(actual), expected - actual


def test_generator_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        res = subprocess.run(
            [sys.executable, str(GEN),
             "--out-dir", str(out), "--force"],
            capture_output=True, text=True, check=False,
        )
        assert res.returncode == 0, (res.stdout, res.stderr)
    for fname in ("batch_01_cc18_tiny_3_tasks.csv",
                   "batch_02_cc18_small_12_tasks.csv",
                   "batch_03_cc18_representative_18_tasks.csv"):
        assert (out_a / fname).read_bytes() == (out_b / fname).read_bytes(), fname
    # And against the committed copy.
    for fname in ("batch_01_cc18_tiny_3_tasks.csv",
                   "batch_02_cc18_small_12_tasks.csv",
                   "batch_03_cc18_representative_18_tasks.csv"):
        assert (out_a / fname).read_bytes() == (BATCHES_DIR / fname).read_bytes(), (
            f"committed {fname} drifted from generator output"
        )


def test_tiny_batch_has_exactly_3_tasks() -> None:
    assert len(_load_csv_task_ids(TINY_FILE)) == 3


def test_small_batch_has_exactly_12_tasks() -> None:
    assert len(_load_csv_task_ids(SMALL_FILE)) == 12


def test_representative_batch_has_exactly_18_tasks() -> None:
    assert len(_load_csv_task_ids(REP_FILE)) == 18


def test_every_batch_task_id_exists_in_tasks_csv() -> None:
    known = set(_load_tasks_csv().keys())
    for path in (TINY_FILE, SMALL_FILE, REP_FILE):
        for tid in _load_csv_task_ids(path):
            assert tid in known, (path.name, tid)


def test_every_batch_task_id_is_unique() -> None:
    for path in (TINY_FILE, SMALL_FILE, REP_FILE):
        ids = _load_csv_task_ids(path)
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Coverage minima
# ---------------------------------------------------------------------------


def test_tiny_batch_covers_required_clauses() -> None:
    """tiny = 1 binary numeric balanced + 1 binary categorical + 1 multiclass."""
    tasks = _load_tasks_csv()
    rows = [tasks[t] for t in _load_csv_task_ids(TINY_FILE)]
    binary = sum(1 for r in rows if r["task_type"] == "binary")
    multi = sum(1 for r in rows if r["task_type"] == "multiclass")
    cat = sum(1 for r in rows if _ti(r["categorical_feature_count"]) > 0)
    assert binary >= 2, binary
    assert multi >= 1, multi
    assert cat >= 1, cat


def test_small_batch_meets_stratification_minima() -> None:
    tasks = _load_tasks_csv()
    rows = [tasks[t] for t in _load_csv_task_ids(SMALL_FILE)]
    binary = sum(1 for r in rows if r["task_type"] == "binary")
    multi = sum(1 for r in rows if r["task_type"] == "multiclass")
    cat = sum(1 for r in rows if _ti(r["categorical_feature_count"]) > 0)
    imb = sum(1 for r in rows if _tf(r["class_imbalance_ratio"]) >= 5.0)
    # Spec: stratified across binary/multiclass, categorical, balance.
    assert binary >= 4 and multi >= 4
    assert cat >= 3
    assert imb >= 2


def test_representative_batch_meets_documented_coverage_minima() -> None:
    """>=6 binary, >=6 multiclass, >=4 categorical, >=4 imbalanced,
    >=3 large where available."""
    tasks = _load_tasks_csv()
    rows = [tasks[t] for t in _load_csv_task_ids(REP_FILE)]
    binary = sum(1 for r in rows if r["task_type"] == "binary")
    multi = sum(1 for r in rows if r["task_type"] == "multiclass")
    cat = sum(1 for r in rows if _ti(r["categorical_feature_count"]) > 0)
    imb = sum(1 for r in rows if _tf(r["class_imbalance_ratio"]) >= 5.0)
    large = sum(1 for r in rows if _ti(r["n_rows"]) > 30000)
    assert binary >= 6, binary
    assert multi >= 6, multi
    assert cat >= 4, cat
    assert imb >= 4, imb
    # CC18 has 11 large tasks; the representative batch should pick at
    # least 3 of them.
    assert large >= 3, large


# ---------------------------------------------------------------------------
# Synthetic canary + stage0 pointer manifests
# ---------------------------------------------------------------------------


def test_synthetic_canary_manifest_is_well_formed() -> None:
    payload = json.loads(
        (BATCHES_DIR / "batch_00_synthetic_canary.json").read_text()
    )
    assert payload["uses_openml"] is False
    assert sorted(payload["methods"]) == sorted([
        "default_gbdt", "random_search", "tpe_optuna",
        "doe_rsm_vrf_true_nbi",
    ])
    assert payload["task_kind"] == "synthetic_binary"


def test_stage0_shard_pointer_references_committed_shard() -> None:
    payload = json.loads(
        (BATCHES_DIR / "batch_04_stage0_shard00_only.json").read_text()
    )
    rel = payload["source_shard"]
    assert (REPO / rel).exists(), rel
    assert payload["stage"] == "stage0_replica_001"


# ---------------------------------------------------------------------------
# Shard filter: immutability + correctness
# ---------------------------------------------------------------------------


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def test_filter_shard_does_not_mutate_source(tmp_path: Path) -> None:
    """Filtering opens the source via SQLite URI mode=ro, so the
    committed shard's bytes must be identical before and after the
    filter call."""
    from scripts.filter_cc18_shard_for_batch import filter_shard

    before = _md5(SHARD)
    out = tmp_path / "tiny.sqlite"
    filter_shard(
        source=SHARD, out=out,
        task_ids=_load_csv_task_ids(TINY_FILE),
        force=True,
    )
    assert _md5(SHARD) == before


def test_filter_shard_contains_only_requested_task_ids(tmp_path: Path) -> None:
    from scripts.filter_cc18_shard_for_batch import filter_shard

    out = tmp_path / "tiny.sqlite"
    requested = _load_csv_task_ids(TINY_FILE)
    filter_shard(source=SHARD, out=out, task_ids=requested, force=True)
    cx = sqlite3.connect(out)
    found = {t for (t,) in cx.execute("SELECT DISTINCT openml_task_id FROM cc18_jobs")}
    cx.close()
    assert found <= set(requested), f"unexpected ids: {found - set(requested)}"


def test_filter_shard_row_count_is_strictly_lower(tmp_path: Path) -> None:
    from scripts.filter_cc18_shard_for_batch import filter_shard

    out = tmp_path / "tiny.sqlite"
    cx = sqlite3.connect(SHARD)
    n_src = cx.execute("SELECT count(*) FROM cc18_jobs").fetchone()[0]
    cx.close()
    filter_shard(source=SHARD, out=out,
                 task_ids=_load_csv_task_ids(TINY_FILE), force=True)
    cx = sqlite3.connect(out)
    n_dst = cx.execute("SELECT count(*) FROM cc18_jobs").fetchone()[0]
    cx.close()
    assert n_dst < n_src
    assert n_dst >= 0


def test_filter_shard_preserves_cc18_jobs_schema(tmp_path: Path) -> None:
    from scripts.filter_cc18_shard_for_batch import filter_shard

    src_cols = {row[1] for row in
                sqlite3.connect(SHARD).execute("PRAGMA table_info(cc18_jobs)")}
    out = tmp_path / "tiny.sqlite"
    filter_shard(source=SHARD, out=out,
                 task_ids=_load_csv_task_ids(TINY_FILE), force=True)
    dst_cols = {row[1] for row in
                sqlite3.connect(out).execute("PRAGMA table_info(cc18_jobs)")}
    assert dst_cols == src_cols


def test_filter_shard_writes_shard_meta_row(tmp_path: Path) -> None:
    from scripts.filter_cc18_shard_for_batch import filter_shard

    out = tmp_path / "tiny.sqlite"
    filter_shard(source=SHARD, out=out,
                 task_ids=_load_csv_task_ids(TINY_FILE), force=True)
    cx = sqlite3.connect(out)
    n = cx.execute("SELECT count(*) FROM shard_meta").fetchone()[0]
    cx.close()
    assert n == 1


def test_filter_with_method_subset(tmp_path: Path) -> None:
    """Filtering by method should keep only those rows."""
    from scripts.filter_cc18_shard_for_batch import filter_shard

    out = tmp_path / "filtered.sqlite"
    filter_shard(
        source=SHARD, out=out,
        task_ids=_load_csv_task_ids(TINY_FILE),
        methods=["default_gbdt", "random_search"],
        force=True,
    )
    cx = sqlite3.connect(out)
    methods = {m for (m,) in cx.execute("SELECT DISTINCT method FROM cc18_jobs")}
    cx.close()
    assert methods <= {"default_gbdt", "random_search"}


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_filter_cli_help_runs() -> None:
    res = subprocess.run(
        [sys.executable, str(FILTER), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "--source" in res.stdout
    assert "--batch-file" in res.stdout


def test_filter_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    out = tmp_path / "wont_be_written.sqlite"
    res = subprocess.run(
        [sys.executable, str(FILTER),
         "--source", str(SHARD),
         "--batch-file", str(TINY_FILE),
         "--out", str(out), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert not out.exists()


def test_create_cc18_batches_cli_help() -> None:
    res = subprocess.run(
        [sys.executable, str(GEN), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "--out-dir" in res.stdout


def test_create_cc18_batches_force_regenerates_to_same_bytes(tmp_path: Path) -> None:
    """Re-running with --force must produce byte-identical CSVs."""
    out = tmp_path / "out"
    a = subprocess.run(
        [sys.executable, str(GEN), "--out-dir", str(out), "--force"],
        capture_output=True, text=True, check=False,
    )
    assert a.returncode == 0
    sigs_a = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
              for p in out.glob("batch_0*.csv")}
    b = subprocess.run(
        [sys.executable, str(GEN), "--out-dir", str(out), "--force"],
        capture_output=True, text=True, check=False,
    )
    assert b.returncode == 0
    sigs_b = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
              for p in out.glob("batch_0*.csv")}
    assert sigs_a == sigs_b


# ---------------------------------------------------------------------------
# Source shard global immutability check
# ---------------------------------------------------------------------------


def test_committed_source_shard_md5_matches_commit_28_baseline() -> None:
    """The committed stage-0 shard 00 must still hash to the value
    pinned by Commit 28; if this fails, something earlier in the
    pipeline mutated it."""
    assert _md5(SHARD) == "91e7a861ea73daf82694029d6c590e54"
