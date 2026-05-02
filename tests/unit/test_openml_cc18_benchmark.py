"""Tests for the OpenML-CC18 doctoral benchmark reframe (Commit 25).

Covers:
- tasks.csv / datasets.csv / metadata.json structural invariants;
- the 72-task constraint and uniqueness of openml_task_id;
- new openml_cc18_72_tasks_* cost-estimator presets;
- DeprecationWarning emitted when resolving doctoral_82_* aliases;
- job-count arithmetic 72 x 3 x 30 = 6,480 with staged 216/864/1,080/4,320;
- the SQLite cc18_jobs schema is well-formed (sqlite3 can apply it);
- the importer's --validate-only flag passes on the committed tasks.csv.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from doe_xgb.cost_estimator import (
    DEPRECATED_PRESETS,
    PRESETS,
    get_preset,
)

REPO = Path(__file__).resolve().parents[2]
CC18_DIR = REPO / "benchmarks" / "doctoral" / "openml_cc18"
TASKS_CSV = CC18_DIR / "tasks.csv"
DATASETS_CSV = CC18_DIR / "datasets.csv"
METADATA_JSON = CC18_DIR / "openml_cc18_metadata.json"
SCHEMA_SQL = REPO / "jobs" / "doctoral" / "openml_cc18" / "schema.sql"
IMPORTER = REPO / "scripts" / "import_openml_cc18.py"
INTERNAL_PANEL_CSV = (
    REPO / "benchmarks" / "doctoral" / "internal_smoke_panel" / "datasets.csv"
)


EXPECTED_N_TASKS = 72
EXPECTED_N_ALGORITHMS = 3
EXPECTED_N_METHODS = 1
EXPECTED_REPLICAS_FULL = 30
EXPECTED_TOTAL_JOBS_FULL = (
    EXPECTED_N_TASKS * EXPECTED_N_ALGORITHMS * EXPECTED_N_METHODS * EXPECTED_REPLICAS_FULL
)


# ---------------------------------------------------------------------------
# tasks.csv / datasets.csv / metadata.json
# ---------------------------------------------------------------------------


def _load_task_rows() -> list[dict[str, str]]:
    with TASKS_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_tasks_csv_has_exactly_72_rows() -> None:
    rows = _load_task_rows()
    assert len(rows) == EXPECTED_N_TASKS


def test_tasks_csv_task_ids_are_unique_ints() -> None:
    rows = _load_task_rows()
    ids = [int(r["openml_task_id"]) for r in rows]
    assert len(ids) == len(set(ids)) == EXPECTED_N_TASKS


def test_tasks_csv_task_type_is_binary_or_multiclass() -> None:
    rows = _load_task_rows()
    types = {r["task_type"] for r in rows}
    assert types <= {"binary", "multiclass"}
    # Both classes should be represented.
    assert "binary" in types and "multiclass" in types


def test_tasks_csv_has_required_columns() -> None:
    rows = _load_task_rows()
    required = {
        "openml_task_id", "openml_dataset_id", "dataset_name", "target_name",
        "task_type", "n_rows", "n_features", "n_classes",
        "categorical_feature_count", "numeric_feature_count",
        "class_imbalance_ratio", "license", "version", "status", "url",
    }
    assert required.issubset(rows[0].keys())


def test_datasets_csv_one_row_per_unique_dataset() -> None:
    with DATASETS_CSV.open("r", encoding="utf-8") as f:
        ds_rows = list(csv.DictReader(f))
    ds_ids = [int(r["openml_dataset_id"]) for r in ds_rows]
    assert len(ds_ids) == len(set(ds_ids))
    # Every dataset id in datasets.csv is referenced by at least one task.
    task_rows = _load_task_rows()
    ref_ids = {int(r["openml_dataset_id"]) for r in task_rows}
    assert set(ds_ids) == ref_ids


def test_datasets_csv_n_tasks_in_cc18_sums_to_72() -> None:
    with DATASETS_CSV.open("r", encoding="utf-8") as f:
        ds_rows = list(csv.DictReader(f))
    total = sum(int(r["n_tasks_in_cc18"]) for r in ds_rows)
    assert total == EXPECTED_N_TASKS


def test_metadata_json_records_suite_99_and_72_tasks() -> None:
    meta = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    assert meta["suite_id"] == 99
    assert meta["n_tasks"] == EXPECTED_N_TASKS
    assert len(meta["task_ids"]) == EXPECTED_N_TASKS
    assert len(set(meta["task_ids"])) == EXPECTED_N_TASKS


# ---------------------------------------------------------------------------
# Internal smoke panel (12 datasets) is separate from CC18
# ---------------------------------------------------------------------------


def test_internal_smoke_panel_has_12_rows() -> None:
    with INTERNAL_PANEL_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # The panel ships at least the 12 v1 datasets seeded with include=true.
    included = [r for r in rows if r.get("include", "").strip().lower() in {"true", "1"}]
    assert len(included) >= 12


# ---------------------------------------------------------------------------
# Cost-estimator presets
# ---------------------------------------------------------------------------


def test_openml_cc18_presets_resolve_to_72_tasks() -> None:
    expected_keys = {
        "openml_cc18_72_tasks_3_algorithms_1_replicas",
        "openml_cc18_72_tasks_3_algorithms_5_replicas",
        "openml_cc18_72_tasks_3_algorithms_10_replicas",
        "openml_cc18_72_tasks_3_algorithms_30_replicas",
    }
    assert expected_keys.issubset(PRESETS.keys())
    for r in (1, 5, 10, 30):
        spec = PRESETS[f"openml_cc18_72_tasks_3_algorithms_{r}_replicas"]
        assert spec.n_datasets == EXPECTED_N_TASKS
        assert spec.n_algorithms == EXPECTED_N_ALGORITHMS
        assert spec.n_replicas == r


def test_doctoral_82_presets_are_marked_deprecated() -> None:
    expected = {
        "doctoral_82_datasets_3_algorithms_1_replicas",
        "doctoral_82_datasets_3_algorithms_5_replicas",
        "doctoral_82_datasets_3_algorithms_10_replicas",
        "doctoral_82_datasets_3_algorithms_30_replicas",
    }
    assert expected.issubset(set(DEPRECATED_PRESETS))
    # And they still resolve via PRESETS so old call sites do not break.
    assert expected.issubset(PRESETS.keys())


def test_get_preset_emits_deprecationwarning_for_doctoral_82() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = get_preset("doctoral_82_datasets_3_algorithms_30_replicas")
    assert spec.n_datasets == 82  # the alias still preserves its spec
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "openml_cc18" in str(w.message).lower()
        for w in caught
    ), [str(w.message) for w in caught]


def test_get_preset_does_not_warn_for_cc18_presets() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_preset("openml_cc18_72_tasks_3_algorithms_30_replicas")
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


# ---------------------------------------------------------------------------
# Job-count arithmetic for the headline doctoral target
# ---------------------------------------------------------------------------


def test_full_doctoral_job_count_is_6480() -> None:
    assert EXPECTED_TOTAL_JOBS_FULL == 6_480


def test_staged_topup_job_counts_match() -> None:
    # Stage cumulative counts from docs/DOCTORAL_BENCHMARK.md.
    per_unit = EXPECTED_N_TASKS * EXPECTED_N_ALGORITHMS * EXPECTED_N_METHODS  # 216
    stage_replica_counts = {
        "stage0_replica_001": 1,
        "stage1_topup_to_005": 4,
        "stage2_topup_to_010": 5,
        "stage3_topup_to_030": 20,
    }
    added = {s: per_unit * r for s, r in stage_replica_counts.items()}
    assert added == {
        "stage0_replica_001": 216,
        "stage1_topup_to_005": 864,
        "stage2_topup_to_010": 1_080,
        "stage3_topup_to_030": 4_320,
    }
    cumulative = []
    running = 0
    for s in (
        "stage0_replica_001",
        "stage1_topup_to_005",
        "stage2_topup_to_010",
        "stage3_topup_to_030",
    ):
        running += added[s]
        cumulative.append(running)
    assert cumulative == [216, 1_080, 2_160, 6_480]


# ---------------------------------------------------------------------------
# SQLite schema is well-formed
# ---------------------------------------------------------------------------


def test_cc18_jobs_schema_applies_in_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "cc18_jobs.sqlite"
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    with sqlite3.connect(db) as cx:
        cx.executescript(sql)
        # Required tables exist.
        names = {
            row[0]
            for row in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"cc18_jobs", "shard_meta"}.issubset(names)
        # Insert + uniqueness probe.
        cx.execute(
            "INSERT INTO cc18_jobs (job_id, openml_task_id, openml_dataset_id, "
            "dataset_name, algorithm, method, replica, stage, config_path, "
            "output_dir) VALUES "
            "('aaaa', 3, 3, 'kr-vs-kp', 'xgboost', 'doe_nbi', 1, "
            "'stage0_replica_001', 'cfg.yaml', 'out/')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            cx.execute(
                "INSERT INTO cc18_jobs (job_id, openml_task_id, openml_dataset_id, "
                "dataset_name, algorithm, method, replica, stage, config_path, "
                "output_dir) VALUES "
                "('bbbb', 3, 3, 'kr-vs-kp', 'xgboost', 'doe_nbi', 1, "
                "'stage0_replica_001', 'cfg.yaml', 'out/')"
            )


# ---------------------------------------------------------------------------
# Importer CLI surface
# ---------------------------------------------------------------------------


def test_importer_help_runs() -> None:
    res = subprocess.run(
        [sys.executable, str(IMPORTER), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "openml" in res.stdout.lower() or "suite" in res.stdout.lower()


def test_importer_validate_only_accepts_committed_tasks_csv() -> None:
    res = subprocess.run(
        [sys.executable, str(IMPORTER), "--validate-only",
         "--out-dir", str(CC18_DIR)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "72" in res.stdout
