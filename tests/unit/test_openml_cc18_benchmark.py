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
METHOD_MATRIX_CSV = CC18_DIR / "method_matrix.csv"
EXECUTION_POLICY_CSV = CC18_DIR / "execution_policy.csv"
PAREGO_SUBSET_CSV = CC18_DIR / "parego_subset.csv"


METHOD_MATRIX_REQUIRED_COLUMNS = (
    "method_id",
    "method_family",
    "primary_or_ablation",
    "objective_mode",
    "implementation",
    "package",
    "full_cc18",
    "subset_only",
    "budget_unit",
    "budget_equivalence_rule",
    "supports_multiclass",
    "supports_categorical_native",
    "notes",
)

EXECUTION_POLICY_REQUIRED_COLUMNS = (
    "method_id",
    "execution_tier",
    "run_scope",
    "replica_policy",
    "stage0",
    "stage1_topup_to_005",
    "stage2_topup_to_010",
    "stage3_topup_to_030",
    "requires_manual_signoff_before_stage3",
    "notes",
)

EXPECTED_PAREGO_SUBSET_SIZE = 48


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


# ---------------------------------------------------------------------------
# method_matrix.csv (Commit 27 freeze)
# ---------------------------------------------------------------------------


def _bool(v: str) -> bool:
    if v.strip().lower() in {"true", "false"}:
        return v.strip().lower() == "true"
    raise AssertionError(f"non-boolean: {v!r}")


def _load_method_matrix() -> list[dict[str, str]]:
    with METHOD_MATRIX_CSV.open() as f:
        return list(csv.DictReader(f))


def _load_execution_policy() -> list[dict[str, str]]:
    with EXECUTION_POLICY_CSV.open() as f:
        return list(csv.DictReader(f))


def test_method_matrix_csv_parses() -> None:
    rows = _load_method_matrix()
    assert len(rows) >= 13


def test_method_matrix_csv_has_required_columns() -> None:
    rows = _load_method_matrix()
    cols = set(rows[0].keys())
    missing = [c for c in METHOD_MATRIX_REQUIRED_COLUMNS if c not in cols]
    assert not missing, f"missing columns: {missing}"


def test_method_matrix_method_ids_are_unique() -> None:
    rows = _load_method_matrix()
    ids = [r["method_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_method_matrix_full_and_subset_are_booleans() -> None:
    rows = _load_method_matrix()
    for r in rows:
        _bool(r["full_cc18"])
        _bool(r["subset_only"])


def test_method_matrix_no_method_is_both_full_and_subset() -> None:
    rows = _load_method_matrix()
    for r in rows:
        if _bool(r["full_cc18"]) and _bool(r["subset_only"]):
            raise AssertionError(
                f"{r['method_id']!r} is both full_cc18 and subset_only"
            )


def test_method_matrix_literature_only_is_not_full_cc18() -> None:
    rows = _load_method_matrix()
    for r in rows:
        if r["primary_or_ablation"] == "literature_only":
            assert not _bool(r["full_cc18"]), r["method_id"]
            assert not _bool(r["subset_only"]), r["method_id"]


def test_method_matrix_benchmarked_methods_have_budget_rule() -> None:
    rows = _load_method_matrix()
    for r in rows:
        if r["primary_or_ablation"] == "literature_only":
            continue
        assert r["budget_equivalence_rule"], r["method_id"]
        assert r["budget_equivalence_rule"] != "not_in_comparison", (
            f"{r['method_id']!r} is benchmarked but its budget rule "
            "is 'not_in_comparison'"
        )


def test_method_matrix_freeze_decisions_applied() -> None:
    """Commit 27 freeze: rename hyperband_or_asha -> asha; FLAML stays
    literature_only."""
    rows = _load_method_matrix()
    by_id = {r["method_id"]: r for r in rows}
    assert "asha" in by_id
    assert "hyperband_or_asha" not in by_id
    assert by_id["flaml_optional"]["primary_or_ablation"] == "literature_only"


# ---------------------------------------------------------------------------
# execution_policy.csv
# ---------------------------------------------------------------------------


def test_execution_policy_csv_parses() -> None:
    rows = _load_execution_policy()
    assert rows


def test_execution_policy_csv_has_required_columns() -> None:
    rows = _load_execution_policy()
    cols = set(rows[0].keys())
    missing = [c for c in EXECUTION_POLICY_REQUIRED_COLUMNS if c not in cols]
    assert not missing, f"missing columns: {missing}"


def test_execution_policy_covers_every_method_in_matrix() -> None:
    mm_ids = {r["method_id"] for r in _load_method_matrix()}
    ep_ids = {r["method_id"] for r in _load_execution_policy()}
    missing = mm_ids - ep_ids
    extra = ep_ids - mm_ids
    assert not missing, f"execution_policy missing rows for: {missing}"
    assert not extra, f"execution_policy has extra rows for: {extra}"


def test_execution_policy_benchmarked_methods_have_at_least_one_stage_true() -> None:
    mm_by = {r["method_id"]: r for r in _load_method_matrix()}
    for r in _load_execution_policy():
        mid = r["method_id"]
        if mm_by[mid]["primary_or_ablation"] == "literature_only":
            continue
        flags = [_bool(r[s]) for s in (
            "stage0", "stage1_topup_to_005",
            "stage2_topup_to_010", "stage3_topup_to_030",
        )]
        assert any(flags), f"{mid!r} has no stage enabled"


def test_execution_policy_literature_only_methods_run_no_stage() -> None:
    mm_by = {r["method_id"]: r for r in _load_method_matrix()}
    for r in _load_execution_policy():
        mid = r["method_id"]
        if mm_by[mid]["primary_or_ablation"] != "literature_only":
            continue
        for s in ("stage0", "stage1_topup_to_005",
                  "stage2_topup_to_010", "stage3_topup_to_030"):
            assert not _bool(r[s]), f"{mid!r} runs stage {s}"
        assert r["run_scope"] == "not_in_comparison"


def test_execution_policy_run_scope_values_are_valid() -> None:
    valid = {"full_cc18", "parego_subset", "not_in_comparison"}
    for r in _load_execution_policy():
        assert r["run_scope"] in valid, (r["method_id"], r["run_scope"])


# ---------------------------------------------------------------------------
# parego_subset.csv
# ---------------------------------------------------------------------------


def test_parego_subset_csv_size_matches_rule() -> None:
    with PAREGO_SUBSET_CSV.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == EXPECTED_PAREGO_SUBSET_SIZE


def test_parego_subset_task_ids_are_subset_of_tasks_csv() -> None:
    with PAREGO_SUBSET_CSV.open() as f:
        sub_ids = {int(r["openml_task_id"]) for r in csv.DictReader(f)}
    with TASKS_CSV.open() as f:
        all_ids = {int(r["openml_task_id"]) for r in csv.DictReader(f)}
    assert sub_ids <= all_ids
    assert len(sub_ids) == EXPECTED_PAREGO_SUBSET_SIZE


def test_parego_subset_obeys_selection_rule() -> None:
    """imbalance >= 5 OR n_classes >= 5 OR (categorical > 0 AND
    5000 <= n_rows <= 50000)."""
    with TASKS_CSV.open() as f:
        all_rows = {int(r["openml_task_id"]): r for r in csv.DictReader(f)}
    with PAREGO_SUBSET_CSV.open() as f:
        for r in csv.DictReader(f):
            tid = int(r["openml_task_id"])
            t = all_rows[tid]
            imb = float(t["class_imbalance_ratio"]) if t["class_imbalance_ratio"] else None
            n_cls = int(t["n_classes"]) if t["n_classes"] else None
            cat = int(t["categorical_feature_count"] or "0")
            n_rows_v = int(t["n_rows"]) if t["n_rows"] else None
            ok = (
                (imb is not None and imb >= 5.0)
                or (n_cls is not None and n_cls >= 5)
                or (cat > 0 and n_rows_v is not None
                    and 5000 <= n_rows_v <= 50000)
            )
            assert ok, f"task {tid} in subset but does not match rule"


# ---------------------------------------------------------------------------
# Job-count projection
# ---------------------------------------------------------------------------


def _stage_replicas() -> dict[str, int]:
    return {
        "stage0": 1,
        "stage1_topup_to_005": 4,
        "stage2_topup_to_010": 5,
        "stage3_topup_to_030": 20,
    }


def _run_scope_size(scope: str) -> int:
    if scope == "full_cc18":
        return EXPECTED_N_TASKS
    if scope == "parego_subset":
        return EXPECTED_PAREGO_SUBSET_SIZE
    return 0


def _projected_jobs() -> dict[str, int]:
    reps = _stage_replicas()
    totals = {s: 0 for s in reps}
    for r in _load_execution_policy():
        nt = _run_scope_size(r["run_scope"])
        for s in reps:
            if _bool(r[s]):
                totals[s] += nt * EXPECTED_N_ALGORITHMS * reps[s]
    return totals


def test_stage_job_counts_match_projection_doc() -> None:
    totals = _projected_jobs()
    assert totals["stage0"] == 2_304
    assert totals["stage1_topup_to_005"] == 9_216
    assert totals["stage2_topup_to_010"] == 13_680
    assert totals["stage3_topup_to_030"] == 54_720


def test_cumulative_job_count_at_stage3_matches_projection_doc() -> None:
    totals = _projected_jobs()
    cum = sum(totals.values())
    assert cum == 79_920


def test_full_cc18_method_count_matches_documented_partition() -> None:
    rows = _load_method_matrix()
    n_full = sum(1 for r in rows if _bool(r["full_cc18"]))
    n_subset = sum(1 for r in rows if _bool(r["subset_only"]))
    n_literature = sum(
        1 for r in rows if r["primary_or_ablation"] == "literature_only"
    )
    # The matrix partitions into exactly these three buckets (the literature
    # bucket has both flags false; the other two are mutually exclusive).
    assert n_full == 12
    assert n_subset == 1
    assert n_literature == 3
    assert n_full + n_subset + n_literature == len(rows)
