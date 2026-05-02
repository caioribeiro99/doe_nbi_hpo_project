"""Tests for the OpenML-CC18 SQLite shard generator (Commit 28).

Covers:
- dry-run does not write SQLite files (only summary JSON/MD);
- a tiny mocked panel materializes the expected number of rows;
- the full real-metadata run yields exactly 79,920 rows across the
  four stages (2,304 / 9,216 / 13,680 / 54,720);
- (openml_task_id, algorithm, method, replica) is unique across all
  shards;
- literature-only methods produce zero rows;
- ParEGO rows reference only the 48 task IDs in parego_subset.csv;
- ablation rows have stage only in stage2/stage3;
- stage-3 jobs of tier-1+ methods carry the manual-signoff note;
- shard_meta is populated in every shard.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.generate_cc18_job_shards import (
    ALGORITHMS,
    STAGES,
    deterministic_job_id,
    generate,
    shard_index,
)

REPO = Path(__file__).resolve().parents[2]
CC18_DIR = REPO / "benchmarks" / "doctoral" / "openml_cc18"
TASKS_CSV = CC18_DIR / "tasks.csv"
MATRIX_CSV = CC18_DIR / "method_matrix.csv"
POLICY_CSV = CC18_DIR / "execution_policy.csv"
PAREGO_CSV = CC18_DIR / "parego_subset.csv"
SCHEMA_SQL = REPO / "jobs" / "doctoral" / "openml_cc18" / "schema.sql"
GENERATOR = REPO / "scripts" / "generate_cc18_job_shards.py"


def _real_generate(out_dir: Path, *, dry_run=False, n_shards=10, stages=None):
    return generate(
        tasks_csv=TASKS_CSV,
        matrix_csv=MATRIX_CSV,
        policy_csv=POLICY_CSV,
        parego_csv=PAREGO_CSV,
        schema_sql_path=SCHEMA_SQL,
        out_dir=out_dir,
        n_shards=n_shards,
        stages=stages or STAGES,
        dry_run=dry_run,
        force=True,
    )


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def test_deterministic_job_id_is_stable() -> None:
    a = deterministic_job_id(3, "xgboost", "doe_rsm_vrf_true_nbi", 1)
    b = deterministic_job_id(3, "xgboost", "doe_rsm_vrf_true_nbi", 1)
    assert a == b
    assert len(a) == 16
    assert a != deterministic_job_id(3, "xgboost", "doe_rsm_vrf_true_nbi", 2)
    assert a != deterministic_job_id(3, "lightgbm", "doe_rsm_vrf_true_nbi", 1)


def test_shard_index_is_deterministic_and_in_range() -> None:
    for tid in (3, 6, 11, 167141):
        for alg in ALGORITHMS:
            for n in (1, 2, 5, 10):
                idx = shard_index(tid, alg, n)
                assert 0 <= idx < n
                assert idx == shard_index(tid, alg, n)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_only_summary(tmp_path: Path) -> None:
    res = _real_generate(tmp_path, dry_run=True)
    sqlites = list(tmp_path.rglob("*.sqlite"))
    assert sqlites == []
    assert res.summary_json.exists()
    assert res.summary_md.exists()
    payload = json.loads(res.summary_json.read_text())
    assert payload["dry_run"] is True
    assert payload["total_rows"] == 79_920


# ---------------------------------------------------------------------------
# Tiny synthetic fixture
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def tiny_panel(tmp_path: Path) -> dict[str, Path]:
    """Two tasks × two methods (one full, one literature-only)."""
    tasks = tmp_path / "tasks.csv"
    matrix = tmp_path / "method_matrix.csv"
    policy = tmp_path / "execution_policy.csv"
    parego = tmp_path / "parego_subset.csv"
    out = tmp_path / "shards"

    _write_csv(
        tasks,
        ["openml_task_id","openml_dataset_id","dataset_name","target_name",
         "task_type","n_rows","n_features","n_classes",
         "categorical_feature_count","numeric_feature_count","class_distribution",
         "class_imbalance_ratio","license","version","status","url","notes"],
        [["1","1","alpha","class","binary","100","5","2","0","5","",
          "1.0","Public","1","ok","",""],
         ["2","2","beta","class","binary","200","6","2","1","5","",
          "1.0","Public","1","ok","",""]],
    )
    _write_csv(
        matrix,
        ["method_id","method_family","primary_or_ablation","objective_mode",
         "implementation","package","full_cc18","subset_only","budget_unit",
         "budget_equivalence_rule","supports_multiclass",
         "supports_categorical_native","notes"],
        [["m_full","x","primary","single_objective","x","x","true","false",
          "evaluations","B","true","encoded","x"],
         ["m_lit","automl_context","literature_only","single_objective",
          "x","x","false","false","seconds","not_in_comparison",
          "true","encoded","x"]],
    )
    _write_csv(
        policy,
        ["method_id","execution_tier","run_scope","replica_policy","stage0",
         "stage1_topup_to_005","stage2_topup_to_010","stage3_topup_to_030",
         "requires_manual_signoff_before_stage3","notes"],
        [["m_full","tier1","full_cc18","full_30","true","true","true","true",
          "true","x"],
         ["m_lit","tier_inf","not_in_comparison","none","false","false",
          "false","false","false","x"]],
    )
    _write_csv(parego, ["openml_task_id"], [])
    return {
        "tasks": tasks, "matrix": matrix, "policy": policy,
        "parego": parego, "out": out,
    }


def test_tiny_fixture_with_two_shards(tiny_panel: dict[str, Path]) -> None:
    res = generate(
        tasks_csv=tiny_panel["tasks"],
        matrix_csv=tiny_panel["matrix"],
        policy_csv=tiny_panel["policy"],
        parego_csv=tiny_panel["parego"],
        schema_sql_path=SCHEMA_SQL,
        out_dir=tiny_panel["out"],
        n_shards=2,
        stages=STAGES,
        dry_run=False,
        force=True,
    )
    # 2 tasks × 3 algs × 1 method × (1+4+5+20)=30 replicas = 180 rows.
    assert res.counts.total_rows == 180
    assert res.counts.literature_rows == 0
    # 4 stages × 2 shards = 8 sqlite files.
    assert len(res.shard_paths) == 8
    sqlites = list(tiny_panel["out"].rglob("*.sqlite"))
    assert len(sqlites) == 8


# ---------------------------------------------------------------------------
# Full real-metadata run (write to tmpdir)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("cc18_shards")
    res = _real_generate(out, dry_run=False, n_shards=10)
    return out, json.loads(res.summary_json.read_text())


def test_real_total_row_count(real_run) -> None:
    _, payload = real_run
    assert payload["total_rows"] == 79_920


def test_real_per_stage_counts(real_run) -> None:
    _, payload = real_run
    assert payload["rows_by_stage"]["stage0_replica_001"] == 2_304
    assert payload["rows_by_stage"]["stage1_topup_to_005"] == 9_216
    assert payload["rows_by_stage"]["stage2_topup_to_010"] == 13_680
    assert payload["rows_by_stage"]["stage3_topup_to_030"] == 54_720


def test_real_no_literature_rows(real_run) -> None:
    _, payload = real_run
    assert payload["literature_rows"] == 0
    for mid in ("flaml_optional", "auto_sklearn_context", "autogluon_context"):
        assert payload["rows_by_method"].get(mid, 0) == 0


def test_real_unique_natural_key_across_shards(real_run) -> None:
    out, _ = real_run
    seen: set[tuple[int, str, str, int]] = set()
    total = 0
    for sq in out.rglob("*.sqlite"):
        with sqlite3.connect(sq) as cx:
            rows = cx.execute(
                "SELECT openml_task_id, algorithm, method, replica FROM cc18_jobs"
            ).fetchall()
            total += len(rows)
            seen.update(rows)
    assert total == 79_920
    assert len(seen) == 79_920


def test_real_parego_uses_only_subset_task_ids(real_run) -> None:
    out, _ = real_run
    with PAREGO_CSV.open() as f:
        subset = {int(r["openml_task_id"]) for r in csv.DictReader(f)}
    parego_task_ids: set[int] = set()
    for sq in out.rglob("*.sqlite"):
        with sqlite3.connect(sq) as cx:
            for tid, in cx.execute(
                "SELECT openml_task_id FROM cc18_jobs WHERE method='parego'"
            ).fetchall():
                parego_task_ids.add(tid)
    assert parego_task_ids
    assert parego_task_ids <= subset


def test_real_ablation_rows_only_at_stage_2_or_3(real_run) -> None:
    out, _ = real_run
    abl_methods = ("doe_rsm_vrf_true_nbi_no_mbpa",
                   "legacy_weighted_sum_scalarization")
    bad_stages: set[str] = set()
    for sq in out.rglob("*.sqlite"):
        with sqlite3.connect(sq) as cx:
            for stage, in cx.execute(
                "SELECT DISTINCT stage FROM cc18_jobs WHERE method IN (?, ?)",
                abl_methods,
            ).fetchall():
                if stage not in {"stage2_topup_to_010", "stage3_topup_to_030"}:
                    bad_stages.add(stage)
    assert not bad_stages, bad_stages


def test_real_stage3_signoff_notes_applied(real_run) -> None:
    out, payload = real_run
    # Tier-1+ methods (every benchmarked method except tier 0 controls
    # default_gbdt and random_search) must carry the manual-signoff note
    # on stage-3 rows.
    by_method: dict[str, int] = {}
    for sq in out.rglob("*.sqlite"):
        with sqlite3.connect(sq) as cx:
            for method, n in cx.execute(
                "SELECT method, count(*) FROM cc18_jobs "
                "WHERE stage='stage3_topup_to_030' "
                "AND notes='requires_manual_signoff_before_stage3' "
                "GROUP BY method"
            ).fetchall():
                by_method[method] = by_method.get(method, 0) + n
    # Tier-0 controls must NOT carry the note.
    assert "default_gbdt" not in by_method
    assert "random_search" not in by_method
    # Tier-1+ methods must carry the note on every stage-3 row.
    expected_methods = {
        "tpe_optuna", "smac3", "asha", "bohb", "dehb", "nsga2", "motpe",
        "doe_rsm_vrf_true_nbi", "doe_rsm_vrf_true_nbi_no_mbpa",
        "legacy_weighted_sum_scalarization", "parego",
    }
    assert set(by_method.keys()) == expected_methods
    # Total stage-3 sign-off rows match the summary JSON.
    assert sum(by_method.values()) == payload["stage3_signoff_rows"]


def test_real_shard_meta_populated(real_run) -> None:
    out, _ = real_run
    files = sorted(out.rglob("*.sqlite"))
    assert len(files) == 40  # 4 stages × 10 shards
    for sq in files:
        with sqlite3.connect(sq) as cx:
            n = cx.execute("SELECT count(*) FROM shard_meta").fetchone()[0]
            assert n == 1, sq
            row = cx.execute(
                "SELECT suite_id, panel_version, n_algorithms FROM shard_meta"
            ).fetchone()
            assert row[0] == 99
            assert row[1] == "cc18_v1"
            assert row[2] == 3


def test_real_status_defaults_to_pending(real_run) -> None:
    out, _ = real_run
    for sq in out.rglob("*.sqlite"):
        with sqlite3.connect(sq) as cx:
            distinct = {
                s for (s,) in cx.execute(
                    "SELECT DISTINCT status FROM cc18_jobs"
                ).fetchall()
            }
            assert distinct in ({"pending"}, set()), (sq, distinct)


def test_real_committed_shards_match_projection() -> None:
    """The shards committed to the repo match the projection. Cheap sanity
    check that the on-disk artifacts haven't drifted from the freeze."""
    committed_dir = REPO / "jobs" / "doctoral" / "openml_cc18" / "shards"
    if not committed_dir.exists():
        pytest.skip("committed shards not present; run the generator first")
    files = sorted(committed_dir.rglob("*.sqlite"))
    if not files:
        pytest.skip("no committed shards")
    total = 0
    for sq in files:
        with sqlite3.connect(sq) as cx:
            total += cx.execute("SELECT count(*) FROM cc18_jobs").fetchone()[0]
    assert total == 79_920


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_help_runs() -> None:
    res = subprocess.run(
        [sys.executable, str(GENERATOR), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    assert "shards" in res.stdout.lower()


def test_cli_dry_run_succeeds(tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(GENERATOR),
         "--out-dir", str(tmp_path), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "79920" in res.stdout
    sqlites = list(tmp_path.rglob("*.sqlite"))
    assert sqlites == []
