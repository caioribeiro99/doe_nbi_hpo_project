"""Unit tests for the doctoral benchmark registry, job-matrix planner,
and multi-machine cost profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from doe_xgb.cost_estimator import (
    PRESETS,
    MachineProfile,
    MultiMachineProfile,
    caio_mac_profile,
    dedicated_mac_profile,
    two_macs_combined,
)
from doe_xgb.doctoral_benchmark import (
    DatasetRow,
    RegistryError,
    canonical_row,
    generate_job_rows,
    job_id,
    load_registry_csv,
    merge_registries,
    stage_topup_replicas,
    validate_registry,
)
from doe_xgb.doctoral_benchmark.jobs import STAGE_NAMES
from doe_xgb.doctoral_benchmark.registry import write_registry_csv

REPO = Path(__file__).resolve().parents[2]
REGISTRY_CSV = REPO / "benchmarks" / "doctoral_82" / "datasets.csv"


# ---------------------------------------------------------------------------
# Registry CSV
# ---------------------------------------------------------------------------


def test_committed_registry_loads_and_validates() -> None:
    rows = load_registry_csv(REGISTRY_CSV)
    validate_registry(rows)
    # The 12 v1 datasets must be seeded with include=True.
    by_id = {r.dataset_id: r for r in rows}
    expected = {
        "magic", "breast_cancer", "pima_diabetes", "spambase",
        "adult", "bank_marketing", "credit_card_default", "german_credit",
        "wine_quality", "dry_bean", "mushroom", "phishing",
    }
    assert expected.issubset(by_id.keys())
    for did in expected:
        assert by_id[did].include is True
        assert by_id[did].loader_status == "registered"


def test_dry_bean_is_only_multiclass_in_committed_registry() -> None:
    rows = load_registry_csv(REGISTRY_CSV)
    multiclass = [r for r in rows if r.task_type == "multiclass"]
    assert [r.dataset_id for r in multiclass] == ["dry_bean"]


def test_canonical_row_rejects_invalid_source() -> None:
    raw = {
        "dataset_id": "x", "display_name": "X", "source": "bogus",
        "task_type": "binary", "include": "True", "loader_status": "registered",
    }
    with pytest.raises(RegistryError, match="invalid source"):
        canonical_row(raw)


def test_canonical_row_rejects_invalid_task_type() -> None:
    raw = {
        "dataset_id": "x", "display_name": "X", "source": "uci",
        "task_type": "regression", "include": "True", "loader_status": "registered",
    }
    with pytest.raises(RegistryError, match="invalid task_type"):
        canonical_row(raw)


def test_canonical_row_rejects_include_true_with_pending_loader() -> None:
    raw = {
        "dataset_id": "x", "display_name": "X", "source": "uci",
        "task_type": "binary", "include": "True", "loader_status": "pending",
    }
    with pytest.raises(RegistryError, match="loader_status=registered"):
        canonical_row(raw)


def test_canonical_row_rejects_dataset_id_with_spaces() -> None:
    raw = {
        "dataset_id": "bad id", "display_name": "X", "source": "uci",
        "task_type": "binary", "include": "False", "loader_status": "pending",
    }
    with pytest.raises(RegistryError, match="must not contain spaces"):
        canonical_row(raw)


def test_canonical_row_rejects_binary_with_three_classes() -> None:
    raw = {
        "dataset_id": "x", "display_name": "X", "source": "uci",
        "task_type": "binary", "include": "False", "loader_status": "pending",
        "n_classes": "3",
    }
    with pytest.raises(RegistryError, match="task_type=binary requires"):
        canonical_row(raw)


def test_validate_registry_detects_duplicate_dataset_id(tmp_path: Path) -> None:
    a = DatasetRow(dataset_id="dup", display_name="A", source="uci",
                   task_type="binary", include=False, loader_status="pending")
    b = DatasetRow(dataset_id="dup", display_name="B", source="openml",
                   task_type="binary", include=False, loader_status="pending")
    with pytest.raises(RegistryError, match="duplicate dataset_id"):
        validate_registry([a, b])


def test_validate_registry_detects_duplicate_openml_id() -> None:
    a = DatasetRow(dataset_id="alpha", display_name="A", source="openml",
                   openml_id=42, task_type="binary",
                   include=False, loader_status="pending")
    b = DatasetRow(dataset_id="beta", display_name="B", source="openml",
                   openml_id=42, task_type="binary",
                   include=False, loader_status="pending")
    with pytest.raises(RegistryError, match="duplicate openml_id"):
        validate_registry([a, b])


def test_merge_registries_is_additive_by_default() -> None:
    base = [DatasetRow(dataset_id="a", display_name="A", source="uci",
                       task_type="binary", include=False, loader_status="pending")]
    incoming = [DatasetRow(dataset_id="a", display_name="A_OVERWRITE", source="uci",
                           task_type="binary", include=False, loader_status="pending"),
                DatasetRow(dataset_id="b", display_name="B", source="uci",
                           task_type="binary", include=False, loader_status="pending")]
    merged = {r.dataset_id: r for r in merge_registries(base, incoming)}
    # Base preserved, b added.
    assert merged["a"].display_name == "A"
    assert "b" in merged


def test_merge_registries_overwrite_flag() -> None:
    base = [DatasetRow(dataset_id="a", display_name="A", source="uci",
                       task_type="binary", include=False, loader_status="pending")]
    incoming = [DatasetRow(dataset_id="a", display_name="A_NEW", source="uci",
                           task_type="binary", include=False, loader_status="pending")]
    merged = {r.dataset_id: r for r in merge_registries(base, incoming, overwrite_existing=True)}
    assert merged["a"].display_name == "A_NEW"


def test_write_registry_csv_round_trip(tmp_path: Path) -> None:
    rows = load_registry_csv(REGISTRY_CSV)
    out = tmp_path / "out.csv"
    write_registry_csv(rows, out)
    again = load_registry_csv(out)
    assert {r.dataset_id for r in rows} == {r.dataset_id for r in again}


# ---------------------------------------------------------------------------
# Job matrix
# ---------------------------------------------------------------------------


def test_job_id_is_deterministic() -> None:
    a = job_id(dataset_id="magic", algorithm="xgboost", method="doe_nbi", replica=1)
    b = job_id(dataset_id="magic", algorithm="xgboost", method="doe_nbi", replica=1)
    assert a == b
    assert len(a) == 16
    # Different inputs => different ids.
    c = job_id(dataset_id="magic", algorithm="xgboost", method="doe_nbi", replica=2)
    assert a != c


def test_stage_topup_replicas_partitions_1_to_30() -> None:
    seen: list[int] = []
    for stage in STAGE_NAMES:
        lo, hi = stage_topup_replicas(stage)
        seen.extend(range(lo, hi + 1))
    assert seen == list(range(1, 31))


def test_stage_topup_replicas_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        stage_topup_replicas("stage99")


def test_generate_job_rows_only_includes_active_datasets() -> None:
    rows = [
        DatasetRow(dataset_id="active", display_name="A", source="uci",
                   task_type="binary", include=True, loader_status="registered"),
        DatasetRow(dataset_id="excluded", display_name="E", source="uci",
                   task_type="binary", include=False, loader_status="pending"),
    ]
    jobs = generate_job_rows(
        datasets=rows,
        algorithms=["xgboost"],
        methods=["doe_nbi"],
    )
    dataset_ids = {j.dataset_id for j in jobs}
    assert dataset_ids == {"active"}
    # 1 dataset x 1 algo x 1 method x 30 replicas = 30 jobs across 4 stages.
    assert len(jobs) == 30


def test_generate_job_rows_total_for_target_panel() -> None:
    rows = [
        DatasetRow(dataset_id=f"d{i:03d}", display_name=f"D{i}", source="uci",
                   task_type="binary", include=True, loader_status="registered")
        for i in range(82)
    ]
    jobs = generate_job_rows(
        datasets=rows,
        algorithms=["xgboost", "lightgbm", "catboost"],
        methods=["doe_nbi"],
    )
    # Expected: 82 datasets * 3 algorithms * 1 method * 30 replicas = 7,380.
    assert len(jobs) == 82 * 3 * 1 * 30
    # Stage breakdown.
    counts = {stage: 0 for stage in STAGE_NAMES}
    for j in jobs:
        counts[j.stage] += 1
    assert counts["stage0_replica_001"] == 82 * 3
    assert counts["stage1_topup_to_005"] == 82 * 3 * 4
    assert counts["stage2_topup_to_010"] == 82 * 3 * 5
    assert counts["stage3_topup_to_030"] == 82 * 3 * 20


def test_generate_job_rows_have_unique_job_ids() -> None:
    rows = load_registry_csv(REGISTRY_CSV)
    jobs = generate_job_rows(
        datasets=rows,
        algorithms=["xgboost", "lightgbm", "catboost"],
        methods=["doe_nbi"],
    )
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Multi-machine cost profiles
# ---------------------------------------------------------------------------


def test_dedicated_mac_default_is_85_percent() -> None:
    p = dedicated_mac_profile()
    assert p.workers == 10
    assert p.hours_per_day == 24.0
    assert p.efficiency_factor == pytest.approx(0.85)
    assert p.model_n_jobs == 1


def test_dedicated_mac_efficiency_scenarios_distinct() -> None:
    cons = dedicated_mac_profile(0.75)
    realistic = dedicated_mac_profile(0.85)
    optimistic = dedicated_mac_profile(0.90)
    assert cons.efficiency_factor < realistic.efficiency_factor < optimistic.efficiency_factor
    def cph(p: MachineProfile) -> float:
        return p.workers * p.hours_per_day * p.efficiency_factor

    assert cph(cons) < cph(realistic) < cph(optimistic)


def test_caio_mac_default_is_70_percent() -> None:
    p = caio_mac_profile()
    assert p.workers == 6
    assert p.efficiency_factor == pytest.approx(0.70)


def test_two_macs_combined_aggregates_daily_cpu_hours() -> None:
    combined = two_macs_combined(dedicated_efficiency=0.85,
                                 caio_efficiency=0.70,
                                 caio_hours_per_day=14.0)
    assert isinstance(combined, MultiMachineProfile)
    # 10 * 24 * 0.85 + 6 * 14 * 0.70 = 204 + 58.8 = 262.8
    assert combined.daily_cpu_hours() == pytest.approx(204.0 + 58.8)
    # Wall-day projection.
    assert combined.wall_days_for_cpu_hours(262.8) == pytest.approx(1.0)
    assert combined.wall_days_for_cpu_hours(0.0) == pytest.approx(0.0)


def test_multi_machine_profile_handles_zero_machines() -> None:
    empty = MultiMachineProfile(machines=())
    assert empty.daily_cpu_hours() == 0.0
    assert empty.wall_days_for_cpu_hours(10.0) == float("inf")


# ---------------------------------------------------------------------------
# Doctoral presets
# ---------------------------------------------------------------------------


def test_doctoral_presets_present() -> None:
    expected = {
        "doctoral_82_datasets_3_algorithms_1_replicas",
        "doctoral_82_datasets_3_algorithms_5_replicas",
        "doctoral_82_datasets_3_algorithms_10_replicas",
        "doctoral_82_datasets_3_algorithms_30_replicas",
    }
    assert expected.issubset(PRESETS.keys())


def test_doctoral_presets_strictly_grow() -> None:
    fits = []
    for r in (1, 5, 10, 30):
        spec = PRESETS[f"doctoral_82_datasets_3_algorithms_{r}_replicas"]
        fits.append((spec.n_replicas, spec.doe_evaluations))
    # n_replicas strictly grow.
    assert [a for a, _ in fits] == [1, 5, 10, 30]


# ---------------------------------------------------------------------------
# Importer CLI surface
# ---------------------------------------------------------------------------


def test_importer_module_exists_and_help_runs() -> None:
    """Smoke check: the importer script can be invoked with --help."""
    import subprocess
    import sys

    script = REPO / "scripts" / "import_doctoral_benchmark_datasets.py"
    assert script.exists()
    res = subprocess.run([sys.executable, str(script), "--help"],
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "doctoral" in res.stdout.lower() or "dataset" in res.stdout.lower()
