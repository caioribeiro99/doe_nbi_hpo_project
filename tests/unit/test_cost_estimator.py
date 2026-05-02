"""Unit tests for the experiment cost estimator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doe_xgb.cost_estimator import (
    ARTIFACTS_PER_REPLICA,
    BatchingPlan,
    BenchmarkSpec,
    CalibrationResult,
    CloudProfile,
    LocalProfile,
    calibrate,
    estimate_cost,
    get_preset,
    list_presets,
)


def _spec(**overrides) -> BenchmarkSpec:
    base = dict(
        n_datasets=1,
        n_algorithms=1,
        n_replicas=2,
        n_folds=5,
        doe_evaluations=10,
        nbi_candidates=5,
        benchmark_evaluations=15,
        n_optimization_methods=4,
        avg_seconds_per_fit=0.1,
        overhead_factor=1.0,
    )
    base.update(overrides)
    return BenchmarkSpec(**base)


def test_total_fits_arithmetic_is_exact() -> None:
    spec = _spec(n_replicas=3, n_folds=5, doe_evaluations=10, nbi_candidates=5,
                 benchmark_evaluations=20, n_optimization_methods=4)
    est = estimate_cost(spec)
    # evals/replica = 10 + 5 + 4*20 = 95
    # fold-fits/replica = 95 * 5 = 475
    # units = 1 * 1 * 3 = 3
    assert est.total_model_fits == 95 * 3
    assert est.total_fold_fits == 475 * 3


def test_cpu_hours_uses_overhead_factor() -> None:
    spec = _spec(avg_seconds_per_fit=2.0, overhead_factor=1.5)
    est = estimate_cost(spec)
    # cpu_seconds = total_fold_fits * 2.0 * 1.5
    expected = est.total_fold_fits * 2.0 * 1.5 / 3600.0
    assert est.cpu_hours == pytest.approx(expected, rel=1e-9)


def test_local_daily_cpu_hours_respects_reserve() -> None:
    profile = LocalProfile(
        max_workers_when_idle=8,
        max_workers_while_working=2,
        hours_idle_per_day=10,
        hours_working_per_day=6,
        reserve_cores_for_user=2,
        efficiency_factor=1.0,
    )
    # Idle: (8-2)*10 = 60; working: (2 - max(0, 2-1))*6 = (2-1)*6 = 6; total = 66.
    assert profile.daily_cpu_hours() == pytest.approx(66.0)


def test_local_efficiency_factor_scales_daily_cpu_hours() -> None:
    profile = LocalProfile(efficiency_factor=0.5)
    base = LocalProfile(efficiency_factor=1.0)
    assert profile.daily_cpu_hours() == pytest.approx(0.5 * base.daily_cpu_hours())


def test_cloud_effective_workers_caps_at_max_concurrent() -> None:
    cloud = CloudProfile(workers=128, max_concurrent_jobs=32)
    assert cloud.effective_workers() == 32


def test_cloud_cost_scales_with_price_and_cpu_hours() -> None:
    spec = _spec(n_replicas=10, avg_seconds_per_fit=1.0)
    est_cheap = estimate_cost(
        spec,
        cloud=CloudProfile(workers=8, instance_hourly_price_per_worker_usd=0.10, efficiency_factor=1.0, max_concurrent_jobs=8),
    )
    est_expensive = estimate_cost(
        spec,
        cloud=CloudProfile(workers=8, instance_hourly_price_per_worker_usd=1.00, efficiency_factor=1.0, max_concurrent_jobs=8),
    )
    assert est_expensive.cloud_wall.cost_usd == pytest.approx(10.0 * est_cheap.cloud_wall.cost_usd, rel=1e-9)


def test_cloud_wall_hours_decrease_with_more_workers() -> None:
    spec = _spec(n_replicas=20, avg_seconds_per_fit=1.0)
    small = estimate_cost(spec, cloud=CloudProfile(workers=4, max_concurrent_jobs=4, efficiency_factor=1.0))
    big = estimate_cost(spec, cloud=CloudProfile(workers=32, max_concurrent_jobs=32, efficiency_factor=1.0))
    assert big.cloud_wall.wall_hours < small.cloud_wall.wall_hours


def test_storage_estimate_is_proportional_to_units() -> None:
    spec_small = _spec(n_datasets=1, n_replicas=10)
    spec_big = _spec(n_datasets=2, n_replicas=10)
    e_small = estimate_cost(spec_small)
    e_big = estimate_cost(spec_big)
    assert e_big.storage.total_artifacts == 2 * e_small.storage.total_artifacts
    assert e_big.storage.total_storage_mb == pytest.approx(2.0 * e_small.storage.total_storage_mb)
    assert e_small.storage.artifacts_per_replica == ARTIFACTS_PER_REPLICA


def test_warning_fires_when_wall_days_exceed_threshold() -> None:
    spec = _spec(n_datasets=82, n_replicas=30, avg_seconds_per_fit=2.0)
    est = estimate_cost(spec, local=LocalProfile(warn_if_wall_days_above=1.0))
    assert any("wall-clock" in w for w in est.warnings)


def test_zero_replicas_yields_zero_work() -> None:
    spec = _spec(n_replicas=0)
    est = estimate_cost(spec)
    assert est.total_model_fits == 0
    assert est.total_fold_fits == 0
    assert est.cpu_hours == 0.0


def test_batching_plan_chunk_count() -> None:
    spec = _spec(n_replicas=23)
    est = estimate_cost(spec, local=LocalProfile(checkpoint_frequency_replicas=5))
    # ceil(23/5) = 5 chunks; replicas_per_chunk = 5.
    assert isinstance(est.batching, BatchingPlan)
    assert est.batching.n_chunks == 5
    assert est.batching.replicas_per_chunk == 5


def test_presets_exist_and_are_specs() -> None:
    names = list_presets()
    assert "article_v1_8_datasets_3_algorithms_10_replicas" in names
    assert "article_v1_12_datasets_3_algorithms_10_replicas" in names
    assert "thesis_82_datasets_3_algorithms_10_replicas" in names
    assert "thesis_82_datasets_3_algorithms_30_replicas" in names
    for name in names:
        spec = get_preset(name)
        assert isinstance(spec, BenchmarkSpec)
        assert spec.n_replicas > 0
        assert spec.n_datasets > 0


def test_presets_are_strictly_growing_in_workload() -> None:
    cheapest = estimate_cost(get_preset("article_v1_8_datasets_3_algorithms_10_replicas"))
    bigger = estimate_cost(get_preset("article_v1_12_datasets_3_algorithms_10_replicas"))
    biggest_q = estimate_cost(get_preset("thesis_82_datasets_3_algorithms_10_replicas"))
    biggest = estimate_cost(get_preset("thesis_82_datasets_3_algorithms_30_replicas"))
    assert cheapest.total_fold_fits < bigger.total_fold_fits
    assert bigger.total_fold_fits < biggest_q.total_fold_fits
    assert biggest_q.total_fold_fits < biggest.total_fold_fits


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        get_preset("does_not_exist")


def test_to_dict_round_trip_is_json_serializable() -> None:
    est = estimate_cost(_spec())
    blob = est.to_dict()
    s = json.dumps(blob)
    parsed = json.loads(s)
    assert parsed["spec"]["n_replicas"] == est.spec.n_replicas
    assert parsed["total_fold_fits"] == est.total_fold_fits
    assert "warnings" in parsed
    assert "batching" in parsed


def test_invalid_spec_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkSpec(n_replicas=-1)
    with pytest.raises(ValueError):
        BenchmarkSpec(avg_seconds_per_fit=0.0)
    with pytest.raises(ValueError):
        BenchmarkSpec(overhead_factor=0.5)


def test_calibrate_returns_a_result_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "cost_estimate_calibration.json"
    res = calibrate(n_samples=300, n_features=4, n_repeats=1, output=out)
    assert isinstance(res, CalibrationResult)
    # XGBoost should be available in this environment (it's a runtime dep).
    assert "xgboost" in res.timings_per_algorithm
    assert res.timings_per_algorithm["xgboost"] > 0.0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["timings_per_algorithm"]["xgboost"] == res.timings_per_algorithm["xgboost"]


def test_estimate_cost_uses_calibration_value() -> None:
    spec_low = _spec(avg_seconds_per_fit=0.01)
    spec_high = _spec(avg_seconds_per_fit=10.0)
    e_low = estimate_cost(spec_low)
    e_high = estimate_cost(spec_high)
    assert e_high.cpu_hours > e_low.cpu_hours * 100
