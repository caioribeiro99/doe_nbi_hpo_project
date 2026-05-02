"""Experiment cost estimator (no real benchmark runs).

Estimates the wall-clock time, CPU-hours, storage footprint, and cloud
spend of an article-track / thesis-scale experiment campaign before
committing to it. Two execution profiles are modeled:

- **Local mode** (a Mac laptop): split between idle hours
  (laptop unattended, more workers) and working hours (user typing,
  fewer workers).
- **Cloud mode**: a fleet of workers running concurrently up to a
  per-account ``max_concurrent_jobs`` cap, billed per CPU-hour.

The arithmetic is pure-Python and import-light. An opt-in calibration
helper runs a tiny synthetic fit per algorithm to estimate
``avg_seconds_per_fit``; the helper is the only path that touches
XGBoost / LightGBM / CatBoost / HistGB and is therefore lazy.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpec:
    """One unit of work to estimate.

    All counts are per-replica unless stated otherwise. Cost
    multiplies through ``n_datasets * n_algorithms * n_replicas``.
    """

    n_datasets: int = 1
    n_algorithms: int = 1
    n_replicas: int = 30
    n_folds: int = 5
    doe_evaluations: int = 88              # DoE+NBI proposed method
    nbi_candidates: int = 50               # NBI confirmation evals
    benchmark_evaluations: int = 138       # per benchmark method (e.g., DOE_runs + NBI_eval_k)
    n_optimization_methods: int = 4        # grid / random / bayes / hyperopt
    avg_seconds_per_fit: float = 0.5       # measured by calibration or guessed
    overhead_factor: float = 1.10          # multiply CPU-time to cover I/O, logging, etc.

    def __post_init__(self) -> None:  # pragma: no cover - validation
        for name in (
            "n_datasets", "n_algorithms", "n_replicas", "n_folds",
            "doe_evaluations", "nbi_candidates", "benchmark_evaluations",
            "n_optimization_methods",
        ):
            v = getattr(self, name)
            if int(v) < 0:
                raise ValueError(f"{name} must be >= 0; got {v}")
        if self.avg_seconds_per_fit <= 0:
            raise ValueError("avg_seconds_per_fit must be > 0")
        if self.overhead_factor < 1.0:
            raise ValueError("overhead_factor must be >= 1.0")


@dataclass(frozen=True)
class LocalProfile:
    """Local Mac availability profile."""

    max_workers_when_idle: int = 8         # laptop unattended overnight
    max_workers_while_working: int = 2     # user typing in foreground
    hours_idle_per_day: float = 10.0
    hours_working_per_day: float = 6.0
    reserve_cores_for_user: int = 2        # always subtracted from idle/working caps
    efficiency_factor: float = 0.70        # XGBoost+ML rarely scales linearly
    model_n_jobs: int = 1                  # one CV fit per worker process
    checkpoint_frequency_replicas: int = 5
    warn_if_wall_days_above: float = 14.0  # raise a warning above this

    @property
    def available_hours_per_day(self) -> float:
        return self.hours_idle_per_day + self.hours_working_per_day

    def daily_cpu_hours(self) -> float:
        idle_workers = max(0, self.max_workers_when_idle - self.reserve_cores_for_user)
        work_workers = max(0, self.max_workers_while_working - max(0, self.reserve_cores_for_user - 1))
        idle = self.hours_idle_per_day * idle_workers
        work = self.hours_working_per_day * work_workers
        return (idle + work) * self.efficiency_factor


@dataclass(frozen=True)
class CloudProfile:
    """Cloud fleet profile."""

    workers: int = 32
    instance_hourly_price_per_worker_usd: float = 0.10
    efficiency_factor: float = 0.85
    max_concurrent_jobs: int = 32
    checkpoint_frequency_replicas: int = 10

    def effective_workers(self) -> int:
        return max(1, min(self.workers, self.max_concurrent_jobs))


# Module-level singletons used as safe defaults in function signatures
# (avoids ruff B008).
DEFAULT_LOCAL_PROFILE: LocalProfile = LocalProfile()
DEFAULT_CLOUD_PROFILE: CloudProfile = CloudProfile()


# ---------------------------------------------------------------------------
# Multi-machine profile (Commit 24)
#
# Sums the daily CPU-hours of an arbitrary set of MachineProfile entries
# so a "dedicated Mac + Caio Mac" combined estimate is computable without
# duplicating the LocalProfile arithmetic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineProfile:
    machine_id: str
    workers: int
    hours_per_day: float
    efficiency_factor: float
    model_n_jobs: int = 1


@dataclass(frozen=True)
class MultiMachineProfile:
    machines: tuple[MachineProfile, ...]

    def daily_cpu_hours(self) -> float:
        return float(sum(
            m.workers * m.hours_per_day * m.efficiency_factor for m in self.machines
        ))

    def wall_days_for_cpu_hours(self, cpu_hours: float) -> float:
        d = self.daily_cpu_hours()
        return cpu_hours / d if d > 0 else float("inf")


def dedicated_mac_profile(efficiency: float = 0.85) -> MachineProfile:
    """Default dedicated MacBook Pro profile.

    Efficiency presets: 0.75 (conservative), 0.85 (realistic with
    cooling, **default**), 0.90 (optimistic). The 0.70 figure used
    elsewhere is reserved for the Caio personal Mac, not the
    dedicated machine.
    """
    return MachineProfile(
        machine_id="mac_dedicado",
        workers=10,
        hours_per_day=24.0,
        efficiency_factor=float(efficiency),
        model_n_jobs=1,
    )


def caio_mac_profile(efficiency: float = 0.70, hours_per_day: float = 14.0) -> MachineProfile:
    """Default Caio personal Mac profile (opportunistic supplement)."""
    return MachineProfile(
        machine_id="macbook_caio",
        workers=6,
        hours_per_day=float(hours_per_day),
        efficiency_factor=float(efficiency),
        model_n_jobs=1,
    )


def two_macs_combined(
    *,
    dedicated_efficiency: float = 0.85,
    caio_efficiency: float = 0.70,
    caio_hours_per_day: float = 14.0,
) -> MultiMachineProfile:
    """Convenience constructor for the dedicated + Caio Mac duo."""
    return MultiMachineProfile(machines=(
        dedicated_mac_profile(efficiency=dedicated_efficiency),
        caio_mac_profile(efficiency=caio_efficiency, hours_per_day=caio_hours_per_day),
    ))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageEstimate:
    artifacts_per_replica: int
    artifact_kb_per_replica: float
    total_artifacts: int
    total_storage_mb: float
    total_storage_gb: float


@dataclass(frozen=True)
class WallClockEstimate:
    cpu_hours: float
    cpu_days: float
    wall_hours: float
    wall_days: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CloudEstimate(WallClockEstimate):
    cost_usd: float = 0.0
    effective_workers: int = 0


@dataclass(frozen=True)
class BatchingPlan:
    n_chunks: int
    replicas_per_chunk: int
    chunk_wall_hours_local: float
    chunk_wall_hours_cloud: float


@dataclass
class CostEstimate:
    spec: BenchmarkSpec
    local: LocalProfile
    cloud: CloudProfile
    total_model_fits: int
    total_fold_fits: int
    cpu_hours: float
    cpu_days: float
    local_wall: WallClockEstimate
    cloud_wall: CloudEstimate
    storage: StorageEstimate
    batching: BatchingPlan
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "local": asdict(self.local),
            "cloud": asdict(self.cloud),
            "total_model_fits": int(self.total_model_fits),
            "total_fold_fits": int(self.total_fold_fits),
            "cpu_hours": float(self.cpu_hours),
            "cpu_days": float(self.cpu_days),
            "local_wall": asdict(self.local_wall),
            "cloud_wall": asdict(self.cloud_wall),
            "storage": asdict(self.storage),
            "batching": asdict(self.batching),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------


# A typical replica writes:
#   manifest.json, doe_results.csv, doe_results_with_scores.csv,
#   factor_loadings.csv, factor_scores.csv, factor_diagnostics.json,
#   rsm_coefficients_<obj>.csv (xN), nbi_anchors.csv, nbi_chim.json,
#   nbi_candidates.csv, nbi_subproblem_diagnostics.csv,
#   nbi_candidate_evaluations.csv, confirmation_summary.csv,
#   confirmation_vrf.csv, frontier_quality.json,
#   post_optimization_diagnostics.json, fold_metrics.csv, run_replica.log.
ARTIFACTS_PER_REPLICA: int = 18
KB_PER_REPLICA_AVG: float = 350.0  # rough average of CSV/JSON/log sizes


def _evaluations_per_replica(spec: BenchmarkSpec) -> int:
    proposed = spec.doe_evaluations + spec.nbi_candidates
    benchmarks = spec.n_optimization_methods * spec.benchmark_evaluations
    return proposed + benchmarks


def _fold_fits_per_replica(spec: BenchmarkSpec) -> int:
    return _evaluations_per_replica(spec) * spec.n_folds


def estimate_cost(
    spec: BenchmarkSpec,
    local: LocalProfile = DEFAULT_LOCAL_PROFILE,
    cloud: CloudProfile = DEFAULT_CLOUD_PROFILE,
) -> CostEstimate:
    """Estimate cost for the given spec under both local and cloud profiles."""

    fold_fits_per_replica = _fold_fits_per_replica(spec)
    evals_per_replica = _evaluations_per_replica(spec)
    units = spec.n_datasets * spec.n_algorithms * spec.n_replicas

    total_model_fits = units * evals_per_replica
    total_fold_fits = units * fold_fits_per_replica

    cpu_seconds = total_fold_fits * spec.avg_seconds_per_fit * spec.overhead_factor
    cpu_hours = cpu_seconds / 3600.0
    cpu_days = cpu_hours / 24.0

    # Local wall clock.
    daily_cpu_hours_local = local.daily_cpu_hours()
    local_wall_days = (
        cpu_hours / daily_cpu_hours_local if daily_cpu_hours_local > 0 else float("inf")
    )
    local_wall_hours = local_wall_days * 24.0
    local_notes: list[str] = []
    if daily_cpu_hours_local <= 0:
        local_notes.append("Local profile yields zero usable CPU-hours per day; check workers/efficiency.")
    local_wall = WallClockEstimate(
        cpu_hours=cpu_hours,
        cpu_days=cpu_days,
        wall_hours=local_wall_hours,
        wall_days=local_wall_days,
        notes=tuple(local_notes),
    )

    # Cloud wall clock.
    eff_workers = cloud.effective_workers()
    cloud_wall_hours = cpu_hours / max(1, eff_workers) / max(0.01, cloud.efficiency_factor)
    cloud_cpu_hours_billed = cpu_hours / max(0.01, cloud.efficiency_factor)
    cloud_cost = cloud_cpu_hours_billed * cloud.instance_hourly_price_per_worker_usd
    cloud_wall = CloudEstimate(
        cpu_hours=cpu_hours,
        cpu_days=cpu_days,
        wall_hours=cloud_wall_hours,
        wall_days=cloud_wall_hours / 24.0,
        notes=tuple([
            f"Effective fleet size: {eff_workers} workers."
            + (" (capped by max_concurrent_jobs)" if cloud.workers > eff_workers else "")
        ]),
        cost_usd=cloud_cost,
        effective_workers=eff_workers,
    )

    # Storage.
    total_artifacts = ARTIFACTS_PER_REPLICA * units
    storage_kb = KB_PER_REPLICA_AVG * units
    storage = StorageEstimate(
        artifacts_per_replica=ARTIFACTS_PER_REPLICA,
        artifact_kb_per_replica=KB_PER_REPLICA_AVG,
        total_artifacts=total_artifacts,
        total_storage_mb=storage_kb / 1024.0,
        total_storage_gb=storage_kb / (1024.0 * 1024.0),
    )

    # Batching.
    chunk_replicas_local = max(1, local.checkpoint_frequency_replicas)
    chunk_replicas_cloud = max(1, cloud.checkpoint_frequency_replicas)
    chunk_replicas = max(chunk_replicas_local, 1)
    n_chunks = max(1, math.ceil(spec.n_replicas / chunk_replicas))
    chunk_wall_local = local_wall_hours / n_chunks if n_chunks else 0.0
    chunk_wall_cloud = cloud_wall_hours / max(1, math.ceil(spec.n_replicas / chunk_replicas_cloud))
    batching = BatchingPlan(
        n_chunks=n_chunks,
        replicas_per_chunk=chunk_replicas,
        chunk_wall_hours_local=chunk_wall_local,
        chunk_wall_hours_cloud=chunk_wall_cloud,
    )

    warnings: list[str] = []
    if local_wall.wall_days > local.warn_if_wall_days_above:
        warnings.append(
            f"Local wall-clock estimate is {local_wall.wall_days:.1f} days "
            f"(> warn_if_wall_days_above={local.warn_if_wall_days_above}). "
            "Consider scaling out to cloud or reducing replicas/datasets."
        )
    if cloud.workers > cloud.max_concurrent_jobs:
        warnings.append(
            f"cloud.workers ({cloud.workers}) > cloud.max_concurrent_jobs "
            f"({cloud.max_concurrent_jobs}); effective fleet capped at {eff_workers}."
        )

    return CostEstimate(
        spec=spec,
        local=local,
        cloud=cloud,
        total_model_fits=total_model_fits,
        total_fold_fits=total_fold_fits,
        cpu_hours=cpu_hours,
        cpu_days=cpu_days,
        local_wall=local_wall,
        cloud_wall=cloud_wall,
        storage=storage,
        batching=batching,
        warnings=warnings,
        metadata={
            "evaluations_per_replica": evals_per_replica,
            "fold_fits_per_replica": fold_fits_per_replica,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    )


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------


PRESETS: dict[str, BenchmarkSpec] = {
    "article_v1_8_datasets_3_algorithms_5_replicas": BenchmarkSpec(
        n_datasets=8,
        n_algorithms=3,
        n_replicas=5,
    ),
    "article_v1_8_datasets_3_algorithms_10_replicas": BenchmarkSpec(
        n_datasets=8,
        n_algorithms=3,
        n_replicas=10,
        n_folds=5,
        doe_evaluations=88,
        nbi_candidates=50,
        benchmark_evaluations=138,
        n_optimization_methods=4,
    ),
    "article_v1_12_datasets_3_algorithms_5_replicas": BenchmarkSpec(
        n_datasets=12,
        n_algorithms=3,
        n_replicas=5,
    ),
    "article_v1_12_datasets_3_algorithms_10_replicas": BenchmarkSpec(
        n_datasets=12,
        n_algorithms=3,
        n_replicas=10,
    ),
    "thesis_82_datasets_3_algorithms_10_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=10,
    ),
    "thesis_82_datasets_3_algorithms_30_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=30,
    ),
    # Doctoral campaign staged presets (Commit 24).
    "doctoral_82_datasets_3_algorithms_1_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=1,
    ),
    "doctoral_82_datasets_3_algorithms_5_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=5,
    ),
    "doctoral_82_datasets_3_algorithms_10_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=10,
    ),
    "doctoral_82_datasets_3_algorithms_30_replicas": BenchmarkSpec(
        n_datasets=82,
        n_algorithms=3,
        n_replicas=30,
    ),
}


def list_presets() -> list[str]:
    return sorted(PRESETS.keys())


def get_preset(name: str) -> BenchmarkSpec:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list_presets()}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Calibration (opt-in; lazy imports so this module stays light)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    timings_per_algorithm: dict[str, float]
    n_samples: int
    n_features: int
    n_repeats: int
    platform: str
    python: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_synthetic(n: int = 1500, k: int = 12, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    coeffs = rng.normal(size=k)
    logits = X @ coeffs
    y = (logits > 0).astype(int)
    return X, y


def _time_one(fit_fn, n_repeats: int = 3) -> float:
    timings = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fit_fn()
        timings.append(time.perf_counter() - t0)
    return float(min(timings))  # take best to filter cold-cache outliers


def _try_xgboost(X, y, n_estimators: int = 100) -> float | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    def fit() -> None:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X, y)

    return _time_one(fit)


def _try_lightgbm(X, y, n_estimators: int = 100) -> float | None:
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None

    def fit() -> None:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=-1,
            learning_rate=0.1,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(X, y)

    return _time_one(fit)


def _try_catboost(X, y, n_estimators: int = 100) -> float | None:
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None

    def fit() -> None:
        model = CatBoostClassifier(
            iterations=n_estimators,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            thread_count=1,
        )
        model.fit(X, y)

    return _time_one(fit)


def _try_histgb(X, y, n_estimators: int = 100) -> float | None:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        return None

    def fit() -> None:
        model = HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=0.1)
        model.fit(X, y)

    return _time_one(fit)


def calibrate(
    *,
    n_samples: int = 1500,
    n_features: int = 12,
    n_repeats: int = 3,
    output: Path | None = None,
) -> CalibrationResult:
    """Run a tiny synthetic fit per available algorithm and report timings.

    Writes ``cost_estimate_calibration.json`` to ``output`` if provided.
    """

    X, y = _make_synthetic(n=n_samples, k=n_features)
    timings: dict[str, float] = {}
    for name, fn in (
        ("xgboost", _try_xgboost),
        ("lightgbm", _try_lightgbm),
        ("catboost", _try_catboost),
        ("histgb", _try_histgb),
    ):
        t = fn(X, y)
        if t is not None:
            timings[name] = float(t)

    result = CalibrationResult(
        timings_per_algorithm=timings,
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_repeats=int(n_repeats),
        platform=platform.platform(),
        python=platform.python_version(),
    )

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


__all__ = [
    "ARTIFACTS_PER_REPLICA",
    "KB_PER_REPLICA_AVG",
    "BenchmarkSpec",
    "LocalProfile",
    "CloudProfile",
    "StorageEstimate",
    "WallClockEstimate",
    "CloudEstimate",
    "BatchingPlan",
    "CostEstimate",
    "PRESETS",
    "list_presets",
    "get_preset",
    "estimate_cost",
    "CalibrationResult",
    "calibrate",
]
