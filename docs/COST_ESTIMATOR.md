# Experiment cost estimator

`doe_xgb.cost_estimator` (and the `doe-xgb estimate-cost` CLI subcommand)
estimates how big a benchmark campaign will be **before** committing to
running it. It does not run any real training. The arithmetic is pure
Python; an optional calibration step does run a tiny synthetic fit per
algorithm to measure ``avg_seconds_per_fit``.

## What it answers

For a given (n_datasets, n_algorithms, n_replicas, n_folds, evaluations
per method) workload, the estimator reports:

- total model fits and total CV fold fits;
- CPU-hours and CPU-days;
- local Mac wall-clock time under an availability profile that splits
  the day between idle hours (laptop unattended) and working hours
  (user typing in the foreground);
- cloud wall-clock time under a fleet of workers, capped by
  `max_concurrent_jobs`;
- estimated cloud cost in USD;
- storage footprint (MB / GB);
- number of output artifacts;
- a recommended batching plan tied to the checkpoint frequency.

## CLI

```bash
# List built-in presets
doe-xgb estimate-cost --list-presets

# Article v1: 8 datasets, 3 algorithms, 10 replicas
doe-xgb estimate-cost --preset article_v1_8_datasets_3_algorithms_10_replicas

# Thesis full sweep
doe-xgb estimate-cost --preset thesis_82_datasets_3_algorithms_30_replicas

# Custom workload
doe-xgb estimate-cost \
    --datasets 12 --algorithms 3 --replicas 10 --folds 5 \
    --doe-evaluations 88 --nbi-candidates 50 \
    --benchmark-evaluations 138 --n-methods 4 \
    --avg-seconds-per-fit 0.5 --overhead-factor 1.10 \
    --max-workers-when-idle 8 --max-workers-while-working 2 \
    --hours-idle-per-day 10 --hours-working-per-day 6 \
    --reserve-cores-for-user 2 --local-efficiency-factor 0.70 \
    --cloud-workers 32 --cloud-price-per-hour 0.10 \
    --max-concurrent-jobs 32 \
    --output cost_estimate.json

# Calibrate avg_seconds_per_fit on the local machine first
doe-xgb estimate-cost --preset thesis_82_datasets_3_algorithms_10_replicas \
    --calibrate --calibration-output cost_estimate_calibration.json \
    --algorithm xgboost
```

Output is JSON to stdout; pass `--output PATH` to also save it.

## Presets

| Preset | Datasets | Algorithms | Replicas |
|---|---:|---:|---:|
| `article_v1_8_datasets_3_algorithms_10_replicas` | 8 | 3 | 10 |
| `article_v1_12_datasets_3_algorithms_10_replicas` | 12 | 3 | 10 |
| `thesis_82_datasets_3_algorithms_10_replicas` | 82 | 3 | 10 |
| `thesis_82_datasets_3_algorithms_30_replicas` | 82 | 3 | 30 |

All presets default to: `n_folds=5`, `doe_evaluations=88`,
`nbi_candidates=50`, `benchmark_evaluations=138`,
`n_optimization_methods=4` (matching the dissertation conventions).

## Local profile fields

| Field | Default | Meaning |
|---|---|---|
| `max_workers_when_idle` | 8 | Workers when the laptop is unattended (e.g., overnight). |
| `max_workers_while_working` | 2 | Workers while the user is typing in the foreground. |
| `hours_idle_per_day` | 10 | Idle hours in a day. |
| `hours_working_per_day` | 6 | Working hours in a day. |
| `reserve_cores_for_user` | 2 | Cores always subtracted from the idle/working cap. |
| `efficiency_factor` | 0.70 | Empirical scaling factor (XGBoost rarely scales linearly). |
| `model_n_jobs` | 1 | One CV fit per worker process (avoids thread oversubscription). |
| `checkpoint_frequency_replicas` | 5 | Drives the batching plan. |
| `warn_if_wall_days_above` | 14 | Emits a warning if local wall-time exceeds this. |

The estimator computes daily CPU-hours as
`(idle_hours * (idle_workers - reserve)) + (working_hours * (working_workers - max(0, reserve - 1)))`,
multiplied by `efficiency_factor`.

## Cloud profile fields

| Field | Default | Meaning |
|---|---|---|
| `workers` | 32 | Fleet size requested. |
| `instance_hourly_price_per_worker_usd` | 0.10 | Hourly price *per worker*. |
| `efficiency_factor` | 0.85 | Scaling efficiency (parallel inefficiency, network, etc.). |
| `max_concurrent_jobs` | 32 | Account-level concurrency cap. |
| `checkpoint_frequency_replicas` | 10 | Drives the cloud batching plan. |

`effective_workers = min(workers, max_concurrent_jobs)`. Cloud cost is
`(cpu_hours / efficiency_factor) * instance_hourly_price_per_worker_usd`.

## Calibration

`--calibrate` runs a tiny synthetic fit (default 1500 rows × 12 features)
for each available algorithm (XGBoost / LightGBM / CatBoost / scikit
HistGB) and reports the best-of-N seconds per fit. Outputs go to
`--calibration-output cost_estimate_calibration.json`. The selected
algorithm's timing replaces the default `avg_seconds_per_fit`.

```python
from doe_xgb.cost_estimator import calibrate
res = calibrate(n_samples=1500, n_features=12, n_repeats=3,
                output="cost_estimate_calibration.json")
# res.timings_per_algorithm == {"xgboost": 0.21, "histgb": 0.18, ...}
```

The calibration step is the only place in this module that imports
`xgboost`, `lightgbm`, `catboost`, or `sklearn.ensemble`. The pure
arithmetic path (`estimate_cost`) does not pull any heavy dependency.

## Programmatic use

```python
from doe_xgb.cost_estimator import (
    BenchmarkSpec, LocalProfile, CloudProfile, estimate_cost
)

spec = BenchmarkSpec(
    n_datasets=12, n_algorithms=3, n_replicas=10,
    avg_seconds_per_fit=0.4, overhead_factor=1.10,
)
local = LocalProfile(max_workers_when_idle=8, max_workers_while_working=2,
                     hours_idle_per_day=10, hours_working_per_day=6,
                     reserve_cores_for_user=2, efficiency_factor=0.70)
cloud = CloudProfile(workers=32, instance_hourly_price_per_worker_usd=0.10,
                     max_concurrent_jobs=32, efficiency_factor=0.85)
estimate = estimate_cost(spec, local=local, cloud=cloud)
print(estimate.local_wall.wall_days, estimate.cloud_wall.cost_usd)
```

## Caveats

The estimator is intentionally conservative:

- `avg_seconds_per_fit` is single-fold, single-process. Real runs see
  XGBoost overhead, I/O, and benchmark-specific surrogate fits — the
  default `overhead_factor=1.10` partially absorbs that.
- Cloud price is per worker; multiply by your account's billing model
  if it differs.
- The storage footprint assumes the dissertation-style artifact set
  (~18 files, ~350 KB per replica). Different orchestrators (e.g., a
  campaign that emits per-fold predictions) will produce more.
- `efficiency_factor` is a fudge factor, not a measurement. Tune it on
  your machine or fleet by running a small workload and dividing
  observed CPU-hours by predicted CPU-hours.
