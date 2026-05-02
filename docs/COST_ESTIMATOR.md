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

| Preset | Datasets / Tasks | Algorithms | Replicas | Notes |
|---|---:|---:|---:|---|
| `article_v1_8_datasets_3_algorithms_10_replicas` | 8 | 3 | 10 | v1 binary panel |
| `article_v1_12_datasets_3_algorithms_10_replicas` | 12 | 3 | 10 | v1 panel (smoke / profiling fixture) |
| `thesis_82_datasets_3_algorithms_10_replicas` | 82 | 3 | 10 | legacy thesis sweep |
| `thesis_82_datasets_3_algorithms_30_replicas` | 82 | 3 | 30 | legacy thesis sweep |
| `openml_cc18_72_tasks_3_algorithms_1_replicas` | 72 | 3 | 1 | **CC18 doctoral, stage 0** |
| `openml_cc18_72_tasks_3_algorithms_5_replicas` | 72 | 3 | 5 | CC18 doctoral, stage 1 cumulative |
| `openml_cc18_72_tasks_3_algorithms_10_replicas` | 72 | 3 | 10 | CC18 doctoral, stage 2 cumulative |
| `openml_cc18_72_tasks_3_algorithms_30_replicas` | 72 | 3 | 30 | **CC18 doctoral, headline** |
| `doctoral_82_datasets_3_algorithms_{1,5,10,30}_replicas` | 82 | 3 | 1 / 5 / 10 / 30 | **deprecated** — emits `DeprecationWarning`; superseded by `openml_cc18_72_tasks_*` (Commit 25) |

All presets default to: `n_folds=5`, `doe_evaluations=88`,
`nbi_candidates=50`, `benchmark_evaluations=138`,
`n_optimization_methods=4` (matching the dissertation conventions).
Resolving any of the deprecated `doctoral_82_*` presets via
`get_preset(...)` raises a `DeprecationWarning` redirecting callers to
the `openml_cc18_72_tasks_*` keys; the `BenchmarkSpec` itself is
unchanged so existing tests still resolve.

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

## Article-track v1 sizing (Commit 17)

A full sizing study for the v1 panel lives at
`experiments/_cost_calibration/article_v1_cost_estimates.md` (and
JSON). Headlines:

- Calibrated worst-case `avg_seconds_per_fit = 0.135 s` (CatBoost,
  Apple Silicon Mac, 1500 rows × 12 feats, default hyperparameters).
- At 4× inflation (~0.54 s/fit), the headline 12 × 3 × 10 panel
  finishes in **1.2 days on a dedicated Mac**, **0.8 days on
  combined two Macs (16 workers × 24 h, eff 0.70)**, or
  **~7.5 h / $24 on a 32-worker $0.10/h cloud**.
- At 8× inflation (~1.08 s/fit), the same panel takes 2.4 days
  dedicated or $48 cloud.
- The full 12 × 3 × 10 panel is feasible without dropping replicas.

Re-run the sizing report at any time with:

```bash
python scripts/article_v1_sizing.py
```

This script does not run the experiment.

## Article v1 full-dataset runtime profile (Commit 22)

A real 5-fold CV pass over the full data on every (dataset, algorithm)
pair lives at
`experiments/_runtime_profile/v1_full_dataset_5fold_profile.{json,md}`.
Headlines (Apple Silicon Mac, single safe HP point per algorithm,
`n_jobs=1` / `thread_count=1`):

- 36 / 36 pairs OK; 0 failures.
- Fastest pair: `pima_diabetes / lightgbm` ≈ 0.04 s.
- Slowest pair: `bank_marketing / catboost` ≈ 5.8 s (native categorical
  on a 30k-row mixed-type dataset).
- Sum of measured per-pair times across the panel: ~26 s.

### Projections at the realistic 4× inflation multiplier

| Scope | Replicas | Total CPU-h | Dedicated Mac (eff 0.70) | Cloud 32w eff 0.85 | Cloud cost |
|---|---:|---:|---:|---:|---:|
| article v1 (12 × 3) | 10 | **207** | **1.23 d** | 0.32 d | **$24** |
| article v1 (12 × 3) | 30 | 622 | 3.70 d | 0.95 d | $73 |
| article v1 (11 binary × 3) | 10 | 164 | 0.98 d | 0.25 d | $19 |
| article v1 (11 binary × 3) | 30 | 492 | 2.93 d | 0.75 d | $58 |
| thesis 82 × 3 | 10 | 1,416 | 8.43 d | 2.17 d | $167 |
| thesis 82 × 3 | 30 | 4,248 | **25.3 d** | 6.51 d | **$500** |

The thesis projections scale by the *mean* per-pair time of our v1
panel (we have no measurements for the other 70 datasets). They are
order-of-magnitude estimates, not commitments.

### Recommendation

Run the full **12 × 3 × 10** article v1 locally on the dedicated Mac.
Reserve 30 replicas for *selected* datasets, not the whole panel. For
the doctoral 82-dataset benchmark, run **1 replica locally first** as
a sizing check before scaling out to cloud.

Re-run the profiler with:

```bash
python scripts/profile_v1_full_dataset_runtime.py
```

The profiler does not run DOE / RSM / NBI / MBPA.

## Doctoral benchmark profiles (Commit 24, retargeted in Commit 25)

The doctoral campaign assumes a **dedicated MacBook Pro** and an
optional **Caio personal Mac**. Helpers in
`doe_xgb.cost_estimator`:

```python
from doe_xgb.cost_estimator import (
    dedicated_mac_profile, caio_mac_profile, two_macs_combined,
)

dedicated = dedicated_mac_profile(efficiency=0.85)   # default doctoral profile
combined  = two_macs_combined(dedicated_efficiency=0.85,
                              caio_efficiency=0.70,
                              caio_hours_per_day=14.0)

combined.daily_cpu_hours()
combined.wall_days_for_cpu_hours(1243)  # CC18 72 x 3 x 10 at 4x inflation
```

Dedicated-Mac efficiency presets:

- **0.75** conservative,
- **0.85** realistic with cooling (default),
- **0.90** optimistic.

The 0.70 figure is reserved for the Caio personal Mac, not the
dedicated machine.

### Provisional OpenML-CC18 72 × 3 × R estimates (Commit 25, primary)

> **Note (Commit 26).** The estimates below cover one method per
> (task, algorithm) cell. The CC18 doctoral campaign runs a
> comparative protocol of multiple HPO baselines plus the proposed
> method (see `docs/COMPARATIVE_PROTOCOL.md` and
> `benchmarks/doctoral/openml_cc18/method_matrix.csv`); the
> headline wall-clock therefore scales roughly linearly with the
> number of methods that run on the full 72 tasks. Multiply the rows
> below by the count of `full_cc18=true` rows in `method_matrix.csv`
> (after protocol freeze) to project the campaign-level wall-clock.
> Methods with subset-only coverage (currently ParEGO) contribute
> proportionally less than a full-coverage row.

Anchored on the v1 mean per-pair 5-fold runtime
(~0.75 s; full table in
`experiments/_runtime_profile/v1_full_dataset_5fold_profile.json`)
and 690 evaluations per replica per pair, at the realistic 4×
inflation multiplier:

| Scope | Total CPU-h | Dedicated 0.75 | Dedicated 0.85 | Dedicated 0.90 | Two Macs (0.85+0.70) |
|---|---:|---:|---:|---:|---:|
| 72 × 3 × 1 | 124 | 0.69 d | 0.61 d | 0.58 d | 0.47 d |
| 72 × 3 × 5 | 622 | 3.46 d | 3.05 d | 2.88 d | 2.37 d |
| 72 × 3 × 10 | 1,243 | 6.91 d | 6.09 d | 5.76 d | 4.73 d |
| 72 × 3 × 30 | **3,730** | **20.72 d** | **18.28 d** | **17.27 d** | **14.19 d** |

These are **provisional** until a real CC18 profiler runs. CC18 is
heavier in expectation than the 12-dataset v1 panel (it includes
larger datasets such as `adult`, `bank-marketing`, `nomao`,
`numerai28.6`, etc.), so the realistic 4× multiplier may underestimate
the headline 30-replica wall-clock. Re-anchor after the first real
CC18 sweep on the dedicated Mac.

### Deprecated 82 × 3 × R estimates (Commit 24, kept for historical context)

The 82-dataset projections from Commit 24 — anchored on the same v1
mean per-pair runtime — were:

| Scope | Total CPU-h | Dedicated 0.75 | Dedicated 0.85 | Dedicated 0.90 | Two Macs (0.85+0.70) |
|---|---:|---:|---:|---:|---:|
| 82 × 3 × 1 | 142 | 0.79 d | 0.69 d | 0.66 d | 0.54 d |
| 82 × 3 × 5 | 708 | 3.93 d | 3.47 d | 3.28 d | 2.69 d |
| 82 × 3 × 10 | 1,416 | 7.87 d | 6.94 d | 6.56 d | 5.39 d |
| 82 × 3 × 30 | 4,248 | 23.60 d | 20.82 d | 19.67 d | 16.16 d |

These are kept only as historical record; the doctoral primary target
is the CC18 72-task panel above.

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
