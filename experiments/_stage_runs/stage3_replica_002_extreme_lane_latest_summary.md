# CC18 stage-run summary -- `stage3_replica_002_extreme_lane_latest`

- run_id: `stage3_replica_002_extreme_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage3_replica_002_extreme_lane`
- exported_at: `2026-05-22T09:04:37Z`
- source_git_sha: `ad0215ea155c`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage3_replica_002_extreme_lane_latest` (gitignored)

- total jobs: **2304**
- success: **24**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 2280
- runtime: total 34156.1s, max 10888.56s across 24 recorded jobs
- started_at_min: `2026-05-21T23:35:09.817Z`, finished_at_max: `2026-05-22T09:04:31.645Z`

- source_shards_unchanged: **True**
- stage3_signoff_present: True
- archive: _(none; large artifacts stay on the publishing machine)_

## Package versions

| package | version |
|---|---|
| `xgboost` | 3.2.0 |
| `lightgbm` | 4.6.0 |
| `catboost` | 1.2.10 |
| `optuna` | 4.8.0 |
| `scikit-learn` | 1.8.0 |
| `openml` | 0.15.1 |
| `smac` | 2.4.0 |
| `pymoo` | 0.6.1.6 |
| `dehb` | 0.1.2 |
| `numpy` | 1.26.4 |
| `pandas` | 3.0.2 |

## Per-shard

| shard | total | success | failed | runtime_s | sha256 |
|---|---:|---:|---:|---:|---|
| `shard_00.execution.sqlite` | 219 | 4 | 0 | 25.80 | `e8d8f63ca2b41e92` |
| `shard_01.execution.sqlite` | 230 | 0 | 0 | 0.00 | `9c97b176cc78f0aa` |
| `shard_02.execution.sqlite` | 223 | 0 | 0 | 0.00 | `8cc2db1bd6b9f9b8` |
| `shard_03.execution.sqlite` | 235 | 4 | 0 | 15284.02 | `e0524daa006503cd` |
| `shard_04.execution.sqlite` | 226 | 4 | 0 | 2261.15 | `84befe20cf88f7a7` |
| `shard_05.execution.sqlite` | 226 | 4 | 0 | 16535.41 | `dcaf4a92ff782fee` |
| `shard_06.execution.sqlite` | 228 | 0 | 0 | 0.00 | `eebab96c37f0de68` |
| `shard_07.execution.sqlite` | 270 | 0 | 0 | 0.00 | `8228bc55ae845d60` |
| `shard_08.execution.sqlite` | 237 | 4 | 0 | 14.34 | `255e95c3c0aadc6f` |
| `shard_09.execution.sqlite` | 210 | 4 | 0 | 35.34 | `568d6b1183b907aa` |

## Failures (grouped)

_(none)_

## Verdict: **NOT GREEN**

Re-run / archive the failures; investigate any source-shard drift; do not promote downstream until this summary clears.

---

## stage 3 / replica_002 extreme-lane summary (Commit 51)

- run_id: `stage3_replica_002_extreme_lane_latest`
- batch_id: `stage3_replica_002_extreme_lane`
- stage: `stage1_topup_to_005`
- topup_tier: `topup_to_5_partial`
- replica: **2** (source template replica = 1)
- lane: `extreme`
- n_source_shards: 10
- run_dir: `runs/cc18/stage3_replica_002_extreme_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- policy_version_pinned: `47b6b50c6d1e1d09`
- signoff_sha256: `3f3f1b1fd6819344`
- stage3_topup_plan_summary_sha256: `bb79eaee6b5a4da2`
- commit48_standard_lane_summary_sha256: `6d028cef7c3715cf`
- commit49_heavy_lane_summary_sha256: `95fba21f38db3453`
- commit50_extreme_plan_summary_sha256: `f22cdd6c1974d629`

- n_jobs_total (across 10 shards): 2304
- expected runnable extreme-lane canary cells: **24**
- executed: **24**, success: **24**, deferred_standard: **1815**, deferred_heavy: **423**, refused_non_canary: **42**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 34156.1 s

- Extreme lane executed with policy-defined max_evaluations=1.
- extreme_lane_timeout_seconds_per_cell_used: 14400 s

### Per-shard status

| shard | total | success | failed | failed_to | pending | skipped | def_std | def_heavy | refused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_00.execution.sqlite` | 219 | 4 | 0 | 0 | 0 | 215 | 177 | 31 | 7 |
| `shard_01.execution.sqlite` | 230 | 0 | 0 | 0 | 0 | 230 | 199 | 31 | 0 |
| `shard_02.execution.sqlite` | 223 | 0 | 0 | 0 | 0 | 223 | 192 | 31 | 0 |
| `shard_03.execution.sqlite` | 235 | 4 | 0 | 0 | 0 | 231 | 202 | 22 | 7 |
| `shard_04.execution.sqlite` | 226 | 4 | 0 | 0 | 0 | 222 | 193 | 22 | 7 |
| `shard_05.execution.sqlite` | 226 | 4 | 0 | 0 | 0 | 222 | 160 | 55 | 7 |
| `shard_06.execution.sqlite` | 228 | 0 | 0 | 0 | 0 | 228 | 162 | 66 | 0 |
| `shard_07.execution.sqlite` | 270 | 0 | 0 | 0 | 0 | 270 | 182 | 88 | 0 |
| `shard_08.execution.sqlite` | 237 | 4 | 0 | 0 | 0 | 233 | 182 | 44 | 7 |
| `shard_09.execution.sqlite` | 210 | 4 | 0 | 0 | 0 | 206 | 166 | 33 | 7 |

### Per-task status breakdown (extreme universe)

| task_id | dataset | total | success | failed | failed_to | runtime_total_s |
|---:|---|---:|---:|---:|---:|---:|
| 6 | `letter` | 33 | 12 | 0 | 0 | 75.5 |
| 167121 | `Devnagari-Script` | 33 | 12 | 0 | 0 | 34080.6 |

### Per-method status breakdown (extreme universe)

| method | total | success | failed | failed_to | runtime_total_s |
|---|---:|---:|---:|---:|---:|
| `asha` | 6 | 0 | 0 | 0 | 0.0 |
| `bohb` | 6 | 0 | 0 | 0 | 0.0 |
| `default_gbdt` | 6 | 6 | 0 | 0 | 3200.9 |
| `dehb` | 6 | 0 | 0 | 0 | 0.0 |
| `doe_rsm_vrf_true_nbi` | 6 | 6 | 0 | 0 | 22881.1 |
| `motpe` | 6 | 0 | 0 | 0 | 0.0 |
| `nsga2` | 6 | 0 | 0 | 0 | 0.0 |
| `parego` | 6 | 0 | 0 | 0 | 0.0 |
| `random_search` | 6 | 6 | 0 | 0 | 3806.7 |
| `smac3` | 6 | 0 | 0 | 0 | 0.0 |
| `tpe_optuna` | 6 | 6 | 0 | 0 | 4267.4 |

### Per-algorithm status breakdown (extreme universe)

| algorithm | total | success | failed | failed_to | runtime_total_s |
|---|---:|---:|---:|---:|---:|
| `catboost` | 22 | 8 | 0 | 0 | 16561.2 |
| `lightgbm` | 22 | 8 | 0 | 0 | 2296.5 |
| `xgboost` | 22 | 8 | 0 | 0 | 15298.4 |

### Extreme tasks executed (success only)

2 tasks: [6, 167121]

### Standard tasks deferred (Commit 48 stands)

57 tasks deferred.
### Heavy tasks deferred (Commit 49 stands)

13 tasks deferred.
### Non-canary methods refused

['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 167121 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_03.execution.sqlite` | 10888.56 |
| 167121 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_05.execution.sqlite` | 10417.05 |
| 167121 | `tpe_optuna` | `catboost` | `shard_05.execution.sqlite` | 2651.19 |
| 167121 | `random_search` | `catboost` | `shard_05.execution.sqlite` | 1941.45 |
| 167121 | `random_search` | `xgboost` | `shard_03.execution.sqlite` | 1535.22 |
| 167121 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_04.execution.sqlite` | 1525.81 |
| 167121 | `default_gbdt` | `catboost` | `shard_05.execution.sqlite` | 1525.72 |
| 167121 | `tpe_optuna` | `xgboost` | `shard_03.execution.sqlite` | 1466.54 |
| 167121 | `default_gbdt` | `xgboost` | `shard_03.execution.sqlite` | 1393.69 |
| 167121 | `random_search` | `lightgbm` | `shard_04.execution.sqlite` | 321.40 |
| 167121 | `default_gbdt` | `lightgbm` | `shard_04.execution.sqlite` | 273.78 |
| 167121 | `tpe_optuna` | `lightgbm` | `shard_04.execution.sqlite` | 140.17 |

### stage 3 replica_002 extreme lane verdict: **GATE PASS — operator review required**

Run finished cleanly: every extreme-lane canary cell on replica_002 across all 10 shards reached a terminal status, every committed source shard is byte-identical to its pre-run MD5, standard / heavy lanes were not rerun, no other replica was executed, and the full topup_to_5 tier was not triggered. Commit 52 may aggregate / review / signoff replica_002.
