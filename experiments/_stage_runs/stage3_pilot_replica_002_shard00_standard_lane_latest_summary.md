# CC18 stage-run summary -- `stage3_pilot_replica_002_shard00_standard_lane_latest`

- run_id: `stage3_pilot_replica_002_shard00_standard_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage3_pilot_replica_002_shard00_standard_lane`
- exported_at: `2026-05-20T17:36:27Z`
- source_git_sha: `fd81eaac7049`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage3_pilot_replica_002_shard00_standard_lane_latest` (gitignored)

- total jobs: **219**
- success: **68**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 151
- runtime: total 341.6s, max 59.84s across 68 recorded jobs
- started_at_min: `2026-05-20T17:30:44.100Z`, finished_at_max: `2026-05-20T17:36:25.686Z`

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
| `shard_00.execution.sqlite` | 219 | 68 | 0 | 341.63 | `49471199492a5c69` |

## Failures (grouped)

_(none)_

## Verdict: **NOT GREEN**

Re-run / archive the failures; investigate any source-shard drift; do not promote downstream until this summary clears.

---

## stage 3 / top-up pilot summary (Commit 47)

- run_id: `stage3_pilot_replica_002_shard00_standard_lane_latest`
- batch_id: `stage3_pilot_replica_002_shard00_standard_lane`
- stage: `stage1_topup_to_005`
- topup_tier: `topup_to_5_pilot`
- replica: **2** (source template replica = 1)
- lane: `standard`
- run_dir: `runs/cc18/stage3_pilot_replica_002_shard00_standard_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- policy_version_pinned: `47b6b50c6d1e1d09`
- signoff_path: `jobs/doctoral/openml_cc18/stage3_signoff.json`
- signoff_sha256: `3f3f1b1fd6819344`
- stage3_topup_plan_summary_path: `experiments/_stage_runs/stage3_topup_plan_latest_summary.json`
- stage3_topup_plan_summary_sha256: `bb79eaee6b5a4da2`
- execution_sqlite_sha256_after_rewrite: `61e5e63534d91172`

- n_jobs_total (shard_00): 219
- expected runnable standard-lane canary cells: **68**
- executed: **68**, success: **68**, deferred_heavy: **31**, deferred_extreme: **11**, refused_non_canary: **109**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 341.6 s

### Per-shard status (single-shard pilot)

| shard | total | success | failed | failed_to | pending | skipped | def_heavy | def_extreme | refused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_00.execution.sqlite` | 219 | 68 | 0 | 0 | 0 | 151 | 31 | 11 | 109 |

### Standard tasks executed (success only)

17 tasks: [3, 16, 23, 43, 53, 3560, 3903, 3913, 9946, 9960, 9976, 10093, 14970, 125920, 146800, 146820, 167140]

### Heavy tasks deferred in this shard

3 tasks: [3573, 9910, 167120]

### Extreme tasks deferred in this shard

1 tasks: [6]

### Non-canary methods refused in this shard

['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 14970 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_00.execution.sqlite` | 59.84 |
| 16 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_00.execution.sqlite` | 39.86 |
| 14970 | `random_search` | `xgboost` | `shard_00.execution.sqlite` | 36.32 |
| 16 | `tpe_optuna` | `catboost` | `shard_00.execution.sqlite` | 34.58 |
| 14970 | `tpe_optuna` | `xgboost` | `shard_00.execution.sqlite` | 33.95 |
| 16 | `random_search` | `catboost` | `shard_00.execution.sqlite` | 31.52 |
| 9976 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_00.execution.sqlite` | 21.35 |
| 9976 | `tpe_optuna` | `catboost` | `shard_00.execution.sqlite` | 18.23 |
| 9976 | `random_search` | `catboost` | `shard_00.execution.sqlite` | 15.64 |
| 14970 | `default_gbdt` | `xgboost` | `shard_00.execution.sqlite` | 10.65 |
| 16 | `default_gbdt` | `catboost` | `shard_00.execution.sqlite` | 5.09 |
| 146800 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_00.execution.sqlite` | 4.42 |

### stage 3 pilot verdict: **GATE PASS — operator review required**

Pilot finished cleanly: every standard-lane canary cell on shard_00 reached a terminal status, the committed source shard is byte-identical to its pre-pilot MD5, no heavy / extreme cell ran, and no full topup_to_5 dispatch was triggered. Commit 48 may plan or run a slightly larger Stage-3 pilot, but **not** the full topup_to_5 tier without explicit operator sign-off.
