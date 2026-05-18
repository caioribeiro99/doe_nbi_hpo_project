# CC18 stage-run summary -- `stage0_extreme_lane_latest`

- run_id: `stage0_extreme_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage0_extreme_lane`
- exported_at: `2026-05-17T23:10:59Z`
- source_git_sha: `4933dbbea628`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage0_extreme_lane_latest` (gitignored)

- total jobs: **2304**
- success: **24**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 2280
- runtime: total 30844.5s, max 10663.53s across 24 recorded jobs
- started_at_min: `2026-05-17T14:36:43.394Z`, finished_at_max: `2026-05-17T23:10:53.502Z`

- source_shards_unchanged: **True**
- stage3_signoff_present: False
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
| `shard_00.execution.sqlite` | 219 | 4 | 0 | 26.04 | `26c8cdb484141b57` |
| `shard_01.execution.sqlite` | 230 | 0 | 0 | 0.00 | `b7e6dc0f0a297ab2` |
| `shard_02.execution.sqlite` | 223 | 0 | 0 | 0.00 | `213568c15930e6f8` |
| `shard_03.execution.sqlite` | 235 | 4 | 0 | 14481.51 | `13b56d5fe3735f6a` |
| `shard_04.execution.sqlite` | 226 | 4 | 0 | 2032.39 | `b99e1e87d45f516b` |
| `shard_05.execution.sqlite` | 226 | 4 | 0 | 14259.38 | `9b10c9e7eb87796f` |
| `shard_06.execution.sqlite` | 228 | 0 | 0 | 0.00 | `daddb277081e3675` |
| `shard_07.execution.sqlite` | 270 | 0 | 0 | 0.00 | `ef6914171f769d66` |
| `shard_08.execution.sqlite` | 237 | 4 | 0 | 13.68 | `23f89f8b856af27f` |
| `shard_09.execution.sqlite` | 210 | 4 | 0 | 31.47 | `2c5006b552345338` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## stage0 extreme-lane execution summary

- batch_id: `stage0_extreme_lane`
- lane: `extreme`
- **execution_status: `executed`**
- n_source_shards: 10
- run_dir: `runs/cc18/stage0_extreme_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09` (pinned: `47b6b50c6d1e1d09`)
- plan_doc: `docs/EXTREME_LANE_PLAN.md`

- n_jobs_total: 2304
- expected extreme-canary cells: 24
- executed: **24**, success: **24**, deferred_standard: 1815, deferred_heavy: 423, refused: 42, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 30844.5 s

- extreme lane executed with policy-defined stage0_max_evaluations=1
- extreme lane executed with policy-defined timeout_seconds_per_cell=14400.0

- task_lane_counts_universe: {'standard': 57, 'extreme': 2, 'heavy': 13}
- extreme tasks executed: [6, 167121]

### Commit 42 plan pre-flight

- plan_summary_path: `/Users/caiotertulianoribeiro/Projects/doe_nbi_hpo_project/experiments/_stage_runs/stage0_extreme_lane_plan_latest_summary.json`
- exported_at: `2026-05-17T13:48:17Z`
- age_days: 0.03
- runnable_extreme_canary (planned): 24
- extreme_tasks_to_execute (planned): [6, 167121]

### stage0 standard-lane pre-flight

- exported_at: `2026-05-16T04:44:08Z`
- age_days: 1.41
- n_executed=684, failed=0, pending=0
- policy_version: `47b6b50c6d1e1d09`

### stage0 heavy-lane pre-flight

- exported_at: `2026-05-17T04:31:54Z`
- age_days: 0.42
- n_executed=156, failed=0, pending=0
- policy_version: `47b6b50c6d1e1d09`

### Per-task breakdown (extreme only)

| task_id | total | success | failed | failed_timeout | skipped | runtime_total_s | runtime_max_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 33 | 12 | 0 | 0 | 21 | 71.2 | 24.5 |
| 167121 | 33 | 12 | 0 | 0 | 21 | 30773.3 | 10663.5 |

### Per-algorithm breakdown (executed extreme rows)

| algorithm | n_total | success | failed | runtime_total_s | runtime_max_s |
|---|---:|---:|---:|---:|---:|
| `catboost` | 8 | 8 | 0 | 14285.4 | 10218.7 |
| `lightgbm` | 8 | 8 | 0 | 2063.9 | 1473.9 |
| `xgboost` | 8 | 8 | 0 | 14495.2 | 10663.5 |

### Per-method breakdown (executed extreme rows)

| method | n_total | success | failed | runtime_total_s | runtime_max_s |
|---|---:|---:|---:|---:|---:|
| `default_gbdt` | 6 | 6 | 0 | 3241.5 | 1541.2 |
| `doe_rsm_vrf_true_nbi` | 6 | 6 | 0 | 22408.9 | 10663.5 |
| `random_search` | 6 | 6 | 0 | 2605.7 | 1250.9 |
| `tpe_optuna` | 6 | 6 | 0 | 2588.4 | 1248.6 |

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

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 167121 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_03.execution.sqlite` | 10663.53 |
| 167121 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_05.execution.sqlite` | 10218.71 |
| 167121 | `default_gbdt` | `catboost` | `shard_05.execution.sqlite` | 1541.16 |
| 167121 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_04.execution.sqlite` | 1473.92 |
| 167121 | `default_gbdt` | `xgboost` | `shard_03.execution.sqlite` | 1417.14 |
| 167121 | `random_search` | `catboost` | `shard_05.execution.sqlite` | 1250.91 |
| 167121 | `tpe_optuna` | `catboost` | `shard_05.execution.sqlite` | 1248.59 |
| 167121 | `random_search` | `xgboost` | `shard_03.execution.sqlite` | 1205.55 |
| 167121 | `tpe_optuna` | `xgboost` | `shard_03.execution.sqlite` | 1195.29 |
| 167121 | `default_gbdt` | `lightgbm` | `shard_04.execution.sqlite` | 275.48 |
| 167121 | `random_search` | `lightgbm` | `shard_04.execution.sqlite` | 143.96 |
| 167121 | `tpe_optuna` | `lightgbm` | `shard_04.execution.sqlite` | 139.03 |

### Signoff note

stage0_replica_001 now has standard, heavy, and extreme lane summaries. stage3_signoff.json is intentionally absent until a later signoff commit.

### stage0 extreme-lane verdict: **GATE PASS**

Stage 0 replica 1 now has standard + heavy + extreme stage-run summaries pinned to the same policy_version. Commit 44 may begin the aggregate signoff plan; stage3_signoff.json should NOT be created until that planning step ships.
