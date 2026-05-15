# CC18 stage-run summary -- `batch_04_stage0_shard00_only_latest`

- run_id: `batch_04_stage0_shard00_only_latest`
- stage: `stage0_replica_001`
- batch_id: `batch_04_stage0_shard00_only`
- exported_at: `2026-05-15T21:31:21Z`
- source_git_sha: `beb423d81b30`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/batch_04_stage0_shard00_only_latest` (gitignored)

- total jobs: **219**
- success: **80**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 139
- runtime: total 748.4s, max 206.70s across 80 recorded jobs
- started_at_min: `2026-05-15T21:18:50.173Z`, finished_at_max: `2026-05-15T21:31:19.812Z`

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
| `shard_00.execution.sqlite` | 219 | 80 | 0 | 748.40 | `7566b6bf9df47278` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## batch_04-specific summary

- batch_id: `batch_04_stage0_shard00_only`
- source_shard: `jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite`
- execution_shard: `runs/cc18/batch_04_stage0_shard00_only_latest/shards/stage0_replica_001/shard_00.execution.sqlite` (gitignored)
- run_dir: `runs/cc18/batch_04_stage0_shard00_only_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- include_extreme_tasks: False
- n_jobs_total_in_shard: 219
- runnable: standard=68, heavy=12, deferred (extreme)=11
- executed: **80**, deferred: **11**, refused: **128**, failed_timeout: **0**, failed (other): **0**, pending_after: 0
- runtime (runner only): 748.4 s

- task_lane_counts_in_shard: {'standard': 17, 'extreme': 1, 'heavy': 3}
- deferred_extreme_tasks: [6]
- non_canary_methods_refused: ['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### batch_03 pre-flight

- exported_at: `2026-05-04T04:29:26Z`
- age_days: 11.70
- success: 216/216 (failed=0, pending=0)
- source_shards_unchanged: True
- run_id: `batch_03_cc18_representative_18_tasks_latest`

### Extended status counts

| status | count |
|---|---:|
| `success` | 80 |
| `failed` | 0 |
| `pending` | 0 |
| `running` | 0 |
| `claimed` | 0 |
| `skipped` | 139 |
| `deferred_extreme_lane` | 11 |
| `refused_not_in_canary_set` | 128 |
| `failed_timeout` | 0 |

### Slowest executed cells

| task_id | method | algorithm | lane | runtime_s |
|---:|---|---|---|---:|
| 3573 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `heavy` | 206.70 |
| 3573 | `tpe_optuna` | `lightgbm` | `heavy` | 96.44 |
| 3573 | `random_search` | `lightgbm` | `heavy` | 93.39 |
| 14970 | `doe_rsm_vrf_true_nbi` | `xgboost` | `standard` | 55.12 |
| 16 | `doe_rsm_vrf_true_nbi` | `catboost` | `standard` | 41.95 |
| 14970 | `tpe_optuna` | `xgboost` | `standard` | 29.19 |
| 16 | `random_search` | `catboost` | `standard` | 28.13 |
| 3573 | `default_gbdt` | `lightgbm` | `heavy` | 27.92 |

### batch_04 verdict: **GATE PASS**

stage-0 standard / heavy / extreme split may now be planned per `docs/HEAVY_TASK_POLICY.md`. Do not run full stage 0 without explicit operator sign-off.
