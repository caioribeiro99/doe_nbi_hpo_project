# CC18 stage-run summary -- `stage0_heavy_lane_latest`

- run_id: `stage0_heavy_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage0_heavy_lane`
- exported_at: `2026-05-17T04:31:54Z`
- source_git_sha: `daae8ab95aa8`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage0_heavy_lane_latest` (gitignored)

- total jobs: **2304**
- success: **156**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 2148
- runtime: total 34889.3s, max 3478.23s across 156 recorded jobs
- started_at_min: `2026-05-16T13:40:45.551Z`, finished_at_max: `2026-05-17T04:31:17.363Z`

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
| `shard_00.execution.sqlite` | 219 | 12 | 0 | 555.29 | `af744f3d3c1c4660` |
| `shard_01.execution.sqlite` | 230 | 12 | 0 | 2480.50 | `21992f27e583a93f` |
| `shard_02.execution.sqlite` | 223 | 12 | 0 | 7004.19 | `11a4df1676b7e014` |
| `shard_03.execution.sqlite` | 235 | 8 | 0 | 5071.82 | `d5bd9e01bcfa1c48` |
| `shard_04.execution.sqlite` | 226 | 8 | 0 | 8918.45 | `230e43705ea4704e` |
| `shard_05.execution.sqlite` | 226 | 20 | 0 | 2433.04 | `36d32273060df94e` |
| `shard_06.execution.sqlite` | 228 | 24 | 0 | 985.74 | `b89c8ec17d6424a1` |
| `shard_07.execution.sqlite` | 270 | 32 | 0 | 4268.11 | `1e82261c32b438b9` |
| `shard_08.execution.sqlite` | 237 | 16 | 0 | 39.65 | `716128ce0c446cd5` |
| `shard_09.execution.sqlite` | 210 | 12 | 0 | 3132.48 | `17dc60ce2dc389eb` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## stage0 heavy-lane summary

- batch_id: `stage0_heavy_lane`
- lane: `heavy`
- n_source_shards: 10
- run_dir: `runs/cc18/stage0_heavy_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09` (pinned from Commit 40)
- n_jobs_total (across shards): 2304
- expected heavy-lane canary cells: 156
- executed: **156**, deferred_standard: **1815**, deferred_extreme: **66**, refused_non_canary: **267**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 34889.3 s

- task_lane_counts_universe: {'standard': 57, 'extreme': 2, 'heavy': 13}
- non_canary_methods_refused: ['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### stage0 standard-lane pre-flight

- exported_at: `2026-05-16T04:44:08Z`
- age_days: 0.37
- n_executed=684, failed=0, pending=0
- source_shards_unchanged: True
- run_id: `stage0_standard_lane_latest`
- policy_version (standard-lane): `47b6b50c6d1e1d09`

### Per-shard status

| shard | total | success | failed | failed_to | pending | skipped | def_std | def_extreme | refused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_00.execution.sqlite` | 219 | 12 | 0 | 0 | 0 | 207 | 177 | 11 | 19 |
| `shard_01.execution.sqlite` | 230 | 12 | 0 | 0 | 0 | 218 | 199 | 0 | 19 |
| `shard_02.execution.sqlite` | 223 | 12 | 0 | 0 | 0 | 211 | 192 | 0 | 19 |
| `shard_03.execution.sqlite` | 235 | 8 | 0 | 0 | 0 | 227 | 202 | 11 | 14 |
| `shard_04.execution.sqlite` | 226 | 8 | 0 | 0 | 0 | 218 | 193 | 11 | 14 |
| `shard_05.execution.sqlite` | 226 | 20 | 0 | 0 | 0 | 206 | 160 | 11 | 35 |
| `shard_06.execution.sqlite` | 228 | 24 | 0 | 0 | 0 | 204 | 162 | 0 | 42 |
| `shard_07.execution.sqlite` | 270 | 32 | 0 | 0 | 0 | 238 | 182 | 0 | 56 |
| `shard_08.execution.sqlite` | 237 | 16 | 0 | 0 | 0 | 221 | 182 | 11 | 28 |
| `shard_09.execution.sqlite` | 210 | 12 | 0 | 0 | 0 | 198 | 166 | 11 | 21 |

### Heavy tasks executed

13 tasks: [32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825, 167119, 167120, 167124, 167125]

### Standard tasks deferred (Commit 40 already ran them)

57 tasks. (See stage0_standard_lane_latest_summary.md for the executed list.)

### Extreme tasks deferred (require explicit opt-in)

2 tasks: [6, 167121]

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 167124 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_04.execution.sqlite` | 3478.23 |
| 167124 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_02.execution.sqlite` | 3241.06 |
| 167124 | `random_search` | `catboost` | `shard_04.execution.sqlite` | 2534.20 |
| 167124 | `tpe_optuna` | `catboost` | `shard_04.execution.sqlite` | 2376.30 |
| 167124 | `tpe_optuna` | `xgboost` | `shard_02.execution.sqlite` | 2042.48 |
| 167124 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_03.execution.sqlite` | 1940.72 |
| 167124 | `tpe_optuna` | `lightgbm` | `shard_03.execution.sqlite` | 1471.57 |
| 146825 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_07.execution.sqlite` | 1444.90 |
| 3573 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_09.execution.sqlite` | 1421.24 |
| 167124 | `random_search` | `lightgbm` | `shard_03.execution.sqlite` | 1392.06 |
| 167124 | `random_search` | `xgboost` | `shard_02.execution.sqlite` | 1142.05 |
| 146825 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_05.execution.sqlite` | 1125.46 |

### isolet (task 3481) recalibration note

isolet (task 3481) was observed in Commit 40 standard-lane at 1078.6 s. It remains in the standard lane under this policy_version. Future recalibration may promote it to heavy via the observed-runtime>=900 rule; do that between replicas, not mid-replica.

### stage0 heavy-lane verdict: **GATE PASS**

The extreme-lane pass remains gated behind explicit `--include-extreme-tasks` and operator review of `docs/HEAVY_TASK_POLICY.md`. Do NOT run the extreme lane without a planning step that anticipates Devnagari-Script runtime.
