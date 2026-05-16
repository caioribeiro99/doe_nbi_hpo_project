# CC18 stage-run summary -- `stage0_standard_lane_latest`

- run_id: `stage0_standard_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage0_standard_lane`
- exported_at: `2026-05-16T04:44:08Z`
- source_git_sha: `a95e7bef84d4`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage0_standard_lane_latest` (gitignored)

- total jobs: **2304**
- success: **684**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 1620
- runtime: total 6801.3s, max 1078.63s across 684 recorded jobs
- started_at_min: `2026-05-16T02:48:22.678Z`, finished_at_max: `2026-05-16T04:41:55.156Z`

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
| `shard_00.execution.sqlite` | 219 | 68 | 0 | 295.77 | `bfb9a7a3e96082f2` |
| `shard_01.execution.sqlite` | 230 | 76 | 0 | 279.88 | `dee6b16ddca500fd` |
| `shard_02.execution.sqlite` | 223 | 72 | 0 | 929.76 | `5a39aeab6ddafa53` |
| `shard_03.execution.sqlite` | 235 | 76 | 0 | 602.47 | `995d267c9a70a0f2` |
| `shard_04.execution.sqlite` | 226 | 72 | 0 | 1107.89 | `87f3c7576271f0b2` |
| `shard_05.execution.sqlite` | 226 | 60 | 0 | 2707.24 | `0e6bb2a580f8fa6e` |
| `shard_06.execution.sqlite` | 228 | 60 | 0 | 156.33 | `d24c29cff190d360` |
| `shard_07.execution.sqlite` | 270 | 68 | 0 | 189.51 | `473ae2975c6adbf5` |
| `shard_08.execution.sqlite` | 237 | 68 | 0 | 443.28 | `e6c2e0fc45a4f0db` |
| `shard_09.execution.sqlite` | 210 | 64 | 0 | 89.19 | `1397044a46186c52` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## stage0 standard-lane summary

- batch_id: `stage0_standard_lane`
- lane: `standard`
- n_source_shards: 10
- run_dir: `runs/cc18/stage0_standard_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- n_jobs_total (across shards): 2304
- expected standard-lane canary cells: 684
- executed: **684**, deferred_heavy: **423**, deferred_extreme: **66**, refused_non_canary: **1131**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 6801.3 s

- task_lane_counts_universe: {'standard': 57, 'extreme': 2, 'heavy': 13}
- non_canary_methods_refused: ['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### batch_04 pre-flight

- exported_at: `2026-05-15T21:31:21Z`
- age_days: 0.22
- failed_timeout=0, failed_other=0, pending_after=0
- source_shards_unchanged: True
- run_id: `batch_04_stage0_shard00_only_latest`

### Per-shard status

| shard | total | success | failed | failed_to | pending | skipped | def_heavy | def_extreme | refused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_00.execution.sqlite` | 219 | 68 | 0 | 0 | 0 | 151 | 31 | 11 | 109 |
| `shard_01.execution.sqlite` | 230 | 76 | 0 | 0 | 0 | 154 | 31 | 0 | 123 |
| `shard_02.execution.sqlite` | 223 | 72 | 0 | 0 | 0 | 151 | 31 | 0 | 120 |
| `shard_03.execution.sqlite` | 235 | 76 | 0 | 0 | 0 | 159 | 22 | 11 | 126 |
| `shard_04.execution.sqlite` | 226 | 72 | 0 | 0 | 0 | 154 | 22 | 11 | 121 |
| `shard_05.execution.sqlite` | 226 | 60 | 0 | 0 | 0 | 166 | 55 | 11 | 100 |
| `shard_06.execution.sqlite` | 228 | 60 | 0 | 0 | 0 | 168 | 66 | 0 | 102 |
| `shard_07.execution.sqlite` | 270 | 68 | 0 | 0 | 0 | 202 | 88 | 0 | 114 |
| `shard_08.execution.sqlite` | 237 | 68 | 0 | 0 | 0 | 169 | 44 | 11 | 114 |
| `shard_09.execution.sqlite` | 210 | 64 | 0 | 0 | 0 | 146 | 33 | 11 | 102 |

### Standard tasks executed

57 tasks: [3, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 37, 43, 45, 49, 53, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3902, 3903, 3904, 3913, 3917, 3918, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9985, 10093, 10101, 14952, 14954, 14969, 14970, 125920, 125922, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 167140, 167141]

### Heavy tasks deferred (Commit 41 will run them)

13 tasks: [32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825, 167119, 167120, 167124, 167125]

### Extreme tasks deferred (require explicit opt-in)

2 tasks: [6, 167121]

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 3481 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_05.execution.sqlite` | 1078.63 |
| 3481 | `random_search` | `catboost` | `shard_05.execution.sqlite` | 741.58 |
| 3481 | `tpe_optuna` | `catboost` | `shard_05.execution.sqlite` | 675.29 |
| 3481 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_04.execution.sqlite` | 258.13 |
| 14970 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_02.execution.sqlite` | 246.81 |
| 3481 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_03.execution.sqlite` | 189.38 |
| 14970 | `random_search` | `catboost` | `shard_02.execution.sqlite` | 170.59 |
| 3481 | `tpe_optuna` | `lightgbm` | `shard_04.execution.sqlite` | 164.50 |
| 3481 | `random_search` | `lightgbm` | `shard_04.execution.sqlite` | 162.78 |
| 14970 | `tpe_optuna` | `catboost` | `shard_02.execution.sqlite` | 153.63 |
| 3481 | `default_gbdt` | `catboost` | `shard_05.execution.sqlite` | 135.72 |
| 9964 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_04.execution.sqlite` | 106.40 |

### stage0 standard-lane verdict: **GATE PASS**

Commit 41 may prepare the heavy-lane pass. Do NOT run the heavy / extreme lanes without explicit operator sign-off; the extreme lane still requires --include-extreme-tasks.
