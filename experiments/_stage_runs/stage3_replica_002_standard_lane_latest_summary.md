# CC18 stage-run summary -- `stage3_replica_002_standard_lane_latest`

- run_id: `stage3_replica_002_standard_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage3_replica_002_standard_lane`
- exported_at: `2026-05-20T20:42:05Z`
- source_git_sha: `20cc7a311e97`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage3_replica_002_standard_lane_latest` (gitignored)

- total jobs: **2304**
- success: **684**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 1620
- runtime: total 7287.9s, max 1143.68s across 684 recorded jobs
- started_at_min: `2026-05-20T18:38:17.039Z`, finished_at_max: `2026-05-20T20:39:56.937Z`

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
| `shard_00.execution.sqlite` | 219 | 68 | 0 | 347.05 | `9c000cc98596eb79` |
| `shard_01.execution.sqlite` | 230 | 76 | 0 | 269.74 | `87629221b8b63dab` |
| `shard_02.execution.sqlite` | 223 | 72 | 0 | 1005.83 | `0dd72e99ca7755d9` |
| `shard_03.execution.sqlite` | 235 | 76 | 0 | 582.49 | `2df32d9943637ded` |
| `shard_04.execution.sqlite` | 226 | 72 | 0 | 1181.49 | `9bde88b5349956c1` |
| `shard_05.execution.sqlite` | 226 | 60 | 0 | 3039.01 | `0dbcaffb32d0c67b` |
| `shard_06.execution.sqlite` | 228 | 60 | 0 | 110.97 | `1bfd3f359df10940` |
| `shard_07.execution.sqlite` | 270 | 68 | 0 | 160.25 | `00a9f94bbffe08f9` |
| `shard_08.execution.sqlite` | 237 | 68 | 0 | 496.19 | `4c0dccb9abc72dc2` |
| `shard_09.execution.sqlite` | 210 | 64 | 0 | 94.87 | `d69246d0fa9ed71d` |

## Failures (grouped)

_(none)_

## Verdict: **NOT GREEN**

Re-run / archive the failures; investigate any source-shard drift; do not promote downstream until this summary clears.

---

## stage 3 / replica_002 standard-lane summary (Commit 48)

- run_id: `stage3_replica_002_standard_lane_latest`
- batch_id: `stage3_replica_002_standard_lane`
- stage: `stage1_topup_to_005`
- topup_tier: `topup_to_5_partial`
- replica: **2** (source template replica = 1)
- lane: `standard`
- n_source_shards: 10
- run_dir: `runs/cc18/stage3_replica_002_standard_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- policy_version_pinned: `47b6b50c6d1e1d09`
- signoff_path: `jobs/doctoral/openml_cc18/stage3_signoff.json`
- signoff_sha256: `3f3f1b1fd6819344`
- stage3_topup_plan_summary_sha256: `bb79eaee6b5a4da2`
- commit47_pilot_summary_sha256: `848c2f67ca832e00`

- n_jobs_total (across 10 shards): 2304
- expected runnable standard-lane canary cells: **684**
- executed: **684**, success: **684**, deferred_heavy: **423**, deferred_extreme: **66**, refused_non_canary: **1131**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 7287.9 s

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

### Standard tasks executed (success only)

57 tasks: [3, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 37, 43, 45, 49, 53, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3902, 3903, 3904, 3913, 3917, 3918, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9985, 10093, 10101, 14952, 14954, 14969, 14970, 125920, 125922, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 167140, 167141]

### Heavy tasks deferred

13 tasks: [32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825, 167119, 167120, 167124, 167125]

### Extreme tasks deferred

2 tasks: [6, 167121]

### Non-canary methods refused

['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 3481 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_05.execution.sqlite` | 1143.68 |
| 3481 | `tpe_optuna` | `catboost` | `shard_05.execution.sqlite` | 909.01 |
| 3481 | `random_search` | `catboost` | `shard_05.execution.sqlite` | 796.12 |
| 3481 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_04.execution.sqlite` | 267.20 |
| 14970 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_02.execution.sqlite` | 263.64 |
| 14970 | `tpe_optuna` | `catboost` | `shard_02.execution.sqlite` | 211.22 |
| 3481 | `tpe_optuna` | `lightgbm` | `shard_04.execution.sqlite` | 204.21 |
| 3481 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_03.execution.sqlite` | 194.65 |
| 14970 | `random_search` | `catboost` | `shard_02.execution.sqlite` | 183.01 |
| 3481 | `random_search` | `lightgbm` | `shard_04.execution.sqlite` | 158.93 |
| 3481 | `default_gbdt` | `catboost` | `shard_05.execution.sqlite` | 136.68 |
| 3481 | `random_search` | `xgboost` | `shard_03.execution.sqlite` | 122.89 |

### stage 3 replica_002 standard lane verdict: **GATE PASS — operator review required**

Run finished cleanly: every standard-lane canary cell on replica_002 across all 10 shards reached a terminal status, every committed source shard is byte-identical to its pre-run MD5, and no heavy / extreme cell ran. Commit 49 may plan or run replica_002 heavy lane (or a selected heavy probe first), but **not** the full topup_to_5 tier without explicit operator sign-off.
