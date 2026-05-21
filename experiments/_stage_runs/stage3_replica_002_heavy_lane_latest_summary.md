# CC18 stage-run summary -- `stage3_replica_002_heavy_lane_latest`

- run_id: `stage3_replica_002_heavy_lane_latest`
- stage: `stage0_replica_001`
- batch_id: `stage3_replica_002_heavy_lane`
- exported_at: `2026-05-21T11:02:39Z`
- source_git_sha: `8bf0ce8cb3d2`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/stage3_replica_002_heavy_lane_latest` (gitignored)

- total jobs: **2304**
- success: **156**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 2148
- runtime: total 37300.6s, max 3750.18s across 156 recorded jobs
- started_at_min: `2026-05-21T00:40:12.517Z`, finished_at_max: `2026-05-21T11:02:04.287Z`

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
| `shard_00.execution.sqlite` | 219 | 12 | 0 | 563.80 | `292eb02ad9382734` |
| `shard_01.execution.sqlite` | 230 | 12 | 0 | 2549.17 | `dca497f26b02c0d8` |
| `shard_02.execution.sqlite` | 223 | 12 | 0 | 7991.38 | `405ab1599ee656c4` |
| `shard_03.execution.sqlite` | 235 | 8 | 0 | 5162.49 | `514e139803f50351` |
| `shard_04.execution.sqlite` | 226 | 8 | 0 | 9710.76 | `b43ed7895ea6ef8f` |
| `shard_05.execution.sqlite` | 226 | 20 | 0 | 2553.82 | `846a419d5accece7` |
| `shard_06.execution.sqlite` | 228 | 24 | 0 | 947.47 | `d6e3f38b2ce1a316` |
| `shard_07.execution.sqlite` | 270 | 32 | 0 | 4396.96 | `cf88447c6d8c20a2` |
| `shard_08.execution.sqlite` | 237 | 16 | 0 | 41.04 | `9c335cf77eeacba9` |
| `shard_09.execution.sqlite` | 210 | 12 | 0 | 3383.72 | `9f210b679b8a082e` |

## Failures (grouped)

_(none)_

## Verdict: **NOT GREEN**

Re-run / archive the failures; investigate any source-shard drift; do not promote downstream until this summary clears.

---

## stage 3 / replica_002 heavy-lane summary (Commit 49)

- run_id: `stage3_replica_002_heavy_lane_latest`
- batch_id: `stage3_replica_002_heavy_lane`
- stage: `stage1_topup_to_005`
- topup_tier: `topup_to_5_partial`
- replica: **2** (source template replica = 1)
- lane: `heavy`
- n_source_shards: 10
- run_dir: `runs/cc18/stage3_replica_002_heavy_lane_latest` (gitignored)
- policy_version: `47b6b50c6d1e1d09`
- policy_version_pinned: `47b6b50c6d1e1d09`
- signoff_path: `jobs/doctoral/openml_cc18/stage3_signoff.json`
- signoff_sha256: `3f3f1b1fd6819344`
- stage3_topup_plan_summary_sha256: `bb79eaee6b5a4da2`
- commit48_standard_lane_summary_sha256: `6d028cef7c3715cf`

- n_jobs_total (across 10 shards): 2304
- expected runnable heavy-lane canary cells: **156**
- executed: **156**, success: **156**, deferred_standard: **1815**, deferred_extreme: **66**, refused_non_canary: **267**, failed_timeout: **0**, failed_other: **0**, pending_after: 0
- runtime (runner only): 37300.6 s

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

### Heavy tasks executed (success only)

13 tasks: [32, 219, 3573, 7592, 9910, 9981, 14965, 146195, 146825, 167119, 167120, 167124, 167125]

### Standard tasks deferred (already handled by Commit 48)

57 tasks: [3, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 37, 43, 45, 49, 53, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3902, 3903, 3904, 3913, 3917, 3918, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9985, 10093, 10101, 14952, 14954, 14969, 14970, 125920, 125922, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 167140, 167141]

### Extreme tasks deferred (later commit)

2 tasks: [6, 167121]

### Non-canary methods refused

['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

### isolet recalibration note

- isolet/task 3481 is currently `standard` under the pinned policy_version and is NOT promoted to heavy by this commit.

### Slowest executed cells

| task_id | method | algorithm | shard | runtime_s |
|---:|---|---|---|---:|
| 167124 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_04.execution.sqlite` | 3750.18 |
| 167124 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_02.execution.sqlite` | 3431.06 |
| 167124 | `tpe_optuna` | `catboost` | `shard_04.execution.sqlite` | 2922.09 |
| 167124 | `random_search` | `catboost` | `shard_04.execution.sqlite` | 2508.65 |
| 167124 | `tpe_optuna` | `xgboost` | `shard_02.execution.sqlite` | 2221.35 |
| 167124 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `shard_03.execution.sqlite` | 1986.55 |
| 167124 | `random_search` | `xgboost` | `shard_02.execution.sqlite` | 1747.34 |
| 167124 | `random_search` | `lightgbm` | `shard_03.execution.sqlite` | 1509.71 |
| 3573 | `doe_rsm_vrf_true_nbi` | `xgboost` | `shard_09.execution.sqlite` | 1484.33 |
| 146825 | `doe_rsm_vrf_true_nbi` | `catboost` | `shard_07.execution.sqlite` | 1458.98 |
| 167124 | `tpe_optuna` | `lightgbm` | `shard_03.execution.sqlite` | 1406.30 |
| 146825 | `tpe_optuna` | `catboost` | `shard_07.execution.sqlite` | 1199.23 |

### stage 3 replica_002 heavy lane verdict: **GATE PASS — operator review required**

Run finished cleanly: every heavy-lane canary cell on replica_002 across all 10 shards reached a terminal status, every committed source shard is byte-identical to its pre-run MD5, standard lane was not rerun, extreme lane was not executed, and isolet was not promoted to heavy. Commit 50 may plan the replica_002 extreme lane; do NOT scale to replica_003-005 without an aggregate review of replica_002 standard + heavy + extreme.
