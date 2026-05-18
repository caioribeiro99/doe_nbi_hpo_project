# stage 0 replica 001 — aggregate signoff plan

- run_id: `stage0_replica_001_signoff_plan_latest`
- stage: `stage0_replica_001`
- exported_at: `2026-05-18T18:17:24Z`
- **signoff_status: `signed`**
- stage3_signoff_present: True
- final_recommendation: **signed_ready_for_next_stage_planning**
- signoff_plan_doc: `docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md`

## Lane summary references

| lane | path | sha256 |
|---|---|---|
| `standard` | `experiments/_stage_runs/stage0_standard_lane_latest_summary.json` | `2f8c7f051c566f95d14c6956b87814783ea6273de261d30eae1ed02897f19404` |
| `heavy` | `experiments/_stage_runs/stage0_heavy_lane_latest_summary.json` | `cb75565820bf21a221e14075d7ae9cb6e39946765accbd52aa9db1fe77a69dbb` |
| `extreme` | `experiments/_stage_runs/stage0_extreme_lane_latest_summary.json` | `e502346357cbe3063cb78ed06790375b2a1e7e6ec8e620430de7a5789c977fdf` |
| `extreme_plan` | `experiments/_stage_runs/stage0_extreme_lane_plan_latest_summary.json` | `f0fbaf3095cfa1c5c00d68a0800608f82df0cb74b265211ffbee5716cefac813` |

## Aggregate invariants

- policy_version: `47b6b50c6d1e1d09` (consistent across lanes: True)
- all_lane_summaries_green: **True**
- source_shards_unchanged_all_lanes: **True**
- no_pending_running_failed_all_lanes: **True**

## Aggregate counts

- n_jobs_total_expected: 2304
- n_standard_success: 684
- n_heavy_success: 156
- n_extreme_success: 24
- **n_canary_success_total: 864**
- n_failed_total: 0
- n_failed_timeout_total: 0
- n_pending_total: 0
- n_running_total: 0
- non_canary_refused_total: 1440

## Task coverage

- n_tasks_total_expected: 72
- n_standard_tasks_expected: 57
- n_heavy_tasks_expected: 13
- n_extreme_tasks_expected: 2
- task_lane_counts_universe: {'extreme': 2, 'heavy': 13, 'standard': 57}
- n_tasks_observed_success: 72

## Method + algorithm coverage

- methods expected: `['default_gbdt', 'random_search', 'tpe_optuna', 'doe_rsm_vrf_true_nbi']`
- methods observed: `['default_gbdt', 'doe_rsm_vrf_true_nbi', 'random_search', 'tpe_optuna']`
- algorithms expected: `['xgboost', 'lightgbm', 'catboost']`
- algorithms observed: `['catboost', 'lightgbm', 'xgboost']`

## Runtime summary

| lane | runtime (s) | runtime (h) |
|---|---:|---:|
| `standard` | 6801.3 | 1.89 |
| `heavy` | 34889.3 | 9.69 |
| `extreme` | 30844.5 | 8.57 |
| **total** | **72535.0** | **20.15** |

### Slowest cells (success, all lanes)

| task_id | method | algorithm | lane | runtime_s |
|---:|---|---|---|---:|
| 167121 | `doe_rsm_vrf_true_nbi` | `xgboost` | `extreme` | 10663.53 |
| 167121 | `doe_rsm_vrf_true_nbi` | `catboost` | `extreme` | 10218.71 |
| 167124 | `doe_rsm_vrf_true_nbi` | `catboost` | `heavy` | 3478.23 |
| 167124 | `doe_rsm_vrf_true_nbi` | `xgboost` | `heavy` | 3241.06 |
| 167124 | `random_search` | `catboost` | `heavy` | 2534.20 |
| 167124 | `tpe_optuna` | `catboost` | `heavy` | 2376.30 |
| 167124 | `tpe_optuna` | `xgboost` | `heavy` | 2042.48 |
| 167124 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `heavy` | 1940.72 |
| 167121 | `default_gbdt` | `catboost` | `extreme` | 1541.16 |
| 167121 | `doe_rsm_vrf_true_nbi` | `lightgbm` | `extreme` | 1473.92 |
| 167124 | `tpe_optuna` | `lightgbm` | `heavy` | 1471.57 |
| 146825 | `doe_rsm_vrf_true_nbi` | `catboost` | `heavy` | 1444.90 |

### Slowest tasks (success, all lanes)

| task_id | n_cells | runtime_total_s | runtime_max_s |
|---:|---:|---:|---:|
| 167121 | 12 | 30773.3 | 10663.5 |
| 167124 | 12 | 20868.6 | 3478.2 |
| 146825 | 12 | 7101.1 | 1444.9 |
| 3573 | 12 | 6111.7 | 1421.2 |
| 3481 | 12 | 3718.1 | 1078.6 |
| 14970 | 12 | 911.9 | 246.8 |
| 167125 | 12 | 336.4 | 134.5 |
| 9964 | 12 | 272.5 | 106.4 |
| 12 | 12 | 251.4 | 84.6 |
| 14 | 12 | 170.7 | 49.9 |

## Metric availability (by task type)

### `binary` (35 tasks, 420 cells)

| metric | mean across cells |
|---|---:|
| `accuracy` | 0.8686 |
| `precision` | 0.7707 |
| `recall` | 0.6750 |
| `specificity` | 0.8747 |
| `balanced_accuracy` | — |

_missing keys_: `['balanced_accuracy']`

### `multiclass` (37 tasks, 444 cells)

| metric | mean across cells |
|---|---:|
| `accuracy` | 0.8291 |
| `balanced_accuracy` | 0.8033 |
| `f1_macro` | 0.8043 |
| `mcc` | 0.7815 |
| `roc_auc_ovr_macro` | 0.9460 |
| `pr_auc_ovr_macro` | 0.8480 |
| `brier_multiclass` | 0.2444 |
| `ece_multiclass` | 0.0761 |

## Metric aggregates per lane

| lane | n_cells | accuracy | balanced_accuracy | f1_macro | mcc |
|---|---:|---:|---:|---:|---:|
| `extreme` | 24 | 0.7340 | 0.7337 | 0.7294 | 0.7269 |
| `heavy` | 156 | 0.8227 | 0.7837 | 0.7861 | 0.7646 |
| `standard` | 684 | 0.8582 | 0.8132 | 0.8142 | 0.7896 |

## Metric aggregates per method

| method | n_cells | accuracy | balanced_accuracy | f1_macro |
|---|---:|---:|---:|---:|
| `default_gbdt` | 216 | 0.8445 | 0.7980 | 0.7991 |
| `doe_rsm_vrf_true_nbi` | 216 | 0.8523 | 0.8092 | 0.8103 |
| `random_search` | 216 | 0.8483 | 0.8031 | 0.8037 |
| `tpe_optuna` | 216 | 0.8482 | 0.8030 | 0.8042 |

## Metric aggregates per algorithm

| algorithm | n_cells | accuracy | balanced_accuracy | f1_macro |
|---|---:|---:|---:|---:|
| `catboost` | 288 | 0.8465 | 0.7957 | 0.7963 |
| `lightgbm` | 288 | 0.8457 | 0.8001 | 0.8010 |
| `xgboost` | 288 | 0.8528 | 0.8143 | 0.8156 |

## Caveats

- isolet (task 3481) ran in the standard lane but its doe_rsm_vrf_true_nbi catboost cell took ~1078.6 s. The observed-runtime>=900 rule would promote it to heavy in a future replica via scripts/build_cc18_heavy_task_policy.py; this replica intentionally stays on the Commit 38 policy_version to keep all four stage-0 artifacts consistent.
- Devnagari-Script (task 167121) ran in the extreme lane under the policy-defined extreme.stage0_max_evaluations=1 (YAML default). Standard- and heavy-lane cells ran at max_evaluations=5. random_search / tpe_optuna / default_gbdt cells on Devnagari therefore exercised fewer configurations than the rest of the panel, but doe_rsm_vrf_true_nbi (which uses n_doe=max(2*d, max_evaluations)) hit the d=4 floor and is unchanged from batch_03. Headline metrics for Devnagari are NOT directly budget-equivalent to the other 70 tasks.
- Heavy- and extreme-lane handling is a reproducibility feature (timeout protection, per-lane budgets), not a post-hoc result filter. The full replica is the 864 canary successes across 72 tasks; subsetting analyses should respect lane assignments only as a runtime budget marker.

## What the later signoff commit should do

- create jobs/doctoral/openml_cc18/stage3_signoff.json (operator-reviewed)
- record final policy_version (currently 47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36)
- record the three lane summary SHA-256 hashes recorded in this signoff plan
- record explicit approval metadata (operator, timestamp, justification)

## Verdict: **SIGNED — READY FOR NEXT-STAGE PLANNING**

Stage 0 replica 1 was signed off at `2026-05-18T18:17:24Z` by `Caio Tertuliano Ribeiro` (`caioribeiro99`) on branch `repo-publication-readiness` at git_sha `77035d066283`. downstream_execution_authorized_in_this_commit: False.

- stage3_signoff_path: `jobs/doctoral/openml_cc18/stage3_signoff.json`
- stage3_signoff_sha256: `3f3f1b1fd681934404078ea9d9ad665c274eb1394dfd05950c24a760fd7608b1`

Caveats remain in force (see above). A future commit may plan the next execution tier (stage-3 top-ups) but must do so explicitly — this signoff is the gate, not the trigger.
