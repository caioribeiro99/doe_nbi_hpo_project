# stage 3 / top-up dispatch plan (planning-only)

- run_id: `stage3_topup_plan_latest`
- stage: `stage3_topup_plan`
- exported_at: `2026-05-19T14:53:19Z`
- **execution_status: `planned_not_executed`**
- drift_report_only: `False`
- policy_drift_detected: `False`

## Signoff context

- signoff_path: `jobs/doctoral/openml_cc18/stage3_signoff.json`
- signoff_status: **signed** (`stage0_replica_001`)
- operator: `Caio Tertuliano Ribeiro` (`caioribeiro99`)
- signed_at_utc: `2026-05-18T18:17:24Z`
- branch: `repo-publication-readiness` git_sha: `77035d066283`
- downstream_execution_authorized_in_this_commit: `False`

## Policy + lane summaries

- policy_version (live): `47b6b50c6d1e1d09…`
- signed policy_version: `47b6b50c6d1e1d09…`
- heavy_task_policy.csv path: `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv`
- runtime_guardrails.yaml path: `benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml`

| lane | live sha256 | signed sha256 |
|---|---|---|
| `standard` | `2f8c7f051c566f95…` | `2f8c7f051c566f95…` |
| `heavy` | `cb75565820bf21a2…` | `cb75565820bf21a2…` |
| `extreme` | `e502346357cbe306…` | `e502346357cbe306…` |

## Observed stage-0 runtime per replica per lane

| lane | total (s) | total (h) | per-cell p50 | per-cell p90 | per-cell max |
|---|---:|---:|---:|---:|---:|
| `standard` | 6801.3 | 1.89 | 0.6 s | 10.3 s | 18.0 min |
| `heavy` | 34889.3 | 9.69 | 3.2 s | 12.6 min | 58.0 min |
| `extreme` | 30844.5 | 8.57 | 1.4 min | 25.3 min | 2.96 h |

## Per-tier plan

Cells per replica: standard=684, heavy=156, extreme=24 — total 864 canary cells per replica.

| tier | replicas | canary cells | est. runtime (p50, h) | est. runtime (max, h) | shard subdir | shard rows total |
|---|---:|---:|---:|---:|---|---:|
| `topup_to_5` | 2–5 (4) | 3,456 | 80.59 | 1707.01 | `stage1_topup_to_005` | 9,216 |
| `topup_to_10` | 6–10 (5) | 4,320 | 100.74 | 2133.77 | `stage2_topup_to_010` | 13,680 |
| `topup_to_30` | 11–30 (20) | 17,280 | 402.97 | 8535.07 | `stage3_topup_to_030` | 54,720 |

### `topup_to_5` — lane breakdown

| lane | task_count | canary cells/replica | total canary cells | p50 (h) | max (h) |
|---|---:|---:|---:|---:|---:|
| `standard` | 57 | 684 | 2,736 | 7.56 | 819.76 |
| `heavy` | 13 | 156 | 624 | 38.77 | 602.89 |
| `extreme` | 2 | 24 | 96 | 34.27 | 284.36 |

### `topup_to_10` — lane breakdown

| lane | task_count | canary cells/replica | total canary cells | p50 (h) | max (h) |
|---|---:|---:|---:|---:|---:|
| `standard` | 57 | 684 | 3,420 | 9.45 | 1024.70 |
| `heavy` | 13 | 156 | 780 | 48.46 | 753.62 |
| `extreme` | 2 | 24 | 120 | 42.84 | 355.45 |

### `topup_to_30` — lane breakdown

| lane | task_count | canary cells/replica | total canary cells | p50 (h) | max (h) |
|---|---:|---:|---:|---:|---:|
| `standard` | 57 | 684 | 13,680 | 37.79 | 4098.80 |
| `heavy` | 13 | 156 | 3,120 | 193.83 | 3014.46 |
| `extreme` | 2 | 24 | 480 | 171.36 | 1421.80 |

## Aggregate across all tiers

- executable_canary_cells_total_all_tiers: **25,056**
- estimated_runtime_total_all_tiers (p50): 584.31 h
- estimated_runtime_total_all_tiers (max): 12375.85 h

## High-risk cells

_Cells flagged as high-risk for top-up scheduling (threshold 600s or known dataset)._

| task_id | dataset | lane | method | algorithm | runtime_s | reasons |
|---:|---|---|---|---|---:|---|
| 167121 | `Devnagari-Script` | `extreme` | `doe_rsm_vrf_true_nbi` | `xgboost` | 10663.5 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=10663.5>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `doe_rsm_vrf_true_nbi` | `catboost` | 10218.7 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=10218.7>=600 |
| 167124 | `CIFAR_10` | `heavy` | `doe_rsm_vrf_true_nbi` | `catboost` | 3478.2 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=3478.2>=600 |
| 167124 | `CIFAR_10` | `heavy` | `doe_rsm_vrf_true_nbi` | `xgboost` | 3241.1 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=3241.1>=600 |
| 167124 | `CIFAR_10` | `heavy` | `random_search` | `catboost` | 2534.2 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=2534.2>=600 |
| 167124 | `CIFAR_10` | `heavy` | `tpe_optuna` | `catboost` | 2376.3 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=2376.3>=600 |
| 167124 | `CIFAR_10` | `heavy` | `tpe_optuna` | `xgboost` | 2042.5 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=2042.5>=600 |
| 167124 | `CIFAR_10` | `heavy` | `doe_rsm_vrf_true_nbi` | `lightgbm` | 1940.7 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=1940.7>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `default_gbdt` | `catboost` | 1541.2 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1541.2>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `doe_rsm_vrf_true_nbi` | `lightgbm` | 1473.9 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1473.9>=600 |
| 167124 | `CIFAR_10` | `heavy` | `tpe_optuna` | `lightgbm` | 1471.6 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=1471.6>=600 |
| 146825 | `Fashion-MNIST` | `heavy` | `doe_rsm_vrf_true_nbi` | `catboost` | 1444.9 | heavy lane; n_features=784, n_rows=70000; observed_runtime_s=1444.9>=600 |
| 3573 | `mnist_784` | `heavy` | `doe_rsm_vrf_true_nbi` | `xgboost` | 1421.2 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=1421.2>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `default_gbdt` | `xgboost` | 1417.1 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1417.1>=600 |
| 167124 | `CIFAR_10` | `heavy` | `random_search` | `lightgbm` | 1392.1 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=1392.1>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `random_search` | `catboost` | 1250.9 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1250.9>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `tpe_optuna` | `catboost` | 1248.6 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1248.6>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `random_search` | `xgboost` | 1205.6 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1205.6>=600 |
| 167121 | `Devnagari-Script` | `extreme` | `tpe_optuna` | `xgboost` | 1195.3 | devnagari_extreme_budget_non_equivalence (signoff caveat 2); extreme lane; observed_runtime_s=1195.3>=600 |
| 167124 | `CIFAR_10` | `heavy` | `random_search` | `xgboost` | 1142.0 | heavy lane; n_features=3072, n_rows=60000; observed_runtime_s=1142.0>=600 |
| 146825 | `Fashion-MNIST` | `heavy` | `doe_rsm_vrf_true_nbi` | `xgboost` | 1125.5 | heavy lane; n_features=784, n_rows=70000; observed_runtime_s=1125.5>=600 |
| 146825 | `Fashion-MNIST` | `heavy` | `random_search` | `catboost` | 1084.1 | heavy lane; n_features=784, n_rows=70000; observed_runtime_s=1084.1>=600 |
| 3481 | `isolet` | `standard` | `doe_rsm_vrf_true_nbi` | `catboost` | 1078.6 | isolet_future_recalibration_candidate (signoff caveat 1); observed_runtime_s=1078.6>=600 |
| 146825 | `Fashion-MNIST` | `heavy` | `tpe_optuna` | `catboost` | 1065.8 | heavy lane; n_features=784, n_rows=70000; observed_runtime_s=1065.8>=600 |
| 3573 | `mnist_784` | `heavy` | `doe_rsm_vrf_true_nbi` | `catboost` | 941.6 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=941.6>=600 |
| 3573 | `mnist_784` | `heavy` | `tpe_optuna` | `xgboost` | 823.6 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=823.6>=600 |
| 3481 | `isolet` | `standard` | `random_search` | `catboost` | 741.6 | isolet_future_recalibration_candidate (signoff caveat 1); observed_runtime_s=741.6>=600 |
| 3573 | `mnist_784` | `heavy` | `random_search` | `catboost` | 693.0 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=693.0>=600 |
| 3573 | `mnist_784` | `heavy` | `random_search` | `xgboost` | 688.0 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=688.0>=600 |
| 3573 | `mnist_784` | `heavy` | `tpe_optuna` | `catboost` | 679.1 | heavy lane; observed_max_runtime_s=1507.2 at R=1; observed_runtime_s=679.1>=600 |

## Signoff caveats acknowledged

- **isolet_future_recalibration_candidate** (task 3481, `isolet`): isolet ran in the standard lane but its observed runtime crosses the 900 s heavy-promotion threshold (1078.6 s in Commit 40). It is a candidate for future policy recalibration via scripts/build_cc18_heavy_task_policy.py but lane assignments…
- **devnagari_extreme_budget_non_equivalence** (task 167121, `Devnagari-Script`): Devnagari-Script ran under the policy-defined extreme.stage0_max_evaluations=1 (YAML default). Standard and heavy lanes ran at max_evaluations=5. random_search / tpe_optuna / default_gbdt cells on Devnagari therefore exercised fewer configu…

## Source shards (committed) — read-only

- n_committed_shards: 40
- no_committed_shard_modified_by_this_script: `True`
- no_execution_sqlite_created_by_this_script: `True`
- no_training_run_by_this_script: `True`

## Next steps

- review this plan + `docs/STAGE3_POLICY_DECISION.md`
- read `docs/STAGE3_TOPUP_EXECUTION_PLAN.md` for the strategic context
- when ready, follow `docs/STAGE3_DISTRIBUTED_RUNBOOK.md` to run the Commit 47 pilot (replica 002, shard 00, standard lane, canary only)
