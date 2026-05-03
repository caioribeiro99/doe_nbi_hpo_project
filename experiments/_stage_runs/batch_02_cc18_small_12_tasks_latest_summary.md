# CC18 stage-run summary -- `batch_02_cc18_small_12_tasks_latest`

- run_id: `batch_02_cc18_small_12_tasks_latest`
- stage: `stage0_replica_001`
- batch_id: `batch_02_cc18_small_12_tasks`
- exported_at: `2026-05-03T05:24:12Z`
- source_git_sha: `58a351f3a615`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/batch_02_cc18_small_12_tasks_latest` (gitignored)

- total jobs: **144**
- success: **144**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 0
- runtime: total 6934.2s, max 1507.17s across 144 recorded jobs
- started_at_min: `2026-05-03T03:28:26.839Z`, finished_at_max: `2026-05-03T05:24:11.614Z`

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
| `shard_00.execution.sqlite` | 28 | 28 | 0 | 678.01 | `9deb596120caa59c` |
| `shard_01.execution.sqlite` | 24 | 24 | 0 | 2599.41 | `8e03dcc434384e4c` |
| `shard_02.execution.sqlite` | 8 | 8 | 0 | 2.09 | `6e5c865b697097e3` |
| `shard_03.execution.sqlite` | 8 | 8 | 0 | 1.15 | `a76c51b424d8d475` |
| `shard_04.execution.sqlite` | 8 | 8 | 0 | 2.25 | `b40d0e5ee7ad6888` |
| `shard_05.execution.sqlite` | 12 | 12 | 0 | 7.60 | `8d44421867ccf4e9` |
| `shard_06.execution.sqlite` | 12 | 12 | 0 | 25.26 | `3c532e9a9957987f` |
| `shard_07.execution.sqlite` | 12 | 12 | 0 | 49.10 | `1b5fa5aec10b0629` |
| `shard_08.execution.sqlite` | 12 | 12 | 0 | 237.04 | `1a59c8a8a1f3b051` |
| `shard_09.execution.sqlite` | 20 | 20 | 0 | 3332.28 | `375142a70aed40ac` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## batch_02-specific summary

- batch_id: `batch_02_cc18_small_12_tasks`
- run_dir: `runs/cc18/batch_02_cc18_small_12_tasks_latest` (gitignored)
- n_cells_expected: 144, in_temp_shard: 144, success: **144**, failed: **0**, pending: 0
- runtime (runner only): 6945.9 s
- shards_unchanged_after_download: **True**
- openml_payloads_committed: False
- execution_shards_committed: False

### batch_01 pre-flight

- run_timestamp: `2026-05-03T02:45:45Z`
- age_days: 0.03
- success: 36/36 (failed=0)
- source_shards_unchanged: True

### Tasks

| task_id | dataset | type | rows | features | classes | categorical | sha256 |
|---:|---|---|---:|---:|---:|---:|---|
| 12 | `mfeat-factors` | multiclass | 2000 | 216 | 10 | 0 | `b99fddd3f517` |
| 16 | `mfeat-karhunen` | multiclass | 2000 | 64 | 10 | 0 | `b637ff0ade4e` |
| 53 | `vehicle` | multiclass | 846 | 18 | 4 | 0 | `0d4d67025d6c` |
| 3022 | `vowel` | multiclass | 990 | 27 | 11 | 2 | `689fa09d2781` |
| 3573 | `mnist_784` | multiclass | 70000 | 784 | 10 | 0 | `5c9b24f1f5d9` |
| 3903 | `pc3` | binary | 1563 | 37 | 2 | 0 | `60b1ec1afad1` |
| 3913 | `kc2` | binary | 522 | 21 | 2 | 0 | `6a2a0aa88fd1` |
| 10101 | `blood-transfusion-service-center` | binary | 748 | 4 | 2 | 0 | `2fec3325b326` |
| 14965 | `bank-marketing` | binary | 45211 | 51 | 2 | 9 | `3d9e54cf9e1d` |
| 125920 | `dresses-sales` | binary | 500 | 156 | 2 | 11 | `7e3aece63974` |
| 146820 | `wilt` | binary | 4839 | 5 | 2 | 0 | `be09b10c9ea3` |
| 167141 | `churn` | binary | 5000 | 33 | 2 | 4 | `8e2cd7d80a0e` |

### Slowest cells

| task_id | method | algorithm | runtime_s |
|---:|---|---|---:|
| 3573 | `doe_rsm_vrf_true_nbi` | `xgboost` | 1507.17 |
| 3573 | `doe_rsm_vrf_true_nbi` | `catboost` | 994.42 |
| 3573 | `tpe_optuna` | `xgboost` | 872.68 |
| 3573 | `random_search` | `xgboost` | 729.97 |
| 3573 | `random_search` | `catboost` | 725.29 |
| 3573 | `tpe_optuna` | `catboost` | 716.11 |
| 3573 | `doe_rsm_vrf_true_nbi` | `lightgbm` | 213.86 |
| 3573 | `default_gbdt` | `xgboost` | 196.26 |

### Per-cell results

| task_id | method | algorithm | status | runtime_s | metric_keys | last_error |
|---:|---|---|---|---:|---|---|
| 16 | `default_gbdt` | `catboost` | success | 5.23 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 43.11 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `random_search` | `catboost` | success | 29.13 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `tpe_optuna` | `catboost` | success | 26.32 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `default_gbdt` | `lightgbm` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.42 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `lightgbm` | success | 0.29 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `lightgbm` | success | 0.29 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `default_gbdt` | `lightgbm` | success | 28.71 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 213.86 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `random_search` | `lightgbm` | success | 159.65 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `tpe_optuna` | `lightgbm` | success | 168.73 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3903 | `default_gbdt` | `lightgbm` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 3903 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.75 | accuracy, precision, recall, specificity | — |
| 3903 | `random_search` | `lightgbm` | success | 0.23 | accuracy, precision, recall, specificity | — |
| 3903 | `tpe_optuna` | `lightgbm` | success | 0.27 | accuracy, precision, recall, specificity | — |
| 3913 | `default_gbdt` | `lightgbm` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 3913 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.09 | accuracy, precision, recall, specificity | — |
| 3913 | `random_search` | `lightgbm` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 3913 | `tpe_optuna` | `lightgbm` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `xgboost` | success | 0.08 | accuracy, precision, recall, specificity | — |
| 146820 | `default_gbdt` | `xgboost` | success | 0.03 | accuracy, precision, recall, specificity | — |
| 146820 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.17 | accuracy, precision, recall, specificity | — |
| 146820 | `random_search` | `xgboost` | success | 0.09 | accuracy, precision, recall, specificity | — |
| 146820 | `tpe_optuna` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 53 | `default_gbdt` | `catboost` | success | 0.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.46 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `catboost` | success | 1.00 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `catboost` | success | 0.92 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `default_gbdt` | `catboost` | success | 153.24 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 994.42 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `random_search` | `catboost` | success | 725.29 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `tpe_optuna` | `catboost` | success | 716.11 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3903 | `default_gbdt` | `catboost` | success | 0.18 | accuracy, precision, recall, specificity | — |
| 3903 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.37 | accuracy, precision, recall, specificity | — |
| 3903 | `random_search` | `catboost` | success | 0.97 | accuracy, precision, recall, specificity | — |
| 3903 | `tpe_optuna` | `catboost` | success | 0.95 | accuracy, precision, recall, specificity | — |
| 3913 | `default_gbdt` | `catboost` | success | 0.09 | accuracy, precision, recall, specificity | — |
| 3913 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.86 | accuracy, precision, recall, specificity | — |
| 3913 | `random_search` | `catboost` | success | 0.45 | accuracy, precision, recall, specificity | — |
| 3913 | `tpe_optuna` | `catboost` | success | 0.44 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `lightgbm` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.53 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `lightgbm` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `lightgbm` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 146820 | `default_gbdt` | `lightgbm` | success | 0.04 | accuracy, precision, recall, specificity | — |
| 146820 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.30 | accuracy, precision, recall, specificity | — |
| 146820 | `random_search` | `lightgbm` | success | 0.23 | accuracy, precision, recall, specificity | — |
| 146820 | `tpe_optuna` | `lightgbm` | success | 0.26 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `catboost` | success | 0.15 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.20 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `catboost` | success | 0.15 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `catboost` | success | 0.19 | accuracy, precision, recall, specificity | — |
| 146820 | `default_gbdt` | `catboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 146820 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.57 | accuracy, precision, recall, specificity | — |
| 146820 | `random_search` | `catboost` | success | 0.39 | accuracy, precision, recall, specificity | — |
| 146820 | `tpe_optuna` | `catboost` | success | 0.36 | accuracy, precision, recall, specificity | — |
| 10101 | `default_gbdt` | `xgboost` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 10101 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.08 | accuracy, precision, recall, specificity | — |
| 10101 | `random_search` | `xgboost` | success | 0.04 | accuracy, precision, recall, specificity | — |
| 10101 | `tpe_optuna` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 167141 | `default_gbdt` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 167141 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.46 | accuracy, precision, recall, specificity | — |
| 167141 | `random_search` | `xgboost` | success | 0.19 | accuracy, precision, recall, specificity | — |
| 167141 | `tpe_optuna` | `xgboost` | success | 0.24 | accuracy, precision, recall, specificity | — |
| 10101 | `default_gbdt` | `lightgbm` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 10101 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 10101 | `random_search` | `lightgbm` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 10101 | `tpe_optuna` | `lightgbm` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 167141 | `default_gbdt` | `lightgbm` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 167141 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.57 | accuracy, precision, recall, specificity | — |
| 167141 | `random_search` | `lightgbm` | success | 0.92 | accuracy, precision, recall, specificity | — |
| 167141 | `tpe_optuna` | `lightgbm` | success | 0.48 | accuracy, precision, recall, specificity | — |
| 10101 | `default_gbdt` | `catboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 10101 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.11 | accuracy, precision, recall, specificity | — |
| 10101 | `random_search` | `catboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 10101 | `tpe_optuna` | `catboost` | success | 0.16 | accuracy, precision, recall, specificity | — |
| 14965 | `default_gbdt` | `xgboost` | success | 0.22 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.79 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `xgboost` | success | 0.88 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `xgboost` | success | 1.08 | accuracy, precision, recall, specificity | — |
| 167141 | `default_gbdt` | `catboost` | success | 0.17 | accuracy, precision, recall, specificity | — |
| 167141 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.33 | accuracy, precision, recall, specificity | — |
| 167141 | `random_search` | `catboost` | success | 0.91 | accuracy, precision, recall, specificity | — |
| 167141 | `tpe_optuna` | `catboost` | success | 0.86 | accuracy, precision, recall, specificity | — |
| 12 | `default_gbdt` | `xgboost` | success | 1.60 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 7.82 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `random_search` | `xgboost` | success | 4.31 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `tpe_optuna` | `xgboost` | success | 3.59 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `default_gbdt` | `xgboost` | success | 0.22 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `xgboost` | success | 0.64 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `xgboost` | success | 0.62 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 14965 | `default_gbdt` | `lightgbm` | success | 0.69 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.90 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `lightgbm` | success | 1.32 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `lightgbm` | success | 1.35 | accuracy, precision, recall, specificity | — |
| 12 | `default_gbdt` | `lightgbm` | success | 2.96 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 14.14 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `random_search` | `lightgbm` | success | 9.33 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `tpe_optuna` | `lightgbm` | success | 10.05 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `default_gbdt` | `lightgbm` | success | 0.26 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.51 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `lightgbm` | success | 0.93 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `lightgbm` | success | 0.98 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 14965 | `default_gbdt` | `catboost` | success | 0.49 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 3.57 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `catboost` | success | 2.47 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `catboost` | success | 2.42 | accuracy, precision, recall, specificity | — |
| 12 | `default_gbdt` | `catboost` | success | 10.35 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 86.88 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `random_search` | `catboost` | success | 56.41 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 12 | `tpe_optuna` | `catboost` | success | 51.01 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `default_gbdt` | `xgboost` | success | 1.42 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 6.16 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `random_search` | `xgboost` | success | 3.17 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `tpe_optuna` | `xgboost` | success | 2.87 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `default_gbdt` | `catboost` | success | 0.91 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 8.03 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `catboost` | success | 5.17 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `catboost` | success | 4.65 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `default_gbdt` | `lightgbm` | success | 1.82 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 9.39 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `random_search` | `lightgbm` | success | 6.66 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 16 | `tpe_optuna` | `lightgbm` | success | 6.59 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `default_gbdt` | `xgboost` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.39 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `xgboost` | success | 0.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `xgboost` | success | 0.21 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `default_gbdt` | `xgboost` | success | 196.26 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1507.17 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `random_search` | `xgboost` | success | 729.97 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3573 | `tpe_optuna` | `xgboost` | success | 872.68 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3903 | `default_gbdt` | `xgboost` | success | 0.04 | accuracy, precision, recall, specificity | — |
| 3903 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.26 | accuracy, precision, recall, specificity | — |
| 3903 | `random_search` | `xgboost` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 3903 | `tpe_optuna` | `xgboost` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 3913 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 3913 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 3913 | `random_search` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 3913 | `tpe_optuna` | `xgboost` | success | 0.11 | accuracy, precision, recall, specificity | — |

### batch_02 verdict: **GATE PASS**

batch_03_cc18_representative_18_tasks may proceed (only after manual review and only via the same handoff protocol).
