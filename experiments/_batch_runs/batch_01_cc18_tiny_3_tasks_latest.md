# batch_01_cc18_tiny_3_tasks -- dedicated Mac gate

- batch_id: `batch_01_cc18_tiny_3_tasks`
- run_timestamp: `2026-05-03T02:45:45Z`
- git_sha: `fd36601d3894`
- hostname: `Factored-LWCTW4633L`
- uname: `macOS-26.4.1-arm64-arm-64bit`
- python: `3.12.13` (/Users/caiotertulianoribeiro/Projects/doe_nbi_hpo_project/.venv/bin/python)
- runtime: 10.8 s

- temp_shard: `/var/folders/lm/1_jq3_6j2k1d13pz46lgscwm0000gn/T/cc18_batch01_g5qtus1u/shard_batch_01.sqlite`
- n_cells_in_temp_shard: 36
- n_cells_expected: 36
- success: **36**, failed: **0**, pending: 0

- source_shards_unchanged: **True**
- shards_unchanged_after_download: **True**
- stage3_signoff_present: False
- openml_cache_root: `data/source/openml_cc18`
- openml_payloads_committed: False

## batch_00 pre-flight

- run_timestamp: `2026-05-03T02:20:17Z`
- age_days: 0.02
- success: 12/12 (failed=0)
- source_shard_unchanged: True

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

## Tasks

| task_id | dataset | type | rows | features | classes | categorical | sha256 |
|---:|---|---|---:|---:|---:|---:|---|
| 9946 | `wdbc` | binary | 569 | 30 | 2 | 0 | `c6a6f5bd7a19` |
| 125920 | `dresses-sales` | binary | 500 | 156 | 2 | 11 | `7e3aece63974` |
| 11 | `balance-scale` | multiclass | 625 | 4 | 3 | 0 | `64d1c2175950` |

## Capability audit

- smoke_ready: ['default_gbdt', 'random_search', 'tpe_optuna', 'doe_rsm_vrf_true_nbi']
- dispatch_only: ['doe_rsm_vrf_true_nbi_no_mbpa', 'legacy_weighted_sum_scalarization']
- stub_only: ['smac3', 'asha', 'bohb', 'dehb', 'nsga2', 'motpe', 'parego']
- missing_packages: []

## 36-cell canary results

| task_id | method | algorithm | status | runtime_s | metric_keys | last_error |
|---:|---|---|---|---:|---|---|
| 11 | `default_gbdt` | `catboost` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `default_gbdt` | `lightgbm` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.12 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.13 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.14 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `catboost` | success | 0.09 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `lightgbm` | success | 0.12 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `xgboost` | success | 0.08 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `catboost` | success | 0.09 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `lightgbm` | success | 0.10 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `xgboost` | success | 0.08 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9946 | `default_gbdt` | `catboost` | success | 0.34 | accuracy, precision, recall, specificity | — |
| 9946 | `default_gbdt` | `lightgbm` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 9946 | `default_gbdt` | `xgboost` | success | 0.03 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 2.52 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.17 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `catboost` | success | 1.77 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `lightgbm` | success | 0.12 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `catboost` | success | 1.65 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `lightgbm` | success | 0.52 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `xgboost` | success | 0.09 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `catboost` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `lightgbm` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.21 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `catboost` | success | 0.15 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `lightgbm` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `xgboost` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `catboost` | success | 0.15 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `lightgbm` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |

## Source shard MD5 (before / after)

| shard | md5_before | md5_after |
|---|---|---|
| `shard_00.sqlite` | `91e7a861ea73daf82694029d6c590e54` | `91e7a861ea73daf82694029d6c590e54` |
| `shard_01.sqlite` | `b94e71ccb24d5d184c3346d336c2691d` | `b94e71ccb24d5d184c3346d336c2691d` |
| `shard_02.sqlite` | `38e0208538432577d82840d356ca039d` | `38e0208538432577d82840d356ca039d` |
| `shard_03.sqlite` | `198c30f36e040c18af674eb6510ccd1d` | `198c30f36e040c18af674eb6510ccd1d` |
| `shard_04.sqlite` | `c5eb54e008f90abf7a3e47e7f4a22584` | `c5eb54e008f90abf7a3e47e7f4a22584` |
| `shard_05.sqlite` | `ff9d67f50910ba1753602a5eac16905c` | `ff9d67f50910ba1753602a5eac16905c` |
| `shard_06.sqlite` | `4f6d062e42e4df8b72c82803fec1b814` | `4f6d062e42e4df8b72c82803fec1b814` |
| `shard_07.sqlite` | `83fb2d1e840aff2376ee70959d1961dd` | `83fb2d1e840aff2376ee70959d1961dd` |
| `shard_08.sqlite` | `711d28b2ce61381a4b72e24a90b107af` | `711d28b2ce61381a4b72e24a90b107af` |
| `shard_09.sqlite` | `f2c5f528ad680b0c4c670b8bdc11bde7` | `f2c5f528ad680b0c4c670b8bdc11bde7` |

## Verdict: **GATE PASS**

batch_02_cc18_small_12_tasks may proceed (only on the dedicated Mac, only after manual review of this artifact).
