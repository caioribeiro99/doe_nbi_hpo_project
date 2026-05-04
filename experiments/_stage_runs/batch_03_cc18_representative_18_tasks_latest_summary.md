# CC18 stage-run summary -- `batch_03_cc18_representative_18_tasks_latest`

- run_id: `batch_03_cc18_representative_18_tasks_latest`
- stage: `stage0_replica_001`
- batch_id: `batch_03_cc18_representative_18_tasks`
- exported_at: `2026-05-04T04:29:26Z`
- source_git_sha: `87e3526727ba`
- host: `Factored-LWCTW4633L`
- python: `3.12.13`
- run_dir: `runs/cc18/batch_03_cc18_representative_18_tasks_latest` (gitignored)

- total jobs: **216**
- success: **216**, failed: **0**, pending: 0, running: 0, claimed: 0, skipped: 0
- runtime: total 56697.1s, max 11090.57s across 216 recorded jobs
- started_at_min: `2026-05-03T12:44:16.718Z`, finished_at_max: `2026-05-04T04:29:25.130Z`

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
| `shard_00.execution.sqlite` | 20 | 20 | 0 | 57.56 | `45adfb02670b72dc` |
| `shard_01.execution.sqlite` | 20 | 20 | 0 | 9.58 | `3aab25043d5c5d61` |
| `shard_02.execution.sqlite` | 16 | 16 | 0 | 13.10 | `01ea4afa2b38dd9d` |
| `shard_03.execution.sqlite` | 24 | 24 | 0 | 24551.97 | `74ce8712e0244c8b` |
| `shard_04.execution.sqlite` | 20 | 20 | 0 | 3785.65 | `94dec7b5427d90fe` |
| `shard_05.execution.sqlite` | 24 | 24 | 0 | 27783.43 | `864ad4005e5a2501` |
| `shard_06.execution.sqlite` | 16 | 16 | 0 | 22.74 | `38017e69751e4ee7` |
| `shard_07.execution.sqlite` | 24 | 24 | 0 | 350.42 | `eb30a8ac41211310` |
| `shard_08.execution.sqlite` | 24 | 24 | 0 | 46.81 | `9671e168702dd796` |
| `shard_09.execution.sqlite` | 28 | 28 | 0 | 75.84 | `4abfbfb3a47e5c18` |

## Failures (grouped)

_(none)_

## Verdict: **GREEN**

All jobs landed in a terminal status, the committed shards are byte-identical to the recorded MD5s, and no stage-3 sign-off file was created.

---

## batch_03-specific summary

- batch_id: `batch_03_cc18_representative_18_tasks`
- run_dir: `runs/cc18/batch_03_cc18_representative_18_tasks_latest` (gitignored)
- n_cells_expected: 216, in_temp_shard: 216, success: **216**, failed: **0**, pending: 0
- runtime (runner only): 56709.6 s
- shards_unchanged_after_download: **True**
- openml_payloads_committed: False
- execution_shards_committed: False

### batch_02 pre-flight

- exported_at: `2026-05-03T05:24:12Z`
- age_days: 0.30
- success: 144/144 (failed=0, pending=0)
- source_shards_unchanged: True
- run_id: `batch_02_cc18_small_12_tasks_latest`

### Tasks

| task_id | dataset | type | rows | features | classes | categorical | sha256 |
|---:|---|---|---:|---:|---:|---:|---|
| 6 | `letter` | multiclass | 20000 | 16 | 26 | 0 | `1af4f64d14cb` |
| 11 | `balance-scale` | multiclass | 625 | 4 | 3 | 0 | `64d1c2175950` |
| 53 | `vehicle` | multiclass | 846 | 18 | 4 | 0 | `0d4d67025d6c` |
| 219 | `electricity` | binary | 45312 | 14 | 2 | 1 | `b30530c72c7c` |
| 2074 | `satimage` | multiclass | 6430 | 36 | 6 | 0 | `604168e6eb36` |
| 2079 | `eucalyptus` | multiclass | 736 | 91 | 5 | 5 | `f1ee63434bf9` |
| 3022 | `vowel` | multiclass | 990 | 27 | 11 | 2 | `689fa09d2781` |
| 3917 | `kc1` | binary | 2109 | 21 | 2 | 0 | `170afab5bc2b` |
| 9946 | `wdbc` | binary | 569 | 30 | 2 | 0 | `c6a6f5bd7a19` |
| 9978 | `ozone-level-8hr` | binary | 2534 | 72 | 2 | 0 | `ffa1826e7951` |
| 10093 | `banknote-authentication` | binary | 1372 | 4 | 2 | 0 | `f20df51da488` |
| 14965 | `bank-marketing` | binary | 45211 | 51 | 2 | 9 | `3d9e54cf9e1d` |
| 125920 | `dresses-sales` | binary | 500 | 156 | 2 | 11 | `7e3aece63974` |
| 146817 | `steel-plates-fault` | multiclass | 1941 | 27 | 7 | 0 | `d7fe1cec5b34` |
| 146819 | `climate-model-simulation-crashes` | binary | 540 | 18 | 2 | 0 | `127d843d259a` |
| 146821 | `car` | multiclass | 1728 | 21 | 4 | 6 | `0c324006773f` |
| 167121 | `Devnagari-Script` | multiclass | 92000 | 1024 | 46 | 0 | `9806e75a40d0` |
| 167125 | `Internet-Advertisements` | binary | 3279 | 3113 | 2 | 1555 | `a7ad48949dc7` |

### Slowest cells

| task_id | method | algorithm | runtime_s |
|---:|---|---|---:|
| 167121 | `doe_rsm_vrf_true_nbi` | `xgboost` | 11090.57 |
| 167121 | `doe_rsm_vrf_true_nbi` | `catboost` | 10575.00 |
| 167121 | `tpe_optuna` | `catboost` | 7943.96 |
| 167121 | `random_search` | `catboost` | 7646.93 |
| 167121 | `tpe_optuna` | `xgboost` | 6524.32 |
| 167121 | `random_search` | `xgboost` | 5435.72 |
| 167121 | `default_gbdt` | `catboost` | 1594.79 |
| 167121 | `doe_rsm_vrf_true_nbi` | `lightgbm` | 1505.01 |

### Per-cell results

| task_id | method | algorithm | status | runtime_s | metric_keys | last_error |
|---:|---|---|---|---:|---|---|
| 6 | `default_gbdt` | `catboost` | success | 3.01 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 17.56 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `random_search` | `catboost` | success | 13.77 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `tpe_optuna` | `catboost` | success | 14.70 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `default_gbdt` | `lightgbm` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.43 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `lightgbm` | success | 0.31 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `lightgbm` | success | 0.30 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9946 | `default_gbdt` | `catboost` | success | 0.33 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 2.53 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `catboost` | success | 1.80 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `catboost` | success | 1.66 | accuracy, precision, recall, specificity | — |
| 10093 | `default_gbdt` | `lightgbm` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 10093 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 10093 | `random_search` | `lightgbm` | success | 0.09 | accuracy, precision, recall, specificity | — |
| 10093 | `tpe_optuna` | `lightgbm` | success | 0.59 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 53 | `default_gbdt` | `catboost` | success | 0.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.47 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `catboost` | success | 1.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `catboost` | success | 0.95 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3917 | `default_gbdt` | `xgboost` | success | 0.03 | accuracy, precision, recall, specificity | — |
| 3917 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.21 | accuracy, precision, recall, specificity | — |
| 3917 | `random_search` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 3917 | `tpe_optuna` | `xgboost` | success | 0.11 | accuracy, precision, recall, specificity | — |
| 10093 | `default_gbdt` | `catboost` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 10093 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.48 | accuracy, precision, recall, specificity | — |
| 10093 | `random_search` | `catboost` | success | 0.27 | accuracy, precision, recall, specificity | — |
| 10093 | `tpe_optuna` | `catboost` | success | 0.25 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `lightgbm` | success | 0.01 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.48 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `lightgbm` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `lightgbm` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 146817 | `default_gbdt` | `xgboost` | success | 0.31 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.71 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `random_search` | `xgboost` | success | 0.88 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `tpe_optuna` | `xgboost` | success | 0.95 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `default_gbdt` | `xgboost` | success | 0.32 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.98 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `random_search` | `xgboost` | success | 1.03 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `tpe_optuna` | `xgboost` | success | 1.15 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3917 | `default_gbdt` | `lightgbm` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 3917 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.31 | accuracy, precision, recall, specificity | — |
| 3917 | `random_search` | `lightgbm` | success | 0.62 | accuracy, precision, recall, specificity | — |
| 3917 | `tpe_optuna` | `lightgbm` | success | 0.26 | accuracy, precision, recall, specificity | — |
| 125920 | `default_gbdt` | `catboost` | success | 0.11 | accuracy, precision, recall, specificity | — |
| 125920 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.21 | accuracy, precision, recall, specificity | — |
| 125920 | `random_search` | `catboost` | success | 0.16 | accuracy, precision, recall, specificity | — |
| 125920 | `tpe_optuna` | `catboost` | success | 0.15 | accuracy, precision, recall, specificity | — |
| 146817 | `default_gbdt` | `lightgbm` | success | 0.40 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 2.65 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `random_search` | `lightgbm` | success | 1.75 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `tpe_optuna` | `lightgbm` | success | 1.94 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.16 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `xgboost` | success | 0.08 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `xgboost` | success | 0.09 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `default_gbdt` | `lightgbm` | success | 0.48 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 3.75 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `random_search` | `lightgbm` | success | 2.65 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `tpe_optuna` | `lightgbm` | success | 3.39 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3917 | `default_gbdt` | `catboost` | success | 0.25 | accuracy, precision, recall, specificity | — |
| 3917 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.94 | accuracy, precision, recall, specificity | — |
| 3917 | `random_search` | `catboost` | success | 0.68 | accuracy, precision, recall, specificity | — |
| 3917 | `tpe_optuna` | `catboost` | success | 0.65 | accuracy, precision, recall, specificity | — |
| 146817 | `default_gbdt` | `catboost` | success | 1.29 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 9.77 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `random_search` | `catboost` | success | 6.85 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146817 | `tpe_optuna` | `catboost` | success | 6.31 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `default_gbdt` | `xgboost` | success | 0.03 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.33 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `random_search` | `xgboost` | success | 0.14 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `tpe_optuna` | `xgboost` | success | 0.17 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `default_gbdt` | `xgboost` | success | 1463.30 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 11090.57 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `random_search` | `xgboost` | success | 5435.72 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `tpe_optuna` | `xgboost` | success | 6524.32 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `default_gbdt` | `lightgbm` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.14 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `lightgbm` | success | 0.11 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `lightgbm` | success | 0.10 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `default_gbdt` | `catboost` | success | 0.78 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 5.95 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `random_search` | `catboost` | success | 4.13 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2074 | `tpe_optuna` | `catboost` | success | 3.93 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9978 | `default_gbdt` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 9978 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.59 | accuracy, precision, recall, specificity | — |
| 9978 | `random_search` | `xgboost` | success | 0.31 | accuracy, precision, recall, specificity | — |
| 9978 | `tpe_optuna` | `xgboost` | success | 0.31 | accuracy, precision, recall, specificity | — |
| 146821 | `default_gbdt` | `lightgbm` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.40 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `random_search` | `lightgbm` | success | 0.28 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `tpe_optuna` | `lightgbm` | success | 0.32 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `default_gbdt` | `lightgbm` | success | 281.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1505.01 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `random_search` | `lightgbm` | success | 1006.63 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `tpe_optuna` | `lightgbm` | success | 975.28 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `default_gbdt` | `catboost` | success | 0.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.12 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `random_search` | `catboost` | success | 0.08 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 11 | `tpe_optuna` | `catboost` | success | 0.09 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9978 | `default_gbdt` | `lightgbm` | success | 0.60 | accuracy, precision, recall, specificity | — |
| 9978 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.88 | accuracy, precision, recall, specificity | — |
| 9978 | `random_search` | `lightgbm` | success | 0.67 | accuracy, precision, recall, specificity | — |
| 9978 | `tpe_optuna` | `lightgbm` | success | 0.73 | accuracy, precision, recall, specificity | — |
| 14965 | `default_gbdt` | `xgboost` | success | 0.22 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.80 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `xgboost` | success | 0.87 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `xgboost` | success | 1.01 | accuracy, precision, recall, specificity | — |
| 146821 | `default_gbdt` | `catboost` | success | 0.21 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.61 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `random_search` | `catboost` | success | 1.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 146821 | `tpe_optuna` | `catboost` | success | 1.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `default_gbdt` | `catboost` | success | 1594.79 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 10575.00 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `random_search` | `catboost` | success | 7646.93 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167121 | `tpe_optuna` | `catboost` | success | 7943.96 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 167125 | `default_gbdt` | `xgboost` | success | 0.67 | accuracy, precision, recall, specificity | — |
| 167125 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 5.29 | accuracy, precision, recall, specificity | — |
| 167125 | `random_search` | `xgboost` | success | 2.76 | accuracy, precision, recall, specificity | — |
| 167125 | `tpe_optuna` | `xgboost` | success | 3.06 | accuracy, precision, recall, specificity | — |
| 3022 | `default_gbdt` | `xgboost` | success | 0.22 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.18 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `xgboost` | success | 0.63 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `xgboost` | success | 0.63 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9978 | `default_gbdt` | `catboost` | success | 0.64 | accuracy, precision, recall, specificity | — |
| 9978 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 4.17 | accuracy, precision, recall, specificity | — |
| 9978 | `random_search` | `catboost` | success | 2.94 | accuracy, precision, recall, specificity | — |
| 9978 | `tpe_optuna` | `catboost` | success | 2.72 | accuracy, precision, recall, specificity | — |
| 14965 | `default_gbdt` | `lightgbm` | success | 0.78 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.84 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `lightgbm` | success | 1.29 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `lightgbm` | success | 1.31 | accuracy, precision, recall, specificity | — |
| 167125 | `default_gbdt` | `lightgbm` | success | 0.22 | accuracy, precision, recall, specificity | — |
| 167125 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.55 | accuracy, precision, recall, specificity | — |
| 167125 | `random_search` | `lightgbm` | success | 1.45 | accuracy, precision, recall, specificity | — |
| 167125 | `tpe_optuna` | `lightgbm` | success | 1.16 | accuracy, precision, recall, specificity | — |
| 219 | `default_gbdt` | `xgboost` | success | 0.18 | accuracy, precision, recall, specificity | — |
| 219 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 1.13 | accuracy, precision, recall, specificity | — |
| 219 | `random_search` | `xgboost` | success | 0.52 | accuracy, precision, recall, specificity | — |
| 219 | `tpe_optuna` | `xgboost` | success | 0.62 | accuracy, precision, recall, specificity | — |
| 2079 | `default_gbdt` | `xgboost` | success | 0.08 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.52 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `random_search` | `xgboost` | success | 0.26 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `tpe_optuna` | `xgboost` | success | 0.30 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `default_gbdt` | `lightgbm` | success | 0.25 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.86 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `lightgbm` | success | 0.88 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `lightgbm` | success | 0.96 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 14965 | `default_gbdt` | `catboost` | success | 0.48 | accuracy, precision, recall, specificity | — |
| 14965 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 3.47 | accuracy, precision, recall, specificity | — |
| 14965 | `random_search` | `catboost` | success | 2.37 | accuracy, precision, recall, specificity | — |
| 14965 | `tpe_optuna` | `catboost` | success | 2.36 | accuracy, precision, recall, specificity | — |
| 146819 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 146819 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.13 | accuracy, precision, recall, specificity | — |
| 146819 | `random_search` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 146819 | `tpe_optuna` | `xgboost` | success | 0.07 | accuracy, precision, recall, specificity | — |
| 167125 | `default_gbdt` | `catboost` | success | 16.92 | accuracy, precision, recall, specificity | — |
| 167125 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 139.48 | accuracy, precision, recall, specificity | — |
| 167125 | `random_search` | `catboost` | success | 94.37 | accuracy, precision, recall, specificity | — |
| 167125 | `tpe_optuna` | `catboost` | success | 83.15 | accuracy, precision, recall, specificity | — |
| 6 | `default_gbdt` | `xgboost` | success | 1.16 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 10.41 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `random_search` | `xgboost` | success | 5.01 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `tpe_optuna` | `xgboost` | success | 5.88 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 219 | `default_gbdt` | `lightgbm` | success | 0.21 | accuracy, precision, recall, specificity | — |
| 219 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 1.50 | accuracy, precision, recall, specificity | — |
| 219 | `random_search` | `lightgbm` | success | 1.04 | accuracy, precision, recall, specificity | — |
| 219 | `tpe_optuna` | `lightgbm` | success | 1.07 | accuracy, precision, recall, specificity | — |
| 2079 | `default_gbdt` | `lightgbm` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.42 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `random_search` | `lightgbm` | success | 0.31 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `tpe_optuna` | `lightgbm` | success | 0.29 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `default_gbdt` | `catboost` | success | 0.90 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 7.82 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `random_search` | `catboost` | success | 5.02 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 3022 | `tpe_optuna` | `catboost` | success | 4.62 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9946 | `default_gbdt` | `xgboost` | success | 0.03 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.17 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `xgboost` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 146819 | `default_gbdt` | `lightgbm` | success | 0.40 | accuracy, precision, recall, specificity | — |
| 146819 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 146819 | `random_search` | `lightgbm` | success | 0.08 | accuracy, precision, recall, specificity | — |
| 146819 | `tpe_optuna` | `lightgbm` | success | 0.08 | accuracy, precision, recall, specificity | — |
| 6 | `default_gbdt` | `lightgbm` | success | 3.86 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 26.35 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `random_search` | `lightgbm` | success | 13.80 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 6 | `tpe_optuna` | `lightgbm` | success | 18.60 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `default_gbdt` | `xgboost` | success | 0.06 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.37 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `random_search` | `xgboost` | success | 0.19 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 53 | `tpe_optuna` | `xgboost` | success | 0.20 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 219 | `default_gbdt` | `catboost` | success | 0.37 | accuracy, precision, recall, specificity | — |
| 219 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.74 | accuracy, precision, recall, specificity | — |
| 219 | `random_search` | `catboost` | success | 1.24 | accuracy, precision, recall, specificity | — |
| 219 | `tpe_optuna` | `catboost` | success | 1.25 | accuracy, precision, recall, specificity | — |
| 2079 | `default_gbdt` | `catboost` | success | 0.18 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.41 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `random_search` | `catboost` | success | 0.97 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 2079 | `tpe_optuna` | `catboost` | success | 0.90 | accuracy, balanced_accuracy, brier_multiclass, ece_multiclass, f1_macro, mcc, pr_auc_ovr_macro, roc_auc_ovr_macro | — |
| 9946 | `default_gbdt` | `lightgbm` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 9946 | `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.14 | accuracy, precision, recall, specificity | — |
| 9946 | `random_search` | `lightgbm` | success | 0.11 | accuracy, precision, recall, specificity | — |
| 9946 | `tpe_optuna` | `lightgbm` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 10093 | `default_gbdt` | `xgboost` | success | 0.02 | accuracy, precision, recall, specificity | — |
| 10093 | `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.10 | accuracy, precision, recall, specificity | — |
| 10093 | `random_search` | `xgboost` | success | 0.05 | accuracy, precision, recall, specificity | — |
| 10093 | `tpe_optuna` | `xgboost` | success | 0.06 | accuracy, precision, recall, specificity | — |
| 146819 | `default_gbdt` | `catboost` | success | 0.19 | accuracy, precision, recall, specificity | — |
| 146819 | `doe_rsm_vrf_true_nbi` | `catboost` | success | 1.54 | accuracy, precision, recall, specificity | — |
| 146819 | `random_search` | `catboost` | success | 1.05 | accuracy, precision, recall, specificity | — |
| 146819 | `tpe_optuna` | `catboost` | success | 0.97 | accuracy, precision, recall, specificity | — |

### batch_03 verdict: **GATE PASS**

batch_04_stage0_shard00_only may proceed (only after manual review and only via the same handoff protocol).
