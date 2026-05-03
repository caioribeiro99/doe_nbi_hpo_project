# batch_00_synthetic_canary -- dedicated Mac gate

- batch_id: `batch_00_synthetic_canary`
- run_timestamp: `2026-05-03T02:20:17Z`
- git_sha: `9658ab21ffc7`
- hostname: `Factored-LWCTW4633L`
- uname: `macOS-26.4.1-arm64-arm-64bit`
- python: `3.12.13` (/Users/caiotertulianoribeiro/Projects/doe_nbi_hpo_project/.venv/bin/python)
- runtime: 3.0 s

- temp_shard: `/var/folders/lm/1_jq3_6j2k1d13pz46lgscwm0000gn/T/cc18_batch00_0y98gw9z/shard_00.sqlite`
- n_cells_in_temp_shard: 12
- n_cells_expected: 12
- success: **12**, failed: **0**, pending: 0

- source_shard_unchanged: **True**
- source_shard_md5_before: `91e7a861ea73daf82694029d6c590e54`
- source_shard_md5_after:  `91e7a861ea73daf82694029d6c590e54`
- stage3_signoff_present:  False

## Package versions

| package | version |
|---|---|
| `xgboost` | 3.2.0 |
| `lightgbm` | 4.6.0 |
| `catboost` | 1.2.10 |
| `optuna` | 4.8.0 |
| `scikit-learn` | MISSING |
| `openml` | 0.15.1 |
| `smac` | 2.4.0 |
| `pymoo` | 0.6.1.6 |
| `dehb` | 0.1.2 |

## Capability audit

- smoke_ready: ['default_gbdt', 'random_search', 'tpe_optuna', 'doe_rsm_vrf_true_nbi']
- dispatch_only: ['doe_rsm_vrf_true_nbi_no_mbpa', 'legacy_weighted_sum_scalarization']
- stub_only: ['smac3', 'asha', 'bohb', 'dehb', 'nsga2', 'motpe', 'parego']
- missing_packages: []

## 12-cell canary results

| method | algorithm | status | runtime_s | last_error |
|---|---|---|---:|---|
| `default_gbdt` | `catboost` | success | 0.05 | — |
| `default_gbdt` | `lightgbm` | success | 0.01 | — |
| `default_gbdt` | `xgboost` | success | 0.01 | — |
| `doe_rsm_vrf_true_nbi` | `catboost` | success | 0.38 | — |
| `doe_rsm_vrf_true_nbi` | `lightgbm` | success | 0.45 | — |
| `doe_rsm_vrf_true_nbi` | `xgboost` | success | 0.10 | — |
| `random_search` | `catboost` | success | 0.32 | — |
| `random_search` | `lightgbm` | success | 0.05 | — |
| `random_search` | `xgboost` | success | 0.05 | — |
| `tpe_optuna` | `catboost` | success | 0.25 | — |
| `tpe_optuna` | `lightgbm` | success | 0.04 | — |
| `tpe_optuna` | `xgboost` | success | 0.08 | — |

## Verdict: **GATE PASS**

batch_01_cc18_tiny_3_tasks may proceed.
