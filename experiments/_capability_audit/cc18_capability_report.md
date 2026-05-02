# OpenML-CC18 method capability audit

- generated_at: `2026-05-02T22:15:47Z`
- python: `3.9.6` (Darwin/arm64)
- methods in matrix: 16
- benchmarked: 13
- literature_only (skipped): 3

## Blockers before stage 0

- default_gbdt: missing lightgbm,catboost
- tpe_optuna: missing optuna
- smac3: missing smac
- asha: missing optuna
- bohb: missing smac
- dehb: missing dehb
- nsga2: missing pymoo
- motpe: missing optuna
- parego: missing smac,pymoo

## Adapter run-status

- **full_ready** (0): _(none)_
- **smoke_ready** (0): _(none)_
- **dispatch_only** (3): `doe_rsm_vrf_true_nbi`, `doe_rsm_vrf_true_nbi_no_mbpa`, `legacy_weighted_sum_scalarization`
- **stub_only** (10): `default_gbdt`, `random_search`, `tpe_optuna`, `smac3`, `asha`, `bohb`, `dehb`, `nsga2`, `motpe`, `parego`

## Per-method

| method | import_ok | run_status | required | missing | notes |
|---|---|---|---|---|---|
| `default_gbdt` | True | `stub_only` | xgboost,lightgbm,catboost | lightgbm,catboost | No search; one fit per CV split with library defaults. supports_categorical_nati |
| `random_search` | True | `stub_only` | scipy,joblib | — | Bergstra & Bengio 2012. Headline single-objective baseline. |
| `tpe_optuna` | True | `stub_only` | optuna | optuna | Akiba et al. 2019. Default-prior TPESampler. |
| `smac3` | True | `stub_only` | smac | smac | Lindauer et al. 2022. RF surrogate with intensification. Native categorical hand |
| `asha` | True | `stub_only` | optuna | optuna | Li et al. 2017/2020. Fidelity dimension = boosting iterations (n_estimators). Fr |
| `bohb` | True | `stub_only` | smac | smac | Falkner et al. 2018. Implementation route: SMAC3 multi-fidelity facade. The hpba |
| `dehb` | True | `stub_only` | dehb | dehb | Awad, Mallik & Hutter 2021. DE inner loop on a Hyperband schedule. |
| `nsga2` | True | `stub_only` | pymoo | pymoo | Deb et al. 2002. Reference evolutionary multi-objective baseline. |
| `motpe` | True | `stub_only` | optuna | optuna | Ozaki et al. 2020. Optuna MOTPESampler. |
| `parego` | True | `stub_only` | smac,pymoo | smac,pymoo | Knowles 2006. Subset-only (48 of 72 CC18 tasks); the subset is frozen by benchma |
| `doe_rsm_vrf_true_nbi` | True | `dispatch_only` | doe_xgb | — | Headline proposed method. All implementation lives in-tree under src/doe_xgb (fa |
| `doe_rsm_vrf_true_nbi_no_mbpa` | True | `dispatch_only` | doe_xgb | — | Ablation: same pipeline as doe_rsm_vrf_true_nbi but with MBPA stage disabled. Qu |
| `legacy_weighted_sum_scalarization` | True | `dispatch_only` | doe_xgb | — | Ablation: dissertation-era weighted-sum solver kept verbatim (doe_xgb.scalarizat |

## Package versions

| package | version |
|---|---|
| `catboost` | _missing_ |
| `dehb` | _missing_ |
| `doe_xgb` | 0.2.0.dev0 |
| `joblib` | 1.5.3 |
| `lightgbm` | _missing_ |
| `optuna` | _missing_ |
| `pymoo` | _missing_ |
| `scipy` | 1.13.1 |
| `smac` | _missing_ |
| `xgboost` | 2.1.4 |
