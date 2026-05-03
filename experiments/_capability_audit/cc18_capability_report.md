# OpenML-CC18 method capability audit

- generated_at: `2026-05-03T02:20:13Z`
- python: `3.12.13` (Darwin/arm64)
- methods in matrix: 16
- benchmarked: 13
- literature_only (skipped): 3

## Blockers before stage 0

_(none)_

## Adapter run-status

- **full_ready** (0): _(none)_
- **smoke_ready** (4): `default_gbdt`, `random_search`, `tpe_optuna`, `doe_rsm_vrf_true_nbi`
- **dispatch_only** (2): `doe_rsm_vrf_true_nbi_no_mbpa`, `legacy_weighted_sum_scalarization`
- **stub_only** (7): `smac3`, `asha`, `bohb`, `dehb`, `nsga2`, `motpe`, `parego`

## Per-method

| method | import_ok | run_status | required | missing | notes |
|---|---|---|---|---|---|
| `default_gbdt` | True | `smoke_ready` | xgboost,lightgbm,catboost | — | No search; one fit per CV split with library defaults. Smoke-ready (Commit 30) — |
| `random_search` | True | `smoke_ready` | scipy,joblib | — | Bergstra & Bengio 2012. Smoke-ready (Commit 30) over the canary search space; th |
| `tpe_optuna` | True | `smoke_ready` | optuna | — | Akiba et al. 2019. Default-prior TPESampler. Smoke-ready (Commit 30) over the ca |
| `smac3` | True | `stub_only` | smac | — | Lindauer et al. 2022. RF surrogate with intensification. Native categorical hand |
| `asha` | True | `stub_only` | optuna | — | Li et al. 2017/2020. Fidelity dimension = boosting iterations (n_estimators). Fr |
| `bohb` | True | `stub_only` | smac | — | Falkner et al. 2018. Implementation route: SMAC3 multi-fidelity facade. The hpba |
| `dehb` | True | `stub_only` | dehb | — | Awad, Mallik & Hutter 2021. DE inner loop on a Hyperband schedule. |
| `nsga2` | True | `stub_only` | pymoo | — | Deb et al. 2002. Reference evolutionary multi-objective baseline. |
| `motpe` | True | `stub_only` | optuna | — | Ozaki et al. 2020. Optuna MOTPESampler. |
| `parego` | True | `stub_only` | smac,pymoo | — | Knowles 2006. Subset-only (48 of 72 CC18 tasks); the subset is frozen by benchma |
| `doe_rsm_vrf_true_nbi` | True | `smoke_ready` | doe_xgb | — | Headline proposed method. Smoke-ready (Commit 30) on the 2-objective canary path |
| `doe_rsm_vrf_true_nbi_no_mbpa` | True | `dispatch_only` | doe_xgb | — | Ablation: same pipeline as doe_rsm_vrf_true_nbi but with MBPA stage disabled. Qu |
| `legacy_weighted_sum_scalarization` | True | `dispatch_only` | doe_xgb | — | Ablation: dissertation-era weighted-sum solver kept verbatim (doe_xgb.scalarizat |

## Package versions

| package | version |
|---|---|
| `catboost` | 1.2.10 |
| `dehb` | 0.1.2 |
| `doe_xgb` | 0.2.0.dev0 |
| `joblib` | 1.5.3 |
| `lightgbm` | 4.6.0 |
| `optuna` | 4.8.0 |
| `pymoo` | 0.6.1.6 |
| `scipy` | 1.17.1 |
| `smac` | 2.4.0 |
| `xgboost` | 3.2.0 |
