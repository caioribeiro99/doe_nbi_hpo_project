# PCO213 post-work benchmark — aggregated results

git commit: `5d1d5f4fa0f7c566e078ea7ae3e19e01e9ea1d68` · completed replications: {'santander': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 'bnp': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 'porto': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 'uci_credit': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]} · effective R: 30
total runtime (sum of stage times): 81.14 h · per dataset (h): {'bnp': 16.286, 'porto': 28.844, 'santander': 30.419, 'uci_credit': 5.588}
counts: {'model_fits': 3600, 'doe_evaluations': 19920, 'nbi_subproblems': 23760, 'nbi_real_objective_evals_variant_C': 50150376, 'reference_points': 15467680, 'direct_auc_search_evals': 4844156}


## Scheffé selected orders / reliability (selected surface, unseen Dirichlet points)

| dataset | response | order freq | reliable frac | R2ext median | rel-RMSE median | Spearman median |
|---|---|---|---|---|---|---|
| santander | roc_auc | {'quadratic': 26, 'linear': 4} | 0.00 | -0.333 | 0.202 | 0.830 |
| santander | log_loss | {'quadratic': 29, 'special_cubic': 1} | 1.00 | 0.971 | 0.033 | 0.994 |
| santander | brier | {'quadratic': 30} | 1.00 | 1.000 | 0.000 | 1.000 |
| santander | pr_auc | {'quadratic': 26, 'linear': 4} | 0.00 | -0.222 | 0.214 | 0.760 |
| bnp | roc_auc | {'linear': 22, 'quadratic': 7, 'special_cubic': 1} | 0.17 | 0.118 | 0.176 | 0.908 |
| bnp | log_loss | {'linear': 21, 'quadratic': 7, 'special_cubic': 2} | 0.10 | -0.180 | 0.177 | 0.955 |
| bnp | brier | {'quadratic': 30} | 1.00 | 1.000 | 0.000 | 1.000 |
| bnp | pr_auc | {'linear': 26, 'quadratic': 3, 'special_cubic': 1} | 0.43 | 0.461 | 0.157 | 0.929 |
| porto | roc_auc | {'quadratic': 26, 'special_cubic': 2, 'linear': 2} | 0.53 | 0.519 | 0.130 | 0.924 |
| porto | log_loss | {'linear': 20, 'quadratic': 8, 'special_cubic': 2} | 0.40 | 0.506 | 0.121 | 0.995 |
| porto | brier | {'quadratic': 30} | 1.00 | 1.000 | 0.000 | 1.000 |
| porto | pr_auc | {'quadratic': 26, 'linear': 3, 'special_cubic': 1} | 0.90 | 0.779 | 0.095 | 0.938 |
| uci_credit | roc_auc | {'quadratic': 20, 'special_cubic': 10} | 1.00 | 0.992 | 0.021 | 0.995 |
| uci_credit | log_loss | {'quadratic': 27, 'special_cubic': 3} | 1.00 | 0.993 | 0.016 | 0.993 |
| uci_credit | brier | {'quadratic': 30} | 1.00 | 1.000 | 0.000 | 1.000 |
| uci_credit | pr_auc | {'quadratic': 29, 'special_cubic': 1} | 1.00 | 0.930 | 0.055 | 0.984 |

## Pareto quality vs empirical reference (weighted cost; median over replications)

| dataset | set | n_front | GD | IGD | IGD+ | HV ratio | joint-ND frac | spacing CV | size-matched spacing pct |
|---|---|---|---|---|---|---|---|---|---|
| santander | nbi_A | 13 | 0.6149 | 0.1487 | 0.1362 | 0.789 | 0.000 | 1.695 | 1.00 |
| santander | nbi_B | 20 | 0.0113 | 0.0420 | 0.0105 | 0.981 | 0.045 | 1.036 | 0.69 |
| santander | nbi_C | 35 | 0.0068 | 0.0430 | 0.0095 | 0.989 | 0.440 | 2.377 | 0.99 |
| santander | ws_random_scalarization | 57 | 0.0009 | 0.1849 | 0.0073 | 0.978 | 0.735 | 2.846 | 0.00 |
| santander | random_dirichlet_budget | 6 | 3.8260 | 0.7674 | 0.7148 | 0.215 | 0.000 | 0.935 | 1.00 |
| santander | design_runs | 7 | 0.0019 | 0.0900 | 0.0113 | 0.981 | 0.061 | 1.002 | 0.66 |
| santander | single_objective_refs | 4 | 0.0000 | 0.1264 | 0.0146 | 0.971 | 0.429 | 0.893 | 0.91 |
| bnp | nbi_A | 8 | 0.0016 | 0.1892 | 0.0271 | 0.913 | 0.076 | 0.827 | 0.53 |
| bnp | nbi_B | 44 | 0.0004 | 0.0557 | 0.0120 | 0.971 | 0.576 | 2.948 | 0.99 |
| bnp | nbi_C | 34 | 0.0085 | 0.0275 | 0.0136 | 0.983 | 0.339 | 0.889 | 0.54 |
| bnp | ws_random_scalarization | 1 | 0.0860 | 0.4578 | 0.3078 | 0.474 | 0.000 | 1.575 | 0.04 |
| bnp | random_dirichlet_budget | 5 | 5.0725 | 1.5510 | 1.5254 | 0.000 | 0.000 | 0.990 | 1.00 |
| bnp | design_runs | 6 | 0.0819 | 0.1698 | 0.0289 | 0.918 | 0.030 | 1.050 | 0.84 |
| bnp | single_objective_refs | 4 | 0.0231 | 0.2047 | 0.1634 | 0.759 | 0.286 | 0.500 | 0.71 |
| porto | nbi_A | 11 | 0.0368 | 0.2501 | 0.0974 | 0.769 | 0.015 | 0.502 | 0.09 |
| porto | nbi_B | 60 | 0.0155 | 0.0599 | 0.0354 | 0.914 | 0.371 | 2.570 | 1.00 |
| porto | nbi_C | 56 | 0.0033 | 0.0270 | 0.0072 | 0.982 | 0.712 | 1.558 | 0.96 |
| porto | ws_random_scalarization | 12 | 0.0362 | 0.3296 | 0.1043 | 0.762 | 0.000 | 2.139 | 0.94 |
| porto | random_dirichlet_budget | 5 | 47.5843 | 1.8216 | 1.7924 | 0.000 | 0.000 | 1.126 | 1.00 |
| porto | design_runs | 3 | 0.0707 | 0.3542 | 0.1538 | 0.647 | 0.000 | 0.777 | 0.69 |
| porto | single_objective_refs | 5 | 0.0091 | 0.2339 | 0.0827 | 0.818 | 0.286 | 1.440 | 0.72 |
| uci_credit | nbi_A | 11 | 0.0070 | 0.2492 | 0.2165 | 0.622 | 0.273 | 1.394 | 0.49 |
| uci_credit | nbi_B | 27 | 0.0023 | 0.1804 | 0.1543 | 0.706 | 0.550 | 1.949 | 0.41 |
| uci_credit | nbi_C | 65 | 0.0014 | 0.0287 | 0.0095 | 0.976 | 0.780 | 1.919 | 0.41 |
| uci_credit | ws_random_scalarization | 44 | 0.0005 | 0.1358 | 0.0086 | 0.978 | 0.606 | 1.511 | 0.05 |
| uci_credit | random_dirichlet_budget | 10 | 0.2930 | 0.2381 | 0.1496 | 0.733 | 0.000 | 0.940 | 0.93 |
| uci_credit | design_runs | 8 | 0.0126 | 0.0828 | 0.0324 | 0.922 | 0.045 | 1.087 | 0.79 |
| uci_credit | single_objective_refs | 5 | 0.0237 | 0.4305 | 0.4022 | 0.337 | 0.286 | 1.825 | 1.00 |

## Pareto quality vs empirical reference (support cost; median over replications)

| dataset | set | n_front | IGD+ | HV ratio | joint-ND frac |
|---|---|---|---|---|---|
| santander | nbi_A | 8 | 0.0672 | 0.853 | 0.000 |
| santander | nbi_B | 11 | 0.0082 | 0.986 | 0.018 |
| santander | nbi_C | 30 | 0.0015 | 0.995 | 0.422 |
| santander | ws_random_scalarization | 12 | 0.0094 | 0.986 | 0.000 |
| santander | random_dirichlet_budget | 4 | 0.6629 | 0.077 | 0.000 |
| santander | design_runs | 4 | 0.0080 | 0.988 | 0.015 |
| santander | single_objective_refs | 4 | 0.0113 | 0.984 | 0.286 |
| bnp | nbi_A | 27 | 0.0405 | 0.904 | 0.015 |
| bnp | nbi_B | 10 | 0.0186 | 0.971 | 0.106 |
| bnp | nbi_C | 13 | 0.0298 | 0.969 | 0.187 |
| bnp | ws_random_scalarization | 66 | 0.3496 | 0.524 | 0.000 |
| bnp | random_dirichlet_budget | 2 | 0.9832 | 0.050 | 0.000 |
| bnp | design_runs | 5 | 0.0363 | 0.935 | 0.030 |
| bnp | single_objective_refs | 6 | 0.0270 | 0.981 | 0.286 |
| porto | nbi_A | 3 | 0.0894 | 0.839 | 0.000 |
| porto | nbi_B | 6 | 0.0166 | 0.869 | 0.015 |
| porto | nbi_C | 12 | 0.0069 | 0.933 | 0.152 |
| porto | ws_random_scalarization | 4 | 0.0916 | 0.831 | 0.000 |
| porto | random_dirichlet_budget | 2 | 1.2325 | 0.000 | 0.000 |
| porto | design_runs | 3 | 0.1485 | 0.710 | 0.000 |
| porto | single_objective_refs | 5 | 0.0129 | 0.859 | 0.286 |
| uci_credit | nbi_A | 8 | 0.0400 | 0.911 | 0.050 |
| uci_credit | nbi_B | 14 | 0.0490 | 0.895 | 0.191 |
| uci_credit | nbi_C | 16 | 0.0366 | 0.942 | 0.165 |
| uci_credit | ws_random_scalarization | 15 | 0.0056 | 0.983 | 0.121 |
| uci_credit | random_dirichlet_budget | 2 | 0.5527 | 0.097 | 0.000 |
| uci_credit | design_runs | 4 | 0.0124 | 0.975 | 0.030 |
| uci_credit | single_objective_refs | 6 | 0.0370 | 0.905 | 0.286 |

## Empirical reference convergence

| dataset | points (median) | rounds | displaced by independent check (median) | front size weighted | front size support |
|---|---|---|---|---|---|
| santander | 100564 | 1 | 0.029 | 135 | 40 |
| bnp | 100564 | 1 | 0.022 | 120 | 39 |
| porto | 100564 | 1 | 0.033 | 94 | 14 |
| uci_credit | 100564 | 1 | 0.039 | 481 | 87 |

## Active-support frequency on the empirical reference front (weighted cost)

- santander: {'lr': 0.3985518410106301, 'gnb': 0.9710368202126021, 'knn': 0.08796795563087352, 'rf': 0.21398859959944538, 'xgb': 0.6911107687567402}
- bnp: {'lr': 1.0, 'gnb': 0.5101331789229878, 'knn': 0.004825323296660877, 'rf': 0.5344528083381587, 'xgb': 0.9585022196487165}
- porto: {'lr': 1.0, 'gnb': 0.1512554802710243, 'knn': 0.39418094858509367, 'rf': 0.8160621761658031, 'xgb': 0.9386209645277003}
- uci_credit: {'lr': 0.9908107805233373, 'gnb': 0.5939933882445229, 'knn': 0.24071272482770215, 'rf': 0.698492743878523, 'xgb': 0.9760744102650305}

## AUC vs log-loss conflict (direct-AUC optimum minus SLSQP optimum, OOF)

- santander: ΔAUC mean +0.00068 (sd 0.00007); Δlog-loss mean +0.00321; Δcost +2.861 ms/1k
- bnp: ΔAUC mean +0.00080 (sd 0.00008); Δlog-loss mean +0.00089; Δcost +0.539 ms/1k
- porto: ΔAUC mean +0.00006 (sd 0.00005); Δlog-loss mean +0.00000; Δcost +2.693 ms/1k
- uci_credit: ΔAUC mean +0.00034 (sd 0.00011); Δlog-loss mean +0.00049; Δcost +0.885 ms/1k

## Diversity vs beta_ij

- {'roc_auc': {'spearman_beta_vs_error_corr_all': -0.4863314905079796, 'spearman_beta_vs_disagreement_all': 0.2529773239796548, 'per_dataset': {'santander': -0.532620584673163, 'bnp': -0.835937288192091, 'porto': 0.0315772397471083, 'uci_credit': -0.7724930277003078}}, 'log_loss': {'spearman_beta_vs_error_corr_all': 0.813067793797079, 'spearman_beta_vs_disagreement_all': -0.609035053751958, 'per_dataset': {'santander': 0.6478507538972655, 'bnp': 0.8684038711541238, 'porto': 0.8151588350981677, 'uci_credit': 0.9287489860998456}}}

Tables: `tables/*.csv`; full numbers: `summary.json`.