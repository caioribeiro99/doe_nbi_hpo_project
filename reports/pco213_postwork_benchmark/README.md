# PCO213 post-work benchmark — aggregated results

git commit: `6fa7877cf399a5fabe09bb0634bdf381db8d4b08` · completed replications: {'santander': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'bnp': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'porto': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'uci_credit': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} · effective R: 10
total runtime (sum of stage times): 27.53 h · per dataset (h): {'bnp': 5.497, 'porto': 9.642, 'santander': 10.566, 'uci_credit': 1.821}
counts: {'model_fits': 1200, 'doe_evaluations': 6640, 'nbi_subproblems': 7920, 'nbi_real_objective_evals_variant_C': 17004670, 'reference_points': 5822560, 'direct_auc_search_evals': 1614624}


## Scheffé selected orders / reliability (selected surface, unseen Dirichlet points)

| dataset | response | order freq | reliable frac | R2ext median | rel-RMSE median | Spearman median |
|---|---|---|---|---|---|---|
| santander | roc_auc | {'quadratic': 8, 'linear': 2} | 0.00 | -0.403 | 0.185 | 0.818 |
| santander | log_loss | {'quadratic': 9, 'special_cubic': 1} | 1.00 | 0.973 | 0.033 | 0.994 |
| santander | brier | {'quadratic': 10} | 1.00 | 1.000 | 0.000 | 1.000 |
| santander | pr_auc | {'quadratic': 8, 'linear': 2} | 0.00 | -0.383 | 0.214 | 0.745 |
| bnp | roc_auc | {'linear': 7, 'quadratic': 2, 'special_cubic': 1} | 0.20 | 0.027 | 0.198 | 0.908 |
| bnp | log_loss | {'linear': 7, 'special_cubic': 2, 'quadratic': 1} | 0.20 | 0.081 | 0.155 | 0.963 |
| bnp | brier | {'quadratic': 10} | 1.00 | 1.000 | 0.000 | 1.000 |
| bnp | pr_auc | {'linear': 7, 'quadratic': 2, 'special_cubic': 1} | 0.30 | 0.439 | 0.173 | 0.933 |
| porto | roc_auc | {'quadratic': 10} | 0.60 | 0.519 | 0.127 | 0.912 |
| porto | log_loss | {'linear': 7, 'special_cubic': 2, 'quadratic': 1} | 0.50 | 0.622 | 0.106 | 0.996 |
| porto | brier | {'quadratic': 10} | 1.00 | 1.000 | 0.000 | 1.000 |
| porto | pr_auc | {'quadratic': 7, 'linear': 2, 'special_cubic': 1} | 0.80 | 0.766 | 0.097 | 0.933 |
| uci_credit | roc_auc | {'quadratic': 5, 'special_cubic': 5} | 1.00 | 0.994 | 0.016 | 0.996 |
| uci_credit | log_loss | {'quadratic': 8, 'special_cubic': 2} | 1.00 | 0.995 | 0.014 | 0.995 |
| uci_credit | brier | {'quadratic': 10} | 1.00 | 1.000 | 0.000 | 1.000 |
| uci_credit | pr_auc | {'quadratic': 9, 'special_cubic': 1} | 1.00 | 0.946 | 0.041 | 0.987 |

## Pareto quality vs empirical reference (weighted cost; median over replications)

| dataset | set | n_front | GD | IGD | IGD+ | HV ratio | joint-ND frac | spacing CV | size-matched spacing pct |
|---|---|---|---|---|---|---|---|---|---|
| santander | nbi_A | 13 | 0.2504 | 0.1166 | 0.0833 | 0.852 | 0.000 | 1.500 | 0.99 |
| santander | nbi_B | 19 | 0.0113 | 0.0529 | 0.0070 | 0.983 | 0.047 | 1.407 | 0.78 |
| santander | nbi_C | 37 | 0.0049 | 0.0431 | 0.0051 | 0.993 | 0.459 | 2.775 | 0.99 |
| santander | ws_random_scalarization | 54 | 0.0009 | 0.2150 | 0.0073 | 0.978 | 0.720 | 2.226 | 0.00 |
| santander | random_dirichlet_budget | 5 | 2.3773 | 0.4331 | 0.3606 | 0.562 | 0.000 | 0.934 | 1.00 |
| santander | design_runs | 7 | 0.0014 | 0.0926 | 0.0066 | 0.982 | 0.061 | 1.068 | 0.64 |
| santander | single_objective_refs | 4 | 0.0000 | 0.1258 | 0.0092 | 0.974 | 0.429 | 0.903 | 0.91 |
| bnp | nbi_A | 8 | 0.0018 | 0.1897 | 0.0275 | 0.917 | 0.083 | 0.807 | 0.49 |
| bnp | nbi_B | 44 | 0.0002 | 0.0603 | 0.0136 | 0.968 | 0.583 | 3.059 | 0.99 |
| bnp | nbi_C | 34 | 0.0094 | 0.0271 | 0.0140 | 0.983 | 0.336 | 0.919 | 0.54 |
| bnp | ws_random_scalarization | 2 | 0.0841 | 0.4447 | 0.2945 | 0.493 | 0.000 | 1.575 | 0.44 |
| bnp | random_dirichlet_budget | 4 | 5.5691 | 1.5510 | 1.5254 | 0.000 | 0.000 | 1.267 | 1.00 |
| bnp | design_runs | 6 | 0.0841 | 0.1699 | 0.0308 | 0.916 | 0.030 | 1.035 | 0.84 |
| bnp | single_objective_refs | 4 | 0.0210 | 0.2059 | 0.1631 | 0.765 | 0.286 | 0.508 | 0.73 |
| porto | nbi_A | 12 | 0.0331 | 0.2190 | 0.0646 | 0.823 | 0.015 | 0.589 | 0.10 |
| porto | nbi_B | 56 | 0.0154 | 0.0708 | 0.0478 | 0.883 | 0.432 | 2.606 | 0.99 |
| porto | nbi_C | 56 | 0.0026 | 0.0308 | 0.0082 | 0.982 | 0.727 | 1.558 | 0.98 |
| porto | ws_random_scalarization | 14 | 0.0354 | 0.3123 | 0.0829 | 0.786 | 0.076 | 2.195 | 0.87 |
| porto | random_dirichlet_budget | 5 | 56.1440 | 1.7599 | 1.7574 | 0.000 | 0.000 | 1.369 | 1.00 |
| porto | design_runs | 3 | 0.0815 | 0.3548 | 0.1594 | 0.640 | 0.008 | 0.786 | 0.39 |
| porto | single_objective_refs | 5 | 0.0127 | 0.2344 | 0.0713 | 0.835 | 0.357 | 1.521 | 0.79 |
| uci_credit | nbi_A | 16 | 0.0092 | 0.1987 | 0.1701 | 0.669 | 0.292 | 1.635 | 0.46 |
| uci_credit | nbi_B | 28 | 0.0033 | 0.1799 | 0.1567 | 0.688 | 0.491 | 2.058 | 0.40 |
| uci_credit | nbi_C | 63 | 0.0008 | 0.0313 | 0.0094 | 0.977 | 0.785 | 2.186 | 0.45 |
| uci_credit | ws_random_scalarization | 55 | 0.0005 | 0.1325 | 0.0084 | 0.980 | 0.720 | 1.508 | 0.01 |
| uci_credit | random_dirichlet_budget | 10 | 0.3984 | 0.2255 | 0.1411 | 0.750 | 0.000 | 1.024 | 0.97 |
| uci_credit | design_runs | 8 | 0.0123 | 0.0827 | 0.0333 | 0.920 | 0.045 | 1.056 | 0.78 |
| uci_credit | single_objective_refs | 4 | 0.0192 | 0.4251 | 0.4000 | 0.344 | 0.286 | 1.682 | 1.00 |

## Pareto quality vs empirical reference (support cost; median over replications)

| dataset | set | n_front | IGD+ | HV ratio | joint-ND frac |
|---|---|---|---|---|---|
| santander | nbi_A | 8 | 0.0626 | 0.860 | 0.000 |
| santander | nbi_B | 13 | 0.0085 | 0.985 | 0.030 |
| santander | nbi_C | 32 | 0.0022 | 0.992 | 0.459 |
| santander | ws_random_scalarization | 10 | 0.0102 | 0.985 | 0.000 |
| santander | random_dirichlet_budget | 4 | 0.6077 | 0.078 | 0.000 |
| santander | design_runs | 4 | 0.0080 | 0.988 | 0.030 |
| santander | single_objective_refs | 4 | 0.0114 | 0.984 | 0.286 |
| bnp | nbi_A | 24 | 0.0405 | 0.929 | 0.000 |
| bnp | nbi_B | 10 | 0.0206 | 0.843 | 0.114 |
| bnp | nbi_C | 13 | 0.0594 | 0.663 | 0.198 |
| bnp | ws_random_scalarization | 66 | 0.3530 | 0.566 | 0.000 |
| bnp | random_dirichlet_budget | 2 | 7.8178 | 0.000 | 0.000 |
| bnp | design_runs | 5 | 0.0384 | 0.965 | 0.030 |
| bnp | single_objective_refs | 6 | 0.1137 | 0.799 | 0.357 |
| porto | nbi_A | 3 | 0.0659 | 0.870 | 0.000 |
| porto | nbi_B | 7 | 0.0183 | 0.845 | 0.015 |
| porto | nbi_C | 12 | 0.0077 | 0.932 | 0.144 |
| porto | ws_random_scalarization | 5 | 0.0808 | 0.852 | 0.015 |
| porto | random_dirichlet_budget | 2 | 1.3582 | 0.000 | 0.000 |
| porto | design_runs | 3 | 0.1723 | 0.700 | 0.000 |
| porto | single_objective_refs | 5 | 0.0113 | 0.885 | 0.286 |
| uci_credit | nbi_A | 10 | 0.0418 | 0.917 | 0.051 |
| uci_credit | nbi_B | 14 | 0.0509 | 0.882 | 0.245 |
| uci_credit | nbi_C | 18 | 0.0424 | 0.938 | 0.165 |
| uci_credit | ws_random_scalarization | 16 | 0.0058 | 0.982 | 0.121 |
| uci_credit | random_dirichlet_budget | 2 | 0.5458 | 0.098 | 0.000 |
| uci_credit | design_runs | 4 | 0.0124 | 0.974 | 0.023 |
| uci_credit | single_objective_refs | 6 | 0.0370 | 0.903 | 0.286 |

## Empirical reference convergence

| dataset | points (median) | rounds | displaced by independent check (median) | front size weighted | front size support |
|---|---|---|---|---|---|
| santander | 100564 | 1 | 0.015 | 142 | 46 |
| bnp | 100564 | 1 | 0.027 | 122 | 40 |
| porto | 150564 | 2 | 0.035 | 100 | 14 |
| uci_credit | 100564 | 1 | 0.038 | 457 | 80 |

## Active-support frequency on the empirical reference front (weighted cost)

- santander: {'lr': 0.40459670424978317, 'gnb': 0.9748482220294883, 'knn': 0.11925411968777103, 'rf': 0.24154379878577623, 'xgb': 0.696877710320902}
- bnp: {'lr': 1.0, 'gnb': 0.4914772727272727, 'knn': 0.002840909090909091, 'rf': 0.5392045454545454, 'xgb': 0.9602272727272727}
- porto: {'lr': 1.0, 'gnb': 0.20978627671541059, 'knn': 0.38976377952755903, 'rf': 0.8149606299212598, 'xgb': 0.9381327334083239}
- uci_credit: {'lr': 0.985337726523888, 'gnb': 0.5958813838550248, 'knn': 0.22932454695222404, 'rf': 0.6980230642504118, 'xgb': 0.9747940691927512}

## AUC vs log-loss conflict (direct-AUC optimum minus SLSQP optimum, OOF)

- santander: ΔAUC mean +0.00068 (sd 0.00005); Δlog-loss mean +0.00351; Δcost +5.329 ms/1k
- bnp: ΔAUC mean +0.00082 (sd 0.00008); Δlog-loss mean +0.00091; Δcost +0.533 ms/1k
- porto: ΔAUC mean +0.00006 (sd 0.00006); Δlog-loss mean +0.00001; Δcost +3.271 ms/1k
- uci_credit: ΔAUC mean +0.00030 (sd 0.00009); Δlog-loss mean +0.00044; Δcost +0.824 ms/1k

## Diversity vs beta_ij

- {'roc_auc': {'spearman_beta_vs_error_corr_all': -0.4951375321095757, 'spearman_beta_vs_disagreement_all': 0.25821178444699755, 'per_dataset': {'santander': -0.5299369936993699, 'bnp': -0.8548934893489348, 'porto': -0.03354335433543354, 'uci_credit': -0.7534353435343534}}, 'log_loss': {'spearman_beta_vs_error_corr_all': 0.8176101100631878, 'spearman_beta_vs_disagreement_all': -0.6141566616008282, 'per_dataset': {'santander': 0.6494689468946893, 'bnp': 0.8726312631263126, 'porto': 0.8311431143114311, 'uci_credit': 0.9401740174017401}}}

Tables: `tables/*.csv`; full numbers: `summary.json`.