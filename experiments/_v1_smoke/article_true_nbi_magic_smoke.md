# Article-track true NBI smoke -- MAGIC + XGBoost

Wrote experiments/_v1_smoke/article_true_nbi_magic_smoke.json.

- ok: **True**
- total_wall_seconds: 86.34
- platform: macOS-26.4.1-arm64-arm-64bit

## Stages

| Stage | wall_seconds | summary |
|---|---:|---|
| config | 0.015 | resolved_dataset_id=magic |
| load_dataset | 0.013 | n_rows=19020, n_features=10, task_type=binary |
| doe | 84.281 | n_design_rows=88, doe_columns=['subsample', 'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'max_depth', 'gamma', 'n_estimators', 'Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'Specifici ... |
| factor_model | 0.007 | n_factors=2, loadings_shape=[5, 2], explained_variance=[0.6691, 0.1987], cumulative_variance=[0.6691, 0.8678], construct_map={'Factor1': ['Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'Specificity ... |
| rsm | 0.017 | models=[{'factor': 'FACTOR1_SCORE', 'n_terms': 36, 'r2': 0.9613, 'r2_adj': 0.9352, 'rank': 36, 'condition_number': 94500600.43}, {'factor': 'FACTOR2_SCORE', 'n_terms': 36, 'r2': 0.5922, 'r2_adj': 0.31 ... |
| nbi_core | 0.568 | n_subproblems=11, n_weights=11, max_residual_norm=1.5921234064300628e-10, median_residual_norm=4.922406748522946e-12, p95_residual_norm=1.4757115111729668e-10, min_t=1.2339687985883461e-10, max_t=2.72 ... |
| confirmation | 1.414 | n_candidates_confirmed=11, selection_rule=distance_to_utopia, selection_distance=0.1508786912195578, chosen_index=2, chosen_metrics={'Accuracy_Mean': 0.801472134595163, 'Precision_Mean': 0.89448876791 ... |
| mbpa | 0.005 | enabled=conditional, triggered=True, frontier_diagnostics={'avg_pairwise_distance': 2.417526628312845, 'unique_nondominated': 11, 'weight_concentration': 1.0, 'curvature_score': 0.07032328228999524, ' ... |

