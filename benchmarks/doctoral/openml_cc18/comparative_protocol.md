# OpenML-CC18 comparative protocol — CC18-scoped slice

This is the CC18-scoped slice of the comparative protocol. The
authoritative cross-cutting document is `docs/COMPARATIVE_PROTOCOL.md`;
the canonical machine-readable method list is `method_matrix.csv` in
this directory. If those three documents disagree, the CSV wins.

## Suite anchor

- Suite: OpenML-CC18 (`suite_id = 99`).
- Task count: 72.
- Source of truth for tasks: `tasks.csv` in this directory (one row
  per OpenML task; never identified by dataset id alone).
- Cross-validation folds: OpenML task-defined folds when available;
  stratified 5-fold otherwise.

## Method participation on CC18

| method_id | family | primary/ablation | objective_mode | full 72 tasks? |
|---|---|---|---|---|
| `default_gbdt` | control | primary | single-obj | yes |
| `random_search` | classical | primary | single-obj | yes |
| `tpe_optuna` | classical | primary | single-obj | yes |
| `smac3` | SMAC | primary | single-obj | yes |
| `asha` | multi-fidelity | primary | single-obj | yes |
| `bohb` | multi-fidelity | primary | single-obj | yes |
| `dehb` | multi-fidelity | primary | single-obj | yes |
| `nsga2` | evolutionary MOO | primary | multi-obj | yes |
| `motpe` | MOO | primary | multi-obj | yes |
| `parego` | MOO | subset | multi-obj | **no — subset only** |
| `doe_rsm_vrf_true_nbi` | proposed | primary | multi-obj | yes |
| `doe_rsm_vrf_true_nbi_no_mbpa` | ablation | ablation | multi-obj | yes |
| `legacy_weighted_sum_scalarization` | ablation | ablation | multi-obj | yes |
| `flaml_optional` | AutoML context | literature-only | single-obj | no — TBD |
| `auto_sklearn_context` | AutoML context | literature-only | n/a | no |
| `autogluon_context` | AutoML context | literature-only | n/a | no |

## ParEGO CC18 subset (frozen in Commit 27)

The ParEGO subset is **frozen** as the union of:

1. CC18 tasks with `class_imbalance_ratio >= 5.0` (high-imbalance);
2. CC18 multiclass tasks with `n_classes >= 5` (high-class-count);
3. CC18 tasks with `categorical_feature_count > 0` and
   `5000 <= n_rows <= 50000` (medium-size mixed-type, where ParEGO's
   Tchebycheff scalarization is most informative).

Applied against the committed `tasks.csv`, this rule selects
**48 of 72 tasks** (15 binary + 33 multiclass; 27 satisfy the
high-class-count clause, 19 the imbalance clause, 6 the
categorical mid-size clause; clauses can overlap). The frozen list
of `openml_task_id` values lives at
`benchmarks/doctoral/openml_cc18/parego_subset.csv` and is the
authoritative source consumed by the shard generator.

## Headline job count

For the methods that run on all 72 CC18 tasks (everything except
ParEGO and the AutoML-context entries):

```
n_methods_full_cc18 = 12   # 13 method rows minus parego minus 3 literature_only,
                           # plus the proposed primary count above
```

The exact count is computed from `method_matrix.csv` at shard
generation, not duplicated as a constant here. The headline 30-replica
job count is `72 × n_methods_full_cc18 × 30` plus the per-replica jobs
contributed by ParEGO on its (smaller) subset.

## Budget per CC18 task per replica

Identical to `docs/COMPARATIVE_PROTOCOL.md`:

- Single-objective + multi-objective non-fidelity methods:
  `B = 138` configurations under stratified 5-fold CV.
- Multi-fidelity methods: `B * max_iter` total boosting iterations
  budget per replica per task.

## Multiclass support

The CC18 panel contains 37 multiclass tasks. Every method in the
table above supports multiclass classification under the per-task
metric set defined in `compute_classification_metrics`; the binary
guardrail in `validate_task_metric_compatibility` is enforced before
optimization starts.

## Categorical handling

CatBoost receives raw categorical columns. XGBoost and LightGBM
receive label-encoded ints by default; native categorical handling
is exposed but disabled in the headline run for cross-method parity.
This decision is logged per replica in the manifest.

## Frozen (Commit 27 freeze gate cleared)

- Method list and family assignment.
- Primary / ablation / subset / literature-only labels.
- Budget equivalence rule (formula).
- Multiclass + categorical handling defaults.
- Multi-fidelity slot: **ASHA** (renamed `method_id` from
  `hyperband_or_asha`).
- FLAML stays **literature_only**.
- ParEGO subset rule and the 48-task subset (`parego_subset.csv`).
- Per-method execution tier and per-stage gating
  (`execution_policy.csv` + `execution_tiers.md`).

The two TODO references in `article/references.bib`
(`bischl2021openmlbenchmark`, `rapin2018nevergrad`) remain unverified
but do not block shard generation; they will be resolved at proof
stage.

## Forward link

The next commit generates the SQLite shard files under `shards/`
from `method_matrix.csv` + `execution_policy.csv` +
`parego_subset.csv` + `tasks.csv`, against the
`jobs/doctoral/openml_cc18/schema.sql` schema. No method names,
scope rules, or stage-gating logic are hardcoded in the shard
generator.
