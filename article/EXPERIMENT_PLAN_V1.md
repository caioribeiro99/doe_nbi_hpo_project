# Experiment Plan v1 — first article campaign

This is the **scope of the first article**. It is intentionally smaller
than the doctoral 82-dataset benchmark.

## Headline scope

- **12** public tabular classification datasets.
- **3** GBDT algorithms: XGBoost, LightGBM, CatBoost.
- **10** independent replicas per (dataset, algorithm, method).
- **5** stratified CV folds.
- DOE = Minitab face-centered CCD (`data/design/hyperparameter_design.csv`).
- NBI weight grid = simplex_lattice {3, 10} (66 points) for the article
  3-VRF run; {2, 50} (51 points) for the dissertation-parity ablation.
- Matched evaluation budget across methods (`B = doe_runs +
  nbi_eval_k`).
- Top-up to 30 replicas only on a small set of "headline" datasets if
  the v1 results suggest the inter-replica intervals overlap too much
  to land a confident claim.

## Dataset panel

For each dataset we record: source, task, expected size, feature mix,
class imbalance, expected preprocessing, computational burden, and
whether the dataset enters the v1 run or is deferred. Replacements are
allowed only if licence / availability blockers appear during loader
implementation.

| # | Dataset | Source | Task | Rows | Feats | Categorical | Imbalanced | Preprocessing | Burden | v1? |
|---|---|---|---|---:|---:|---|---|---|---|---|
| 1 | MAGIC Gamma Telescope | UCI 159 | binary | 19,020 | 10 numeric | none | mild | map `{g,h}->{0,1}`; standardize | medium | yes (continuity with dissertation) |
| 2 | Breast Cancer Wisconsin Diagnostic | UCI 17 / sklearn | binary | 569 | 30 numeric | none | mild | standardize | light | yes (small classical) |
| 3 | Pima Indians Diabetes | OpenML 37 | binary | 768 | 8 numeric | none | mild | impute zero-as-missing for glucose / BP / skin / insulin / BMI; standardize | light | yes |
| 4 | Spambase | UCI 94 | binary | 4,601 | 57 numeric | none | mild | standardize | light | yes |
| 5 | Adult / Census Income | UCI 2 | binary | 48,842 | 14 mixed | yes | yes | drop `?`; one-hot or native categorical for CatBoost | medium | yes |
| 6 | Bank Marketing | UCI 222 | binary | 45,211 | 16 mixed | yes | yes | drop `unknown`; native categorical for CatBoost | medium | yes |
| 7 | Default of Credit Card Clients | UCI 350 | binary | 30,000 | 23 numeric | (encoded) | yes | standardize | medium | yes (financial continuity) |
| 8 | German Credit | UCI 144 | binary | 1,000 | 20 mixed | yes | mild | one-hot or native categorical | light | yes |
| 9 | Wine Quality (binarised quality >= 6) | UCI 186 | binary | 6,497 | 11 numeric | none | mild | red+white merged + flag; standardize | light | yes |
| 10 | Dry Bean | UCI 602 | multiclass (7) | 13,611 | 16 numeric | none | mild | standardize | medium | yes (multiclass test) |
| 11 | Mushroom | UCI 73 | binary | 8,124 | 22 categorical | yes | balanced | label-encode or native categorical for CatBoost | light | yes (categorical-heavy showcase) |
| 12 | Phishing Websites | UCI 327 | binary | 11,055 | 30 categorical/integer | mostly | balanced | none beyond label encode | medium | yes (security application) |

### Possible substitutions (only if a licence / hosting issue appears)

- **Higgs (small)** could replace MAGIC for a larger numerical
  baseline if MAGIC ever becomes hard to fetch.
- **Telco Customer Churn** (Kaggle) could replace Bank Marketing if
  the UCI mirror is unstable.
- **Covertype (binarised)** could replace Spambase for a larger
  numerical-only benchmark.

## Dry Bean — multiclass case study (appendix only)

Dry Bean is the only multiclass entry in the v1 panel (7 classes).
`docs/METHODOLOGY_DECISIONS.md` D15 documents the decision:

**Option B.** Dry Bean is reported as a *multiclass case study in the
appendix / supplementary section*, **not a headline dataset**. The v1
headline tables use the **eleven binary datasets**.

Multiclass evaluator (Commit 18):

- `compute_multiclass_metrics(...)` returns `accuracy`, `f1_macro`,
  `balanced_accuracy`, `mcc`, `roc_auc_ovr_macro`, `pr_auc_ovr_macro`,
  `brier_multiclass`, `ece_multiclass`. Aggregated keys:
  `F1Macro_Mean`, `BalancedAccuracy_Mean`, `MCC_Mean`,
  `ROCAUC_OVR_Mean`, `PRAUC_OVR_Mean`, `BrierMC_Mean`, `ECE_Mean`.
- `evaluate_xgb_cv` auto-detects the task type and uses
  `predict_proba` for probability-based metrics.
- `assert_metric_set_compatible_with_task` is the orchestrator-side
  guardrail that refuses to start an FA / NBI run on Dry Bean with the
  binary response defaults.

Dry Bean enters the appendix tables only after a multiclass YAML
config (article track) ships that explicitly selects the eight
multiclass response columns above for FA / NBI.

## Per-dataset metric coverage

For all datasets we report:

- ROC-AUC, PR-AUC, F1 (binary) / F1-macro (multiclass), Balanced
  Accuracy, MCC.
- Brier score, ECE (calibration metrics) -- secondary, reporting only.
- Mean fit-plus-predict time per fold; total optimization wall-time.
- For the proposed method: full Pareto front; NBI sub-problem
  residuals; MBPA trigger diagnostics.

Calibration is **reported, not optimized** in v1. A follow-up paper
may add Brier / ECE as primary objectives via the existing
`ObjectiveSpec` interface.

## Computational sizing

The cost estimator preset
`article_v1_12_datasets_3_algorithms_10_replicas`
(`docs/COST_ESTIMATOR.md`) gives an authoritative wall-clock and
dollar-cost projection. The expected order of magnitude under the
default `avg_seconds_per_fit=0.5`, `overhead_factor=1.10`,
`local_efficiency_factor=0.70`, 8 idle / 2 working workers, 16 hours
available per day:

```bash
doe-xgb estimate-cost --preset article_v1_12_datasets_3_algorithms_10_replicas
```

**Calibrate** before launch with `--calibrate --calibration-output
cost_estimate_calibration.json --algorithm xgboost` (and rerun for
LightGBM / CatBoost). Use the worst-case `avg_seconds_per_fit` of the
three.

## Statistical analysis plan

- **Within-dataset**: paired Wilcoxon signed-rank on each metric
  across the R replicas, comparing the proposed method to each
  baseline. Report effect size + Holm-Bonferroni-corrected p-values.
- **Cross-dataset**: Friedman test on per-(dataset, algorithm) ranks,
  followed by Nemenyi post-hoc with critical-difference plots
  ([Demsar 2006](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)).
- **Frontier quality**: per-replica IGD, spread, and spacing entropy;
  aggregated as median ± inter-quartile range.

## Deliverables

For each (dataset, algorithm, method, replica), the orchestrator writes
the artifact set defined in `docs/REPRODUCIBILITY.md`. The aggregator
produces:

- `article/tables/performance.csv` (Table~\ref{tab:performance}).
- `article/tables/cost.csv` (Table~\ref{tab:cost}).
- `article/tables/nbi_residuals.csv` (Table~\ref{tab:nbi-residuals}).
- `article/tables/mbpa_summary.csv` (Table~\ref{tab:mbpa}).
- `article/figures/cd_diagram.pdf`, `pareto_<dataset>_<alg>.pdf`,
  `nbi_vs_weighted_sum.pdf`.

## Scope guard

- **No** doctoral-scale 82-dataset run in this article.
- **No** non-tree estimators (deep tabular networks, linear models)
  in this article.
- **No** unbounded budget tuning; every method receives the same
  $B = 138$ evaluations per replica.
- **No** new methodology beyond what is implemented and CI-tested on
  `repo-publication-readiness` at the time of submission.
