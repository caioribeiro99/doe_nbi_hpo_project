# The original dissertation protocol, reconstructed from the frozen code

Reconstructed by reading the source at tag `v0.1.0-dissertation` (commit `67d9fe5`) and by executing
it (see `audits/provenance/`). Where the dissertation *text* and the dissertation *code* disagree,
this document reports the **code**, because the code produced the numbers. Textual divergences are
flagged and cross-referenced to `docs/METHODOLOGY_DECISIONS.md` on the article branch, which the
author wrote before this audit began.

---

## 1. Problem

Multiobjective hyperparameter optimization of XGBoost for binary classification, trading predictive
quality against training cost.

## 2. Decision space

Seven hyperparameters, with the bounds in `src/doe_xgb/config.py`:

| Hyperparameter | Low | High | Type |
|---|---|---|---|
| `subsample` | 0.05 | 1.00 | continuous |
| `colsample_bytree` | 0.05 | 1.00 | continuous |
| `colsample_bylevel` | 0.05 | 1.00 | continuous |
| `learning_rate` | 0.01 | 0.30 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.00 | continuous |
| `n_estimators` | 50 | 700 | integer |

Integers are cast by `int(round(.))` at evaluation time, so the surrogate is fitted on a continuous
relaxation whose evaluated points are rounded. This is a real, unremarked discretization gap.

## 3. Design

A face-centred central composite design in seven factors, 88 runs, generated in Minitab and stored
as `data/design/hyperparameter_design.csv` (semicolon-separated, comma decimals, with `StdOrder`,
`RunOrder`, `PtType` and `Blocks` columns). The design is version-controlled, so it is reusable
exactly; its SHA-256 is recorded in `audits/provenance/provenance_run.json`.

## 4. Evaluation

`evaluate_xgb_cv`: stratified 5-fold cross-validation, shuffled, seeded. Per fold it records
accuracy, precision, recall and specificity, plus wall-clock fit-and-predict time from
`time.perf_counter()`. `aggregate_fold_metrics` reports the mean of each across folds, and
`Time_MeanFold` is the mean per-fold time.

Two consequences worth stating plainly:

- **The quality metrics are threshold metrics at 0.5.** No ranking metric (area under the receiver
  operating characteristic curve) and no calibration metric (log loss, Brier score) enters the
  objective set. The cost-quality trade-off is therefore measured entirely at a fixed decision
  threshold.
- **Cost is wall-clock training time on the machine that ran it.** It is not an operation count and
  is not machine-independent, so it cannot be compared across hardware.

## 5. Objective reduction

`run_factor_analysis` with the frozen defaults: log1p on the time column, z-score all five metrics,
principal component analysis with `n_factors = 3` (raised to at least 3 by `force_time_factor`),
Varimax rotation, sign orientation by the sign of the time loading and by the mean sign of the
quality loadings.

Then:

- `Score_Cost` = negated z-score of the time factor's score, so larger is better.
- `Score_Quality` = unweighted mean of the z-scored scores of the remaining factors, because
  `combine_quality_factors` defaults to `True`.

**Three factors are extracted and two objectives are optimized.** `docs/METHODOLOGY_DECISIONS.md`
D8 records the same thing from the other direction: the final dissertation text describes three
Varimax-rotated factors, and the code collapses the non-time factors into one. See
`audits/PCA_VARIMAX_IDENTITY_AUDIT.md` for what else the frozen factor stage does and does not do.

## 6. Surrogate

`fit_rsm_backward`: a full quadratic response surface per objective, fitted in **uncoded** units,
with backward elimination at α = 0.05 and hierarchy enforced. One surface for `Score_Quality`, one
for `Score_Cost`. `docs/METHODOLOGY_DECISIONS.md` D6 records that the dissertation *tables* report
coded coefficients while the code fits uncoded.

Reproduced on MAGIC (`audits/provenance/`): R² = 0.859 with 18 terms for quality, R² = 0.903 with
12 terms for cost.

There is **no external validation of the surrogate**. Nothing in the frozen pipeline holds out
design points, checks external R², or gates the optimizer on surrogate adequacy. Every downstream
result rests on an unvalidated fit.

## 7. Optimizer

`run_nbi_weighted_sum`: normalized weighted-sum scalarization, 20 weight pairs at step 0.05, SLSQP
from 10 multistarts, with an optional inequality keeping both predictions inside the observed
`[nadir, utopia]` box. The box is built from component-wise maxima and minima of the design rows.

This is not Normal Boundary Intersection. See `audits/METHODOLOGICAL_IDENTITY_AUDIT.md`.

## 8. Candidate validation

`evaluate_candidate_list` re-evaluates every returned candidate by the same real 5-fold CV, so the
reported performance of the proposed method is measured, not predicted. This is a genuine strength
of the original protocol and Paper 2 keeps it.

## 9. Selection

`best_row` is chosen by `max(Accuracy_Mean)` for **every** method, including the proposed
multiobjective one. A multiobjective procedure is therefore scored by a single-objective rule, which
discards the trade-off the procedure exists to expose. `docs/METHODOLOGY_DECISIONS.md` D7 records
this and sets `distance_to_utopia` as the article-track default.

## 10. Comparators and budget

`benchmark_budget = 88 (design runs) + number of validated candidates`, and each comparator is given
that many evaluations. Comparators: coarse grid search, random search, Bayesian optimization
(`scikit-optimize`) and tree-structured Parzen estimator (`hyperopt`).

The accounting omits the anchor cost, because weighted sum with an observed-extremes box has none.
Any canonical-NBI arm does have one, which is why `protocol/budget_accounting.md` separates the
ledger into terms rather than a single number.

## 11. Replication

30 replicas, seeded per replica.

## 12. Known text-versus-code divergences

| # | Dissertation text | Frozen code | Recorded in |
|---|---|---|---|
| 1 | NBI with anchors, CHIM, quasi-normal, `max t` | normalized weighted sum over a β grid | D1 |
| 2 | three Varimax-rotated factors | three extracted, two optimized | D8 |
| 3 | coded RSM coefficients in the tables | uncoded fit in the code | D6 |
| 4 | selection by utility, knee point or distance to utopia | `max(Accuracy_Mean)` | D7 |
| 5 | factor analysis with Varimax | PCA on a correlation matrix, Varimax applied to eigenvectors | this audit, `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` |

Divergence 5 is the one not previously recorded.

## 13. Unresolved: equation numbering

`docs/METHODOLOGY_DECISIONS.md` D1 cites the NBI formulation as **§4.4.3, Eq. 4.16**. The planning
note for Paper 2 cites **§2.9, Eqs. 2.107–2.114**. These cannot both be the primary NBI derivation:
§2.x is the theoretical framework chapter and §4.x is results and discussion.

**This could not be resolved here.** The translated dissertation chapters are not on this machine.
`article/DISSERTATION_TO_ARTICLE_MAP.md` records their location as a OneDrive path under a different
user account (`/Users/caiotertuliano/Library/CloudStorage/OneDrive-Pessoal/.../Tradução da
Dissertação/`), holding `Dissertacao_Caio_02_theoretical_framework_en.pdf` and
`Dissertacao_Caio_04_results_discussion_en.pdf` among others. The dissertation is unpublished, so
there is no repository copy to fetch.

**What is needed:** open those two PDFs and record which section states the NBI subproblem
`max t s.t. Φβ + t·n̂ = F(x)` and which merely applies it. The likely reading, given the chapter
structure, is that §2.9 Eqs. 2.107–2.114 is the theoretical derivation and §4.4.3 Eq. 4.16 is its
instantiation on the XGBoost problem, in which case both citations are correct for different
purposes and D1 should cite both.

**Blocking status:** not blocking. Nothing in the Paper-2 protocol depends on which equation number
is primary; the audit rests on the code, not on the text. The citation must be fixed before Paper 2
is submitted, and is tracked in `protocol/EXPERIMENT_PROTOCOL.md` as an open item.
