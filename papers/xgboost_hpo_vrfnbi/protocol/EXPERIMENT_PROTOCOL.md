# Experiment protocol, Paper 2

**Status: DRAFT, NOT FROZEN.** This document is frozen, and the tag `xgboost-hpo-protocol-v1`
applied, only after `protocol_adversarial_review.md` records every MUST FIX as resolved. Nothing in
the campaign runs before that, except the pilot, whose purpose is to supply the numbers this
document still leaves open.

**Tag lineage.** `xgboost-hpo-protocol-v1` froze this document before pilot Stage A ran.
Stage A then revised two parameters — the surrogate gate's external set and, through it, the budget
and the NSGA-II settings — and closed the question of how many objectives to run. Those revisions are
frozen at `xgboost-hpo-protocol-v2`. **`v1` is not moved.** It records what was pre-registered before
any measurement, which is exactly the thing a reader should be able to check the revisions against.
Every difference between the two tags is a measurement reported in
`audits/PILOT_STAGE_A_FINDINGS.md`, not a preference.

**Pre-registration rule.** Every threshold, budget and decision rule below is fixed before any
campaign result is seen. Anything decided after seeing a result is reported as post hoc, in those
words.

---

## 1. Question

Hyperparameter optimization by design of experiments, response surfaces and Normal Boundary
Intersection over Varimax-rotated factor objectives bundles three choices that are usually made
silently:

1. how the objective reference is constructed,
2. which scalarization geometry is used,
3. where the anchors come from.

Which of the three does the outcome depend on, and by how much?

## 2. Arms

Four primary arms — HISTORICAL-WS, WS-S, NBI-S, NBI-R — defined in `protocol/method_arms.md`, with
the two-by-two rationale and the list of what is held fixed. HISTORICAL-WS is run twice, once
bit-faithfully (`HISTORICAL-WS-asrun`) and once under the shared specification, and the two are never
mixed in one table.

## 3. Comparators

Random search, NSGA-II, Bayesian optimization, tree-structured Parzen estimator and coarse grid, per
`baseline_gap_assessment.md`, all at the `B_total_solution` of the most expensive arm, with the
realized ratio reported. NSGA-II is treated as **likely required**, not optional.

**NSGA-II budget — DECIDED (resolution of review finding A2), revised by the pilot.** Population and
generations are fixed jointly in advance so that their product matches the budget as closely as the
arithmetic allows and the generation count is at least ten. At the revised q = 2 budget of 386 the
setting is **population 32 for 12 generations** (384, a shortfall of 2 evaluations, 0.5%, reported as
such; 386 = 2 x 193 admits no exact factorization with ten or more generations).

A starved population method is a budget artifact, not a finding. NSGA-II is therefore also run at
**ten times** the matched budget, 3,860 evaluations, reported separately and labelled as unmatched.
Pilot Stage A measured that doing so on every replication would cost 180 hours by itself, so the
unmatched run is scoped to **one replication per dataset**, about 6 hours, and is reported as a
single-replication check on whether budget starvation explains the matched result. Both runs use the
same deterministic box repair with integer rounding and the same seed policy as the arms.

**Single-objective comparators are not scored as fronts (resolution of A4).** Bayesian optimization
and the tree-structured Parzen estimator as the dissertation configured them produce two
single-objective optima, not a front. They are reported on the single-objective endpoints only,
labelled as single-objective runs, and never appear in the front-indicator table.

## 4. Datasets

MAGIC, Adult, Spambase and Bank Marketing, provisionally, per `protocol/dataset_selection.md`. Each
must pass the four screening measurements in the pilot or be replaced from the registry, with the
measurement that caused the replacement recorded.

## 5. Design and evaluation

The version-controlled 88-run face-centred central composite design on the seven XGBoost
hyperparameters of `protocol/original_thesis_protocol.md` §2. Stratified 5-fold cross-validation.
R = 30 replications per dataset, paired by partition. One seed per replication, with component
offsets recorded in a seed map.

## 6. Objectives — DECIDED

The three sub-decisions below were freeze blockers. Each is settled here, with the evidence that
settled it, before any campaign result exists.

### 6.1 The cost objective — DECIDED

`audits/provenance/README.md` showed the dissertation's wall-clock `Time_MeanFold` does not reproduce
across environments, so it cannot anchor a paired comparison across 30 replications. Candidate
replacements were measured against the 88 measured times of the reproduced MAGIC design:

| Candidate | Spearman against measured time | Deterministic across refits | Range ratio |
|---|---|---|---|
| total **leaf count** of the fitted ensemble | **0.862** | **yes, verified** | 339 |
| `n_estimators` alone | 0.856 | yes | 14 |
| total node count | 0.846 | yes, verified | 423 |
| `n_estimators` x `max_depth` | 0.806 | yes | 84 |
| `n_estimators` x `max_depth` x `subsample` x `colsample_bytree` | 0.475 | yes | 33600 |
| measured wall-clock time | 1.000 | **no**; two fits of one configuration differ by 1.3% (median) | 111 |

Two facts worth stating plainly. **No deterministic proxy reproduces wall-clock time better than
about 0.86**, and elaborating the proxy with the sampling hyperparameters makes it markedly worse,
not better. And the wall-clock measurement is itself noisy at the 1.3% level between two fits of the
same configuration, which bounds how well any proxy could possibly agree with it.

**Decision. The primary cost objective is the total leaf count of the fitted ensemble**, summed over
all trees, averaged over the cross-validation folds. It is deterministic given the seed and the data,
machine-independent, has the widest dynamic range of the deterministic candidates that track time
well, correlates with measured training time better than any of them, and is itself a meaningful
deployment quantity: it is the model's size, and traversal cost at inference scales with the trees
and depths that produce it.

Reproduce the table with `python scripts/cost_objective_selection.py`, which writes
`audits/cost_objective_selection.json`.

**Mean per-fold wall-clock training time is retained as a reported secondary**, for continuity with
the dissertation, with its measured run-to-run variation stated. It is never a primary endpoint.

The relationship between the two, Spearman 0.86 on the MAGIC design, is reported. The two are
related, not interchangeable, and §11.4 pre-registers reporting whether they disagree about the
winning arm.

### 6.2 The quality objectives — DECIDED

Six responses, entering the factor stage alongside the cost objective:

| Response | Why |
|---|---|
| Accuracy, mean over folds | dissertation continuity |
| Precision, mean over folds | dissertation continuity |
| Recall, mean over folds | dissertation continuity |
| Specificity, mean over folds | dissertation continuity |
| Area under the receiver operating characteristic curve | threshold-free ranking; the dissertation had none |
| Log loss | calibration; the dissertation had none |

The first four are threshold metrics at 0.5, which is what the dissertation optimized and what made
its trade-off a fixed-threshold one. The last two are added because a result about hyperparameter
optimization that says nothing about ranking or calibration will not speak to the venue's audience.

### 6.3 The factor stage — DECIDED

One shared specification for all arms. **Every objective is canonicalized to minimization**, with its
direction declared per objective rather than inferred from a loading sign (resolution of review
finding M2); a test asserts that all four arms see elementwise-identical canonicalized objectives on
the design rows.

| Element | Value |
|---|---|
| Extraction | principal components of the correlation matrix of the seven standardized responses |
| Reported loadings | eigenvectors scaled by the square root of their eigenvalues, and the text says so |
| Rotation | Varimax, applied to the **scaled loadings**, not to the eigenvectors |
| Number of components | fixed at 3, with the Kaiser criterion reported alongside |
| Sign orientation | declared per objective, not inferred |
| Quality aggregation | **weighted by each component's share of explained variance** |
| Pre-registered sensitivity | the unweighted mean of z-scored scores, that is the dissertation's own choice |

**Why variance weighting is primary.** `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4 measured the two
weightings ranking the MAGIC design at Spearman 0.374, sharing 1 of 10 top rows and picking different
best rows. The choice is therefore not innocuous and cannot be left to a default. Variance weighting
is chosen because it is the weighting the extraction itself implies: a component's share of explained
variance is the reason it was extracted, and discarding that share by z-scoring to unit variance
throws away the only ordering the method provides. The dissertation's equal weighting becomes the
pre-registered sensitivity analysis of §11.1, which is the honest way to carry a predecessor's choice
forward without either adopting it silently or discarding it silently.

## 7. Surrogate and its gate

Quadratic response surface per objective, backward elimination at α = 0.05, hierarchy enforced,
**coded units**. Fitted identically for every arm that uses a surrogate.

An external reliability gate is applied. Paper 1's central finding was that an unvalidated surrogate
fails in ways nothing else detects, and the dissertation pipeline has no such check at all.

**DECIDED, and revised by the pilot.** The held-out set is the design's **complementary half
fraction plus axial runs at half the design's axial distance**: 78 points per replication, disjoint
from the 88 design rows by construction and evaluated on the real objectives. The gate passes when
**external R² ≥ 0.5 and Spearman ≥ 0.9** per response.

The original specification was 100 Latin-hypercube points, inherited from Paper 1. Pilot Stage A
showed that construction does not transfer: in seven dimensions uniform sampling reaches almost no
corner combinations, so the held-out set carried 2.0 to 9.8 times less response spread than the
design, and R² against it was dominated by a near-zero denominator rather than by the surface. The
same surfaces score R² of 0.775 to 0.953 against the complementary fraction and −23.99 to +0.468
against random sets. See `audits/PILOT_STAGE_A_FINDINGS.md` Finding 4.

The thresholds are still Paper 1's, and adopting them remains a transfer rather than a calibration.
Measured pre-campaign they now pass 7 of 8 dataset-by-response cells, failing only Spambase's quality
surface at Spearman 0.847. **The threshold is not moved.** The honest reading, which goes in the
manuscript, is that on this problem the surrogate is adequate nearly everywhere and the gate is a
check rather than a discriminator. Retuning after seeing campaign results is forbidden.

`protocol/budget_accounting.md` charges this as `B_surrogate_validation = 78` per replication for
every arm that uses a surrogate.

## 8. Budget

The ledger of `protocol/budget_accounting.md`: `B_design`, `B_surrogate_validation`, `B_anchor`,
`B_candidate_validation`, `B_direct_search`, `B_total_solution`, `B_total_audit`,
`B_total_experiment`. Every arm is charged for everything it needs to run alone. `B_total_audit` is
reported and charged to no arm.

**DECIDED, per replication per dataset, at q = 2 objectives:**

| Term | HISTORICAL-WS | WS-S | NBI-S | NBI-R |
|---|---:|---:|---:|---:|
| `B_design` | 88 | 88 | 88 | 88 |
| `B_surrogate_validation` | 0 | 78 | 78 | 78 |
| `B_anchor` | 0 | 0 | 0 | **200** |
| `B_candidate_validation` | 20 | 20 | 20 | 20 |
| **`B_total_solution`** | **108** | **186** | **186** | **386** |

`B_anchor` is **100 real evaluations per objective**, spent by NBI-R's direct search for each
objective's real optimum. The figure is fixed in advance and is identical across datasets and
objectives, so it is a constant of the design and not a tuning knob. At q = 3 it becomes 300 and
NBI-R's total 508.

HISTORICAL-WS is charged no surrogate validation because the dissertation pipeline has no gate; that
is a property of the historical method, not a concession to it, and the manuscript says so.

**The comparator budget is 386** (at q = 2), the `B_total_solution` of the most expensive arm, so no
comparator is handicapped. The realized ratio is reported per run. Note the asymmetry this creates
and state it: the cheaper arms are compared against comparators that received up to 3.6 times their
budget. That is the conservative direction for the paper's own claims and the honest one to report.

**Measured campaign cost (pilot Stage A).** Per-evaluation cost is 1.02 to 1.69 seconds on eight
threads, mean 1.40 s. Evaluations actually performed per replication per dataset are 446 for the arms
and 1,930 for the five comparators, 2,376 in total; over 30 replications and 4 datasets that is
**285,120 evaluations, about 111 hours, 4.6 days run serially**. The same accounting at q = 3 gives
357,120 evaluations and 5.8 days.

**The campaign therefore runs at q = 2.** This closes the open question of whether to use three
objectives with a measurement rather than a preference, and the manuscript reports it that way.

The serial figure assumes no parallelism across replications and is the pessimistic bound. Paper 1
met the same arithmetic and found that running independent replications across eight processes with
single-threaded evaluators cut a projected 106 hours to 15.5 actual. Stage B measures the parallel
throughput before the campaign launches.

## 9. Scoring

The full front is persisted for every method and replication. Primary endpoints are computed against
an empirical reference built independently of any single method, and repeated against a sampled core
the compared methods do not contribute to. Indicators follow Paper 1: inverted generational distance
plus, hypervolume ratio, generational distance, Schott spacing, joint non-dominated fraction.
`max(Accuracy_Mean)` is a legacy continuity column only.

Every NBI subproblem records `success` and `residual_norm`, and the certified fraction is reported
per arm. `audits/NBI_GEOMETRY_AUDIT.md` showed that an infeasible subproblem still returns a
plausible-looking iterate, so publishing the certification rate is the only defence.

**Integer rounding (resolution of review finding M3).** `max_depth` and `n_estimators` are integers,
cast by `int(round(.))` at evaluation time, while the surrogate is fitted on the continuous
relaxation and the NBI equality constraint is certified at a continuous point. The point actually
evaluated is a rounded neighbour that does not satisfy that constraint, which biases exactly the
contrast this paper is about. Two measures, both required:

1. Each subproblem reports the objective displacement induced by rounding,
   `norm(F_hat(x_continuous) - F_hat(x_rounded))`, beside its residual. An arm's certification is
   reported as the certified fraction **and** the distribution of rounding displacement.
2. Each NBI subproblem is re-solved with the integer dimensions fixed at their rounded values and the
   continuous dimensions re-optimized, and the certified fraction of that restricted problem is
   reported. This costs no real evaluations and certifies the point actually evaluated.

**Anchor degeneracy (resolution of review finding M4).** A backward-eliminated quadratic is often
minimized on a box corner, so two objectives' surrogate anchors can coincide, making the payoff
matrix rank-deficient and the quasi-normal ill-defined. Each replication records the anchor decision
vectors, the rank and condition number of the payoff matrix, and whether any two anchors coincide. A
rank-deficient replication is reported and excluded from the geometry contrasts, with the exclusion
count published. It is not silently repaired, because how often it happens is itself a result about
the method.

## 10. Statistics

Paired by partition. Percentile bootstrap intervals on medians, with the interval matched to the
point estimate it accompanies. Win, tie and loss counts with Wilson intervals. Matched-pairs
rank-biserial correlation. Holm correction within each dataset family; no pooling across datasets.
The dataset is the unit of generalization.

**The overlap correction (resolution of review finding S1).** The Nadeau and Bengio corrected t-test
assumes an overlap proportion that follows from the resampling scheme. Paper 1's value came from its
own scheme and is **not** carried over. Here the overlap proportion is derived from the actual
partition scheme used, stated explicitly, and every significance statement is reported as holding
*at* that value, with the effect consistency that does not depend on it reported beside it. If the
scheme makes the proportion ambiguous, the descriptive triple (median with its bootstrap interval,
win fraction with its Wilson interval, rank-biserial correlation) is primary and the test secondary.

**Multiplicity across the four arms (resolution of review finding S2).** Four arms admit six pairwise
comparisons. The three identifying contrasts of `protocol/method_arms.md` --- HISTORICAL-WS to WS-S,
WS-S to NBI-S, NBI-S to NBI-R --- are pre-registered as the **primary family** and carry the Holm
correction within each dataset. The remaining three pairwise comparisons are declared **secondary and
descriptive in advance**, reported without tests, and are never described as significant.

## 11. Planned secondary analyses, pre-registered

1. **Aggregation sensitivity across the panel.** Repeat the Q4 comparison on every dataset and every
   replication, and report how often the two weightings disagree about the **returned front**, not
   only about the design ranking. This is what turns `novelty_matrix.md` element 17 from an anecdote
   into a result.

   **Pre-registered reading (resolution of review finding S4).** If the two weightings disagree about
   the returned front in more than a stated fraction of replications, the paper's conclusions are
   reported as conditional on the weighting and the conditionality appears in the abstract. Below
   that fraction it is a robustness note. **DECIDED: the fraction is 0.20.** If the two weightings
   return different fronts in more than 20% of replications on any dataset, the conclusions for that
   dataset are reported as conditional on the weighting, and the conditionality appears in the
   abstract. "Different fronts" means the two returned non-dominated sets differ by at least one
   member after deduplication at the evaluation tolerance.
2. **The anchor-injection control.** Mandatory, not optional. Rescore NBI-S's candidate set augmented
   with the real anchors and report how much of the NBI-S to NBI-R gap that alone closes.
3. **The restricted-`t` demonstration.** Run NBI-S once with `restrict_t_nonnegative=True` to show on
   the real problem what `audits/NBI_GEOMETRY_AUDIT.md` shows synthetically.
4. **Cost-definition sensitivity.** Whether the primary and secondary cost objectives disagree about
   the winning arm, reported as a frequency over replications.

## 12. Pilot

Two stages (resolution of review finding A1), because the screening measurements are needed for
every candidate dataset and the timing measurement is not.

**Stage A, screening.** The 88 design rows on one partition of **every** candidate dataset. About 88
real evaluations per candidate, and the only cost needed for all four screening measurements of
`protocol/dataset_selection.md`.

**Stage B, timing and certification.** All four arms and all comparators on one partition of one
dataset.

Together they produce:

- the four dataset-screening measurements of `protocol/dataset_selection.md` for each candidate;
- a measured per-evaluation cost, replacing the projection in `protocol/budget_accounting.md`;
- the cost-objective decision of §6.1;
- confirmation that every NBI subproblem certifies on the real problem, under both the continuous
  and the fixed-integer formulations of §9;
- the paired standard deviation of each primary endpoint, and from it the smallest effect the design
  detects at R = 30 with the planned interval (resolution of review finding S3). If that detectable
  effect is larger than any difference Stage B suggests exists, the protocol says so **before** the
  campaign rather than discovering it in the results;
- the panel size the measured cost allows, which may exceed four (resolution of review finding A3);
  the manuscript states the panel size as budget-determined and shows the measurement.

**Gate:** the full campaign proceeds only if the pilot's measured projection is at or under about
five wall-clock days. If it is over, the panel shrinks before the arms do, because the arm set is
what the paper is about.

## 13. Open items, and whether each blocks the freeze

All seven freeze blockers are now decided, each in the section named, from evidence that existed
before any campaign result. What remains does not block the freeze.

| # | Item | Blocks? | Where decided |
|---|---|---|---|
| 1 | The cost objective | resolved | §6.1 — total leaf count, primary; wall-clock time, secondary |
| 2 | The quality objective list | resolved | §6.2 — four threshold metrics plus ROC-AUC and log loss |
| 3 | The aggregation weighting and its sensitivity | resolved | §6.3 — explained-variance weighting, equal weighting as the sensitivity |
| 4 | Gate thresholds and held-out set size | resolved, **revised by the pilot** | §7 — 78 points, the design's complementary half fraction plus half-distance axial runs; R² ≥ 0.5 and Spearman ≥ 0.9, unchanged |
| 5 | `B_anchor` per objective | resolved | §8 — 100 real evaluations per objective; `B_total_solution` 108 / 186 / 186 / 386 |
| 6 | NSGA-II population and generations | resolved, **revised by the pilot** | §3 — 32 x 12 at the revised q = 2 budget, plus an unmatched ten-times run on one replication per dataset |
| 12 | The disagreement fraction for the weighting sensitivity | resolved | §11.1 — 0.20 |

| # | Remaining item | Blocks? | Disposition |
|---|---|---|---|
| 7 | Dataset panel confirmed by the screening measurements | **closed** | Stage A: all four datasets pass all four criteria; no replacement needed. `audits/PILOT_STAGE_A_FINDINGS.md` |
| 8 | Thesis equation numbering, §2.9 Eqs 2.107–2.114 against §4.4.3 Eq 4.16 | no | needed before submission. The chapters are not on this machine; see `protocol/original_thesis_protocol.md` §13 |
| 9 | Whether `pepper_species` exists and is public | no | the panel does not depend on it; see `protocol/dataset_selection.md` |
| 10 | Self-overlap assessment against Pereira et al. (2025) and Paper 1 | no | needed before submission; see §13b |
| 11 | Five failing tests in `test_stage0_extreme_lane_plan.py` | no | a wall-clock staleness gate refusing a 120-day-old summary, unrelated to this work; must not be left failing at submission |
| 13 | Panel size beyond four, and the detectable effect size at R = 30 | partly closed | Stage A measured 4.6 days serial for the four-dataset panel at q = 2, so the panel does not grow; the detectable effect size still comes from Stage B |
| 14 | Whether to run at q = 3 rather than q = 2 | **closed** | Stage A: q = 3 costs 5.8 days against a 5-day ceiling, so the campaign runs at q = 2. §8 |

Items 8 and 9 both need the dissertation chapters, which are on a OneDrive path under a different
user account. Neither affects the experiment.

## 13b. Dependence on the unpublished Paper 1 (resolution of review finding E2)

Four elements of this protocol are inherited from the authors' Paper 1: the external reliability
gate, evaluation-matched budgeting, the NSGA-II harness and the anchor-injection control. Paper 1 is
frozen at `paper-submission-v2` and **has not been submitted**, so it cannot be cited as established
work.

Two rules follow. First, **every inherited element is described self-containedly in Paper 2's
methodology**, so that Paper 2 stands alone if Paper 1 is never published. Second, whichever paper is
submitted second cites the first as under review or as published, as applicable, and each discloses
the other to the editor if both are under review at once. Neither paper cites the other as
established.

## 14. What is already settled and will not be reopened

- The arm set is four, for the reason in `protocol/method_arms.md`.
- `t` is free in the NBI subproblem, per `audits/NBI_GEOMETRY_AUDIT.md`.
- Response surfaces are fitted in coded units.
- The full front is the output; `max(Accuracy_Mean)` is never a primary endpoint.
- The dataset is the unit of generalization; nothing is pooled.
- The claim blacklist in `novelty_matrix.md` is binding, and is enforced by a build-time scan of the
  compiled manuscript rather than by a checklist (review finding E4).
- WS-S's second reference end is the pseudo-nadir; every objective is canonicalized to minimization;
  the shared weight grid is symmetric. See `protocol/method_arms.md`.
- The three identifying contrasts are the primary family; the other three pairwise comparisons are
  descriptive only.
- The outcome-contingent framings in `protocol/protocol_adversarial_review.md` were written before
  the campaign and are what the manuscript uses, whichever outcome occurs.
- Each response declares its transform as well as its direction; the cost response declares `log1p`.
- Factor signs are oriented by the mean loading over the factor's own role block, and the screening
  refuses to report when the composite's objective conflict disagrees in sign with the same quantity
  measured from the responses directly. Both come from `audits/PILOT_STAGE_A_FINDINGS.md`.
