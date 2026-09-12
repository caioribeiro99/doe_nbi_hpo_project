# Evaluation-matched NSGA-II baseline — verified report

Answers one narrow question: **at a matched real-objective evaluation budget, how does metamodel-free NBI (NBI-C)
compare with a canonical evolutionary multiobjective optimizer on the same cached ensemble-weight objectives?**

The frozen R = 30 experiment (tag `pco213-postwork-r30`, commit `b3ed050`) was not modified. No model was retrained,
no out-of-fold prediction regenerated. NSGA-II reads the frozen cached artifacts and writes to a separate tree.

**Headline: NSGA-II attains a better approximation than NBI-C on both primary endpoints, on all four datasets, under
both reference definitions.** The margins are small in absolute terms and the consistency is high. This is reported
plainly; the paper's contribution is diagnostic, not a claim of algorithmic superiority for NBI.

---

## 1. Configuration and provenance

| Item | Value |
|---|---|
| Library | pymoo 0.6.2 (`blank2020pymoo`); algorithm NSGA-II (`deb2002nsga2`) |
| Configuration | `papers/surrogate_nbi_ensemble/nsga2_preregistered_config.json`, committed at `3e64804` **before** any run |
| Objectives | exactly NBI-C's: f1 = −ROC-AUC, f2 = log-loss, f3 = weighted cost Σ wᵢcᵢ, on cached OOF probabilities |
| Encoding | real vector in [0,1]⁵ with deterministic simplex repair after sampling, crossover and mutation |
| Repair | clip to non-negative, divide by the sum; uniform fallback for a degenerate row. A repair, not a Euclidean projection |
| Operators | pymoo defaults: SBX η = 15, p = 0.9; PM η = 20, p = 1/5; rank + crowding survival; duplicate elimination |
| Population | 66, matching the 66 NBI β directions |
| Termination | per dataset × replication, matched to the real-objective evaluation count NBI-C actually consumed |
| Seed | 20260904 + rep + 770000 (offset fixed once, before execution) |
| Tuning | none. No operator parameter, population size or termination rule was chosen by inspecting performance |

## 2. Execution

| Item | Value |
|---|---|
| Runs | **120 / 120**, 0 failures, 0 retries |
| Target evaluations | 50,150,376 |
| Actual evaluations | **50,150,166** (ratio **0.999996**) |
| Per-run ratio | 0.999924 … 1.000099 |
| Wall clock | 15.46 h across 8 worker processes; 123.3 h of CPU |
| Returned set | 66 mutually non-dominated points in every run |

Per-dataset wall clock against NBI-C (means, seconds per replication): BNP 3,040 vs 1,409; Porto 5,541 vs 2,503;
Santander 5,526 vs 2,527; UCI credit 689 vs 306. **NSGA-II costs about 2.2× the wall clock of NBI-C at equal
evaluation count.** Only evaluations are matched; wall clock is not, and is reported separately.

## 3. Two reference definitions

**Reference A — sample-core.** The non-dominated set of the independently sampled real-objective reference only:
Dirichlet samples, simplex lattice, vertices, edge sweeps, the ε-constraint sweep, the 66-run design and the
single-objective references. It contains the search output of **no** optimizer under comparison, so it cannot favour
any of them. This is the clean sensitivity reference.

**Reference B — common augmented union.** The non-dominated set of the sample core together with every candidate set,
NSGA-II included, Pareto-filtered on real objectives. Indicators are recomputed for **all** algorithms against this
same union, so NSGA-II is neither uniquely advantaged nor uniquely disadvantaged.

Contribution to Reference B's front (mean over four probed replications per dataset): NSGA-II 8–29%, NBI-C 8–23%,
sample core 43–75%. Both optimizers are therefore partly graded against their own output in Reference B, which is
precisely why Reference A is reported alongside. **The conclusion is identical under both.**

Scoring reuses `_quality_for_set` imported verbatim from the frozen benchmark runner, so NSGA-II is scored by exactly
the code that produced the R = 30 numbers.

## 4. Primary comparison — NSGA-II vs NBI-C, weighted cost

Δ > 0 favours NSGA-II. ΔIGD⁺ = IGD⁺(NBI-C) − IGD⁺(NSGA-II); ΔHV = HV(NSGA-II) − HV(NBI-C).

### Reference A (sample-core)

| Dataset | Endpoint | mean [95% CI] | median | W/T/L | r_rb | Holm p |
|---|---|---|---|---|---|---|
| Santander | IGD⁺ | +0.0047 [0.0038, 0.0055] | +0.0047 | 29/0/1 | +1.00 | 0.0013 |
| Santander | HV | +0.0060 [0.0050, 0.0070] | +0.0060 | 30/0/0 | +1.00 | 0.0009 |
| BNP Paribas | IGD⁺ | +0.0080 [0.0071, 0.0090] | +0.0075 | 30/0/0 | +1.00 | <0.0001 |
| BNP Paribas | HV | +0.0099 [0.0090, 0.0110] | +0.0098 | 30/0/0 | +1.00 | <0.0001 |
| Porto Seguro | IGD⁺ | +0.0112 [0.0066, 0.0168] | +0.0065 | 28/0/2 | +0.97 | 0.55 |
| Porto Seguro | HV | +0.0209 [0.0118, 0.0318] | +0.0113 | 26/0/4 | +0.94 | 0.55 |
| UCI credit | IGD⁺ | +0.0037 [0.0025, 0.0050] | +0.0034 | 24/0/6 | +0.89 | 0.13 |
| UCI credit | HV | +0.0116 [0.0091, 0.0141] | +0.0114 | 30/0/0 | +1.00 | 0.020 |

### Reference B (common augmented union)

| Dataset | Endpoint | mean [95% CI] | median | W/T/L | r_rb | Holm p |
|---|---|---|---|---|---|---|
| Santander | IGD⁺ | +0.0074 [0.0054, 0.0095] | +0.0060 | 29/0/1 | +1.00 | 0.026 |
| Santander | HV | +0.0107 [0.0082, 0.0134] | +0.0083 | 30/0/0 | +1.00 | 0.026 |
| BNP Paribas | IGD⁺ | +0.0089 [0.0081, 0.0098] | +0.0087 | 30/0/0 | +1.00 | <0.0001 |
| BNP Paribas | HV | +0.0131 [0.0119, 0.0143] | +0.0127 | 30/0/0 | +1.00 | <0.0001 |
| Porto Seguro | IGD⁺ | +0.0090 [0.0051, 0.0139] | +0.0048 | 26/0/4 | +0.94 | 0.58 |
| Porto Seguro | HV | +0.0209 [0.0117, 0.0319] | +0.0115 | 26/0/4 | +0.94 | 0.58 |
| UCI credit | IGD⁺ | +0.0034 [0.0022, 0.0045] | +0.0032 | 23/2/5 | +0.87 | 0.13 |
| UCI credit | HV | +0.0121 [0.0095, 0.0147] | +0.0114 | 30/0/0 | +1.00 | 0.023 |

**Reading.** The direction is unanimous across 16 dataset × endpoint × reference cells. The corrected test reaches
significance on Santander, BNP Paribas and UCI credit hypervolume; on Porto Seguro the paired differences are
heavy-tailed (mean +0.021 against median +0.011), so by the study's own pre-specified rule the median, the win
fraction and the rank-biserial correlation are the evidence there, and all three point the same way. Absolute margins
are small: the median hypervolume-ratio gain is between +0.006 and +0.013.

## 5. Indicator levels (weighted cost, Reference B, median over 30)

| Dataset | NBI-A | NBI-B | NBI-C | **NSGA-II** | Scalarization | Design runs |
|---|---|---|---|---|---|---|
| Santander | 0.803 | 0.981 | 0.990 | **0.998** | 0.978 | 0.981 |
| BNP Paribas | 0.914 | 0.969 | 0.982 | **0.994** | 0.472 | 0.917 |
| Porto Seguro | 0.766 | 0.915 | 0.981 | **0.995** | 0.761 | 0.645 |
| UCI credit | 0.633 | 0.706 | 0.977 | **0.988** | 0.978 | 0.921 |

NSGA-II is the best set on every dataset. Against NBI-B it wins 29–30 of 30 everywhere (up to +0.29 hypervolume on
UCI credit); against random scalarization it wins 30/30 everywhere.

## 6. A spacing result that reverses a negative finding

The frozen study reported that NBI's characteristic even spread does **not** survive revalidation on real objectives,
and could not say whether this was a property of the objective geometry or of the CHIM construction. NSGA-II answers
it. Median spacing coefficient of variation (lower is more even), weighted cost, Reference B:

| Dataset | NBI-C | **NSGA-II** | Scalarization |
|---|---|---|---|
| Santander | 2.283 | **0.737** | 2.846 |
| BNP Paribas | 0.883 | **0.619** | 1.575 |
| Porto Seguro | 1.556 | **0.603** | 2.160 |
| UCI credit | 1.920 | **0.650** | 1.512 |

Crowding-distance selection produces markedly more evenly spaced revalidated fronts than CHIM-based construction, on
every dataset. **The poor spacing of NBI-C after revalidation is therefore specific to the construction, not an
inevitable consequence of the objective geometry.** This is a genuine addition to the paper's negative result.

## 7. Support-cost sensitivity (post hoc)

NSGA-II optimized the weighted cost only; its final set is re-scored under the support cost
Σᵢ cᵢ·1[wᵢ > 10⁻³] without re-optimization, exactly as the frozen study does for the other methods.

| Dataset | ΔIGD⁺ median · W/T/L | ΔHV median · W/T/L |
|---|---|---|
| Santander | −0.0003 · 11/2/17 | +0.0026 · 29/0/1 |
| BNP Paribas | +0.0169 · 28/0/2 | +0.0411 · 30/0/0 |
| Porto Seguro | +0.0032 · 26/0/4 | +0.0280 · 24/0/6 |
| UCI credit | +0.0130 · 21/0/9 | +0.0174 · 25/0/5 |

The advantage largely transfers to the deployment cost, and is largest on BNP Paribas, the dataset where the frozen
study found that linear-cost optimizers miss the cheapest supports. The single reversal is Santander IGD⁺, where
NBI-C is slightly closer to the support-cost reference in 17 of 30 partitions.

## 8. Verification performed

| # | Check | Result |
|---|---|---|
| 1 | Objective-evaluation budgets match | ratio 0.999996 overall; 0.999924–1.000099 per run |
| 2 | All weights are valid simplex compositions | min weight 0; max \|Σw − 1\| = 2.2 × 10⁻¹⁶ |
| 3 | No OOF/holdout leakage | optimizer never references the holdout matrix; rows used = frozen n_train |
| 4 | Same real objectives as NBI-C | max deviation 0 (AUC), 0 (log-loss), 2.7 × 10⁻¹⁵ (cost) |
| 5 | Reference A contains no optimizer output | true by construction from `reference_sample.npz` |
| 6 | Reference B recomputed for all algorithms | yes; contributions NSGA-II 8–29%, NBI-C 8–23%, core 43–75% |
| 7 | All 120 paired comparisons use matching partitions | every set has replications 0–29 on every dataset |
| 8 | Support cost post hoc only | `cost_support` absent from the optimized objective vector |
| 9 | Runtime reported separately from evaluation count | yes; ~2.2× wall clock at matched evaluations |
| 10 | No hyperparameter selection after seeing performance | config committed at `3e64804` before the first run |

## 9. What this changes, and what it does not

**Changes.** The frozen study's statement that NBI-C is "best or tied-best" holds only within the surrogate-derived
comparator set. Once a canonical evolutionary optimizer is run at a matched real-objective budget, it is better on
both endpoints on all four datasets. The manuscript must say so plainly. The spacing negative result also gains a
positive counterpart (§6).

**Does not change.** Every conclusion about the *surrogate pipeline* is untouched, because NSGA-II is not part of it:
surrogate reliability is dataset- and metric-dependent; anchor misplacement is the largest observed
surrogate-mediated failure; the reliability gate detects unusable surfaces but not misplaced anchors; the classical
synergism criterion is satisfied by surfaces whose real blends contradict it; the cost definition changes the winning
method. The paper's contribution is a diagnosis of when a surrogate-assisted pipeline can be trusted, and an external
optimizer that outperforms its metamodel-free arm does not weaken that diagnosis — it removes the reviewer's obvious
question and sharpens the practical recommendation.

## 10. Artifacts

- `nsga2_pareto_quality.csv` — 3,840 rows: 8 sets × 2 costs × 2 references × 4 datasets × 30 replications
- `nsga2_paired_effects.csv`, `nsga2_paired_tests.csv` — paired effects and Holm-corrected tests
- `nsga2_indicator_levels.csv` — median indicator levels per reference, cost, dataset and set
- `nsga2_budget_runtime.csv` — realized evaluation and wall-clock ratios against NBI-C
- `nsga2_runs.csv` — per-run seeds, budgets, realized counts, runtimes, front sizes
- `experiments/pco213_postwork_nsga2/` — raw populations and manifest (unversioned)
