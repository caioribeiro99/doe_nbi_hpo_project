# The third objective: considered, evidenced, and rejected before the confirmatory campaign

**Decision: REJECTED before any confirmatory arm-level result was observed.** The confirmatory study
is two-objective. This decision is final for the campaign and is not reopened on the strength of any
campaign result.

**What this decision is not.** It is not a finding that the third objective is bad, uninformative or
irrelevant. The evidence below shows it carries real structure on three of four datasets. The
finding is narrower and more specific:

> That structure is not represented reliably and consistently enough by the **common surrogate
> architecture** to serve as a **cross-dataset confirmatory objective**.

Three of the four arms optimize a response surface, not the truth. An objective the shared surrogate
cannot represent on part of the panel, and whose meaning changes between datasets, cannot anchor a
comparison whose whole purpose is that the arms differ in one controlled respect.

Every figure below regenerates from `scripts/objective_count_evidence.py` into
`audits/objective_count_evidence.json`. No new real evaluations were used to produce any of it.

---

## 1. Why a third objective was considered

The factor stage already extracts **three** components. The two-objective formulation identifies the
one dominated by leaf count as the cost objective and then **collapses the remaining two into a
single quality composite** by a variance-weighted mean of their z-scored scores.

The third objective is therefore not a new measurement, a new metric or a new experiment. It is the
decision **not to collapse**. Every response is already evaluated; the cost is only the additional
weight vectors, subproblems, anchor searches and candidate revalidations that a three-dimensional
simplex requires.

Adversarial review finding M5 of the v2 protocol review had already flagged this: at two objectives
the convex hull of individual minima is a line segment and the quasi-normal is a single direction,
which is the dimensionality at which NBI and weighted-sum scalarization differ least.

## 2. Evidence that the third objective carries real information

### 2.1 The two quality axes are not redundant, and on two datasets they conflict

Spearman between the two quality factors on the 88 design rows, with percentile bootstrap intervals
from 2,000 resamples:

| Dataset | Spearman | 95% interval |
|---|---:|---|
| MAGIC | **−0.308** | [−0.535, −0.053] |
| Spambase | **−0.533** | [−0.719, −0.303] |
| Adult | +0.649 | [+0.443, +0.812] |
| Bank Marketing | +0.759 | [+0.586, +0.871] |

Two are negative with intervals excluding zero: the axes rank configurations in opposing directions
on MAGIC and Spambase. The correct register is that **redundancy was not measured**, not that
conflict was established everywhere.

### 2.2 Real non-dominated structure, referenced to a null

Adding any third coordinate weakly enlarges a non-dominated set, so a raw growth factor is not
evidence. Permuting the third coordinate against the first two, 400 draws, preserves its marginal
and destroys only its association:

| Dataset | two objectives | three objectives | raw growth | null mean | p | exceeds null |
|---|---:|---:|---:|---:|---:|:--:|
| MAGIC | 21 | 42 | 2.0× | 21.8 | 0.000 | **yes** |
| Spambase | 6 | **65** | 10.8× | 16.2 | 0.000 | **yes** |
| Adult | 6 | 37 | 6.2× | 18.1 | 0.000 | **yes** |
| Bank Marketing | 10 | 13 | 1.3× | **22.7** | **1.000** | **no** |

The null cuts both ways, and both readings belong here. On three datasets it **strengthens** the
case: the excess on Spambase is 65 against 16, not 65 against 6. On Bank Marketing it **reverses**
it — the real third objective yields *fewer* non-dominated points than a random axis.

These counts are measured on **real evaluations of the design**, not on surrogate predictions. The
structure is there in the data.

### 2.3 What each axis is on each dataset

Correlation of each quality factor with the unweighted mean of the six standardized canonicalized
quality responses, with its dominant loading:

| Dataset | quality 1 | | quality 2 | |
|---|---:|---|---:|---|
| MAGIC | **+0.944** | ROC-AUC | −0.021 | precision |
| Spambase | +0.637 | specificity | +0.025 | precision |
| Adult | +0.614 | specificity | **+0.993** | log loss |
| Bank Marketing | +0.779 | specificity | **+0.991** | log loss |

## 3. Compute was not the blocker

The original two-objective choice in protocol v2 was justified by a **serial** projection: 4.6 days
against 5.8, with a 5-day ceiling. Stage B retired that premise entirely.

| | two objectives | three objectives |
|---|---:|---:|
| campaign evaluations | 285,120 | 357,120 |
| projected wall clock at the measured 0.1585 s per evaluation | **0.52 days** | **0.66 days** |
| including the unmatched NSGA-II run | 0.55 days | 0.69 days |

Against a ceiling of 5 days and a threshold for "comfortably under" fixed at 4 days **before any
throughput number was read**, both fit with an order of magnitude to spare. Had the decision been
about compute, the third objective would have been adopted.

## 4. External surrogate validation of the proposed third objective

**This is the measurement that decided it, and it had never been taken.** Every gate number in
Stage A is for the *aggregated* composite. The protocol's own surrogate — quadratic response
surface, coded units, backward elimination at α = 0.05 with hierarchy enforced — fitted on the 88
design rows and scored on the 78-run complementary fraction, **per objective**:

| Dataset | objective | fitted order and size | external R² | Spearman | gate |
|---|---|---|---:|---:|:--:|
| MAGIC | quality 1 | quadratic, 19 terms after elimination | 0.870 | 0.927 | PASS |
| MAGIC | **quality 2** | quadratic, 16 terms | 0.896 | 0.951 | **PASS** |
| MAGIC | cost | quadratic, 18 terms | 0.917 | 0.940 | PASS |
| Spambase | quality 1 | quadratic, 14 terms | 0.969 | **0.861** | **FAIL** |
| Spambase | **quality 2** | quadratic, **5 terms** | **0.080** | **0.252** | **FAIL** |
| Spambase | cost | quadratic, 15 terms | 0.942 | 0.965 | PASS |
| Adult | quality 1 | quadratic, 8 terms | 0.905 | 0.912 | PASS |
| Adult | **quality 2** | quadratic, 13 terms | 0.841 | 0.910 | **PASS** |
| Adult | cost | quadratic, 17 terms | 0.934 | 0.955 | PASS |
| Bank Marketing | quality 1 | quadratic, 9 terms | 0.921 | 0.929 | PASS |
| Bank Marketing | **quality 2** | quadratic, 12 terms | 0.720 | **0.874** | **FAIL** |
| Bank Marketing | cost | quadratic, 17 terms | 0.942 | 0.958 | PASS |

Gate: external R² ≥ 0.5 **and** Spearman ≥ 0.9. **Three of twelve cells fail.** The proposed third
objective fails on **two of four** datasets.

Spambase is the sharp case. Backward elimination leaves **five terms** on a seven-factor quadratic,
and the surface explains 8% of held-out variance with rank correlation 0.252. That is the same
near-intercept failure mode amendment 2 diagnosed for untransformed leaf count. A separate fact:
Spambase's *leading* quality factor also fails, at Spearman 0.861, so **Spambase has no
gate-passing quality objective at three objectives**.

### 4.1 Why a gate failure is disqualifying for an objective but not for a dataset

The distinction matters and the protocol now states it in §7.1. A gate failure on an objective the
campaign *already* optimizes is a **finding**, reported and stratified, with every arm still run —
that is one of the study's scientific questions. A gate failure on an objective **being considered
for promotion** is a **disqualification**, because the choice is still open and making it anyway
would knowingly build three arms' anchors, payoff matrix, convex hull and quasi-normal direction out
of a surface that carries almost no signal.

The perversity is worth stating plainly: Spambase supplies the amendment's strongest evidence, 65
non-dominated points against a null of 16.2, and that structure is measured on real evaluations. It
is therefore **precisely the structure no surrogate-driven arm could reach**.

## 5. Semantic identity across datasets

From §2.3: on MAGIC and Spambase the **leading** axis is overall quality and the second is close to
orthogonal to it (ρ = −0.021 and +0.025). On Adult and Bank Marketing the **second** axis is overall
quality (ρ = +0.993 and +0.991) and the leading axis is a specificity contrast. Dominant loadings:
precision, precision, log loss, log loss.

"Objective 2" does not name the same quantity on any two datasets. A three-objective panel result
would compare a precision axis on MAGIC against a log-loss axis on Adult and call the comparison a
cross-dataset synthesis. The study's own synthesis question requires exactly that, and could not be
answered.

A related constraint from the arms: `run_historical_ws` reproduces the dissertation by calling the
frozen `run_nbi_weighted_sum`, whose signature takes two models and a two-component weight grid, and
reimplementing it is forbidden. **HISTORICAL-WS cannot exist at three objectives**, so one of the
three identifying contrasts would disappear.

## 6. Adversarial review verdict

Four independent roles: a multiobjective methodologist, a statistician, an automated-machine-learning
empiricist, and an editor with a research-integrity brief. Each read the documents, the code and the
committed artifacts.

| Role | Recommendation |
|---|---|
| Multiobjective methodologist | KEEP_Q2 |
| Automated-machine-learning empiricist | KEEP_Q2 |
| Statistician | ADOPT_WITH_CHANGES |
| Editor / integrity | ADOPT_WITH_CHANGES |
| Adjudication | ADOPT_WITH_CHANGES |

All four reached the same per-objective gate numbers and the same permutation-null reversal on Bank
Marketing independently. What separated them was disposition, not evidence.

The adjudication's case rested on one good observation: at the measured throughput both objective
counts fit **together** in about 1.3 days, so the either/or need not be taken. That dissolves the
HISTORICAL-WS problem and stops objective count becoming an unmeasured researcher degree of freedom.

It was refused anyway, on three grounds the run-both proposal does not answer:

1. **Running an arm on an objective it cannot see does not become acceptable by labelling the result
   secondary.** On Spambase three of four arms would steer on a five-term near-flat surface.
2. **The identity problem is not fixable by running more.** No amount of compute makes a precision
   axis and a log-loss axis synthesizable across a panel.
3. **It would add a large amount of unvalidated machinery for a secondary result.** Hypervolume is a
   Monte-Carlo estimate at three objectives with a front-dependent sampling box; the named IGD⁺ was
   computing plain IGD; the spread measure sorts by the first objective and does not traverse a
   surface; no weight lattice existed above two objectives; and nothing filtered dominated points,
   which NBI returns at three objectives and weighted sum does not.

Two of the proposal's own arguments were also wrong, and both errors were the author's: the
aggregation-weighting figure of 0.374 is from the **dissertation** pipeline, while the protocol's own
stage gives 0.865 to 0.964; and the claim that the suppressed axis on Spambase is precision against
specificity is false, since those correlate at **+0.863** there.

## 7. Final decision

**REJECTED BEFORE THE CONFIRMATORY CAMPAIGN.** Recorded as amendment 15 in `PROTOCOL_AMENDMENTS.md`
and settled in `protocol/EXPERIMENT_PROTOCOL.md` §14, which lists it among the decisions not reopened
on the strength of any campaign result.

The decision was taken with **no confirmatory arm-level result in existence**: no arm had been run,
no Pareto front produced, no method comparison made.

## 8. Why rejecting is conservative with respect to this paper's hypothesis

Stated plainly, because it is the part a sceptical reader should check.

NBI's advantage over weighted-sum scalarization is expected to be **larger** at three objectives,
where the convex hull of individual minima is a simplex rather than a line segment and uniform
spread is a genuinely harder property. The paper's first primary contrast, NBI-S against WS-S, is
therefore being run at the dimensionality where the difference it looks for is **smallest**.

Rejecting the amendment is the choice **less favourable** to the paper's own hypothesis. It is taken
because the evidence does not support the objective, not because the objective is inconvenient, and
the direction is recorded here so that a reviewer does not have to infer it.

## 9. What is preserved, and where it goes

The third objective is retained as a **supplementary and future-work finding**, with everything
measured:

- It passes the gate and clearly exceeds its permutation null on **MAGIC and Adult**.
- It fails the gate on Spambase and Bank Marketing, and on Bank Marketing it is below its own null.
- The construction a reviewer would actually want, reached independently by two of the four roles, is
  a **named** third objective — quality, calibration, cost — rather than a latent rotated component.
  That formulation dissolves the semantic-identity problem entirely, at the cost of departing from
  the NBI-VRF construction this paper exists to decompose. It is the right shape for the follow-up
  study.
- The supplementary material reports the per-objective gate table, the permutation nulls and the
  semantic-identity analysis, so that a reader can see what was considered and why it was set aside.
