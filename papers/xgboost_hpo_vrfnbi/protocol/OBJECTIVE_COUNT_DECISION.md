# Two objectives or three: the scientific question, stated before the compute question

> **OUTCOME: the amendment was REFUSED.** See `Q3_AMENDMENT_REVIEW.md`. The confirmatory campaign
> runs at two objectives. This document is retained unedited in structure, with its two factual
> errors corrected in place and marked, because it is the record of what was argued **before** the
> adversarial review and before the per-objective gate was measured. Every figure below is
> regenerable by `../scripts/objective_count_evidence.py`.

**Why this document exists.** Protocol v2 runs the campaign at two objectives, and the stated reason
was cost: 4.6 days serial against 5.8. A serial projection is not a sufficient reason to omit a
load-bearing objective, and the decision is being taken on a machine whose parallel throughput had
not been measured. This settles the scientific part first, in writing, before Stage B's numbers are
allowed to touch it.

---

## A. The two objectives frozen in protocol v2

| # | Objective | Composition |
|---|---|---|
| 1 | **quality composite** | the variance-weighted mean of the z-scored scores of the **two non-cost Varimax factors**, over six responses: accuracy, precision, recall, specificity, area under the receiver operating characteristic curve, and log loss |
| 2 | **cost** | the Varimax factor dominated by `log1p(total leaf count)` |

Both canonicalized to minimization.

## B. The third objective that was excluded

**The second quality factor, kept separate instead of aggregated into objective 1.**

This is the precise statement, and it matters: the third objective is not a new response, a new
metric, or a new measurement. Every response is already evaluated. The factor stage already extracts
**three** components and already identifies which one is cost. The two-objective formulation then
takes the remaining two factors and **collapses them into one number** by a weighted mean. The
three-objective formulation simply does not collapse them.

So the cost of the third objective is not data collection. It is the extra weight vectors, extra
subproblems, extra anchor searches and extra candidate revalidations that a three-dimensional
simplex requires.

Measured on the Stage A designs, the two quality factors and their dominant responses:

| Dataset | leading quality factor | second quality factor | variance shares |
|---|---|---|---|
| MAGIC | area under the curve | precision | 0.610 / 0.254 |
| Spambase | specificity | precision | 0.695 / 0.135 |
| Adult | specificity | log loss | 0.755 / 0.111 |
| Bank Marketing | specificity | log loss | 0.759 / 0.139 |

## C. Why the third objective exists scientifically

Three reasons, in descending strength.

**1. The two quality factors are not redundant, and on two datasets they conflict.**

| Dataset | Spearman between the two quality factors |
|---|---|
| MAGIC | **−0.308** |
| Spambase | **−0.533** |
| Adult | +0.649 |
| Bank Marketing | +0.759 |

A negative correlation means the two factors rank configurations in opposing directions. Collapsing
them into a weighted mean does not summarize a trade-off; it **suppresses** one.

> **CORRECTION.** This paragraph originally continued: "On Spambase the suppressed axis is precision
> against specificity, which is the classical threshold trade-off." That is **wrong**. On the
> canonicalized Spambase design, precision and specificity correlate at **+0.863** — they agree. The
> −0.533 is a property of the rotation, not of those two metrics. The correct statement is that the
> second factor is dominated by precision and is close to orthogonal to overall quality there
> (ρ = +0.025 with the mean of the six standardized quality responses); what it measures is not
> identified by a named metric pair.

> **REGISTER.** With bootstrap intervals the correlations are MAGIC −0.308 [−0.535, −0.053],
> Spambase −0.533 [−0.719, −0.303], Adult +0.649 [+0.443, +0.812], Bank Marketing +0.759
> [+0.586, +0.871]. The defensible claim is that redundancy was **not measured**, not that conflict
> was established.

**2. The aggregation discards Pareto structure — but the raw growth figure is not the evidence.**

> **CORRECTION.** This section originally reported raw growth factors with no null. Adding *any*
> third coordinate, including a pure-noise one, weakly enlarges a non-dominated set — roughly 2× at
> n = 88. The figure is claimable only as an excess over a null. Permuting the third coordinate
> against the first two, 400 draws:

| Dataset | two objectives | three objectives | raw growth | permutation null mean | p | exceeds null |
|---|---:|---:|---:|---:|---:|:--:|
| MAGIC | 21 | 42 | 2.0× | 21.8 | 0.000 | yes |
| Spambase | 6 | 65 | 10.8× | 16.2 | 0.000 | yes |
| Adult | 6 | 37 | 6.2× | 18.1 | 0.000 | yes |
| Bank Marketing | 10 | 13 | 1.3× | **22.7** | **1.000** | **no** |

The null cuts both ways and both readings belong here. On three datasets it **strengthens** the
case, because the excess on Spambase is 65 against 16, not 65 against 6. On Bank Marketing it
**reverses** it: the real third objective produces *fewer* non-dominated points than a random axis,
so on that dataset the evidence runs against the amendment.

**3. It would remove a free parameter — but a much smaller one than stated here originally.**

> **CORRECTION.** This section originally cited Spearman **0.374** for variance-weighted against
> equal-weighted aggregation. That figure comes from `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4 and
> is a measurement of the **dissertation's** pipeline. Under the protocol's own factor stage the
> agreement is **0.865 / 0.874 / 0.884 / 0.964** on MAGIC / Spambase / Adult / Bank Marketing. The
> free parameter is roughly four times less awkward than the argument claimed, and this is the
> amendment's **weakest** argument once corrected, not its strongest.

What remains true: the orientation rule for the aggregated composite had to be corrected in
amendment 3 after it inverted on three datasets, and at three objectives there is no aggregation to
orient. That is a real but modest benefit.

**4. It is where NBI is supposed to earn its keep.** At two objectives the convex hull of individual
minima is a line segment and the quasi-normal is a single direction. At three it is a simplex, and
uniform spread over a simplex is the property Das and Dennis advanced NBI to deliver and the property
weighted-sum scalarization most visibly lacks. Review finding M5 already said this. Running the
geometry contrast at the dimensionality where the geometry barely differs is the conservative choice
for the paper's hypothesis, and a reviewer is entitled to ask why it was made.

**5. A constraint this section originally missed.** `run_historical_ws` reproduces the dissertation by
calling the frozen `run_nbi_weighted_sum`, whose signature takes **two** models and a two-component
weight grid, and `method_arms.md` forbids reimplementing it. **HISTORICAL-WS and HISTORICAL-WS-asrun
cannot exist at three objectives**, so one of the three primary identifying contrasts would
disappear and the composite with its weighting would survive inside that arm regardless.

## D. Classification

**Complementary, and arguably central.**

Not *redundant*: the factors conflict on two of four datasets and the non-dominated set grows by up
to a factor of ten.

Not *exploratory*: it needs no new measurement and no new method, and it removes two known protocol
weaknesses rather than adding an open question.

Not unambiguously *central*, because the paper's two primary contrasts — geometry and anchor
provenance — are well posed at two objectives and would be answerable there. The third objective
makes both contrasts sharper and makes the answer more general; it does not create them.

## E. The decision rule, fixed before Stage B's numbers are read

Taken verbatim from the instruction, and binding:

1. If the third objective were scientifically redundant or not load-bearing, keep two objectives
   regardless of available compute. **Section C establishes it is neither**, so this branch does not
   apply.
2. If it is scientifically important **and** the measured parallel wall-clock projection is
   comfortably at or under five days, it is **not** silently added. A short adversarial protocol
   review runs, the amendment is documented, and `xgboost-hpo-protocol-v3` is cut **before any
   confirmatory method comparison is run**.
3. If it is scientifically important but the projection stays above five days after reasonable
   parallelization, the confirmatory study keeps two objectives and the excluded objective is
   documented as future work with this analysis attached.

"Comfortably at or under five days" is fixed here as **at or under four days**, leaving a fifth of
the ceiling as margin for the campaign's own variance, before any throughput number is consulted.

Nothing in this decision may reference which formulation produces better-looking Pareto fronts, which
arm wins, or any method comparison. Scientific necessity first, compute second, observed method
performance never.

## F. Outcome

**Refused.** Stage B measured 0.1585 s per real evaluation, so both objective counts fit far inside
the ceiling (0.52 and 0.69 days) and **compute was never the binding constraint**. Branch 2 was
therefore entered and the adversarial review ran, as the rule required.

The review found what this section had not measured: the proposed third objective fails the
protocol's own surrogate reliability gate on two of four datasets, and the two quality axes exchange
roles across the panel so that "objective 2" does not name the same quantity on any two datasets.
Full decision and reasoning in `Q3_AMENDMENT_REVIEW.md`; regenerable evidence in
`../audits/objective_count_evidence.json`.

A sixth branch is added to §E, retrospectively and explicitly marked as such, because the rule as
written had no branch for the situation that actually arose:

> **Branch 5 (added after the review).** Load-bearing but **not surrogate-representable**: if the
> candidate objective fails the §7 reliability gate on any panel dataset, it is not promoted to an
> optimization target, because the surrogate-driven arms would steer on a coordinate they cannot
> see. Compute does not enter this branch.

The branch is written down here rather than applied silently, and it is recorded as amendment 14 in
`../PROTOCOL_AMENDMENTS.md` with the acknowledgement that it postdates the situation it governs.
