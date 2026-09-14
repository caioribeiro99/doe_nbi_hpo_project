# Two objectives or three: the scientific question, stated before the compute question

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
them into a weighted mean does not summarize a trade-off; it **suppresses** one. On Spambase the
suppressed axis is precision against specificity, which is the classical threshold trade-off and the
thing a practitioner deploying a spam filter actually cares about.

**2. The aggregation discards a large amount of Pareto structure.** Non-dominated set sizes on the
88 design rows, two objectives against three:

| Dataset | two objectives | three objectives | growth |
|---|---|---|---|
| MAGIC | 21 | 42 | 2.0× |
| Spambase | 6 | **65** | **10.8×** |
| Adult | 6 | 37 | 6.2× |
| Bank Marketing | 10 | 13 | 1.3× |

On Spambase the two-objective formulation declares 6 of 88 design points non-dominated where the
three-objective formulation finds 65. Those 59 points are not noise; they are configurations that
are best-in-class on the suppressed axis.

**3. It eliminates the protocol's most awkward free parameter.** The aggregation weighting is the
subject of `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4, which measured variance weighting and equal
weighting ranking the design at Spearman 0.374 with different best rows, and of amendment 3, where
the orientation rule for the aggregated composite had to be corrected after it inverted on three
datasets. **At three objectives there is no aggregation, so neither problem exists.** The
pre-registered sensitivity analysis of §11.1 also becomes unnecessary.

**4. It is where NBI is supposed to earn its keep.** At two objectives the convex hull of individual
minima is a line segment and the quasi-normal is a single direction. At three it is a simplex, and
uniform spread over a simplex is the property Das and Dennis advanced NBI to deliver and the property
weighted-sum scalarization most visibly lacks. Review finding M5 already said this. Running the
geometry contrast at the dimensionality where the geometry barely differs is the conservative choice
for the paper's hypothesis, and a reviewer is entitled to ask why it was made.

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

Recorded in `STAGE_B_THROUGHPUT.md` once the measurement exists, together with the branch taken and
the projection that justified it.
