### 6.6 The prospectively retained boundary dataset did not resolve

Spambase was separated from the primary inferential family before any result existed.
Screening placed its non-dominated design-row set at two points with undefined
curvature — insufficient interior geometry for a front-construction contrast to act on —
and the protocol recorded in advance that a null result there would be neither evidence
for nor against the geometry mechanism. The replacement rule was considered and
deliberately not exercised: Spambase was executed in full, with the same arms, budgets,
seeds, evaluation machinery and $R = 30$ replicated outer partitions, and excluded only
from the primary family.

On the Claim 2 contrast the boundary dataset did not resolve. The median paired
difference in CORE-relative hypervolume ratio from WS-S to NBI-S is **+0.0500**, which
is numerically **larger** than MAGIC's significant +0.0374, with 18 wins, 2 ties and 10
losses over 30 replications, a percentile bootstrap interval of $[-0.0043, +0.4369]$
spanning zero, and Holm $p = 0.1175$. What distinguishes the two datasets is dispersion,
not location: the per-replication standard deviation is **1.7706** on Spambase against
**0.0594** on MAGIC, roughly thirty times larger, over a range $-5.2874$ to $+5.7457$
(Figure 3).

The interpretation is therefore narrow. This is a failure to resolve the contrast at
$R = 30$, not a demonstration that the effect is absent. It is **consistent with** the
prospectively stated boundary designation and does **not** confirm the boundary
mechanism: we neither read the non-significance as evidence that no geometry effect
exists on Spambase, nor treat it as corroboration of the reasoning behind the
designation. An earlier framing by the present authors asserted the stronger reading and
was refuted by an independent verifier before publication.

Two qualifications belong with it. First, the premise is less clean in execution than at
screening: the two-point front is a **pre-campaign** design-row measurement, whereas
across the 30 confirmatory replications the design-row front has median size 3, range 2
to 5, and is exactly 2 in 9 of them. The degeneracy that motivated the designation is
not a fixed property of every replication. Second, Spambase is not globally null: its
anchor-provenance contrast (Section 6.4) is strongly negative and significant at median
$-0.6932$, the largest such deficit in the panel, and its historical reconstruction gap
is also the panel's largest.

The dispersion should be read with the scale of the endpoint (Section 6.1): the
CORE-relative hypervolume ratio is not bounded above by one, exceeds 1 in 35% of
Spambase method-unit rows, and reaches the campaign's overall maximum of 9.912 there,
against 2.349 outside Spambase. The large excursions concentrate on the dataset whose
finite reference set is least informative about the trade-off surface.

### 6.7 The geometry advantage persisted where the surrogate failed its own gate

The external reliability gate is an audit-only diagnostic. It scores the fitted
surrogates on a 78-point set — the design's complementary half fraction plus 14 axial
runs at half radius — that no fitting stage can reach, passing when external $R^2 \ge
0.5$ and Spearman $\ge 0.9$ for the response. It is **diagnostic, not adaptive**: every
arm ran at every replication whatever the gate said, so its pass rate is a result rather
than a filter, and the primary comparisons are additionally reported conditioned on gate
status.

Over the confirmatory campaign the composite-quality gate passed in **8 of 30**
replications on MAGIC and **9 of 30** on Spambase, and in **0 of 30** on both Adult and
Bank Marketing. The cost gate passed 30/30, 30/30, 22/30 and 27/30 respectively
(Supplement S11).

Adult and Bank Marketing are consequently regimes in which the externally audited
composite-quality surrogate failed the frozen criterion in every replication — and they
also carry the largest geometry effects, +0.2549 and +0.2281, Holm-significant on both.
The relative advantage of NBI front construction over specification-matched weighted
scalarization therefore persisted where the surrogate did not meet its own external
standard. Within MAGIC, the only primary-panel dataset with both strata populated (Spambase, the boundary control, also has both, with 9 of 30 passing), the geometry
difference is positive in each: median +0.1173 with a win fraction of 1.00 across the 8
gate-passing units, and +0.0165 with 0.82 across the 22 failing units. Both strata are
small and are reported descriptively, without a test.

This says nothing about absolute surrogate trustworthiness. Whether a surrogate is
accurate enough to be believed, and whether one front-construction geometry beats
another on the same surrogate, are different questions; the study answers only the
second. We do not claim that the geometry effect requires a reliable surrogate, requires
an unreliable one, or that NBI compensates for surrogate error.

### 6.8 Baselines: the frozen-budget direct grid comparison

At the frozen comparator budget, **the median paired difference favoured the direct grid
baseline over NBI-S on all four datasets**. Its marginal median CORE-relative
hypervolume ratio is the higher of the two on MAGIC, Adult and Bank Marketing, but
**not on Spambase**, where NBI-S is higher at 1.0812 against 1.0192. The paired
statistic is the within-replication comparison; the marginal medians are not, and the
two do not have to agree (Table 5, Figure 6). Against NBI-S:

| dataset | NBI-S median | GRID median | median difference (NBI-S − GRID) | NBI-S wins | raw $p$ |
|---|---:|---:|---:|---:|---:|
| MAGIC | 0.9430 | 1.0068 | −0.0628 | 0/30 | 0.0000 |
| Adult | 0.9951 | 1.0476 | −0.0488 | 5/30 | 0.0001 |
| Bank Marketing | 0.9979 | 1.0625 | −0.1323 | 10/30 | 0.1642 |
| Spambase | 1.0812 | 1.0192 | −0.1286 | 12/30 | 0.8774 |

**Table 5.** Median CORE-relative hypervolume ratio by method. Direct-search comparators operate under the frozen comparator budget of 386 evaluations; WS-S and NBI-S cost 186 standalone.

The gap is significant on MAGIC and Adult and not on Bank Marketing or Spambase. This
contrast is not in the frozen primary family, carries no Holm correction, and is
reported descriptively.

**The budget asymmetry belongs in the same breath.** The frozen fairness rule matched
comparators to the *most expensive arm*, not to each arm: the comparator budget is 386
real evaluations, NBI-R's cost, while WS-S and NBI-S cost 186 standalone, so the
comparators received roughly twice the real evaluations those two arms require. The rule
was fixed before any result existed and is not revised now. The grid result therefore
cannot be stated as an efficiency claim, the budgets not being equal — and equally
cannot be explained away on that basis: at the budget the protocol froze, direct search
led. Any account of the geometry result in Section 6.3 must carry this alongside it.

The other direct-search comparators, under the same frozen budget, sit below the grid: median CORE-relative
hypervolume ratios of 0.8907 (random) and 0.9580 (NSGA-II) on MAGIC, 0.4369 and 0.8650
on Adult, 0.3190 and 0.9674 on Bank Marketing, and 0.0000 and 0.4164 on Spambase.
Matched NSGA-II ran at $32 \times 12 = 384$ evaluations, two short of the comparator
budget, a shortfall stated rather than absorbed. The **unmatched** NSGA-II run — ten
times the matched budget, one replication per dataset — is reported as **context only**:
no fairness claim attaches to it, it enters no budget-matched comparison, and it
contributes to neither reference set. The frozen protocol excludes the single-objective
Bayesian-optimization and TPE endpoints from the front-indicator table, a
single-objective optimizer not returning a front.

Two properties of the comparator qualify this. The budget rule matches comparators to
the most expensive arm rather than to each arm, so the grid received 386 real
evaluations against the 186 WS-S and NBI-S require standalone. And the grid's 128-point
mesh contains the design's 64 factorial corners, so 64 of its evaluations re-measure
points inside the CORE reference it is scored against — an advantage on a CORE-relative
indicator that the surrogate-assisted arms, which propose interior points, do not have.
Neither fact was chosen after seeing the result, and neither is offered as a reason to
set the comparison aside: at the budget and construction this protocol froze, direct
search led on the paired statistic across all four datasets.

### 6.9 Holdout confirmation

Each unit re-measured its selected candidates on a held-out partition whose labels were
structurally unavailable until the confirmation stage, at 5 audit-only real evaluations
per unit, charged to the study and to no arm. Across all four datasets and all five
arms, the **median per-replication difference** between internal and holdout accuracy has
magnitude below **0.012** in every one of the 20 dataset-by-arm cells, the largest
being **0.0117**, and is negative — the held-out partition scoring better than the
internal resampling — in **13 of the 20** cells (Supplement S14). These are descriptive audit figures read from
`analysis/secondary_analysis.json`, not confirmatory claims.

The correct reading is an absence of gross optimism in the returned configurations at
this budget — nothing more. The stage measures internal-versus-holdout optimism and was
never designed to rank arms, so the ordering of these differences is not interpreted and
a smaller drop is not evidence that an arm is better. Every primary claim in this study
is about the **returned front** under the frozen conditions, not about the
generalization of a single deployed configuration.
