### 6.6 The boundary dataset did not resolve

Spambase was separated from the primary inferential family before the confirmatory campaign.
Screening on one pre-campaign pilot partition placed its non-dominated design-row set at two
points with undefined curvature, insufficient interior geometry for a front-construction
contrast to act on, and the protocol recorded before the campaign that a null result there
would be neither evidence for nor against the geometry mechanism. Spambase was retained as a
boundary geometry control by a pre-campaign amendment adopted after this failure was known
(amendment 22): the replacement rule frozen at protocol v1, before the pilot, required a
dataset failing criterion 1 or 2 to be replaced from the registry, and retaining Spambase
departs from that rule (Section 5.1). It was executed in full, with the same arms, budgets,
seeds, evaluation machinery and $R = 30$ replicated outer partitions, and excluded only
from the primary family.

On the geometry contrast the boundary dataset did not resolve. The median paired
difference in CORE-relative hypervolume ratio from WS-S to NBI-S is +0.0500, which
is numerically larger than MAGIC's Holm-significant +0.0374, with 18 wins, 2 ties and 10
losses over 30 replications, a percentile bootstrap interval of $[-0.0043, +0.4369]$
spanning zero, and Holm $p = 0.1175$. What distinguishes the two datasets is dispersion,
not location: the per-replication standard deviation is 1.7706 on Spambase against
0.0594 on MAGIC, roughly thirty times larger, over a range $-5.2874$ to $+5.7457$
(Figure 3).

The interpretation is therefore narrow. This is a failure to resolve the contrast at
$R = 30$, not a demonstration that the effect is absent. It is consistent with the
boundary designation stated before the campaign and does not confirm the boundary
mechanism: we neither read the non-significance as evidence that no geometry effect
exists on Spambase, nor treat it as corroboration of the reasoning behind the
designation. An earlier framing by the present authors asserted the stronger reading and
was refuted before publication by a post-campaign recomputation from the raw unit
artifacts (Section 8.6).

Three qualifications belong with it. First, the premise is less clean in execution than at
screening: the two-point front is a pre-campaign design-row measurement, whereas
across the 30 confirmatory replications the design-row front has median size 3, range 2
to 5, and is exactly 2 in 9 of them and three or more in the other 21. The degeneracy that
motivated the designation is not a fixed property of every replication. The three primary
datasets never showed it: their design-row fronts had at least three points in every
confirmatory replication (median 9, 6 and 5; minimum 6, 4 and 3 on MAGIC, Adult and Bank
Marketing; Supplement S5). Second, Spambase is not globally null: its
anchor-provenance contrast (Section 6.4) is strongly negative and Holm-significant at median
$-0.6932$, the largest such deficit in the panel, and its historical reconstruction gap
is also the panel's largest. Third, all of these results are measured in Spambase's frozen
objective space, whose quality composite agrees only weakly with equal-weight raw quality on
its fitting rows and is negatively rank-correlated with it over configurations drawn
uniformly from the decision box (post hoc; Supplement S4.3 and Section 8.5), so the Spambase
result cannot be read as evidence about front geometry.

The dispersion should be read with the scale of the endpoint (Section 6.1): the
CORE-relative hypervolume ratio is not bounded above by one, exceeds 1 in 35% of
Spambase method-unit rows, and reaches the campaign's overall maximum of 9.912 there,
against 2.349 outside Spambase. The large excursions concentrate on the dataset whose
finite reference set is least informative about the trade-off surface.

### 6.7 The geometry advantage persisted where the surrogate failed its own gate

The external reliability gate is an audit-only diagnostic. It scores the fitted
surrogates on a 78-point set (the design's complementary half fraction plus 14 axial
runs at half radius) that no fitting stage can reach, passing when external $R^2 \ge
0.5$ and Spearman $\rho \ge 0.9$ for the objective. It is diagnostic, not adaptive: every
arm ran at every replication whatever the gate said, so its pass rate is a result rather
than a filter, and the primary comparisons are additionally reported conditioned on gate
status (Supplement S11).

Over the confirmatory campaign the composite-quality gate passed in 8 of 30
replications on MAGIC and 9 of 30 on Spambase, and in 0 of 30 on both Adult and
Bank Marketing. The cost gate, on the complexity objective that leaf count dominates, passed
30/30, 30/30, 22/30 and 27/30 respectively (Supplement S11). Every failure, on either
objective, was on the rank criterion alone: external $R^2$ ranged from 0.574 to 0.962 and
met its cut-off in all 240 unit-by-objective cells, while the composite-quality Spearman
$\rho$ had medians of 0.854, 0.891, 0.867 and 0.855 on MAGIC, Spambase, Adult and Bank
Marketing.

Adult and Bank Marketing are consequently regimes in which the externally audited
composite-quality surrogate failed the frozen criterion in every replication, and they
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

On the CORE-relative hypervolume ratio, the coarse grid, given 386 real evaluations against
NBI-S's 186 under the frozen maximum-arm comparator rule, had the higher paired median on all
four datasets; its lead met the frozen 21-of-30 direction rule (Section 5.15) on MAGIC and
Adult only, its bootstrap interval also excluded zero on Bank Marketing (20/30), and it was
unresolved on Spambase (18/30). The grid's marginal median is the higher of the two on MAGIC,
Adult and Bank Marketing, but not on Spambase, where NBI-S is higher at 1.0812 against
1.0192. The paired statistic is the within-replication comparison; the marginal medians are
not, and the two do not have to agree (Table 5, Figure 6). Against NBI-S:

| dataset | NBI-S median | GRID median | median paired difference (NBI-S − GRID) | NBI-S wins | raw $p$ |
|---|---:|---:|---:|---:|---:|
| MAGIC | 0.9430 | 1.0068 | −0.0628 | 0/30 | 1.86e−09 |
| Adult | 0.9951 | 1.0476 | −0.0488 | 5/30 | 1.37e−04 |
| Bank Marketing | 0.9979 | 1.0625 | −0.1323 | 10/30 | 0.1642 |
| Spambase | 1.0812 | 1.0192 | −0.1286 | 12/30 | 0.8774 |

**Table 5.** NBI-S against the coarse grid: median CORE-relative hypervolume ratio of each method, median paired difference (NBI-S − GRID), NBI-S wins out of 30 replications, and the raw $p$ of a post hoc, unadjusted two-sided Wilcoxon signed-rank test (descriptive; outside the frozen primary family). Direct-search comparators operate under the frozen comparator budget of 386 evaluations; WS-S and NBI-S cost 186 standalone.

The raw $p$ is below 0.05 on MAGIC and Adult and not on Bank Marketing or Spambase. This
contrast is not in the frozen primary family, carries no Holm correction, and is
reported descriptively.

The budgets were not equal. The frozen comparator rule matched comparators to the *most
expensive arm*, not to each arm: the comparator budget is 386 real evaluations, NBI-R's
cost, while WS-S and NBI-S cost 186 standalone, so the comparators received roughly twice
the real evaluations those two arms require (Supplement S7.1). The rule dates from protocol
v1 (2026-09-13) and its value of 386 from protocol v2, adopted the same day after the Stage A
pilot revised two protocol parameters; neither is revised now. The grid result therefore
cannot be stated as an efficiency claim, nor can the asymmetry be used to dismiss it.

The other direct-search comparators, under the same frozen budget, sit below the grid: median
CORE-relative hypervolume ratios of 0.8907 (random) and 0.9580 (NSGA-II) on MAGIC, 0.4369 and
0.8650 on Adult, 0.3190 and 0.9674 on Bank Marketing, and 0.0000 and 0.4164 on Spambase.
Both also sit below NBI-S's marginal median on every dataset except NSGA-II on MAGIC
(Table 5). On the paired statistic only the grid had the higher median against NBI-S on all
four datasets, so that lead belongs to the grid, not to direct search as a class; against
random search NBI-S had the higher paired median on all four datasets, every bootstrap
interval excluding zero. Matched NSGA-II ran at $32 \times 12 = 384$ evaluations, two short
of the comparator budget, a shortfall stated rather than absorbed. The unmatched NSGA-II run,
ten times the matched budget on one replication per dataset, is reported as context only
(Supplement S13.2): no fairness claim attaches to it, it enters no budget-matched
comparison, and it contributes to neither reference set. The frozen protocol excludes the
single-objective Bayesian-optimization and TPE runs from the front-indicator table, because
each optimizes one objective rather than the trade-off; their single-objective endpoints are
reported in Supplement S13.1.

The grid's lead is also qualified by its construction: its 128-point mesh contains the
design's 64 factorial corners, so 64 of its evaluations re-measure points inside the
CORE reference it is scored against, an advantage on a CORE-relative indicator that the
surrogate-assisted arms do not have by construction, since their candidates come from
surrogate subproblems and coincide with a design corner, if at all, only in some units.
Like the budget rule, this construction was fixed before the confirmatory campaign, and
neither is offered as a reason to set the comparison aside: at the budget and construction
this protocol froze, the coarse grid had the higher paired median against NBI-S on all four
datasets.

### 6.9 Holdout confirmation

Each unit re-measured one candidate per arm: the revalidated candidate nearest, after min–max
normalization of the two objectives, to the utopia point of that arm's own revalidated set.
Each selected configuration was refitted on the replication's full training partition and
scored on a held-out partition whose labels were unavailable within the replication until the
confirmation stage (on its overlap with the pre-campaign pilot data, see Supplement S4.1), at
5 audit-only real evaluations per unit, charged to the study and to no arm. Across all four
datasets and all five arms, the median per-replication difference between internal and
holdout accuracy has magnitude below 0.012 in every one of the 20 dataset-by-arm cells, the
largest being 0.0117, and is negative (the held-out partition scoring better than the
internal resampling) in 13 of the 20 cells (Supplement S14). These are descriptive audit
figures read from `analysis/secondary_analysis.json`, not confirmatory claims.

The correct reading is an absence of gross optimism in the returned configurations at
this budget, and nothing more. The stage measures internal-versus-holdout optimism and was
never designed to rank arms, so the ordering of these differences is not interpreted and
a smaller drop is not evidence that an arm is better. Every primary claim in this study
is about the returned front under the frozen conditions, not about the
generalization of a single deployed configuration.

### 6.10 Declared analyses computed after the campaign

Seven analyses that the frozen protocol declared were computed after the campaign from stored
outputs, without new evaluations; Section 8.7 gives their status and the post hoc choices each
required, and none of them enters the primary family. Under the pre-declared equal weighting of the
two quality factors, the returned non-dominated sets differed from the variance-weighted ones in 9
to 23 of 30 replications, more than the 20% threshold of the frozen protocol on every dataset and
for every arm, so the conclusions are conditional on that weighting; every identifying contrast
kept its direction (Supplement S4.5). When measured mean per-fold time replaced the leaf-count cost
axis, the winning arm changed in 9, 17, 16 and 14 of 30 replications on MAGIC, Spambase, Adult and
Bank Marketing; the geometry contrast stayed positive on every dataset, though smaller on Adult and
Bank Marketing, and the anchor-provenance contrast changed sign on Adult (median $+0.0696$, NBI-R
ahead in 18 of 30), where under the leaf-count cost it was a non-detection (Supplement S4.7). Over
all 120 units, raw total leaf count tracked measured time at a median per-unit Spearman correlation
of 0.87 to 0.96 (Supplement S4.6). Under the post hoc optimal matching, factor models refitted on
each replication's own design rows matched the frozen models at a Tucker congruence of at least
0.895 on MAGIC, Spambase and Adult and of 0.751 at worst on Bank Marketing, where 10 of 30
replications fell below the conventional equivalence level of 0.95; the greedy matching recorded by
the runner, which does not resolve factor sign, gives lower values (Supplement S4.4). The
single-objective Bayesian-optimization and TPE endpoints and the unmatched NSGA-II run are context
only and enter no comparison (Supplement S13).
