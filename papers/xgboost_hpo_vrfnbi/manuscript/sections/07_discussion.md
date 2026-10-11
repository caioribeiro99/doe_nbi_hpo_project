## 7. Discussion

### 7.1 What the decomposition establishes

The campaign separates three mechanisms ordinarily bundled together in a
surrogate-assisted multiobjective hyperparameter pipeline: the objective specification and
its normalization reference, the geometry of front construction, and the provenance of the
anchor and payoff information that geometry consumes. The contribution is not a new
algorithm but a prospectively frozen mechanistic decomposition of an existing one.
Because that existing method is a weighted sum normalized by an observed-extrema box
(Section 3), the historical-to-canonical difference decomposes into a normalization row and
a geometry row.

### 7.2 Geometry moved the indicator

On the three primary datasets (those that passed the pre-campaign front-geometry screen on the
design rows, Section 5.1) and 30 replicated outer partitions, under the frozen study
conditions, canonical NBI front construction produced higher real-revalidated
CORE-relative hypervolume than specification-matched weighted scalarization: median paired
differences of $+0.0374$ on MAGIC, $+0.2549$ on Adult and $+0.2281$ on Bank Marketing,
favoured in 26/30, 30/30 and 24/30 replications with every bootstrap interval excluding zero,
rank-biserial $+0.92$, $+1.00$ and $+0.88$, Holm-significant within every primary dataset,
which is the generalization unit (Table 4).

The scope is part of the claim: two objectives, this surrogate architecture, this budget,
these three datasets. Nothing licenses "NBI is better than weighted sum" in general. The
endpoint is a CORE-relative hypervolume ratio against a finite method-independent set of
288 points, where a value above 1 is expected and means only that the returned set
improved on that finite reference.


The strength of the geometry evidence depends on which test is read. Under the
secondary Holm-corrected signed-rank test, the contrast is significant in all
three primary datasets. Under the Nadeau–Bengio correction declared in advance as a
sensitivity, reported unadjusted, it clears $0.05$ on Adult alone ($p = 0.0148$, against
0.0860 and 0.0764). We take the descriptive triple as
primary, as specified before the campaign ran, and report the corrected test because
it was promised, not because it agrees. A reader who weights the corrected test more
heavily than we do should read the geometry result as one clearly resolved dataset and
two suggestive ones.

### 7.3 Why hypervolume and IGD⁺ move while the non-dominated count does not

The advantage is specifically an improvement in convergence and coverage geometry as
measured by hypervolume and IGD⁺: median IGD⁺, NBI-S against WS-S, of 0.0511/0.0930 on
MAGIC, 0.0579/0.1688 on Adult and 0.0915/0.1948 on Bank Marketing. The joint non-dominated
fraction behaves differently: median differences of $+0.0208$, $+0.0714$ and $+0.0000$,
with no detectable difference under CORE on any of the four datasets including Spambase.

This is not a contradiction. Hypervolume and IGD⁺ are positional: they measure proximity
to the reference and dominated space, so moving points closer or redistributing them along
the trade-off changes both. The joint non-dominated fraction is a mutual-dominance count,
and at $q = 2$ points at different positions along one trade-off band are mutually
non-dominated by construction, so an arm can sit nearer the reference while almost none of
its points dominate the other arm's. The geometry result must not be read as NBI returning
more non-dominated solutions, which the joint non-dominated fraction does not show under
CORE on any dataset.

### 7.4 Anchor provenance: a negative result, not a caveat

Holding the NBI formulation fixed and replacing surrogate-derived payoff information with the
pre-specified empirical-real anchor procedure did not improve the confirmatory fronts.
Paired medians are $-0.0741$ on MAGIC, $-0.0192$ on Adult and $-0.4093$ on Bank Marketing,
Holm-significant on MAGIC and Bank Marketing; Adult is a non-detection under CORE with an
interval spanning zero. This carries the same weight here as Section 7.2. Real anchors are the
expensive, better-grounded option, bought with real evaluations by NBI-R alone, and the
result went the other way. The sign is negative under CORE and AUGMENTED alike, with the
panel's one reference disagreement on Adult's anchor contrast, Holm-significant under AUGMENTED
($-0.0535$, Holm $p = 0.0174$) and not under CORE, its bootstrap interval spanning zero under
both references.

### 7.5 CHIM contraction is a post hoc association, and only that

The deficit is not attributable to solver failure. Over the 240 arm-units the
certified fraction has median 1.000 with two exceptions (0.900, 0.950), the
per-unit maximum equality residual has median $6.6\times10^{-10}$ against a
campaign maximum of $6.9\times10^{-1}$ on one unit, and both arms return 20
distinct realized configurations in 239 of 240 arm-units. Comparable, not identical.

The CHIM contraction of Section 6.4.1 is a post hoc association across 30 replications
within a dataset, not a demonstrated cause: it is clear on Adult and Bank Marketing
(within-dataset Spearman between contraction and deficit $+0.683$ and $+0.678$), weak on
Spambase ($+0.320$) and absent on MAGIC ($-0.111$). Nor does the injection control prove the
negative is entirely geometric: the control set is a superset of NBI-S's, so its effect
cannot be negative, and because NBI-R returned none of its own empirical anchors
(Section 5.12) it bounds what adding those anchors to NBI-S's set does rather than
decomposing the gap.

### 7.6 Specification: a non-detection at $R = 30$

Replacing the historical observed-extrema normalization with the payoff-matrix reference
produced no detectable change in CORE-relative hypervolume on any primary dataset: medians
$+0.0000$, $-0.0006$ and $+0.0000$, exact ties in 3, 4 and 8 of 30 replications, Holm $p$
of 0.5165, 0.2087 and 0.3896. This is a non-detection at $R = 30$, not a demonstration
that normalization has no effect; it is where a DoE-range-normalization explanation
of the historical pipeline's behaviour would have shown, had that been the principal
driver.

### 7.7 The grid baseline, and what it costs the argument

The coarse grid, given 386 real evaluations against NBI-S's 186 under the frozen maximum-arm
comparator rule, had the higher paired median on all four datasets; its lead met the frozen
21-of-30 direction rule on MAGIC and Adult only, its bootstrap interval also excluded zero on
Bank Marketing (20/30), and it was unresolved on Spambase (18/30; Section 6.8). Its marginal
median CORE-relative hypervolume ratio was higher than that of every surrogate-assisted arm on
three of the four datasets: NBI-S against GRID, 0.9430/1.0068 on MAGIC, 0.9951/1.0476 on
Adult, 0.9979/1.0625 on Bank Marketing and 1.0812/1.0192 on Spambase, NBI-S winning 0/30,
5/30, 10/30 and 12/30 replications (Supplement S13). The comparator budget is the maximum over
arms, set by NBI-R, so the comparators received roughly twice the real evaluations WS-S and
NBI-S require. The contrast is outside the frozen primary family and uncorrected, and its raw
$p$ is below 0.05 on MAGIC and Adult only.

This is stated plainly rather than deflected: at the budget this protocol froze before the
confirmatory campaign, on this panel, the coarse grid had the higher paired median against
NBI-S on all four datasets. The matched-to-the-most-expensive-arm rule bears on how large the
deficit is; it is not grounds for setting it aside, and is not revised now the result is
known. The geometry finding says which construction to use *given* that a front is built on a
surrogate, not that this family was the right instrument here.

### 7.8 The surrogate-gate regime

The advantage of NBI over weighted scalarization persisted even where the externally
audited surrogate failed the frozen quality criterion. On Adult and Bank Marketing the
composite-quality gate passed in 0 of 30 replications and the geometry effect is
Holm-significant on both, against 8 of 30 on MAGIC and 9 of 30 on Spambase
(Figure 6). The gate was diagnostic: every arm ran at every replication whatever it
said, so it annotates the evidence instead of selecting it. The inference is narrow:
absolute surrogate trustworthiness and relative front-construction geometry are different
questions and this study answers only the second. We do not conclude that the geometry
effect requires a reliable surrogate, still less that NBI compensates for an unreliable
one. The regime removes one alternative explanation: the effect is not confined to
replications whose surface met the frozen criterion.

### 7.9 The estimand is the returned front

Every result above is a property of the set an arm returns, after real revalidation of
every candidate and scoring against a finite method-independent reference, not the
out-of-sample performance of a configuration a practitioner would deploy. No selection
rule collapses the front to a point, and the holdout stage is an audit. A median advantage
of $+0.2549$ on Adult says NBI-S's front occupies more space relative to CORE than WS-S's;
it does not say by how much any chosen model improves.

### 7.10 What the decomposition does not establish

Spambase was separated from the primary family before the confirmatory campaign because
screening showed insufficient interior front geometry in its frozen objective space. Its
geometry contrast, a median of $+0.0500$ against MAGIC's Holm-significant $+0.0374$ but with
roughly thirty times MAGIC's dispersion and an interval spanning zero, did not resolve
(Section 6.6). That is consistent with the boundary designation; it does not confirm the
boundary mechanism. Because Spambase's quality composite is negatively rank-correlated with
raw quality over configurations drawn uniformly from the decision box (post hoc; Section 8.5,
Supplement S4.3), the result also cannot be read as evidence about front geometry. Nor is
Spambase globally null: its
anchor-provenance contrast is strongly negative and Holm-significant.

More broadly: four datasets, two objectives, one surrogate class, one budget rule. Within
that envelope geometry carries the effect on the three primary datasets and is unresolved on
the fourth, specification does not, and provenance is unhelpful in a way that a post hoc
analysis associates with CHIM contraction. Whether those orderings survive three
objectives, another surrogate class, or a budget at which surrogate assistance is
competitive with the coarse grid, this design was not built to answer.
