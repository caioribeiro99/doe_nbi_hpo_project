## 1. Introduction

Surrogate-assisted multiobjective hyperparameter optimization is usually reported as a
pipeline rather than as a set of separable choices. A design of experiments is run, response
surfaces are fitted to reduced objectives, a scalarization is solved over those surfaces on a
weight grid, and the returned candidates are re-evaluated on the real learner. When such a
pipeline is compared against another, the comparison moves several things at once: how the
objectives are normalized and what reference they are measured against; what geometry is used
to construct the front; and where the payoff and anchor information used by that geometry
comes from. A difference in a front indicator can then be attributed to none of them
individually.

These three choices are not interchangeable. The normalization reference fixes the scale on
which any scalarized objective is combined. The construction geometry determines which points
of a nonconvex front are reachable at all: a weighted sum cannot return points in a nonconvex
region, whereas Normal Boundary Intersection (NBI; Das and Dennis 1998) distributes
subproblems along a segment of the convex hull of individual minima (CHIM) and searches along
a quasi-normal direction. Anchor provenance determines the payoff matrix, and therefore the
CHIM itself: surrogate
anchors are free once the surfaces exist, while empirical anchors are bought with real
evaluations. In a pipeline where all three are set together, a reported gain is consistent
with any of them.

This paper does not propose a new optimizer. It reports a prospectively frozen decomposition
of one such pipeline in which the three mechanisms are separated into single-factor contrasts,
executed under a budget ledger fixed in advance, and reported whether or not the result
favours the method under study. The pipeline is the one implemented in the first author's
master's dissertation (Ribeiro 2026) for surrogate-assisted multiobjective XGBoost tuning over
Varimax-rotated factor objectives. Its archived code normalizes the objectives by an
observed-extrema box and constructs the front by weighted scalarization, so the historical
method enters the arm set as a weighted sum (Section 3).

### 1.1 Research questions

Holding the decision space, design, surrogate family, weight grid, candidate realization and
evaluation protocol fixed, and changing exactly one mechanism per contrast (the budget is also
fixed in RQ1 and RQ2; in RQ3 the 200 real anchor evaluations are part of the change):

- **RQ1: specification and normalization.** Does replacing the historical observed-extrema
  normalization box with a payoff-matrix reference change the returned front?
- **RQ2: front construction geometry.** At a fixed reference and fixed surrogates, does
  canonical NBI construction differ from specification-matched weighted-sum construction?
- **RQ3: anchor and payoff provenance.** With the NBI formulation held fixed, what happens when the
  payoff matrix and anchors are obtained by direct search on the real objectives instead of
  from the fitted surrogates?

The study answers these on four binary-classification datasets, seven XGBoost hyperparameters,
an 88-run face-centred central composite design, two minimized latent objectives and $R = 30$
replicated outer partitions per dataset, with every returned candidate re-evaluated on the real
learner. The first objective is a Varimax-rotated composite quality factor, oriented on six
canonicalized quality responses; the second is a leaf-count model-complexity factor. Both are
fixed linear scores of all seven responses, fitted once per dataset on the 88 design rows of a
pre-campaign pilot partition (Section 5.5). The campaign is 396,120 logical
evaluations. The primary endpoint is the CORE-relative hypervolume ratio: hypervolume measured
against a finite, method-independent empirical reference of 288 points that no compared method
contributes to. CORE is not the true Pareto front, and a ratio above 1 simply means a candidate
set improved on that finite reference. Dataset is the generalization unit; nothing is pooled
across datasets.

RQ1, RQ2 and RQ3 form the primary inferential family. Section 4 adds two questions outside it:
the RQ2 contrast on Spambase, which failed the pre-campaign front-geometry screen and was
retained as a boundary control (RQ4), and direct search at the frozen comparator budget (RQ5).
Also declared before the campaign are a secondary question, whether the contrasts behave
differently where a surrogate fails its frozen external reliability gate, a diagnostic
covariate scored on 78 audit-only real evaluations (Section 6.7, Supplement S11); sensitivities
of the primary analysis, of which the AUGMENTED reference (CORE plus every point a compared
method is scored on) and the Nadeau–Bengio corrected test are reported in Sections 6.5 and
6.3.2; and a holdout audit of selection optimism that ranks no arm (Section 6.9). Declared
analyses that were computed only after the campaign, and those that were not executed, are
listed in Section 8.7.

### 1.2 What the study found

**Geometry separates.** On the three primary datasets, those that passed the pre-campaign
front-geometry screen on the design rows (Section 5.1), canonical NBI construction produced
higher CORE-relative hypervolume than specification-matched weighted scalarization: median
paired differences of $+0.0374$ on MAGIC (26/30 replications), $+0.2549$ on Adult (30/30) and
$+0.2281$ on Bank Marketing (24/30). Every bootstrap interval on the median difference excludes
zero, and the secondary signed-rank test, a post hoc choice (Section 5.15), is Holm-significant within every one of the three
datasets (Table 4); under the pre-declared Nadeau–Bengio sensitivity the contrast reaches
unadjusted $p < 0.05$ on Adult only (Section 6.3.2). The geometry effect is the main positive
result, and its scope is these datasets, this surrogate architecture as frozen (its surrogate
anchors are not uniquely determined; Section 8.5), two objectives and this budget. On Adult and Bank Marketing it was obtained with a composite-quality surrogate that
failed the frozen external reliability gate in every replication (Section 6.7). It is also
indicator-specific: the advantage appears in hypervolume and IGD$^{+}$, while the joint
non-dominated fraction shows no detectable difference under CORE on any of the four datasets
(Section 6.3.1).

**Anchor provenance did not help.** Replacing
surrogate-derived payoff information with the frozen empirical-real anchor procedure did not
improve the confirmatory NBI fronts: median differences of $-0.0741$ on MAGIC, $-0.0192$ on
Adult (a non-detection, interval spanning zero) and $-0.4093$ on Bank Marketing. Two of three
primary datasets degraded; under the Nadeau–Bengio sensitivity the deficit reaches unadjusted
$p < 0.05$ on MAGIC only. The paid anchor stage bought nothing measurable here, measured against
the surrogate anchors the frozen pipeline selected, which the fitted surfaces do not determine
uniquely (Section 8.5).

**A frozen-budget direct grid baseline led NBI-S at the frozen comparator budget.** The
median paired difference favoured the direct grid baseline over NBI-S on all four datasets.
The grid's lead met the frozen 21-of-30 direction rule on MAGIC and Adult only (NBI-S won 0
and 5 of 30 replications); on Bank Marketing its bootstrap interval also excluded zero (NBI-S
won 10 of 30), and on Spambase it was unresolved (NBI-S won 12 of 30). The grid held the higher
marginal median on three of the four datasets, but not on Spambase, where NBI-S is the higher
of the two (Table 5). The paired statistic is the within-replication comparison and the
marginal medians are not, so the two need not agree. The budgets were not equal: the frozen
rule matched comparators to the most expensive arm (NBI-R, 386 real evaluations), while WS-S
and NBI-S cost 186 standalone, so the comparators received roughly twice the real evaluations
those arms require. The rule dates from protocol v1 (2026-09-13), before any arm was
implemented, and its value of 386 from protocol v2; both precede the confirmatory campaign and
were applied unchanged, and the asymmetry is not used to dismiss the result.

**Specification and normalization were undetectable.** The RQ1 contrast produced no detectable
change in CORE-relative hypervolume on any primary dataset, reported as a non-detection at
$R = 30$ rather than as an absence of effect.

### 1.3 Contributions

1. A single-factor decomposition of one surrogate-assisted multiobjective HPO pipeline into
   specification and normalization, front-construction geometry, and anchor and payoff
   provenance, with run-time enforcement that each contrast varies only its declared mechanism.
2. A protocol frozen before any confirmatory comparative result was inspected, with its
   dataset roles, endpoints, reference, statistical family and budget ledger fixed in advance,
   every pre-campaign change recorded as a numbered amendment, and the departures identified
   after the campaign listed as deviations (Section 8.7).
3. A confirmatory campaign of 396,120 logical evaluations over four datasets and $R = 30$
   paired replications, with every candidate revalidated on the real learner and every headline
   number traceable to a committed artifact.
4. Negative and scope-limiting results: the negative anchor-provenance contrast, the grid
   baseline that led NBI-S at the frozen comparator budget, and the indicator specificity of
   the geometry effect.
5. A prospectively frozen anchor-injection control that bounds what including the empirical
   anchors adds to NBI-S's set (Section 6.4); a post hoc CHIM-contraction association is
   reported there but, resting on non-unique surrogate anchors, is not interpreted
   (Section 8.5).
