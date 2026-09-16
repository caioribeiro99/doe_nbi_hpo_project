# 1. Introduction

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
region, whereas Normal Boundary Intersection (NBI) distributes subproblems along a segment of
the convex hull of individual minima (CHIM) and searches along a quasi-normal direction.
Anchor provenance determines the payoff matrix, and therefore the CHIM itself — surrogate
anchors are free once the surfaces exist, while empirical anchors are bought with real
evaluations. In a pipeline where all three are set together, a reported gain is consistent
with any of them.

This paper does not propose a new optimizer. It reports a prospectively frozen reconstruction
of one such pipeline in which the three mechanisms are separated into single-factor contrasts,
executed at a fixed budget, and reported whether or not the result favours the method under
study. The reconstruction target is an archived dissertation implementation of
surrogate-assisted multiobjective XGBoost tuning over Varimax-rotated factor objectives. A
code-level identity audit of that archived implementation established that its front
construction used normalized weighted scalarization, although the historical text used NBI
terminology. That fact is stated here once because it is the reason the arm set has the shape
it does; it is not itself a contribution and it is not treated as anything other than a
specification detail to be carried forward correctly.

## 1.1 Research questions

Holding the decision space, design, surrogate family, weight grid, candidate realization,
evaluation protocol and budget fixed, and changing exactly one mechanism per contrast:

- **RQ1 — specification and normalization.** Does replacing the historical observed-extrema
  normalization box with a payoff-matrix reference change the returned front?
- **RQ2 — front construction geometry.** At a fixed reference and fixed surrogates, does
  canonical NBI construction differ from specification-matched weighted-sum construction?
- **RQ3 — anchor and payoff provenance.** With NBI geometry held fixed, what happens when the
  payoff matrix and anchors are obtained by direct search on the real objectives instead of
  from the fitted surrogates?

The study answers these on four binary-classification datasets, seven XGBoost hyperparameters,
an 88-run face-centred central composite design, two minimized objectives — a Varimax-rotated
composite quality factor over six canonicalized quality responses and a leaf-count
model-complexity factor — and $R = 30$ replicated outer partitions per dataset, with every
returned candidate re-evaluated on the real learner. The campaign is 396,120 logical
evaluations. The primary endpoint is the CORE-relative hypervolume ratio: hypervolume measured
against a finite, method-independent empirical reference of 288 points that no compared method
contributes to. CORE is not the true Pareto front, and a ratio above 1 simply means a candidate
set improved on that finite reference. Dataset is the generalization unit; nothing is pooled
across datasets.

## 1.2 What the study found

**Geometry separates, on the datasets where the screening said it could.** On the three
datasets prospectively assigned to the primary geometry panel, canonical NBI construction
produced higher CORE-relative hypervolume than specification-matched weighted scalarization:
median paired differences of $+0.0374$ on MAGIC (26/30 replications), $+0.2549$ on Adult
(30/30) and $+0.2281$ on Bank Marketing (24/30), Holm-significant within every one of the three
datasets Table 4. This is the primary positive result, and its scope is these
datasets, this surrogate architecture, two objectives and this budget. It is also
indicator-specific: the advantage appears in hypervolume and IGD⁺, while the joint
non-dominated fraction comparison is non-significant on all four datasets.

**Anchor provenance did not help, and is reported with the same prominence.** Replacing
surrogate-derived payoff information with the frozen empirical-real anchor procedure did not
improve the confirmatory NBI fronts: median differences of $-0.0741$ on MAGIC, $-0.0192$ on
Adult (a non-detection, interval spanning zero) and $-0.4093$ on Bank Marketing. Two of three
primary datasets degraded. The paid anchor stage bought nothing measurable here.

**A frozen-budget direct grid baseline led every surrogate-assisted arm.** At the frozen
comparator budget, a coarse grid favoured the coarse grid on all four datasets in median paired difference, and exceeded every surrogate-assisted arm in marginal median on three of the four (not Spambase) Table 5. The budget asymmetry
belongs in the same sentence: the frozen rule matched comparators to the most expensive arm
(NBI-R, 386 real evaluations), while WS-S and NBI-S cost 186 standalone, so the comparators
received roughly twice the real evaluations those arms require. The rule was fixed before any
result existed and is not revised now, and the result is not explained away on that basis.

**Specification and normalization were undetectable.** The RQ1 contrast produced no detectable
change in CORE-relative hypervolume on any primary dataset, reported as a non-detection at
$R = 30$ rather than as an absence of effect.

## 1.3 Contributions

1. A single-factor decomposition of one surrogate-assisted multiobjective HPO pipeline into
   specification and normalization, front-construction geometry, and anchor and payoff
   provenance, with run-time enforcement that each contrast varies only its declared mechanism.
2. A protocol frozen before any confirmatory comparative result was inspected, with its
   dataset roles, endpoints, reference, statistical family and budget ledger fixed in advance,
   and every subsequent change recorded as a numbered amendment.
3. A confirmatory campaign of 396,120 logical evaluations over four datasets and $R = 30$
   paired replications, with every candidate revalidated on the real learner and every headline
   number traceable to a committed artifact.
4. Results reported against interest: the negative anchor-provenance contrast, the grid
   baseline that leads at the frozen budget, and the indicator specificity of the geometry
   effect, each given the same standing as the positive result.
5. A mechanistic reading of the negative contrast — an anchor-injection control together with
   a measured contraction of the CHIM segment — reported as an association within datasets, not
   as a demonstrated cause Figure 4.
