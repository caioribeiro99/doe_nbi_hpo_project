# 7. Discussion

## 7.1 What the decomposition establishes

The campaign separates three mechanisms ordinarily bundled together in a
surrogate-assisted multiobjective hyperparameter pipeline: the objective specification and
its normalization reference, the geometry of front construction, and the provenance of the
anchor and payoff information that geometry consumes. The contribution is not a new
algorithm but a prospectively frozen mechanistic decomposition of an existing one.
Code-level reconstruction of the archived dissertation implementation showed that front
construction there used normalized weighted scalarization, although the historical text
used Normal Boundary Intersection terminology, which is why the historical-to-canonical
difference decomposes into a normalization row and a geometry row.

## 7.2 Geometry moved the indicator

On three geometry-positive datasets and 30 replicated outer partitions, under the frozen
study conditions, canonical NBI front construction produced higher real-revalidated
CORE-relative hypervolume than specification-matched weighted scalarization: median paired
differences of $+0.0374$ on MAGIC, $+0.2549$ on Adult and $+0.2281$ on Bank Marketing,
favoured in 26/30, 30/30 and 24/30 replications, rank-biserial $\geq +0.88$ in all three,
Holm-significant within every primary dataset, which is the generalization unit
[TAB:primary_contrasts].

The scope is part of the claim: two objectives, this surrogate architecture, this budget,
these three datasets. Nothing licenses "NBI is better than weighted sum" in general. The
endpoint is a CORE-relative hypervolume ratio against a finite method-independent set of
288 points, where a value above 1 is expected and means only that the returned set
improved on that finite reference.

## 7.3 Why hypervolume and IGD⁺ move while the non-dominated count does not

The advantage is specifically an improvement in convergence and coverage geometry as
measured by hypervolume and IGD⁺: median IGD⁺, NBI-S against WS-S, of 0.0511/0.0930 on
MAGIC, 0.0579/0.1688 on Adult and 0.0915/0.1948 on Bank Marketing. The joint non-dominated
fraction behaves differently — median differences of $+0.0208$, $+0.0714$ and $+0.0000$,
non-significant on all four datasets including Spambase.

This is not a contradiction. Hypervolume and IGD⁺ are positional: they measure proximity
to the reference and dominated space, so moving points closer or redistributing them along
the trade-off changes both. The joint non-dominated fraction is a mutual-dominance count,
and at $q = 2$ points at different positions along one trade-off band are mutually
non-dominated by construction — an arm can sit nearer the reference while almost none of
its points dominate the other arm's. The geometry result must not be read as NBI returning
more non-dominated solutions: it did not, on any dataset.

## 7.4 Anchor provenance: a negative result, not a caveat

Holding NBI geometry fixed and replacing surrogate-derived payoff information with the
pre-specified empirical-real anchor procedure did not improve the confirmatory fronts.
Paired medians are $-0.0741$ on MAGIC, $-0.0192$ on Adult and $-0.4093$ on Bank Marketing,
Holm-significant on MAGIC and Bank Marketing; Adult is a non-detection under CORE with an
interval spanning zero. This carries the same weight here as §7.2. Real anchors are the
expensive, better-grounded option, bought with real evaluations by NBI-R alone, and the
result went the other way. The sign is negative under CORE and AUGMENTED alike, with the
panel's one reference disagreement on Adult's anchor contrast, significant under AUGMENTED
($-0.0535$, Holm $p = 0.0174$) and not under CORE.

## 7.5 CHIM contraction is the associated mechanism, and only that

The deficit is not solver failure: certified fraction 1.000, zero maximum equality
residual, 20 distinct realized configurations per arm on every dataset. Inserting NBI-R's
empirical anchors into NBI-S's returned set and changing nothing else moves the indicator
by $+0.0133$ on MAGIC and $+0.0000$ on the other three, against the full provenance gaps
above and $-0.6932$ on Spambase: the deficit is associated with the geometric consequences
of the relocated payoff matrix rather than with anchor point-set composition. And the CHIM
segment contracts to 20–81% of its surrogate-derived extent under empirical anchors —
extent ratios 0.811, 0.204, 0.272 and 0.370 on MAGIC, Adult, Bank Marketing and Spambase
[FIG:chim_contraction]. At $q = 2$ that segment is what the subproblems distribute targets
along, so a shorter one spreads the set over a narrower band.

This is an association across 30 replications within a dataset, not a demonstrated cause:
the within-dataset Spearman between contraction and deficit is $+0.683$ on Adult, $+0.678$
on Bank Marketing and $+0.320$ on Spambase — and $-0.111$ on MAGIC, where it is absent.
Nor does the injection control prove the negative is entirely geometric: the control set
is a superset of NBI-S's, so its effect cannot be negative.

## 7.6 Specification: a non-detection at $R = 30$

Replacing the historical observed-extrema normalization with the payoff-matrix reference
produced no detectable change in CORE-relative hypervolume on any primary dataset: medians
$+0.0000$, $-0.0006$ and $+0.0000$, exact ties in 3, 4 and 8 of 30 replications, Holm $p$
of 0.5165, 0.2087 and 0.3896. This is a non-detection at $R = 30$, not a demonstration
that normalization has no effect — and it is where a DoE-range-normalization explanation
of the historical pipeline's behaviour would have shown, had that been the principal
driver.

## 7.7 The grid baseline, and what it costs the argument

At the frozen comparator budget, an evaluation-matched coarse grid attained higher median
CORE-relative hypervolume than every surrogate-assisted arm on every dataset in the panel
— NBI-S against GRID, 0.9430/1.0068 on MAGIC, 0.9951/1.0476 on Adult, 0.9979/1.0625 on
Bank Marketing and 1.0812/1.0192 on Spambase, NBI-S winning 0/30, 5/30, 10/30 and 12/30
replications [TAB:baseline_vs_arms]. The budget asymmetry belongs in the same breath: the
comparator budget is the maximum over arms, 386 real evaluations set by NBI-R, while WS-S
and NBI-S cost 186 standalone, so comparators received roughly twice the real evaluations
those arms require. The contrast is outside the frozen primary family, uncorrected, and
significant on raw $p$ for MAGIC and Adult only.

This is stated plainly rather than deflected: at the budget this protocol froze before any
result existed, on this panel, surrogate assistance did not pay for itself against direct
search. The matched-to-the-most-expensive-arm rule bears on how large the deficit is; it
is not grounds for setting it aside, and is not revised now the result is known. The
geometry finding says which construction to use *given* that a front is built on a
surrogate — not that this family was the right instrument here.

## 7.8 The surrogate-gate regime

The advantage of NBI over weighted scalarization persisted even where the externally
audited surrogate failed the frozen quality criterion. On Adult and Bank Marketing the
composite-quality gate passed in 0 of 30 replications and the geometry effect is
Holm-significant on both, against 8 of 30 on MAGIC and 9 of 30 on Spambase
[FIG:gate_regime]. The gate was diagnostic: every arm ran at every replication whatever it
said, so it annotates the evidence instead of selecting it. The inference is narrow:
absolute surrogate trustworthiness and relative front-construction geometry are different
questions and this study answers only the second. We do not conclude that the geometry
effect requires a reliable surrogate, still less that NBI compensates for an unreliable
one. The regime removes one alternative explanation — the effect is not confined to
replications where the surface was accurate.

## 7.9 The estimand is the returned front

Every result above is a property of the set an arm returns, after real revalidation of
every candidate and scoring against a finite method-independent reference — not the
out-of-sample performance of a configuration a practitioner would deploy. No selection
rule collapses the front to a point, and the holdout stage is an audit. A median advantage
of $+0.2549$ on Adult says NBI-S's front occupies more space relative to CORE than WS-S's;
it does not say by how much any chosen model improves.

## 7.10 What the decomposition does not establish

Spambase was prospectively separated from the primary family because screening showed
insufficient interior front geometry. Its geometry contrast returned a median of
$+0.0500$, numerically larger than MAGIC's significant $+0.0374$, with an interval
spanning zero, Holm $p = 0.1175$ and a standard deviation of 1.7706 against MAGIC's 0.0594
— too unstable to resolve the contrast. That is consistent with the boundary designation;
it does not confirm the boundary mechanism. Nor is Spambase globally null: its
anchor-provenance contrast is strongly negative and significant.

More broadly: four datasets, two objectives, one surrogate class, one budget rule. Within
that envelope geometry carries the effect, specification does not, and provenance is
unhelpful in a way associated with CHIM contraction. Whether those orderings survive three
objectives, another surrogate class, or a budget at which surrogate assistance is
competitive with direct search, this design was not built to answer.
