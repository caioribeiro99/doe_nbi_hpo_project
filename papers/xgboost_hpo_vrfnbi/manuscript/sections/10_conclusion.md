## 10. Conclusion

Surrogate-assisted multiobjective hyperparameter optimization pipelines change several
things at once. When such a pipeline is revised, the revision typically alters the
scalarization specification, the geometry by which a front is constructed, and the
information from which anchors and the payoff matrix are built — and a difference in
the returned front cannot then be attributed to any of them. This study separated the
three under a protocol frozen before any comparative result existed, and revalidated
every returned candidate on the real learner.

Two of the three mechanisms produced clear answers, in opposite directions. Correcting
the archived scalarization specification and its observed-extrema normalization changed
nothing detectable. Replacing weighted scalarization with canonical Normal Boundary
Intersection, holding everything else fixed, improved CORE-relative hypervolume on all
three geometry-positive datasets, with 26, 30 and 24 of 30 paired wins and Holm-adjusted
significance within each dataset. The third mechanism answered in the direction opposite
to the one the pipeline's design would suggest: substituting a pre-specified
empirical-real anchor procedure for surrogate-derived payoff information gave no
benefit, and degraded the front on two of the three datasets. The frozen anchor-injection control moves the
indicator by approximately zero on three of four datasets and by $+0.0133$ on MAGIC,
where it accounts for about 18% of that dataset's gap, so the deficit is associated
with the relocated payoff information rather than with which points enter the returned
set,
and the payoff matrices show a contraction of the convex hull of individual minima that
is associated with the deficit on three of four datasets. That association is reported
as an association.

Two results limit how far the geometry finding should be carried. The advantage is
specific to hypervolume and IGD⁺; the joint non-dominated fraction shows no significant
difference on any dataset in the panel. And under the frozen direct-search comparator
budget, the median paired difference favoured a coarse grid over NBI-S on all four
datasets, and the grid held the higher marginal median on three of the four — not on
Spambase — while receiving 386 real evaluations against the 186 the scalarization arms
require standalone. That asymmetry is a property of the frozen
comparator rule, not a defence of the arms: at the budget this protocol fixed in
advance, direct search led.

What the study establishes is therefore narrower than "NBI is better", and more useful.
Within this decision space, this surrogate architecture, two objectives and this budget,
front-construction geometry is the mechanism that moved the returned front;
specification did not; and anchor provenance moved it the wrong way. The prospectively
designated boundary dataset did not resolve, which is consistent with the reasoning
that separated it in advance without confirming that reasoning. Whether the geometry
advantage survives at larger budgets, at more than two objectives, or against direct
search given equal evaluations, this design cannot say, and we have tried not to say it.
