## 10. Conclusion

Surrogate-assisted multiobjective hyperparameter optimization pipelines change several
things at once. When such a pipeline is revised, the revision typically alters the
scalarization specification, the geometry by which a front is constructed, and the
information from which anchors and the payoff matrix are built. A difference in the
returned front cannot then be attributed to any of them. This study separated the three
under a protocol frozen after documented measurement-validation pilots and before the
confirmatory campaign, and revalidated every returned candidate on the real learner.

Two of the three mechanisms produced clear answers, in opposite directions. Correcting
the archived scalarization specification and its observed-extrema normalization changed
nothing detectable. Replacing weighted scalarization with canonical Normal Boundary
Intersection, holding everything else fixed, improved CORE-relative hypervolume on all
three screened datasets, with 26, 30 and 24 of 30 paired wins, every bootstrap interval on the
median difference excluding zero and the secondary test Holm-significant within each dataset;
under the pre-declared Nadeau–Bengio sensitivity the improvement reached unadjusted $p < 0.05$
on Adult alone. The third mechanism answered in the direction opposite
to the one the pipeline's design would suggest: substituting a pre-specified
empirical-real anchor procedure for surrogate-derived payoff information gave no
benefit, and degraded the front on two of the three datasets, of which only MAGIC reached
unadjusted $p < 0.05$ under the Nadeau–Bengio sensitivity. The frozen anchor-injection
control, which adds the empirical anchors to NBI-S's candidates and changes nothing else,
moved the indicator by approximately zero on three of four datasets and by $+0.0133$ on
MAGIC; because NBI-R returned none of its empirical anchors, the control bounds what anchor
inclusion alone adds rather than decomposing the deficit. In a post hoc analysis the payoff
matrices show a contraction of the convex hull of individual minima that is associated with
the deficit, clearly on Adult and Bank Marketing, weakly on Spambase and not on MAGIC. Both
are reported as associations, not as causes.

Two results limit how far the geometry finding should be carried. The advantage is
specific to hypervolume and IGD$^{+}$; the joint non-dominated fraction shows no detectable
difference under CORE on any dataset in the panel. And under the frozen direct-search
comparator budget, the median paired difference favoured a coarse grid over NBI-S on all four
datasets: the grid's lead met the frozen 21-of-30 direction rule on MAGIC and Adult only, its
bootstrap interval also excluded zero on Bank Marketing (20/30 wins), and it was unresolved on
Spambase (18/30); the grid also held the higher marginal median on three of the four datasets,
but not on Spambase. The grid received 386 real evaluations against the 186 that WS-S and NBI-S
cost standalone. That asymmetry is a property of the frozen comparator rule, not a defence of
the arms: at the budget this protocol fixed in advance, the grid led.

What the study establishes is therefore narrower than "NBI is better".
Within this decision space, this surrogate architecture, two objectives and this budget, and on
the three screened datasets, front-construction geometry is the mechanism that moved the
returned front; specification did not; and anchor provenance moved it the wrong way. Spambase,
which failed the pre-campaign front-geometry screen and was retained as a boundary control
before the confirmatory campaign, did not resolve; because its quality composite is negatively
aligned with raw quality over configurations drawn uniformly from the decision box (post hoc; Section 8.5), that result is not
evidence about front geometry in either direction. Whether the geometry advantage survives at
larger budgets, at more than two objectives, or against direct search when the arms receive
the comparators' 386 evaluations, this design cannot say, and we have tried not to say it.
