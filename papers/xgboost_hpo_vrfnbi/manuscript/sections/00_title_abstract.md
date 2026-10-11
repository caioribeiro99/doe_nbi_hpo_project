# Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study

## Abstract

Surrogate-assisted multiobjective hyperparameter optimization pipelines routinely
change scalarization specification, Pareto-front construction geometry, and anchor
provenance together, so an observed difference cannot be attributed to any one of
them. We report a prospectively frozen, provenance-preserving reconstruction that
varies the three one at a time, with seven XGBoost hyperparameters on an 88-run face-centred
central composite design, four datasets, 30 replicated outer partitions and every returned
candidate revalidated on the real learner. Both minimized objectives are latent: a
Varimax-rotated composite quality factor and a leaf-count model-complexity factor, each a fixed
linear score of six quality responses and the log leaf count, fitted per dataset on pilot
design runs. The budget ledger counted every request, repeats included: 396,120 logical evaluations,
377,316 of them unique.

Replacing the archived observed-extrema normalization with a payoff-matrix reference
produced no detectable change. Holding the surrogate model, scaling, decision space,
candidate realization and budget fixed, canonical Normal Boundary Intersection (NBI) on the
surrogate payoff matrix (NBI-S) improved hypervolume over specification-matched weighted
scalarization on all three datasets that passed the pre-campaign front-geometry screen (a
curved front among the non-dominated design runs). Measured against CORE, a 288-point
empirical reference to which no compared method contributes, the median paired differences
were $+0.037$ on MAGIC (26/30 wins), $+0.255$ on Adult (30/30) and $+0.228$ on Bank Marketing
(24/30); all three bootstrap intervals exclude zero, and the secondary signed-rank test is
Holm-significant within each dataset. The composite-quality surrogate met its frozen external
reliability criterion in 8 of 30 replications on MAGIC, none on Adult or Bank Marketing and 9
of 30 on the fourth dataset, Spambase; the criterion was diagnostic and removed nothing, so
the Adult and Bank Marketing effects rest entirely on replications that failed it. Replacing
surrogate-derived payoff information with a pre-specified empirical-real anchor procedure,
costing 200 additional real evaluations, gave no benefit and degraded the front on two of
the three ($-0.074$ on MAGIC; $-0.019$ on Adult, a non-detection; $-0.409$ on Bank
Marketing). A post hoc analysis associates the deficit with contraction of the convex hull of
individual minima, clearly on Adult and Bank Marketing, weakly on Spambase and not on MAGIC.
Under the pre-declared Nadeau–Bengio corrected t-test, a sensitivity that allows for training
data shared across partitions, the geometry improvement reaches unadjusted $p < 0.05$ on
Adult only and the anchor deficit on MAGIC only. An AUGMENTED reference that adds to CORE every
point a compared method is scored on gives the same bootstrap-interval conclusion
in all 12 dataset-by-contrast cells and the same secondary-test conclusion in 11.
On every dataset these conclusions are conditional on the variance-weighted quality composite: rescoring each arm's revalidated candidates after the campaign under the pre-declared equal weighting changed the non-dominated sets in more than 20% of replications; under the equal weighting the WS-S $\rightarrow$ NBI-S and NBI-S $\rightarrow$ NBI-R median differences kept their direction (Supplement S4.5). Under the frozen
direct-search comparator budget the median paired difference favoured a coarse grid over
NBI-S on all four datasets, with the grid receiving 386 real evaluations against 186
standalone for NBI-S; the lead met the frozen 21-of-30 direction rule on MAGIC and Adult
only, its bootstrap interval also excluded zero on Bank Marketing (20/30 wins), and it was
unresolved on Spambase (18/30). On Spambase, which failed the screen and was retained as a boundary
control, the geometry contrast did not resolve (a numerically larger point estimate than the
smallest resolved effect, at roughly thirty times its dispersion), and its quality composite
is negatively rank-correlated with raw quality over uniform draws of the decision box (a post
hoc check), so this result cannot be read as evidence about front geometry.

**Keywords:** multiobjective hyperparameter optimization; Normal Boundary
Intersection; weighted scalarization; response surface methodology; XGBoost;
replicated evaluation; Pareto indicators
