# Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study

## Abstract

Surrogate-assisted multiobjective hyperparameter optimization pipelines routinely
change scalarization specification, Pareto-front construction geometry, and anchor
provenance together, so an observed difference cannot be attributed to any one of
them. We report a prospectively frozen, provenance-preserving reconstruction that
separates the three: seven XGBoost hyperparameters on an 88-run face-centred central
composite design, two minimized objectives — a Varimax-rotated composite quality factor
and a leaf-count model-complexity factor — four datasets, 30 replicated outer
partitions, and every returned candidate revalidated on the real learner (396,120
logical evaluations).

Replacing the archived observed-extrema normalization with a payoff-matrix reference
produced no detectable change. Holding the surrogate model, scaling, decision space,
candidate realization and budget fixed, canonical Normal Boundary Intersection improved
CORE-relative hypervolume over specification-matched weighted scalarization on all three geometry-positive datasets: $+0.037$ (26/30 wins),
$+0.255$ (30/30) and $+0.228$ (24/30), Holm-significant within each dataset. Replacing
surrogate-derived payoff information with a pre-specified empirical-real anchor
procedure gave no benefit and degraded the front on two of the three ($-0.074$;
$-0.019$, not significant under the CORE reference; $-0.409$); the deficit is associated, on three of four
datasets, with contraction of the convex hull of individual minima rather than with
anchor point-set composition. CORE and AUGMENTED references agree in 11 of 12 cells. Under the frozen
direct-search comparator budget the median paired difference favoured a coarse grid
over NBI-S on all four datasets, although the grid received 386 real evaluations
against 186 standalone for the scalarization arms. A prospectively
designated boundary dataset did not resolve: a larger point estimate than the smallest
significant effect, at roughly thirty times its dispersion.

**Keywords:** multiobjective hyperparameter optimization; Normal Boundary
Intersection; weighted scalarization; response surface methodology; XGBoost;
replicated evaluation; Pareto indicators
