# Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study

## Abstract

Surrogate-assisted multiobjective hyperparameter optimization usually changes scalarization
specification, front-construction geometry and anchor provenance at once, confounding
them. A prospectively frozen reconstruction varies the three one at a time for seven XGBoost
hyperparameters (88-run central composite design, four datasets, 30 replications), with
real-learner revalidation of every returned candidate. Both objectives are fixed latent factors
(composite quality and leaf-count complexity), so every effect refers to this frozen objective
space. Specification produced no detectable change. Against CORE, a method-independent empirical
reference, Normal Boundary Intersection on the surrogate payoff
matrix (NBI-S) improved the hypervolume ratio over specification-matched weighted scalarization on
the three datasets passing a pre-campaign front-geometry screen: median paired differences
$+0.037$ (MAGIC), $+0.255$ (Adult) and $+0.228$ (Bank Marketing), with bootstrap intervals
excluding zero and Holm-significant post hoc signed-rank tests; the pre-declared Nadeau–Bengio test reached
$p < 0.05$ only on Adult. On Adult and Bank Marketing every replication's quality surrogate failed its
external reliability gate. Empirical-real anchors degraded the front on MAGIC ($-0.074$) and Bank
Marketing ($-0.409$); on Adult ($-0.019$) it was undetectable (Nadeau–Bengio $p < 0.05$ on MAGIC only). All conclusions
are conditional on the composite's variance weighting: its pre-declared alternative changed the
returned fronts in over 20% of replications, reversing no contrast. A coarse grid led NBI-S by the
paired median on every dataset at 386 evaluations and, post hoc, at NBI-S's 186, meeting the
21-of-30 direction rule on two. On Spambase, the boundary dataset, the composite runs against raw
quality over uniform draws (post hoc), so its unresolved result is uninformative.

**Keywords:** multiobjective hyperparameter optimization; Normal Boundary
Intersection; weighted scalarization; response surface methodology; XGBoost;
replicated evaluation; Pareto indicators
