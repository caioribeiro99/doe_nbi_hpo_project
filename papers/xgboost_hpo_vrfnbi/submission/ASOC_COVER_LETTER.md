# Cover letter

Dear Editor,

We submit *Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study* for consideration as a research article in Applied Soft Computing.

The paper separates three choices that surrogate-assisted multiobjective hyperparameter
optimization pipelines normally change together: the scalarization specification, the
geometry by which a Pareto front is constructed, and the provenance of the anchors and
payoff matrix that geometry uses. Because a pipeline revision typically moves all three
at once, a reported gain cannot be attributed to any one of them. We hold the surrogate
model, scaling, decision space and candidate realization fixed and vary one mechanism at a
time (changing anchor provenance also adds the 200 real evaluations its empirical anchors
cost), under a protocol frozen before the confirmatory campaign, with 30 replicated outer
partitions per dataset and every returned candidate re-evaluated on the real learner.

Relevance to Applied Soft Computing: the work sits in two areas the journal names
explicitly, Machine and Deep Learning (the optimized learner is a gradient-boosted
decision-tree model) and Multi-objective Optimization (the object of the study).
Evolutionary computing enters as an NSGA-II comparator rather than as the contribution,
and we state that plainly rather than overclaiming it.

We also report results that do not favour our own method line. The anchor-provenance
mechanism gave no benefit and degraded the front on two of the three screened datasets. Under
the frozen comparator budget, the median paired difference favoured a coarse grid over NBI-S
(canonical Normal Boundary Intersection on the surrogate payoff matrix) on all four datasets,
and the lead met the frozen 21-of-30 direction rule on two of them and had a bootstrap interval
excluding zero on a third; the grid received 386 real
evaluations against the 186 that NBI-S costs standalone. Both results are reported in full,
with that budget asymmetry stated alongside the second.

The protocol, the analysis code and every committed artifact are version controlled and
tagged, and the manuscript states the tags. A fourth dataset, which failed the pre-campaign
front-geometry screen and was retained as a boundary control, did not resolve; it is reported
as unresolved rather than omitted, together with the construct limitation that prevents
reading it as evidence about front geometry.

The work is original, is not under consideration elsewhere, and has not been published
previously. There is no prior conference version. It extends the first author's master's
dissertation, as disclosed in the Declarations.

Yours sincerely,

Caio Tertuliano Ribeiro, on behalf of all authors
Corresponding author: caio.tertu@hotmail.com
Institute of Production Engineering and Management (IEPG), Federal University of
Itajubá (UNIFEI), Av. BPS 1303, Pinheirinho, Itajubá, MG 37500-903, Brazil
