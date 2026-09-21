# Cover letter

Dear Editor,

We submit *Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study* for consideration as a research article in Applied Soft Computing.

The paper separates three choices that surrogate-assisted multiobjective hyperparameter
optimization pipelines normally change together: the scalarization specification, the
geometry by which a Pareto front is constructed, and the provenance of the anchors and
payoff matrix that geometry uses. Because a pipeline revision typically moves all three
at once, a reported gain cannot be attributed to any one of them. We hold the surrogate
model, scaling, decision space, candidate realization and budget fixed and vary one
mechanism at a time, under a protocol frozen before any comparative result existed, with
30 replicated outer partitions per dataset and every returned candidate re-evaluated on
the real learner.

Relevance to Applied Soft Computing: the work sits in two areas the journal names
explicitly — Machine and Deep Learning, since the optimized learner is a gradient-boosted
decision-tree model, and Multi-objective Optimization, which is the object of the study.
Evolutionary computing enters as an NSGA-II comparator rather than as the contribution,
and we state that plainly rather than overclaiming it.

We also report results that do not favour our own method line. The anchor-provenance
mechanism gave no benefit and degraded the front on two of three datasets, and a
frozen-budget coarse grid led the surrogate-assisted arms on the paired statistic on all
four datasets. Both are reported in full, with the budget asymmetry that qualifies the
second stated alongside it.

The protocol, the analysis code and every committed artifact are version controlled and
tagged, and the manuscript states the tags. A prospectively designated boundary dataset
that did not resolve is reported as unresolved rather than omitted.

The work is original, is not under consideration elsewhere, and has not been published
previously. There is no prior conference version.

Yours sincerely,

Caio Tertuliano Ribeiro, on behalf of all authors
Corresponding author — caio.tertu@hotmail.com
Institute of Production Engineering and Management (IEPG), Federal University of
Itajubá (UNIFEI), Av. BPS 1303, Pinheirinho, Itajubá, MG 37500-903, Brazil
