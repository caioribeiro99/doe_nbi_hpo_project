# Cover letter

To the Editors, *Engineering Applications of Artificial Intelligence*

Dear Editors,

We submit **"When Is Surrogate-Assisted Multiobjective Ensemble Weighting Trustworthy? A Replicated Study of Mixture
Designs, Scheffé Surfaces and Normal Boundary Intersection on the Classifier-Weight Simplex"** for consideration.

**We begin with a disclosure.** This manuscript stress-tests a framework that one of us co-developed and that this
journal published: Pereira, Tertuliano Ribeiro, Mendes, Campos and de Paiva, *EAAI* **162**, 112510 (2025). One of the
three configurations we study, NBI-A — surrogate objectives with anchors taken from the surrogate's own optima — is
that paper's construction, and we report it failing on two of our four datasets. We claim no part of the
DoE–RSM–NBI pipeline as new. That construction is prior art four times over: for classifier ensembles specifically in
Kwon, Lee and Lee (2024), for a neural-network ensemble with NBI in Rocha et al. (2025), for forecast combination in
Bacci et al. (2019), and in our own 2025 paper. A dimension-by-dimension self-overlap analysis accompanies the
submission.

**What the paper contributes is an evaluation architecture and what it reveals.** Transferring the framework to
classifier-ensemble weighting puts it under conditions its originating setting does not have: one objective is a
piecewise-constant rank statistic, the decision variables are themselves the mixture so anchors sit on the boundary
of the feasible region, and deployment cost is a step function of the support rather than a smooth response. We run
the identical Normal Boundary Intersection three ways — surrogate objectives with surrogate anchors, the same
surfaces with anchors recomputed from real out-of-fold optima, and metamodel-free on the real objectives — revalidate
every returned candidate on the true objectives, and score everything against an empirical Pareto reference whose
sampled core the compared methods do not generate. The whole pipeline is replicated over four datasets and 30 outer
partitions each, paired by partition, with the Nadeau–Bengio correction for partition overlap.

Four findings seem to us worth the journal's space:

1. **Anchor placement, not surface accuracy, is where a one-shot surrogate pipeline is most exposed** — and we
   decompose even that. Because a vertex subproblem returns its anchor directly, part of the measured gain is the
   injection of extreme solutions rather than the relocated CHIM; a control that adds the same anchors to the
   surrogate-anchored arm closes a median 80%, 90%, 49% and 1% of the gap on the four datasets. The mechanism is
   therefore mostly set composition on two datasets and mostly geometry on another.
2. **An external reliability gate detects unusable surfaces but not misplaced anchors**, and on one dataset the two
   actively diverge: the partitions where the surrogate-anchored arm collapses are the ones whose selected surface
   fits held-out compositions *better*.
3. **The classical synergism criterion can be satisfied by a surface whose real blend contradicts it.** For the
   leading classifier pair the fitted quadratic meets the criterion in every partition of all four datasets, yet the
   real 50/50 blend beats its better member on only one.
4. **The cost that these methods can optimize disagrees with the cost deployment pays**, changing the winning method
   in up to 24 of 30 partitions.

**We include a result that does not flatter the framework.** Anticipating the obvious question of whether a
conventional optimizer would behave differently on the same objectives, we added NSGA-II at a real-objective
evaluation budget matched per replication to what the metamodel-free arm consumed (realized ratio 0.999996 over 120
runs). It produced a better approximation on both primary endpoints, on all four datasets, under both reference
definitions, and markedly more evenly spaced revalidated fronts. We report this plainly. It bounds what the pipeline
is worth relative to the standard toolbox without disturbing the diagnosis of the pipeline itself, and we think a
reader is better served by knowing it.

The study is fully reproducible: four annotated Git tags freeze the exact states, every stochastic component is
seeded, datasets carry checksums and acquisition commands, the runners are checkpointed and resumable, the library
has 83 unit tests, and one script regenerates every figure and table from the frozen artifacts.

We suggest reviewers with expertise in surrogate-assisted multiobjective optimization, in mixture experiments and
response-surface methodology, and in empirical machine-learning evaluation. We would ask the editor to avoid
reviewers who are co-authors of the 2025 predecessor.

Thank you for your consideration.

Sincerely,

Caio Tertuliano Ribeiro, on behalf of the authors
