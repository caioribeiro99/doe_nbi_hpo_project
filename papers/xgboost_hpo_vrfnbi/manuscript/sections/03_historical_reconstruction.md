## 3. Historical method reconstruction

### 3.1 What the archived implementation computes

The historical pipeline (Ribeiro 2026) follows the sequence set out at the start of the
Introduction, with Varimax-rotated factor objectives as the reduced objectives. Its solver, frozen
at tag `v0.1.0-dissertation` (Section 9.1), exposes a single routine. For each weight pair
$(\beta_1,\beta_2)$ on a fixed
grid it maximizes $\sum_j \beta_j \bar f_j(x)$ by SLSQP from ten multistarts, where
$\bar f_j = (f_j - \text{nadir}_j)/(\text{utopia}_j - \text{nadir}_j)$. That is normalized weighted
scalarization. None of the structural elements of Normal Boundary Intersection is present: there are
no per-objective anchor minimizers, no payoff matrix $\Phi$, no convex hull of individual minima, no
quasi-normal direction $\hat n = -\Phi\mathbf{1}/\lVert\Phi\mathbf{1}\rVert$, no
$\max t \ \text{s.t.}\ \Phi\beta + t\hat n = F(x)$ subproblem, and no auxiliary variable $t$ at all.
In this code $\beta$ indexes a weight grid rather than a CHIM coordinate. An optional inequality
keeps both predictions inside the observed $[\text{nadir},\text{utopia}]$ box; that is a feasibility
restriction on the prediction range, not a scalarization structure.

The dissertation describes this procedure in NBI terms. Where its text and the frozen code
disagree, this study follows the code, which is what can be re-executed and checked. The
divergence is reported as provenance, not as a finding: it was recorded in the project's own
methodology log (`docs/METHODOLOGY_DECISIONS.md`, entry D1) before this study examined the code,
and no claim is made about any other implementation of NBI.

### 3.2 Four properties carried forward verbatim

Four further properties of the archived implementation are relevant because each could, on its own,
displace a returned front (Table 2).

| Property | Archived implementation |
|---|---|
| Front construction | normalized weighted scalarization over a $\beta$ grid |
| Normalization reference | component-wise observed extrema of the 88 design rows |
| Weight grid | asymmetric, 20 points at step 0.05 |
| Objective orientation | maximization, larger-is-better objectives |
| Surrogate fit | full quadratic response surfaces in uncoded natural units |
| Surrogate adequacy | no external validation and no gate |

**Table 2.** Properties of the archived implementation, reproduced rather than
repaired. `HISTORICAL-WS-asrun` calls the frozen `v0.1.0-dissertation` solver
unmodified.

The normalization box is built from the component-wise maxima and minima of the design rows. These
are not the objective values attained at the other objective's minimizer, which is what a payoff
matrix supplies, so the historical method differs from canonical NBI in two independent respects —
geometry and reference — rather than one.

The weight grid is asymmetric. It sweeps $b$ from $0.05$ to $1.00$ and forms $(1-b, b)$, so the
pair $(0.00,1.00)$ is present and the pair $(1.00,0.00)$ is absent: the subproblem is solved at the
pure-cost vertex and never at the pure-quality vertex, and the returned set is short at the quality
end by construction on every run. The property is stated because any comparison against a symmetric
grid would otherwise attribute its consequence to the method.

The objectives are oriented so that larger is better and the solver maximizes, whereas the canonical
implementation used by the other arms canonicalizes to minimization before building a payoff matrix.
The response surfaces are fitted in uncoded natural units across factor ranges spanning $0.29$ to
$650$.

### 3.3 Calling the archived solver rather than reimplementing it

HISTORICAL-WS-asrun calls the frozen `v0.1.0-dissertation` solver unmodified and keeps every
property in Table 2. The solver is the only archived routine the arm calls: the uncoded
surfaces and the observed-extrema box it receives follow the archived specification but are
computed by this study's code, on this study's objectives (Section 3.5). The solver is called at its default grid
step of 0.05, which gives the 20-point grid of Table 2; the archived replication runner defaults
instead to a step of 0.02 (50 pairs), and which of the two steps produced the dissertation's
reported numbers could not be established from the archived code. Because the historical
pipeline has no surrogate gate, the arm is charged no surrogate-validation budget, a property of
the method that is reported as one: its standalone cost is 108 real evaluations, 88 design rows
plus 20 revalidations, against 186 for the shared-specification arms. Reimplementing the solver
would silently substitute this study's reading of the method for the method itself, which is
exactly the conflation this study is built to avoid.

### 3.4 The specification-matched control

Faithfulness and comparability are separate requirements, so the historical weighted sum is run
twice. `HISTORICAL-WS` applies the same weighted-sum solver under the shared specification — the
symmetric weight grid with both vertices, minimization orientation, coded-unit surrogates, and the
gated surrogates WS-S itself uses — and differs from WS-S in the normalization reference alone
(Figure 1). The contrast `HISTORICAL-WS` $\rightarrow$ `WS-S` therefore identifies
normalization as a single factor, isolated from geometry and from anchor provenance. The two
historical entities are separate registry identifiers with different budgets and are never mixed in
one table. Both have every returned candidate revalidated on the real learner, as the archived
protocol did.

### 3.5 What is not carried forward

The reconstruction preserves the archived *optimizer*, not its objective space. The
archived pipeline optimized four threshold quality metrics — accuracy, precision,
recall and specificity — against mean per-fold wall-clock time as its cost response,
and used no ranking or calibration metric. This study optimizes a Varimax-rotated
composite over six quality responses, which adds ROC AUC and log loss, against total
leaf count.

The cost substitution is deliberate and is a limitation of the comparison. A wall-clock
response is not reproducible from a seed: it varies with machine load, thread
scheduling and hardware, so a replicated design cannot hold it fixed across 30
partitions. Total leaf count is a deterministic model-complexity proxy that the same
configuration and seed reproduce exactly. The consequence is that `HISTORICAL-WS-asrun`
reproduces the archived *solver, normalization, weight grid and surface
parameterization* faithfully, while the objectives it optimizes are this study's, not
the dissertation's. Differences between it and the other arms are therefore attributable
to the optimizer's construction and not to the objective space, but no statement here
should be read as reproducing the archived study's reported outcomes.
