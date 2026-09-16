## 3. Historical method reconstruction

### 3.1 What the archived implementation computes

The historical pipeline was reconstructed by reading and executing the code frozen at tag
`v0.1.0-dissertation`, not by reading the dissertation text. Where text and code disagree, this
study reports the code, because the code produced the archived numbers. No historical artifact was
edited; the tag was extracted to a scratch directory and read there.

The frozen solver exposes a single routine. For each weight pair $(\beta_1,\beta_2)$ on a fixed
grid it maximizes $\sum_j \beta_j \bar f_j(x)$ by SLSQP from ten multistarts, where
$\bar f_j = (f_j - \text{nadir}_j)/(\text{utopia}_j - \text{nadir}_j)$. That is normalized weighted
scalarization. None of the structural elements of Normal Boundary Intersection is present: there are
no per-objective anchor minimizers, no payoff matrix $\Phi$, no convex hull of individual minima, no
quasi-normal direction $\hat n = -\Phi\mathbf{1}/\lVert\Phi\mathbf{1}\rVert$, no
$\max t \ \text{s.t.}\ \Phi\beta + t\hat n = F(x)$ subproblem, and no auxiliary variable $t$ at all.
In this code $\beta$ indexes a weight grid rather than a CHIM coordinate. An optional inequality
keeps both predictions inside the observed $[\text{nadir},\text{utopia}]$ box; that is a feasibility
restriction on the prediction range, not a scalarization structure.

The historical text uses NBI terminology for this procedure. The divergence between the text and
the implementation is stated here once, as provenance, and is not the subject of this paper. It was
recorded in the original project's own methodology log before the present reconstruction began, and
divergences of this kind between a write-up and research code are ordinary. Nothing in this study
depends on which section of the historical text states the NBI subproblem, and no claim is made
about any other implementation of NBI in the surrounding literature.

### 3.2 Four properties carried forward verbatim

Four further properties of the archived implementation are relevant because each could, on its own,
displace a returned front [TAB:historical_properties].

| Property | Archived implementation |
|---|---|
| Front construction | normalized weighted scalarization over a $\beta$ grid |
| Normalization reference | component-wise observed extrema of the 88 design rows |
| Weight grid | asymmetric, 20 points at step 0.05 |
| Objective orientation | maximization, larger-is-better objectives |
| Surrogate fit | full quadratic response surfaces in uncoded natural units |
| Surrogate adequacy | no external validation and no gate |

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

### 3.3 Reproduction, not reimplementation

The study does not reimplement the historical method. `HISTORICAL-WS-asrun` calls the frozen
`v0.1.0-dissertation` solver unmodified and retains every property in the table above, including the
asymmetric grid, the maximization orientation, the uncoded surfaces, the observed-extrema
normalization and the absence of a surrogate gate. It is charged no surrogate-validation budget,
because the historical pipeline has no gate; that is a property of the method and is reported as one.
Its standalone budget is 108 real evaluations against 186 for the shared-specification arms. A
reimplementation would silently substitute the reconstructor's reading of the method for the method,
which is exactly the conflation this study is built to avoid.

### 3.4 The specification-matched control

Faithfulness and comparability are separate requirements, so the historical weighted sum is run
twice. `HISTORICAL-WS` applies the same weighted-sum solver under the shared specification — the
symmetric weight grid with both vertices, minimization orientation, coded-unit surrogates, and the
gated surrogates WS-S itself uses — and differs from WS-S in the normalization reference alone
[FIG:arm_lattice]. The contrast `HISTORICAL-WS` $\rightarrow$ `WS-S` therefore identifies
normalization as a single factor, isolated from geometry and from anchor provenance. The two
historical entities are separate registry identifiers with different budgets and are never mixed in
one table. Both have every returned candidate revalidated on the real learner, as the archived
protocol did.
