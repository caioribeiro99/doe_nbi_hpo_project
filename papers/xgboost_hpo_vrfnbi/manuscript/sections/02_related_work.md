## 2. Related work and methodological lineage

### 2.1 Normal Boundary Intersection

Das and Dennis (1998) introduced Normal Boundary Intersection (NBI) as a scalarization that
distributes subproblems geometrically rather than by weight. Each objective is minimized individually
to obtain an anchor; the anchors fill the columns of the payoff matrix $\Phi$, whose diagonal is the
utopia point. The convex hull of individual minima (CHIM) is the simplex spanned by the anchors, a
quasi-normal direction $\hat{n} = -\Phi\mathbf{1}/\lVert\Phi\mathbf{1}\rVert$ points from the CHIM
toward the attainable set, and each subproblem maximizes the step $t$ along $\hat{n}$ from a CHIM
point under an equality constraint. At $q = 2$, as here, the CHIM is a segment whose extent
$\lVert\Phi_{:,0} - \Phi_{:,1}\rVert$ is the interval along which subproblem targets are placed.

The property that motivates NBI is that an even spread of CHIM points tends to produce an even
spread of boundary points, including in regions a weighted sum cannot reach, and that this behaviour
is invariant to affine rescaling of the objectives. Its limits are equally well established in the
original paper: subproblem solutions are boundary points of the attainable set and need not be
Pareto-optimal, so a dominance filter is required downstream; for $q > 2$ the CHIM simplex need not
cover the whole efficient frontier; and the construction is defined relative to the payoff matrix, so
the anchors are part of the problem statement rather than an implementation detail. That last
dependence is treated here as a separate, measurable mechanism.

### 2.2 Weighted-sum scalarization

The weighted sum minimizes $\sum_k w_k \tilde{f}_k(x)$ over a weight grid. Every solution it returns
is a supported efficient point, lying on the convex hull of the attainable set; efficient points in
non-convex regions of the frontier are unreachable at any weight vector, and uniform weight spacing
does not produce uniform spacing of the returned points. The method is also scale-dependent, so it
requires a normalization box, and the choice of that box — observed extrema over the design rows, a
payoff-matrix utopia with a pseudo-nadir, or a true nadir from anti-optimization — is a modelling
decision separate from the choice of geometry. Separating the two is why a specification-matched
weighted-sum arm runs alongside the historical one.

### 2.3 Design of experiments and response surfaces for hyperparameter tuning

Treating hyperparameter tuning as a designed experiment with a fitted response surface is prior art.
Lujan-Moreno, Howard, Rojas and Montgomery (2018) give the canonical statement — screening factorial
then response surface methodology, on a random forest, single objective — and Vasquez-Ramos et al.
(2025) apply response surface methodology to XGBoost hyperparameters with a Box–Behnken design and a
single response, without a Pareto front, factor reduction, or multiobjective scalarization. Neither
the framing nor its application to XGBoost hyperparameters is claimed here.

### 2.4 Multiobjective hyperparameter optimization

Predictive quality against training or inference cost is the standard multiobjective HPO problem,
surveyed by Morales-Hernández, Van Nieuwenhuyse and Rojas Gonzalez (2023) and Karl et al. (2023),
with dedicated benchmark suites (Eggensperger et al. 2021; Pfisterer et al. 2022) and an agreed
baseline set (Guerrero-Viu et al. 2021). Posing HPO as multiobjective, trading quality against cost,
and returning a Pareto front of configurations are therefore all established, and none is claimed
here. The comparators used below — coarse grid, random search (Bergstra and Bengio 2012), Bayesian
optimization, the tree-structured Parzen estimator, and NSGA-II — follow that literature; multi-fidelity
methods are outside the frozen scope and are named as such in §7.

### 2.5 PCA and Varimax-rotated factor objectives

Reducing correlated responses by principal components or factor analysis, rotating the components by
Varimax, and optimizing the resulting factor scores by NBI is a mature line from the present authors'
group: Costa et al. (2016), Luz et al. (2021), Streitenberger et al. (2022), Pereira et al. (2025);
de Azevedo et al. (2026) names the construction NBI-VRF. The composite quality factor and the
leaf-count complexity factor used here are instances of it, inherited and not proposed.

### 2.6 Pareto indicators and paired inference

Hypervolume is used as the primary indicator because it is the standard Pareto-compliant set
indicator, and IGD⁺ (Ishibuchi, Masuda, Tanigaki and Nojima 2015) as a secondary indicator because,
unlike IGD, it is weakly Pareto compliant and does not reward sets that move away from the reference.
Schott spacing and a joint non-dominated fraction complete the frozen secondary set. All are computed
against a finite empirical reference set, so the primary endpoint is reported throughout as a
**CORE-relative hypervolume ratio**: that reference is not a true Pareto front, and a ratio above one
means the returned set improved on a finite reference. Because the replicated
outer partitions share training data, paired differences across replications are not independent.
Nadeau and Bengio (2003) quantify that variance inflation for resampled estimates; their corrected
test is carried here as a pre-declared sensitivity, not as the primary test, since a set indicator
computed on a returned front is not the generalization error it was derived for.

### 2.7 Methodological lineage

This work follows the first author's master's dissertation, which applied the DoE–response-surface–
NBI-VRF construction to XGBoost hyperparameter optimization on the datasets reused here, and the
group's published formulation in Pereira et al. (2025), which formulates the method using canonical
NBI. A code-level reconstruction of the archived dissertation implementation established that front
construction there used normalized weighted scalarization, although the historical text
used NBI terminology; that is stated once as a fact about the archived code, it is why the arm set
contains both a historical weighted-sum arm and a specification-matched one, and no claim is made
about any implementation not reconstructed here, including that of Pereira et al. (2025). A bounded
search located no work applying NBI or NBI-VRF to hyperparameter optimization; non-location is weaker
than absence, and the transfer is not offered as the contribution. NBI, Varimax-rotated factor
objectives, and design of experiments for hyperparameter tuning are all prior art, and none of them
is claimed here — §2.7 records each element with its owner. What this paper contributes is
the prospectively frozen decomposition of that inherited construction into the three mechanisms it
bundles, described in §3.
