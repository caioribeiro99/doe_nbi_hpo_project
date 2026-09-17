### 5.7 HISTORICAL-WS-asrun — the archived solver, called unmodified

`HISTORICAL-WS-asrun` executes the frozen dissertation pipeline (tag `v0.1.0-dissertation`) without
modification: maximization orientation, uncoded natural-unit quadratic response surfaces with
backward elimination at $\alpha = 0.05$, a normalization box built from the component-wise observed
extremes of the design rows, an asymmetric 20-point weight grid at step 0.05 — running $\beta_1$ from
0.95 down to 0.00, so it contains the pure-cost vertex but not the pure-quality vertex — and SLSQP
from ten multistarts. It has no surrogate gate, so it never pays the external-validation term;
its standalone cost is 108 real evaluations, 88 design rows plus 20 revalidations.

One fact is recorded here once and not revisited: a code-level reconstruction establishes that the
archived implementation performed front construction by normalized weighted scalarization, while the
historical text described the procedure in Normal Boundary Intersection terms. The arm is reproduced,
not repaired, and that reconstruction is what makes the decomposition in §5.8–§5.11 possible.
The two historical entities are separate identifiers with different costs (Supplement S6).

### 5.8 HISTORICAL-WS — the same weighted sum under the shared specification

`HISTORICAL-WS` applies the same weighted-sum scalarization, solver and shared symmetric weight grid
as the rest of the arm set, over the same gated surrogates fitted in coded units to the shared,
minimization-canonicalized objectives. It differs from `WS-S` in exactly one respect: it retains the
historical observed-extrema normalization box instead of the payoff-matrix reference, so the
`HISTORICAL-WS` $\to$ `WS-S` contrast isolates specification and normalization and nothing else.
Standalone cost 186 real evaluations: 88 design, 78 external validation, 20 revalidations.

### 5.9 WS-S — weighted sum over the surrogate payoff reference

`WS-S` minimizes a weighted sum of min–max normalized surrogate predictions over the shared
symmetric weight grid, normalized against the payoff reference rather than observed extremes. The
utopia end is the payoff matrix diagonal; the other end is the **pseudo-nadir**, its row-wise
maximum. The pseudo-nadir is used because it is what the CHIM construction itself induces, so `WS-S`
and `NBI-S` share one reference object and differ only in what they do with it. The true nadir is recorded per
replication and the contrast's sensitivity to it is a reported secondary analysis. Standalone cost
186.

### 5.10 NBI-S — canonical Normal Boundary Intersection

Surrogate anchors are obtained by per-objective minimization of the fitted surfaces over the coded
box. With $q = 2$ minimized objectives and anchor vectors $x^{*i}$, the payoff matrix $\Phi$ has
columns $F(x^{*i}) - F^{*}$, where the utopia point $F^{*}$ is the matrix diagonal. The Convex Hull of Individual Minima is $\{\Phi\beta : \beta \ge 0,\ \mathbf{1}^{\top}\beta
= 1\}$, and the quasi-normal direction is

$$\hat{n} \;=\; -\,\frac{\Phi\mathbf{1}}{\lVert \Phi\mathbf{1} \rVert}.$$

Each weight vector $\beta$ on the shared grid defines the subproblem $\max_{x,t} t$ subject to
$\Phi\beta + t\,\hat{n} = F(x) - F^{*}$, $x$ in the coded box, solved by SLSQP. At $q = 2$ the CHIM
extent $\lVert \Phi_{:,0} - \Phi_{:,1} \rVert$ *is* the segment along which the subproblems
distribute their targets, and it is recorded per replication.

Three diagnostics are recorded because this geometry can fail quietly. Every subproblem records
`success` and its equality residual and the certified fraction is published per arm, since an
infeasible subproblem still returns a plausible-looking iterate. Because the two integer hyperparameters are rounded at
evaluation time while the constraint is certified at a continuous point, each subproblem also reports
the objective displacement induced by rounding and is re-solved with those dimensions fixed. And
because a backward-eliminated quadratic is often minimized on a box corner, two anchors can coincide
and leave $\Phi$ rank-deficient; anchor vectors, rank and condition number are recorded, and a
rank-deficient replication is excluded with the count published. Standalone cost 186; `WS-S` $\to$
`NBI-S` changes the scalarization geometry and nothing else (Figure 1).

### 5.11 NBI-R — NBI with empirical-real anchors

`NBI-R` is `NBI-S` with anchors and payoff matrix obtained by direct search on the real objectives
under a pre-declared per-objective budget, 200 real evaluations per replication in total. Those
points are the best found within that budget and are never described as certified optima. Because
$\hat{n}$ is derived from each arm's own payoff matrix, it necessarily differs between `NBI-S` and
`NBI-R`; that displacement is the intervention under test, not a confound. The anchor search is
charged to this arm alone, taking its standalone cost to 386.

**Contrast isolation, enforced at run time.** For each arm a `_contrast_fingerprint` is computed over
the solver configuration, candidate realizer, weight grid, surrogate identities and reference, and
`_require_same` raises a `MethodologicalFailure` at the arm's own call site if any component differs
beyond the one mechanism that contrast declares. The rule runs in the other direction too: an
identical quasi-normal across `NBI-S` and `NBI-R` would mean the contrast varies nothing, and that
also fails. The campaign completed 120 of 120 units with zero failures in 9 h 46 min.

### 5.12 Controls and baselines

The **anchor-injection control** rescores `NBI-S`'s own real-revalidated set augmented with the same
empirical anchors `NBI-R` receives, changing nothing else. That set is a superset of `NBI-S`'s, so
its effect on the indicator is non-negative by construction; it separates the part of any `NBI-S`
$\to$ `NBI-R` gap attributable to anchor set composition from the part attributable to the relocated
payoff matrix and CHIM.

Six direct-search comparators run at the frozen comparator budget of 386 real evaluations: coarse
grid, random search, Bayesian optimization on quality and on cost, and TPE on quality and on cost.
The four single-objective ones are reported on their own endpoint only and enter neither the
front-indicator table nor the AUGMENTED reference. `NSGA2-MATCHED` runs at $32 \times 12 = 384$, a
two-evaluation shortfall that is published rather than absorbed. `NSGA2-UNMATCHED` receives ten times
that budget on one replication per dataset and is a **context baseline only**: no fairness claim
attaches to it, and it enters no budget-matched comparison and neither reference.

That budget was frozen before any result existed as the maximum over arms, `NBI-R`'s 386, and the
asymmetry is stated wherever comparator results appear: `WS-S` and `NBI-S` cost 186 standalone, so
the comparators received roughly twice the real evaluations those arms require. The rule is
matched-to-the-most-expensive-arm, not matched-to-each-arm, and is not revised now
(Supplement S7).


**The grid comparator's composition.** At the frozen 386-evaluation budget over seven
factors the coarse grid admits $\lfloor 386^{1/7}\rfloor = 2$ levels per factor, so it
is a 128-point two-level corner mesh plus 258 uniformly sampled points — **67% of the
comparator is uniform random padding**, and the implementation reports the split rather
than describing the whole set as a grid. The mesh also contains the design's 64
factorial corners, so **64 of the grid's 386 evaluations re-measure points that are
themselves among the 88 design rows inside the CORE reference**. The grid is therefore
scored partly against its own points, which favours it on a CORE-relative indicator
relative to arms that propose elsewhere in the box. This is a property of the frozen
comparator construction, disclosed here and carried into every comparison against it.

### 5.13 Real-model revalidation

No indicator in this study is computed on a surrogate prediction. Every candidate returned by every
arm is re-evaluated on the real learner under the replication's own partition and seed, by the same
stratified five-fold cross-validation used for the design, and all front indicators are computed on
those revalidated points. Seeds are derived per `(dataset, replication, method, stage)`,
retaining the replication seed so that the paired design still pairs. Five holdout confirmations per
unit are audit-only and steer nothing. The campaign charged
396,120 logical evaluations — the ledger unit that enters every fairness comparison — realized as
377,316 unique physical fits at a 4.75 % cache hit rate; the three figures are kept distinct.

### 5.14 References and metrics

The **CORE** reference is the 88 design rows plus the 200 anchor-search rows, 288 points before
Pareto filtering, to which no compared method contributes; it is primary. The **AUGMENTED** reference
adds every compared method's real-revalidated candidates and is a mandatory sensitivity, reported for every
contrast beside its self-grading share of the front, never as an alternate primary test. Where the
two references disagree, that disagreement is itself a result and is reported prominently.

The primary endpoint is the **CORE-relative hypervolume ratio**. CORE is a finite,
method-independent empirical set of 288 points, not the true Pareto front; a ratio above 1 is
expected and means the candidate set improved on that finite reference, and the endpoint is never
presented as a fraction of true Pareto hypervolume. IGD$^{+}$, generational distance, Schott spacing
and the joint non-dominated fraction are secondary and descriptive, reported with intervals and no
tests. Schott spacing is undefined on a single-point front, where NaN is correct; those 140 cells, on
the 70 method-by-reference blocks whose front has one point, are counted and excluded from spacing
summaries rather than propagated through a median (Supplement S10).

### 5.15 Statistical analysis

Comparisons are paired by replication, $R = 30$ per dataset. **The descriptive triple is primary**:
the median paired difference with a percentile bootstrap interval, win/tie/loss counts with Wilson
intervals, and the matched-pairs rank-biserial correlation. The significance test is **secondary**.
Holm correction is applied **within each dataset**, over a family of three identifying contrasts on
one primary indicator. **The dataset is the unit of generalization and there is no pooling**: the
three primary-panel datasets are never combined into a 90-replication analysis, and Spambase's
identical comparisons form a separate, prospectively declared boundary-control family.

As a further sensitivity, the Nadeau–Bengio correction is derived rather than imported. With
$\mathrm{SE}_{\text{corr}}^{2} = (1/R + n_{\text{test}}/n_{\text{train}})\,s^{2}$, $R = 30$ and an
80/20 outer split ($n_{\text{test}}/n_{\text{train}} = 0.25$),

$$\frac{\mathrm{SE}_{\text{corr}}}{\mathrm{SE}_{\text{naive}}} \;=\; \sqrt{\frac{1/30 + 0.25}{1/30}}
\;=\; \sqrt{8.5} \;=\; 2.9155 .$$

This is not an equicorrelation reading of $\rho = 0.25$, which would give 3.3166. It is a sensitivity
and never the primary test: the correction was derived for a
learner's generalization error under repeated resampling, and a Pareto quality indicator computed on
a returned set is not that quantity. The surrogate reliability gate is a covariate, not a filter:
every arm ran at every replication whatever the gate said, and every primary comparison is also
reported conditioned on gate status. All conclusions are scoped to these four datasets,
this surrogate architecture, two objectives and this budget.
