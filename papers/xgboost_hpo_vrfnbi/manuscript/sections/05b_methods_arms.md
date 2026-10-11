### 5.7 HISTORICAL-WS-asrun: the archived solver, called unmodified

HISTORICAL-WS-asrun calls the frozen `v0.1.0-dissertation` solver unmodified: normalized weighted
scalarization by SLSQP from ten multistarts, with the maximization orientation, the observed-extrema
normalization box and the asymmetric 20-point weight grid of Section 3.2. The grid has step 0.05 and
runs $\beta_1$ from 0.95 down to 0.00, so it contains the pure-cost vertex but not the pure-quality
vertex (Section 3.3). What the solver receives is computed by this study's
code to the archived specification: full quadratic surfaces in uncoded natural units with backward
elimination at $\alpha = 0.05$, and the component-wise observed extremes of the design rows, both
taken on this study's objectives of Section 5.5, negated so that the solver maximizes (Section 3.5).
It has no surrogate gate, so it never pays the external-validation term, a property of the
historical method; its standalone cost is 108 real evaluations, 88 design rows plus 20
revalidations. The two historical entities are separate identifiers with different costs and are
never pooled (Supplement S6).

### 5.8 HISTORICAL-WS: the same weighted sum under the shared specification

HISTORICAL-WS applies the same weighted-sum scalarization, solver and shared symmetric weight grid
as the rest of the arm set, over the same surrogates fitted in coded units to the shared,
minimization-canonicalized objectives. It differs from WS-S in exactly one respect: it retains the
historical observed-extrema normalization box instead of the payoff-matrix reference, so the
HISTORICAL-WS $\to$ WS-S contrast isolates specification and normalization and nothing else.
Standalone cost 186 real evaluations: 88 design, 78 external validation, 20 revalidations.

### 5.9 WS-S: weighted sum over the surrogate payoff reference

WS-S minimizes a weighted sum of min–max normalized surrogate predictions over the shared
symmetric weight grid, normalized against the payoff reference rather than observed extremes. The
utopia end is the payoff matrix diagonal; the other end is the *pseudo-nadir*, its row-wise
maximum. The pseudo-nadir is used because it is what the CHIM construction itself induces, so WS-S
and NBI-S share one reference object and differ only in what they do with it. Each objective's worst
surrogate value over the box, found by anti-optimization, is also recorded per replication (the
frozen protocol calls it the true nadir, although it is not the Pareto nadir); the protocol declared
the contrast's sensitivity to using it as the far end of the box a secondary analysis, but that
analysis was not run and is not reported (Section 8.7). Standalone cost 186.

### 5.10 NBI-S: canonical Normal Boundary Intersection

Surrogate anchors are obtained by per-objective minimization of the fitted surfaces over the coded
box. With $q = 2$ minimized objectives, $F(x)$ the vector of their fitted surfaces and anchor
vectors $x^{*i}$, the payoff matrix has columns $F(x^{*i})$ and the utopia point $F^{*}$ on its
diagonal; $\Phi$ is the payoff matrix with $F^{*}$ subtracted from each column, so $\Phi$ has columns
$F(x^{*i}) - F^{*}$ and a zero diagonal. The Convex Hull of Individual Minima is $\{\Phi\beta : \beta \ge 0,\ \mathbf{1}^{\top}\beta
= 1\}$, and the quasi-normal direction is

$$\hat{n} \;=\; -\,\frac{\Phi\mathbf{1}}{\lVert \Phi\mathbf{1} \rVert}. \tag{2}$$

Each weight vector $\beta$ on the shared grid defines the subproblem $\max_{x,t} t$ subject to
$\Phi\beta + t\,\hat{n} = F(x) - F^{*}$, $x$ in the coded box, solved by SLSQP. At $q = 2$ the CHIM
extent $\lVert \Phi_{:,1} - \Phi_{:,2} \rVert$ *is* the segment along which the subproblems
distribute their targets; it follows from $\Phi$, which is recorded per replication, and its use as
a diagnostic in Section 6.4.1 is post hoc.

Three diagnostics are recorded because this geometry can fail quietly. Every subproblem records
the solver's success flag and its equality residual, and the certified fraction is published per
arm, since an infeasible subproblem still returns a plausible-looking iterate. Because the two
integer hyperparameters are rounded at evaluation time while the constraint is certified at a
continuous point, each subproblem also records the objective displacement induced by rounding,
$\lVert F(x_{\text{cont}}) - F(x_{\text{round}}) \rVert$ on the fitted surfaces (Supplement S12.2).
The frozen protocol also required each subproblem to be solved again with those dimensions fixed at
their rounded values, to certify the point actually evaluated; that second solve was not
implemented, so the certified fraction refers to the continuous solution only (Section 8.7). And
because a backward-eliminated quadratic is often minimized on a box corner, two anchors can coincide
and leave $\Phi$ rank-deficient; $\Phi$, its rank and condition number and an anchor-coincidence flag
are recorded, and under the frozen protocol a rank-deficient replication is excluded from the
geometry contrasts with the count published. None occurred: $\Phi$ had rank 2 and no anchors
coincided in all 240 NBI-S and NBI-R arm-units. Standalone cost 186; WS-S $\to$ NBI-S changes the
scalarization geometry and nothing else (Figure 1).

### 5.11 NBI-R: NBI with empirical-real anchors

NBI-R is NBI-S with anchors and payoff matrix obtained by direct search on the real objectives
under a pre-declared per-objective budget: for each objective, the best of 100 configurations drawn
uniformly at random in the coded box, 200 real evaluations per replication in total. Those
points are the best found within that budget and are never described as certified optima. Because
the utopia point $F^{*}$, $\Phi$ and $\hat{n}$ are all built from each arm's own anchors, they move
with the anchor source, and so does every subproblem target $F^{*} + \Phi\beta$; that displacement is
the intervention under test, not a confound. For NBI-R, $F^{*}$ and $\Phi$ are computed from the real
objectives measured at the empirical anchors, whereas its subproblem equality constraint, as in
NBI-S, is imposed on the fitted surfaces. The anchor search is charged to this arm alone, taking its
standalone cost to 386.

**Contrast isolation, enforced at run time.** For each arm a `_contrast_fingerprint` is computed over
the solver configuration, candidate realizer, weight grid, surrogate identities and reference, and
`_require_same` raises a `MethodologicalFailure` at the arm's own call site if any component differs
beyond the one mechanism that contrast declares. The rule runs in the other direction too: an
identical utopia point across NBI-S and NBI-R, or across WS-S and HISTORICAL-WS, would mean the
contrast varies nothing, and that also fails. The campaign completed 120 of 120 units with zero
failures in 9 h 46 min.

### 5.12 Controls and baselines

The *anchor-injection control* rescores NBI-S's own real-revalidated set augmented with the same
empirical anchors NBI-R receives, changing nothing else. That set is a superset of NBI-S's, so
its effect on the indicator is non-negative by construction. It was frozen on the premise that NBI-R
returns its own anchors at the vertex weights; in execution NBI-R returned none of its 240 empirical
anchors, because its targets are built from real measurements while its equality constraint is
imposed on the fitted surfaces (Section 5.11). The control therefore bounds what including those
anchors alone adds to NBI-S's set rather than decomposing the observed NBI-S $\to$ NBI-R gap.

The comparator budget was fixed before the confirmatory campaign as the maximum standalone cost
over arms, NBI-R's 386 real evaluations; the rule dates from protocol v1 and its value from protocol
v2, both of 2026-09-13 (Supplement S1). Six direct-search comparators run at that budget: coarse
grid, random search and four single-objective runs (Bayesian optimization and TPE, each run once on
quality alone and once on cost alone). Every comparator evaluates its configurations on the real
learner only, never on the design's fitted surfaces, which is the sense in which it is direct. Both
single-objective methods are this study's own implementations: Bayesian optimization starts from 77
uniformly drawn configurations and then, one evaluation at a time, refits a Gaussian process (a
constant times a Matérn kernel with $\nu = 2.5$, plus white noise) and evaluates the candidate with
the largest expected improvement among 512 uniform draws; TPE is a simplified TPE-style density
ratio that scores 512 uniform draws by the ratio of isotropic kernel densities fitted to the best
quarter of the configurations evaluated so far and to the rest. Not designed to return a front, the
four single-objective runs enter neither the front-indicator table nor the AUGMENTED reference and
are reported only by the best value each reached on its own objective (Supplement S13.1).
NSGA2-MATCHED is matched to the same budget, not pairwise to NBI-S: it runs at $32 \times 12 = 384$,
two evaluations short of 386, a shortfall that is published rather than absorbed. NSGA-II is the
only comparator whose search is guided by both objectives; no multiobjective Bayesian optimization
method was run, so no arm, NBI included, is compared with that class of methods. NSGA2-UNMATCHED
receives ten times as many, 3,840 evaluations, on one replication per dataset and is a context
baseline only: no fairness claim attaches to it, and it enters no budget-matched comparison and
neither reference.

The asymmetry this creates is stated wherever comparator results appear: WS-S and NBI-S cost 186
standalone, so the comparators received roughly twice the real evaluations those arms require. The
rule is matched-to-the-most-expensive-arm, not matched-to-each-arm, and is not revised now
(Supplement S7).


**The grid comparator's composition.** At the frozen 386-evaluation budget over seven
factors the coarse grid admits $\lfloor 386^{1/7}\rfloor = 2$ levels per factor, so it
is a 128-point two-level corner mesh plus 258 uniformly sampled points — **67% of the
comparator is uniform random padding**, and the implementation reports the split rather
than describing the whole set as a grid. The mesh also contains the design's 64
factorial corners, so **64 of the grid's 386 evaluations re-measure points that are
themselves among the 88 design rows inside the CORE reference**. The grid is therefore
scored partly against its own points, which favours it on a CORE-relative indicator
relative to arms whose candidates come from surrogate subproblems and coincide with a design
corner, if at all, only in some units. This is a property of the frozen
comparator construction, disclosed here and carried into every comparison against it.

### 5.13 Real-model revalidation

No indicator in this study is computed on a surrogate prediction. Every candidate returned by every
arm is re-evaluated on the real learner under the replication's own partition and seed, by the same
stratified five-fold cross-validation used for the design, and all front indicators are computed on
those revalidated points. Within a replication every method shares the outer split, the
cross-validation folds and the XGBoost training seed, all set by the replication's seed, so the
design is paired and a repeated request returns the same evaluation; only the baselines' candidate
streams are seeded per `(dataset, replication, method, stage)` (Section 9.2). Five holdout
confirmations per unit are audit-only and steer nothing. The campaign charged 396,120 logical
evaluations (the ledger unit that enters every fairness comparison), realized as 377,316 unique
physical evaluations at a 4.75% cache hit rate, each a five-fold cross-validation (five XGBoost
fits) except the single-fit holdout confirmations; the three figures are kept distinct.

### 5.14 References and metrics

The CORE reference is the 88 design rows plus the 200 anchor-search rows, 288 points before
Pareto filtering, to which no compared method contributes; it is primary. The AUGMENTED reference
adds the points every compared method is scored on and is a mandatory sensitivity, reported for
every contrast beside its self-grading share of the front (Supplement S9.2), never as an alternate
primary reference. Where the two references disagree, that disagreement is itself a result and is
reported prominently.

The primary endpoint is the *CORE-relative hypervolume ratio*. In every unit (one dataset and
replication), every indicator is computed on objectives min–max normalized by the range of the
reference front, for the primary endpoint the CORE front. The hypervolume is computed exactly with
reference point $1.1$ on each normalized axis, without clipping values better than the CORE front's
best, and the ratio is $\mathrm{HV}(\tilde{S})/\mathrm{HV}(\tilde{R})$, where $\tilde{R}$ is the
normalized CORE front and $\tilde{S}$ the normalized non-dominated subset of the points a method is
scored on: an arm's real-revalidated candidates, or every evaluation of the grid, random-search and
matched NSGA-II comparators. The box and the reference point depend on the CORE front alone, so
every method scored in a unit shares them, whatever normalization its own scalarization uses, and
none can move them; a paired difference is therefore a hypervolume difference divided by that unit's
CORE hypervolume. The AUGMENTED ratio is computed in the same way in the AUGMENTED front's own box;
that front weakly dominates every scored point, so the ratio cannot exceed 1, and switching
references changes the normalization as well as the reference set unless the two fronts span the
same range. The box rule and the reference point were fixed in the scoring code before the
confirmatory campaign. CORE is a finite, method-independent empirical set of 288 points, not the
true Pareto front; a CORE ratio above 1 is expected and means the candidate set improved on that
finite reference in the hypervolume sense, dominating a larger normalized region than the CORE
front, which it can do without weakly dominating every CORE-front point; the endpoint is never
presented as a fraction of true Pareto hypervolume. IGD$^{+}$, generational distance, Schott spacing
and the joint non-dominated fraction are secondary and descriptive: Supplement S10 gives their
medians, the intervals declared for them are not reported (Section 8.7), and the one post hoc test
is labelled as such (Section 6.3.1). Schott spacing is undefined on a single-point front, where NaN is correct; those 140 cells, on
the 70 method-by-reference blocks whose front has one point, are counted and excluded from spacing
summaries rather than propagated through a median (Supplement S10).

### 5.15 Statistical analysis

Comparisons are paired by replication, $R = 30$ per dataset. The descriptive triple is primary:
the median paired difference with a percentile bootstrap interval, win/tie/loss counts with Wilson
intervals, and the matched-pairs rank-biserial correlation; win/tie/loss counts are read as a
direction only when one arm wins at least 21 of the 30 replications, the smallest count whose Wilson
interval excludes one half. The
significance test is secondary: a two-sided Wilcoxon signed-rank test on the 30 paired differences,
zero differences discarded, Holm-corrected within each dataset over a family of three identifying
contrasts on one primary indicator; "Holm-significant" means a Holm-adjusted $p < 0.05$ in that
family. The frozen protocol fixed the family and the within-dataset Holm correction and named only
the Nadeau–Bengio corrected $t$-test, as a sensitivity; the signed-rank form was first recorded in
the analysis code committed with the confirmatory results, and because that choice cannot be dated
before the first result was seen it is reported as post hoc. The dataset is the unit of
generalization and there is no pooling: the three primary-panel datasets are never combined into a
90-replication analysis, and Spambase's identical comparisons form a separate boundary-control
family, declared before the confirmatory campaign (Section 5.1). A claim that an effect holds on all
three primary-panel datasets is therefore a conjunction of per-dataset results: it is made only if
the effect is resolved in each dataset on its own (for the secondary test, within that dataset's
Holm family), so it needs no correction across datasets. A count such as "two of the three" is a
tally of per-dataset results, not a pooled test, and agreement across the three datasets is a
pattern, not a tested claim that the effect holds beyond them.

As a further sensitivity, the Nadeau–Bengio correction is derived rather than imported. With
$\mathrm{SE}_{\text{corr}}^{2} = (1/R + n_{\text{test}}/n_{\text{train}})\,s^{2}$, $R = 30$ and an
80/20 outer split ($n_{\text{test}}/n_{\text{train}} = 0.25$),

$$\frac{\mathrm{SE}_{\text{corr}}}{\mathrm{SE}_{\text{naive}}} \;=\; \sqrt{\frac{1/30 + 0.25}{1/30}}
\;=\; \sqrt{8.5} \;=\; 2.9155 . \tag{3}$$

This is not an equicorrelation reading of $\rho = 0.25$, which would give 3.3166. It is a sensitivity
and never the study's significance test: the correction was derived for a
learner's generalization error under repeated resampling, and a Pareto quality indicator computed on
a returned set is not that quantity. The surrogate reliability gate is a covariate, not a filter:
every arm ran at every replication whatever the gate said, and every primary comparison is also
reported conditioned on gate status (Supplement S11). All conclusions are scoped to these four datasets,
this surrogate architecture, two objectives and this budget.
