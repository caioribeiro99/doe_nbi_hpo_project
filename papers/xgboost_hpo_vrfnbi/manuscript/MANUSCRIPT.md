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


## 1. Introduction

Surrogate-assisted multiobjective hyperparameter optimization is usually reported as a
pipeline rather than as a set of separable choices. A design of experiments is run, response
surfaces are fitted to reduced objectives, a scalarization is solved over those surfaces on a
weight grid, and the returned candidates are re-evaluated on the real learner. When such a
pipeline is compared against another, the comparison moves several things at once: how the
objectives are normalized and what reference they are measured against; what geometry is used
to construct the front; and where the payoff and anchor information used by that geometry
comes from. A difference in a front indicator can then be attributed to none of them
individually.

These three choices are not interchangeable. The normalization reference fixes the scale on
which any scalarized objective is combined. The construction geometry determines which points
of a nonconvex front are reachable at all: a weighted sum cannot return points in a nonconvex
region, whereas Normal Boundary Intersection (NBI) distributes subproblems along a segment of
the convex hull of individual minima (CHIM) and searches along a quasi-normal direction.
Anchor provenance determines the payoff matrix, and therefore the CHIM itself — surrogate
anchors are free once the surfaces exist, while empirical anchors are bought with real
evaluations. In a pipeline where all three are set together, a reported gain is consistent
with any of them.

This paper does not propose a new optimizer. It reports a prospectively frozen reconstruction
of one such pipeline in which the three mechanisms are separated into single-factor contrasts,
executed at a fixed budget, and reported whether or not the result favours the method under
study. The reconstruction target is an archived dissertation implementation of
surrogate-assisted multiobjective XGBoost tuning over Varimax-rotated factor objectives. A
code-level identity audit of that archived implementation established that its front
construction used normalized weighted scalarization, although the historical text used NBI
terminology. That fact is stated here once because it is the reason the arm set has the shape
it does; it is not itself a contribution and it is not treated as anything other than a
specification detail to be carried forward correctly.

### 1.1 Research questions

Holding the decision space, design, surrogate family, weight grid, candidate realization,
evaluation protocol and budget fixed, and changing exactly one mechanism per contrast:

- **RQ1 — specification and normalization.** Does replacing the historical observed-extrema
  normalization box with a payoff-matrix reference change the returned front?
- **RQ2 — front construction geometry.** At a fixed reference and fixed surrogates, does
  canonical NBI construction differ from specification-matched weighted-sum construction?
- **RQ3 — anchor and payoff provenance.** With NBI geometry held fixed, what happens when the
  payoff matrix and anchors are obtained by direct search on the real objectives instead of
  from the fitted surrogates?

The study answers these on four binary-classification datasets, seven XGBoost hyperparameters,
an 88-run face-centred central composite design, two minimized objectives — a Varimax-rotated
composite quality factor over six canonicalized quality responses and a leaf-count
model-complexity factor — and $R = 30$ replicated outer partitions per dataset, with every
returned candidate re-evaluated on the real learner. The campaign is 396,120 logical
evaluations. The primary endpoint is the CORE-relative hypervolume ratio: hypervolume measured
against a finite, method-independent empirical reference of 288 points that no compared method
contributes to. CORE is not the true Pareto front, and a ratio above 1 simply means a candidate
set improved on that finite reference. Dataset is the generalization unit; nothing is pooled
across datasets.

### 1.2 What the study found

**Geometry separates, on the datasets where the screening said it could.** On the three
datasets prospectively assigned to the primary geometry panel, canonical NBI construction
produced higher CORE-relative hypervolume than specification-matched weighted scalarization:
median paired differences of $+0.0374$ on MAGIC (26/30 replications), $+0.2549$ on Adult
(30/30) and $+0.2281$ on Bank Marketing (24/30), Holm-significant within every one of the three
datasets (Table 4). This is the primary positive result, and its scope is these
datasets, this surrogate architecture, two objectives and this budget. It is also
indicator-specific: the advantage appears in hypervolume and IGD⁺, while the joint
non-dominated fraction comparison is non-significant on all four datasets.

**Anchor provenance did not help, and is reported with the same prominence.** Replacing
surrogate-derived payoff information with the frozen empirical-real anchor procedure did not
improve the confirmatory NBI fronts: median differences of $-0.0741$ on MAGIC, $-0.0192$ on
Adult (a non-detection, interval spanning zero) and $-0.4093$ on Bank Marketing. Two of three
primary datasets degraded. The paid anchor stage bought nothing measurable here.

**A frozen-budget direct grid baseline led NBI-S at the frozen comparator budget.** The
median paired difference favoured the direct grid baseline over NBI-S on all four datasets,
and the grid held the higher marginal median on three of the four — not on Spambase, where
NBI-S is the higher of the two (Table 5). The paired statistic is the within-replication
comparison and the marginal medians are not, so the two need not agree. The budget asymmetry
belongs in the same sentence: the frozen rule matched comparators to the most expensive arm
(NBI-R, 386 real evaluations), while WS-S and NBI-S cost 186 standalone, so the comparators
received roughly twice the real evaluations those arms require. The rule was fixed before any
result existed and is not revised now, and the result is not explained away on that basis.

**Specification and normalization were undetectable.** The RQ1 contrast produced no detectable
change in CORE-relative hypervolume on any primary dataset, reported as a non-detection at
$R = 30$ rather than as an absence of effect.

### 1.3 Contributions

1. A single-factor decomposition of one surrogate-assisted multiobjective HPO pipeline into
   specification and normalization, front-construction geometry, and anchor and payoff
   provenance, with run-time enforcement that each contrast varies only its declared mechanism.
2. A protocol frozen before any confirmatory comparative result was inspected, with its
   dataset roles, endpoints, reference, statistical family and budget ledger fixed in advance,
   and every subsequent change recorded as a numbered amendment.
3. A confirmatory campaign of 396,120 logical evaluations over four datasets and $R = 30$
   paired replications, with every candidate revalidated on the real learner and every headline
   number traceable to a committed artifact.
4. Results reported against interest: the negative anchor-provenance contrast, the grid
   baseline that leads at the frozen budget, and the indicator specificity of the geometry
   effect, each given the same standing as the positive result.
5. A mechanistic reading of the negative contrast — an anchor-injection control together with
   a measured contraction of the CHIM segment — reported as an association within datasets, not
   as a demonstrated cause (Figure 4).


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


## 4. Confirmatory questions and prior commitments

Pipelines of this kind bundle three choices usually made silently: how the objective reference
is normalized, which scalarization geometry builds the front, and where the anchors and payoff
matrix come from. The design separates them by changing one factor at a time, everything else
held fixed and checked at run time by a contrast fingerprint.

**RQ1 — Specification and normalization.** Does replacing the historical observed-extrema
normalization with the payoff-matrix reference change the returned front? Contrast:
HISTORICAL-WS $\rightarrow$ WS-S, differing only in that reference.

**RQ2 — Front-construction geometry.** At a fixed reference, does canonical NBI construction
change the returned front relative to specification-matched weighted scalarization? Contrast:
WS-S $\rightarrow$ NBI-S, differing only in scalarization geometry.

**RQ3 — Anchor and payoff provenance.** Holding NBI geometry fixed, what happens when payoff
and anchor information changes from surrogate-derived to the pre-specified empirical-real
procedure? Contrast: NBI-S $\rightarrow$ NBI-R. The quasi-normal is derived from each arm's own
payoff matrix and moves with the reference by construction: that is the intervention, not a
confound. A pre-declared anchor-injection control and the CHIM diagnostics accompany it.

**RQ4 — The boundary case.** What happens in a dataset identified before the campaign as
lacking the interior front geometry the mechanism acts on? The RQ2 contrast on Spambase,
analysed outside the primary family.

**RQ5 — Direct search.** How does the surrogate-assisted family compare with direct search at the
frozen comparator budget — grid, random, Bayesian optimization, TPE, NSGA-II? That budget is matched to
the most expensive arm, not to each arm, so a comparator receives about twice what
WS-S and NBI-S require standalone. §4 maps each question to its contrast and evidence.

### 4.1 What was frozen

Fixed at protocol tag `xgboost-hpo-protocol-v3`, before any confirmatory arm-level result was
inspected.

*Panel and dataset roles.* From pre-campaign screening on the design rows alone:
MAGIC, Adult and Bank Marketing met the interior-geometry criterion and form the primary
geometry-confirmatory panel; Spambase failed it — a two-point non-dominated set, curvature
undefined — and was retained prospectively as a boundary geometry control. The replacement rule
was considered and deliberately not exercised; Spambase executes in full, excluded only from the
primary inferential family.

*Primary endpoint and reference.* The CORE-relative hypervolume ratio, CORE being a finite
method-independent empirical set no compared method contributes to. The AUGMENTED-relative ratio
is a mandatory sensitivity, never an alternate primary test; a direction disagreement between
references is itself a reported result. IGD$^+$, generational distance, spacing and joint
non-dominated fraction are secondary, reported with intervals and no tests.

*Family and multiplicity.* Within each primary-panel dataset, three contrasts $\times$ one
indicator, Holm-corrected within that dataset; the other arm pairs and all baseline comparisons
are descriptive and uncorrected. Primary evidence is the descriptive triple — median difference
with bootstrap interval, win fraction with Wilson interval, rank-biserial — and the test is
secondary.

*Boundary-control interpretation.* A null geometry difference on Spambase is not evidence
against the mechanism, screening having established that the required interior geometry is
absent; a non-null difference is reported and investigated after completion, without revising
the protocol.

*Budget.* A request-based logical ledger charges every evaluation, cache hit or not. The
comparator budget is the maximum over arms, not matched to each arm: NBI-R, the only arm buying
anchors with real evaluations, costs 386 standalone, WS-S and NBI-S cost 186, and every
comparator receives 386. That rule predates any result and is not revised now; wherever a
baseline comparison appears, the asymmetry appears with it.

*Gate status.* The external surrogate reliability gate is diagnostic, not adaptive: every arm
runs at every replication regardless, and every primary comparison is also reported conditioned
on it.

Throughout, **the dataset is the unit of generalization and nothing is pooled**: the primary
datasets are never combined into a 90-replication analysis, and no conclusion extends beyond
this panel, this surrogate architecture, two objectives and this budget.


## 5. Methods

### 5.1 Datasets and their prospectively assigned roles

Four public binary-classification datasets are used, each obtainable under CC BY 4.0 with a
published SHA-256. MAGIC Gamma Telescope is the continuity dataset, the one the historical
pipeline used; Adult / Census Income contributes scale and mixed types; Bank Marketing
contributes class imbalance with independently constructed categorical features; Spambase is
small and entirely numeric.

Roles were assigned by a screening rule frozen before the campaign, measured on the 88 design
rows of each dataset on one partition, before any arm ran. Two of its four criteria are
structural: a raw-response conflict between the six quality responses and the raw leaf-count
response, and a non-dominated set of design rows with detectable curvature. MAGIC, Adult and
Bank Marketing met all four and form the **primary geometry panel**. Spambase failed criterion 2
with a two-point non-dominated set, which was treated by the screening rule as leaving no interior geometry for the front-construction contrast to act on. Whether that premise held in execution is a result, reported in §6.6, not an assumption carried forward here. It was therefore **retained prospectively as a boundary geometry
control**: the replacement rule was considered and deliberately not exercised, no replacement
dataset was selected, and Spambase runs every arm, budget, seed and all $R = 30$ replications,
excluded only from the primary inferential family. These are pre-campaign screening
measurements, not study results (Supplement S5). The dataset is the unit of
generalization; nothing is pooled.

### 5.2 Decision space

Seven XGBoost hyperparameters are optimized, with the bounds of the reconstructed
historical configuration retained unchanged (Table 1).

| Hyperparameter | Low | High | Type |
|---|---:|---:|---|
| `subsample` | 0.05 | 1.00 | continuous |
| `colsample_bytree` | 0.05 | 1.00 | continuous |
| `colsample_bylevel` | 0.05 | 1.00 | continuous |
| `learning_rate` | 0.01 | 0.30 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.00 | continuous |
| `n_estimators` | 50 | 700 | integer |

**Table 1.** The seven XGBoost hyperparameters and their frozen bounds. All enter
the design in coded units on $[-1,1]$; integers are cast by `int(round(·))` at
evaluation time.

A configuration is written in coded units as $x \in [-1, 1]^7$ and mapped to natural units
by $\text{natural}_i(x) = lo_i + (x_i + 1)(hi_i - lo_i)/2$. `max_depth` and `n_estimators`
are cast by `int(round(·))` at evaluation time, so every surrogate is fitted on a continuous
relaxation whose evaluated points are rounded; the gap is carried rather than repaired, with
each subproblem reporting the objective displacement induced by rounding beside its residual.

### 5.3 The 88-run face-centred central composite design

The design is a version-controlled face-centred central composite in the seven factors, 88 runs,
reused byte-identically across datasets and replications and checksummed. It comprises 64
factorial points (a half fraction of $2^7$), 14 axial points and 10 centre points; face-centred
means an axial distance of $\alpha = 1$, so no run leaves the coded box.

One **real evaluation** is one stratified, shuffled, seeded 5-fold cross-validation of one
configuration on one outer partition — five XGBoost fits. Outer partitions are resampled $R = 30$
times per dataset on an 80/20 split, and every method sees the same split at a given replication,
so comparisons are paired by partition.

A separate 78-point set — the design's complementary half fraction plus 14 axial runs at half
the design's axial distance — is evaluated on the real objectives and is **audit-only**:
disjoint from the 88 design rows by construction, reaching the surrogate reliability gate and
nothing else. The gate passes when external $R^2 \ge 0.5$ and Spearman $\rho \ge 0.9$ per
response, and is **diagnostic, not adaptive** — every arm runs at every replication whatever it
returns, and its pass rate is reported as a covariate, never used as a filter.

### 5.4 Responses, transforms and orientation

Seven responses are recorded per evaluation as the mean over folds. Six carry the `quality`
role — `Accuracy_Mean`, `Precision_Mean`, `Recall_Mean`, `Specificity_Mean`, `RocAuc_Mean`,
`LogLoss_Mean`, the threshold metrics computed at 0.5 and log-loss probabilities clipped to
$[10^{-7}, 1 - 10^{-7}]$ — and one carries the `cost` role, `Leaves_Mean`, the total leaf count
of the fitted ensemble summed over all trees. Mean per-fold wall-clock time, `Time_MeanFold`, is
recorded on every evaluation and is a **secondary audit variable entering no objective**.

Transform and orientation are declared per response, not inferred from data.
`Leaves_Mean` is transformed by $\log 1p$; every other response is untransformed. The transform
is not cosmetic: leaf count spans $591\times$ to $4,356\times$ over the design box, a
quadratic fitted to it raw reaches in-sample $R^2$ of only 0.49 to 0.65 against 0.95
transformed, and untransformed it strips the cost surface to an intercept on two datasets.
Responses are then canonicalized to minimization by multiplying by $-1$ where
higher is better, giving $M$, whose columns are standardized to $Z$ using the mean and sample
standard deviation ($\text{ddof} = 1$) computed **on the design side only**, with $\sigma_j
\leftarrow 1$ wherever $\sigma_j = 0$.

### 5.5 Latent objective construction

**Extraction and rotation.** PCA of $Z$ retains $k = 3$ components. Loadings are the eigenvector
matrix scaled by the square roots of the eigenvalues, $\Lambda = V
\operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$; this is the matrix
that is rotated and reported, and Varimax gives $\Lambda_R = \Lambda R$. Scores are formed from
**standardized** component scores,

$$S = Z\,V\,\operatorname{diag}(1/\sqrt{\lambda})\,R,$$

then standardized column-wise to $S_z$. The $\operatorname{diag}(1/\sqrt{\lambda})$ term is
load-bearing: raw component scores have variances equal to the eigenvalues, so rotating them
orthogonally mixes axes of unequal scale and yields correlated factors. On this panel the
maximum off-diagonal correlation of the nominally orthogonal factors ranged from 0.379 to
0.698 across the panel, and is below $10^{-15}$ under the algebra above, which also makes $\Lambda_R$ the loading
matrix of the scores actually in use, so the role and sign rules below are read off a matrix
that describes what is being oriented (Figure 1).

**Role assignment and sign, both deterministic.** The **cost factor** is the component with the
largest absolute rotated loading on `Leaves_Mean`; the other two are **quality factors**. The cost
factor is oriented so that the rotated loading of `Leaves_Mean` on it is positive, each quality
factor so that the **mean rotated loading over the six quality responses** is positive — the
role-block mean rather than the single largest loading, because specificity anti-correlates with
accuracy, recall and AUC and dominates the leading quality factor on three of four datasets.
After orientation, a larger score is a worse configuration on every factor.

**Objective 1, composite quality.** $f_{\text{quality}}(x) = \sum_{j \in Q} w_j S_z[:, j]$
with $w_j = h_j / \sum_{i \in Q} h_i$ and $h_j = \sum_i \Lambda_R[i, j]^2$ — the **rotated**
sums of squared loadings, normalized over the two quality factors. Varimax redistributes variance
across components, so an unrotated eigenvalue $\lambda_j$ and rotated component $j$ are not the
same object and indexing one by the other pairs unrelated quantities. On this panel the rotated
shares are 0.556/0.444, 0.775/0.225, 0.547/0.453 and 0.587/0.413 against unrotated 0.706/0.294,
0.837/0.163, 0.872/0.128 and 0.845/0.155; on Adult the second factor's weight differs by a
factor of 3.5; both shares are persisted under distinct names. Equal weighting — the historical
choice — is the sensitivity declared in advance, and the two agree at Spearman 0.987 to 0.991.

**Objective 2, leaf-count complexity.** $f_{\text{cost}}(x) = S_z[:, c]$, the cost-role factor,
which loads on `Leaves_Mean` at 0.90 to 1.00 on this panel. It is a **deterministic
model-complexity proxy**, never described as training time: it tracks measured per-fold time at
Spearman $\approx 0.86$, an association and not an identity. Both objectives are minimized.

**$k = 3$ is fixed at protocol level, not criterion-selected.** It is inherited from the
historical construction so that this study decomposes that construction rather than re-choosing
its dimensionality. The Kaiser criterion retains 2, 2, 1 and 1 components on MAGIC, Spambase,
Adult and Bank Marketing, with $\lambda_3/\lambda_4$ of 2.93, 2.10, 1.34 and 1.06 — never three.
Eigenvalues and Kaiser counts are reported as diagnostics; the component count is never chosen by
which value optimizes better.

**One frozen model per dataset, fitted on the 88 design rows only.** The tuple $(\mu, \sigma, V,
R, \text{roles}, \text{signs}, \mu_S, \sigma_S, w)$ is fitted once per dataset on the 88 design
rows and **nothing else**, then applied unchanged to the 78 audit-only rows, every arm's
revalidated candidates, every direct baseline, the reference sets and the holdout confirmation
(Supplement S4.1). Two constraints force this scope at once. A per-replication refit
makes the objective a different variable in every pair, so 30 paired indicator values would not
live in one objective space. And an earlier specification adding the 78-point complement — 166
points — was withdrawn, because that complement *is* the audit-only external construction (64 of
its 78 coded points identical to the external set, the other 14 the same axial runs up to integer
rounding of `max_depth`), which would let the gate validate a surface against data that helped
define that surface's target; the intersection is verified as zero in coded space on all four
datasets. The per-replication refit is computed and reported as a Tucker-congruence sensitivity,
flips reported rather than corrected, and is never applied.

### 5.6 Response surface models

Each objective is modelled by a full quadratic response surface in **coded** units, fitted by
ordinary least squares on the 88 design rows only, with backward elimination at $\alpha = 0.05$
and hierarchy enforced. The surfaces are fitted to the objective of §5.5 — the factor score, not
a raw response — so the $\log 1p$ of §5.4 sits inside the objective and the cost surface is fitted
on the transformed scale by construction. The same procedure, elimination rule and coded
parameterization are used identically by every arm that consumes a surrogate, which is what
allows the arm contrasts of §5.7 to vary one mechanism at a time. Model order is fixed by this
rule and never revised after the fact. One consequence is carried rather than smoothed over: a
backward-eliminated quadratic is often minimized on a box corner, so two objectives' surrogate
anchors can coincide and leave the payoff matrix rank-deficient.

The archived historical implementation fits its surfaces in uncoded natural units and constructs
its front by normalized weighted scalarization, although the historical text uses NBI
terminology; it is reproduced unmodified in the as-run historical arm rather than repaired.

Everything specified here is scoped to these four datasets, this surrogate architecture, two
objectives and this budget.


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


## 6. Results

### 6.1 Protocol integrity, accounting and the screened panel

The confirmatory campaign executed **120 of 120 units with 0 failures and 0 methodological
failures**, in 9 h 46 min of wall clock on 14 workers at one thread each. A unit is one
`(dataset, replication)` pair — 4 datasets $\times$ $R = 30$ replicated outer partitions — and every
arm, control and baseline ran at every unit.

The scientific budget is a **request ledger**: every request is charged whether or not it is served
from cache, so duplicate candidates, rounding and cache behaviour cannot reduce a method's charge.
The campaign spent **396,120 logical evaluations**, exactly the declared method-stage
registry total, with **every one of the 120 units reconciling individually**. Physical execution
required **377,316 unique physical fits**, a 4.75% cache hit rate. The two are kept distinct in
every statement below: only the logical figure enters a fairness comparison, and the physical figure
is an engineering quantity that enters none. The 15,360 evaluations of the
unmatched NSGA-II context baseline are **already inside the four `rep_00` ledgers** summing to
396,120; they are not added again. A total that adds them a second time is
wrong by construction.

Objective definitions were frozen before execution and **applied, never refitted**. One factor model
per dataset was fitted on that dataset's 88 face-centred central composite design rows and nothing
else, then applied unchanged to all 30 replications; the per-replication refit is reported only as a
Tucker-congruence sensitivity and never scores a front. Fitting once per dataset keeps the
30 paired indicator values inside one objective space, which a per-replication refit would not, and
the 78-point external validation set is disjoint from the fitting sample in coded space on all four
datasets, so the surrogate gate audits a surface against rows that never helped define its target.

The primary endpoint is the **CORE-relative hypervolume ratio**. The CORE reference is a finite,
method-independent empirical set — the 88 design rows plus the 200 anchor-search rows, **288 points
before Pareto filtering, present in every unit** — that no compared method contributes to. It is not
the true Pareto front, and a ratio above 1 is not an error: it means the returned set improved on
that finite reference.

Dataset roles were assigned by prospective screening and not revised (see Supplement S5).
MAGIC, Adult and Bank Marketing met the nonlinear-front criterion and form the primary geometry
panel; Spambase failed it and was retained in advance as a **boundary geometry control**, executed
in full under identical arms, budgets and seeds, and excluded only from the primary inferential
family. Dataset is the generalization unit; nothing is pooled.

Two integrity items are reported rather than smoothed. First, the campaign output contains **140
non-finite cells**, being Schott spacing and its coefficient of variation on the **70
method-by-reference blocks whose front has one point**. Schott spacing requires at least two gaps and
is undefined on a single-point front; `NaN` is the mathematically correct value there. Those blocks
are excluded from spacing summaries rather than propagated through a median. Second, five independent
verifiers recomputed eleven headline claims from the raw per-unit artifacts, and the central contrast
was additionally rebuilt from `candidate_revalidation` rows through the frozen factor model with an
independent dominance filter and a two-dimensional hypervolume sweep, reproducing all 480 published
per-unit values to $0.000\mathrm{e}{+}00$. Nine claims reproduced exactly; two were corrected, and
both were the author's.

### 6.2 The historical reconstruction: specification and normalization

A code-level reconstruction of the archived dissertation implementation established that its front
construction is a min–max normalized weighted scalarization over component-wise observed extrema of
the design rows, although the historical text uses Normal Boundary Intersection terminology. That
fact fixes the arm set: the historical method enters as a weighted-sum arm, and the geometry question
needs a separate, specification-matched contrast.

**HISTORICAL-WS $\rightarrow$ WS-S isolates normalization alone**: the same weighted sum, the same
surrogates, the same symmetric weight grid, differing only in whether the normalization box comes
from observed design-row extrema or from the payoff matrix. The effect is undetectable on every
primary dataset (Table 3).

| dataset | median difference | bootstrap interval | win/tie/loss | Holm $p$ |
|---|---:|---|---:|---:|
| MAGIC | +0.0000 | [-0.0022, +0.0056] | 14/3/13 | 0.5165 |
| Adult | -0.0006 | [-0.0033, +0.0000] | 9/4/17 | 0.2087 |
| Bank Marketing | +0.0000 | [-0.0088, +0.0000] | 8/8/14 | 0.3896 |

**Table 3.** The historical specification contrast, HISTORICAL-WS $\rightarrow$ WS-S, CORE reference, $R=30$.

This is a **non-detection at $R = 30$**, not a demonstration of no effect: exact ties occur in 3, 4
and 8 of the 30 replications on MAGIC, Adult and Bank Marketing, and the frozen design resolves only
medium effects. It does establish that this contrast — the one place a normalization-reference
hypothesis would have to appear — does not carry the historical-to-canonical difference.

The **as-run to shared-specification gap** is a descriptive control quantity rather than a member of
the frozen primary family, and carries no multiplicity correction. It compares the bit-faithful
frozen dissertation solver, with its own asymmetric weight grid, against the same weighted sum under
this campaign's surrogates, coding and symmetric grid: median paired difference in
CORE-relative hypervolume ratio +0.0994 [+0.0623, +0.1287] on MAGIC, +0.1480 [-0.0815, +0.2702] on
Adult, +0.3126 [+0.1981, +0.4484] on Bank Marketing and +0.5404 [+0.1526, +0.8736] on Spambase, the
panel's largest. That gap is the whole historical reconstruction effect, and it is confounded by
construction: surrogate identity, coding and weight-grid symmetry move together in it. Given the
non-detection above, the normalization reference is not what explains it, and we attribute it no
further (Figure 1).


### 6.3 Front-construction geometry: WS-S versus NBI-S

The second contrast in the frozen decomposition holds the surrogate models, the
canonicalized objective set, the scaling, the decision space, the candidate realizer,
the weight grid and the payoff reference fixed, and changes only the scalarization
geometry: a weighted sum of min–max normalized surrogate objectives (WS-S) against a
canonical NBI construction with a CHIM, a quasi-normal direction and a $\max t$
subproblem (NBI-S). Isolation was enforced at run time by a contrast fingerprint over
the solver configuration, the realizer, the weight grid, the surrogate identities and
the reference. Every returned candidate was re-evaluated on the real learner before any
indicator was computed. Differences are paired by replication and formed as
(second − first), with $R = 30$ per dataset; Holm correction is applied within each
dataset over the family of three primary contrasts, and the dataset — not the
replication — is the generalization unit.

Under these conditions, canonical NBI front construction produced higher
real-revalidated CORE-relative hypervolume than specification-matched weighted
scalarization on all three primary datasets. The medians are **+0.0374** on MAGIC,
**+0.2549** on Adult and **+0.2281** on Bank Marketing; the win/tie/loss counts are
26/0/4, 30/0/0 and 24/0/6; the percentile bootstrap intervals on the median are
$[+0.0144, +0.0687]$, $[+0.1740, +0.5614]$ and $[+0.1670, +0.4741]$, none of them
spanning zero; and the matched-pairs rank-biserial correlation is $+0.92$, $+1.00$ and
$+0.88$. The secondary tests are Holm-significant within every primary dataset
($p = 9.42\times10^{-7}$, $5.59\times10^{-9}$ and $9.72\times10^{-6}$). This is the
primary positive result of the study, and its scope is exactly these datasets, this
surrogate architecture, two objectives and this budget. It is not a claim that NBI
recovers a true Pareto front: the endpoint is a ratio against CORE, a finite
method-independent empirical reference of 288 points before Pareto filtering, so a
ratio above 1 means only that the candidate set improved on that finite reference
(Table 4).

Two qualifications belong in the same breath as the result. First, the geometry
advantage did not require a surrogate that passed the study's own external reliability
criterion: on Adult and Bank Marketing the composite-quality gate passed in 0 of 30
replications, and the geometry effect is Holm-significant on both. The gate was
diagnostic rather than adaptive — every arm ran at every replication regardless of its
outcome — and absolute surrogate trustworthiness and relative front-construction
geometry are different questions, of which this study answers only the second. Second,
under the frozen direct-search comparator budget the median paired difference
favoured the coarse grid over NBI-S on all four datasets ($-0.0628$, $-0.0488$,
$-0.1323$, $-0.1286$), and the grid's marginal median exceeded NBI-S on three of the
four (grid 1.0068 vs NBI-S 0.9430 on MAGIC; 1.0476 vs 0.9951 on Adult; 1.0625 vs
0.9979 on Bank Marketing) but **not on Spambase**, where NBI-S is the higher at 1.0812
against 1.0192. The two statistics answer different questions and only the paired one
is a within-replication comparison. The budget rule is matched to
the most expensive arm (NBI-R, 386 evaluations), while WS-S and NBI-S cost 186
standalone, so the comparators received roughly twice the real evaluations those arms
require; the rule was frozen before any result existed and is not revised here. The
geometry contrast compares two front constructions at a fixed surrogate budget; it is
not a demonstration that the surrogate-assisted family is preferable to direct search.
Section 6.8 reports this baseline in full.

| Contrast | Dataset | Median diff. | Bootstrap interval | Win/tie/loss | Rank-biserial | Holm $p$ |
|---|---|---|---|---|---|---|
| HISTORICAL-WS → WS-S | MAGIC | +0.0000 | [−0.0022, +0.0056] | 14/3/13 | — | 0.5165 |
| HISTORICAL-WS → WS-S | Adult | −0.0006 | [−0.0033, +0.0000] | 9/4/17 | — | 0.2087 |
| HISTORICAL-WS → WS-S | Bank Marketing | +0.0000 | [−0.0088, +0.0000] | 8/8/14 | — | 0.3896 |
| WS-S → NBI-S | MAGIC | **+0.0374** | [+0.0144, +0.0687] | 26/0/4 | +0.92 | 9.42e−07 |
| WS-S → NBI-S | Adult | **+0.2549** | [+0.1740, +0.5614] | 30/0/0 | +1.00 | 5.59e−09 |
| WS-S → NBI-S | Bank Marketing | **+0.2281** | [+0.1670, +0.4741] | 24/0/6 | +0.88 | 9.72e−06 |
| WS-S → NBI-S | Spambase (boundary) | +0.0500 | [−0.0043, +0.4369] | 18/2/10 | — | 0.1175 |
| NBI-S → NBI-R | MAGIC | **−0.0741** | [−0.0957, −0.0482] | 2/0/28 | — | 0.0000 |
| NBI-S → NBI-R | Adult | −0.0192 | [−0.1610, +0.0299] | 11/0/19 | — | 0.1607 |
| NBI-S → NBI-R | Bank Marketing | **−0.4093** | [−0.5821, −0.2059] | 7/0/23 | — | 0.0000 |

**Table 4.** The primary family: three contrasts per dataset on the CORE-relative hypervolume ratio, differences formed as (second − first), Holm-corrected within dataset, $R=30$.

Spambase is shown for completeness and is excluded from the primary inferential family
by prospective designation; it is treated in Section 6.6.

#### 6.3.1 The geometry effect is indicator-specific

The advantage does not extend uniformly across the frozen secondary indicators, and
that limitation is reported here rather than deferred. On convergence and coverage
geometry the direction matches hypervolume: median IGD⁺ is 0.0511 for NBI-S against
0.0930 for WS-S on MAGIC, 0.0579 against 0.1688 on Adult, and 0.0915 against 0.1948 on
Bank Marketing, lower being better. On the joint non-dominated fraction it does not:
the median differences are +0.0208 on MAGIC, +0.0714 on Adult and +0.0000 on Bank
Marketing, and the comparison is non-significant on all four datasets, including
Spambase (raw $p$ = 0.160, 0.093, 0.627 and 0.586). Secondary indicators are reported
descriptively, with intervals and without multiplicity-adjusted tests, and none of them
is described as significant. The defensible statement is therefore narrow: under the
frozen study conditions the advantage is specifically an improvement in convergence and
coverage geometry as measured by hypervolume and IGD⁺. The data do not support a
general better-coverage claim, and in particular do not support a claim that NBI
construction returned more non-dominated solutions (Figure 2).


#### 6.3.2 The pre-declared corrected test

Section 5.15 declared the Nadeau–Bengio corrected resampled $t$ as a sensitivity, with
an inflation of $\sqrt{8.5} = 2.9155$ at $R = 30$ and $n_{\text{test}}/n_{\text{train}} = 0.25$.
It is reported here whichever way it falls. **Under the corrected test the geometry
contrast reaches $0.05$ on Adult ($p = 0.0148$) and on neither MAGIC
($p = 0.0860$) nor Bank Marketing ($p = 0.0764$).** The
anchor-provenance contrast reaches it on MAGIC ($p = 0.0254$) and on neither
Adult ($p = 0.4161$) nor Bank Marketing ($p = 0.0791$).

This correction was derived for the generalization error of a learner under repeated
resampling, and a Pareto quality indicator computed on a returned set is not that
quantity; applying it here is a transfer, which is why it was declared a sensitivity
and not the primary test. The descriptive triple remains primary. But a sensitivity
declared before any result existed is reported whether or not it flatters the
finding, and under this one the geometry contrast clears $0.05$ on one of the three
primary datasets rather than three. Every corrected $p$-value appears in
Supplement S9.

### 6.4 Anchor and payoff provenance: NBI-S versus NBI-R

The third contrast holds the NBI geometry fixed and changes only the provenance of the
payoff matrix and anchors, from surrogate-derived (NBI-S) to the pre-specified
empirical-real procedure that obtains each anchor by direct search on the true
objectives (NBI-R). The quasi-normal direction is derived from each arm's own payoff
matrix and so moves with the reference by construction; that displacement is the
intervention, not a confound.

Replacing surrogate-derived payoff information with the frozen empirical-real anchor
procedure did not improve the confirmatory NBI fronts, and degraded CORE-relative
hypervolume on two of the three primary datasets. The medians are **−0.0741** on MAGIC
(2/0/28, $[-0.0957, -0.0482]$, Holm $p$ = 0.0000), **−0.4093** on Bank Marketing
(7/0/23, $[-0.5821, -0.2059]$, Holm $p$ = 0.0000) and **−0.0192** on Adult (11/0/19,
$[-0.1610, +0.0299]$, Holm $p$ = 0.1607), the last a non-detection under CORE with an
interval spanning zero. This negative result carries the same evidentiary weight as the
geometry result of Section 6.3 and is reported at the same prominence: spending real
evaluations to relocate the payoff matrix onto empirically obtained anchors did not buy
better fronts under these conditions, and no alternative formulation under which it
does is offered here.

#### 6.4.1 What differs mechanically, stated as association

Two prospectively frozen diagnostics narrow where the deficit comes from. The
anchor-injection control inserts NBI-R's empirical anchors into NBI-S's revalidated
candidate set and changes nothing else; because the control set is a superset of
NBI-S's, its effect is non-negative by construction. Its median effect is +0.0133 on
MAGIC and +0.0000 on the other three datasets, against full NBI-S → NBI-R gaps of
−0.0741, −0.0192, −0.4093 and −0.6932. The deficit is therefore associated with the
geometric consequences of the relocated payoff matrix rather than with anchor
point-set composition, though the control does not establish that the negative is
entirely geometric. Solver health is identical between the two arms — certified
fraction has median 1.000 over the 240 arm-units, with two exceptions at 0.900 and
0.950; the per-unit maximum equality residual has median $6.6\times10^{-10}$, with a
campaign maximum of $6.9\times10^{-1}$ on a single MAGIC NBI-S unit; and both arms
return 20 distinct realized configurations in every unit. Solver behaviour is
therefore comparable between the arms and does not account for the deficit, but it
is not identical and we do not state it as a universal.

The second diagnostic is the CHIM extent $\|\Phi_{:,0} - \Phi_{:,1}\|$, which at $q = 2$
is the segment along which the subproblems distribute their targets. Under empirical
anchors it contracts to 20–81% of its surrogate-derived extent: ratio 0.811 on MAGIC,
0.204 on Adult, 0.272 on Bank Marketing and 0.370 on Spambase. Within-dataset Spearman
correlations between the per-replication extent ratio and the per-replication
hypervolume gap are +0.683 on Adult, +0.678 on Bank Marketing, +0.320 on Spambase and
**−0.111 on MAGIC**. This is an association measured across 30 replications within a
dataset, present on three datasets and absent on the fourth; it is not a demonstrated
cause of the deficit, and it is not presented as one (Figure 4).

### 6.5 Reference sensitivity

Every dataset-by-contrast conclusion was recomputed under both the primary
method-independent CORE reference and the mandatory AUGMENTED sensitivity reference.
**Eleven of the twelve conclusions agree in direction and in significance, and no cell
anywhere flips direction.** (Figure 5). The single disagreement is Adult's anchor-provenance
contrast: under CORE the median is −0.0192 with Holm $p$ = 0.1607, non-significant;
under AUGMENTED the median is −0.0535 with Holm $p$ = 0.0174, significant. The sign is
negative under both references, so the disagreement is one of resolution, not of
direction. Under AUGMENTED that cell is significant only on the secondary test: the
primary descriptive triple remains equivocal, with the bootstrap interval on the median
spanning zero.

CORE remains primary because it was prospectively frozen as method-independent before
any result existed; AUGMENTED remains a mandatory sensitivity, not an alternative from
which a reference may be selected after the fact. The reported Adult conclusion is
accordingly the CORE non-detection, with the AUGMENTED result stated alongside it here
and in Supplement S9.


### 6.6 The prospectively retained boundary dataset did not resolve

Spambase was separated from the primary inferential family before any result existed.
Screening placed its non-dominated design-row set at two points with undefined
curvature — insufficient interior geometry for a front-construction contrast to act on —
and the protocol recorded in advance that a null result there would be neither evidence
for nor against the geometry mechanism. The replacement rule was considered and
deliberately not exercised: Spambase was executed in full, with the same arms, budgets,
seeds, evaluation machinery and $R = 30$ replicated outer partitions, and excluded only
from the primary family.

On the Claim 2 contrast the boundary dataset did not resolve. The median paired
difference in CORE-relative hypervolume ratio from WS-S to NBI-S is **+0.0500**, which
is numerically **larger** than MAGIC's significant +0.0374, with 18 wins, 2 ties and 10
losses over 30 replications, a percentile bootstrap interval of $[-0.0043, +0.4369]$
spanning zero, and Holm $p = 0.1175$. What distinguishes the two datasets is dispersion,
not location: the per-replication standard deviation is **1.7706** on Spambase against
**0.0594** on MAGIC, roughly thirty times larger, over a range $-5.2874$ to $+5.7457$
(Figure 3).

The interpretation is therefore narrow. This is a failure to resolve the contrast at
$R = 30$, not a demonstration that the effect is absent. It is **consistent with** the
prospectively stated boundary designation and does **not** confirm the boundary
mechanism: we neither read the non-significance as evidence that no geometry effect
exists on Spambase, nor treat it as corroboration of the reasoning behind the
designation. An earlier framing by the present authors asserted the stronger reading and
was refuted by an independent verifier before publication.

Two qualifications belong with it. First, the premise is less clean in execution than at
screening: the two-point front is a **pre-campaign** design-row measurement, whereas
across the 30 confirmatory replications the design-row front has median size 3, range 2
to 5, and is exactly 2 in 9 of them. The degeneracy that motivated the designation is
not a fixed property of every replication. Second, Spambase is not globally null: its
anchor-provenance contrast (Section 6.4) is strongly negative and significant at median
$-0.6932$, the largest such deficit in the panel, and its historical reconstruction gap
is also the panel's largest.

The dispersion should be read with the scale of the endpoint (Section 6.1): the
CORE-relative hypervolume ratio is not bounded above by one, exceeds 1 in 35% of
Spambase method-unit rows, and reaches the campaign's overall maximum of 9.912 there,
against 2.349 outside Spambase. The large excursions concentrate on the dataset whose
finite reference set is least informative about the trade-off surface.

### 6.7 The geometry advantage persisted where the surrogate failed its own gate

The external reliability gate is an audit-only diagnostic. It scores the fitted
surrogates on a 78-point set — the design's complementary half fraction plus 14 axial
runs at half radius — that no fitting stage can reach, passing when external $R^2 \ge
0.5$ and Spearman $\ge 0.9$ for the response. It is **diagnostic, not adaptive**: every
arm ran at every replication whatever the gate said, so its pass rate is a result rather
than a filter, and the primary comparisons are additionally reported conditioned on gate
status.

Over the confirmatory campaign the composite-quality gate passed in **8 of 30**
replications on MAGIC and **9 of 30** on Spambase, and in **0 of 30** on both Adult and
Bank Marketing. The cost gate passed 30/30, 30/30, 22/30 and 27/30 respectively
(Supplement S11).

Adult and Bank Marketing are consequently regimes in which the externally audited
composite-quality surrogate failed the frozen criterion in every replication — and they
also carry the largest geometry effects, +0.2549 and +0.2281, Holm-significant on both.
The relative advantage of NBI front construction over specification-matched weighted
scalarization therefore persisted where the surrogate did not meet its own external
standard. Within MAGIC, the only primary-panel dataset with both strata populated (Spambase, the boundary control, also has both, with 9 of 30 passing), the geometry
difference is positive in each: median +0.1173 with a win fraction of 1.00 across the 8
gate-passing units, and +0.0165 with 0.82 across the 22 failing units. Both strata are
small and are reported descriptively, without a test.

This says nothing about absolute surrogate trustworthiness. Whether a surrogate is
accurate enough to be believed, and whether one front-construction geometry beats
another on the same surrogate, are different questions; the study answers only the
second. We do not claim that the geometry effect requires a reliable surrogate, requires
an unreliable one, or that NBI compensates for surrogate error.

### 6.8 Baselines: the frozen-budget direct grid comparison

At the frozen comparator budget, **the median paired difference favoured the direct grid
baseline over NBI-S on all four datasets**. Its marginal median CORE-relative
hypervolume ratio is the higher of the two on MAGIC, Adult and Bank Marketing, but
**not on Spambase**, where NBI-S is higher at 1.0812 against 1.0192. The paired
statistic is the within-replication comparison; the marginal medians are not, and the
two do not have to agree (Table 5, Figure 6). Against NBI-S:

| dataset | NBI-S median | GRID median | median difference (NBI-S − GRID) | NBI-S wins | raw $p$ |
|---|---:|---:|---:|---:|---:|
| MAGIC | 0.9430 | 1.0068 | −0.0628 | 0/30 | 0.0000 |
| Adult | 0.9951 | 1.0476 | −0.0488 | 5/30 | 0.0001 |
| Bank Marketing | 0.9979 | 1.0625 | −0.1323 | 10/30 | 0.1642 |
| Spambase | 1.0812 | 1.0192 | −0.1286 | 12/30 | 0.8774 |

**Table 5.** Median CORE-relative hypervolume ratio by method. Direct-search comparators operate under the frozen comparator budget of 386 evaluations; WS-S and NBI-S cost 186 standalone.

The gap is significant on MAGIC and Adult and not on Bank Marketing or Spambase. This
contrast is not in the frozen primary family, carries no Holm correction, and is
reported descriptively.

**The budget asymmetry belongs in the same breath.** The frozen fairness rule matched
comparators to the *most expensive arm*, not to each arm: the comparator budget is 386
real evaluations, NBI-R's cost, while WS-S and NBI-S cost 186 standalone, so the
comparators received roughly twice the real evaluations those two arms require. The rule
was fixed before any result existed and is not revised now. The grid result therefore
cannot be stated as an efficiency claim, the budgets not being equal — and equally
cannot be explained away on that basis: at the budget the protocol froze, direct search
led. Any account of the geometry result in Section 6.3 must carry this alongside it.

The other direct-search comparators, under the same frozen budget, sit below the grid: median CORE-relative
hypervolume ratios of 0.8907 (random) and 0.9580 (NSGA-II) on MAGIC, 0.4369 and 0.8650
on Adult, 0.3190 and 0.9674 on Bank Marketing, and 0.0000 and 0.4164 on Spambase.
Matched NSGA-II ran at $32 \times 12 = 384$ evaluations, two short of the comparator
budget, a shortfall stated rather than absorbed. The **unmatched** NSGA-II run — ten
times the matched budget, one replication per dataset — is reported as **context only**:
no fairness claim attaches to it, it enters no budget-matched comparison, and it
contributes to neither reference set. The frozen protocol excludes the single-objective
Bayesian-optimization and TPE endpoints from the front-indicator table, a
single-objective optimizer not returning a front.

Two properties of the comparator qualify this. The budget rule matches comparators to
the most expensive arm rather than to each arm, so the grid received 386 real
evaluations against the 186 WS-S and NBI-S require standalone. And the grid's 128-point
mesh contains the design's 64 factorial corners, so 64 of its evaluations re-measure
points inside the CORE reference it is scored against — an advantage on a CORE-relative
indicator that the surrogate-assisted arms, which propose interior points, do not have.
Neither fact was chosen after seeing the result, and neither is offered as a reason to
set the comparison aside: at the budget and construction this protocol froze, direct
search led on the paired statistic across all four datasets.

### 6.9 Holdout confirmation

Each unit re-measured its selected candidates on a held-out partition whose labels were
structurally unavailable until the confirmation stage, at 5 audit-only real evaluations
per unit, charged to the study and to no arm. Across all four datasets and all five
arms, the **median per-replication difference** between internal and holdout accuracy has
magnitude below **0.012** in every one of the 20 dataset-by-arm cells, the largest
being **0.0117**, and is negative — the held-out partition scoring better than the
internal resampling — in **13 of the 20** cells (Supplement S14). These are descriptive audit figures read from
`analysis/secondary_analysis.json`, not confirmatory claims.

The correct reading is an absence of gross optimism in the returned configurations at
this budget — nothing more. The stage measures internal-versus-holdout optimism and was
never designed to rank arms, so the ordering of these differences is not interpreted and
a smaller drop is not evidence that an arm is better. Every primary claim in this study
is about the **returned front** under the frozen conditions, not about the
generalization of a single deployed configuration.


## 7. Discussion

### 7.1 What the decomposition establishes

The campaign separates three mechanisms ordinarily bundled together in a
surrogate-assisted multiobjective hyperparameter pipeline: the objective specification and
its normalization reference, the geometry of front construction, and the provenance of the
anchor and payoff information that geometry consumes. The contribution is not a new
algorithm but a prospectively frozen mechanistic decomposition of an existing one.
Code-level reconstruction of the archived dissertation implementation showed that front
construction there used normalized weighted scalarization, although the historical text
used Normal Boundary Intersection terminology, which is why the historical-to-canonical
difference decomposes into a normalization row and a geometry row.

### 7.2 Geometry moved the indicator

On three geometry-positive datasets and 30 replicated outer partitions, under the frozen
study conditions, canonical NBI front construction produced higher real-revalidated
CORE-relative hypervolume than specification-matched weighted scalarization: median paired
differences of $+0.0374$ on MAGIC, $+0.2549$ on Adult and $+0.2281$ on Bank Marketing,
favoured in 26/30, 30/30 and 24/30 replications, rank-biserial $\geq +0.88$ in all three,
Holm-significant within every primary dataset, which is the generalization unit
(Table 4).

The scope is part of the claim: two objectives, this surrogate architecture, this budget,
these three datasets. Nothing licenses "NBI is better than weighted sum" in general. The
endpoint is a CORE-relative hypervolume ratio against a finite method-independent set of
288 points, where a value above 1 is expected and means only that the returned set
improved on that finite reference.


The strength of the geometry evidence depends on which test is read. Under the
Holm-corrected signed-rank test declared primary, the contrast is significant in all
three primary datasets. Under the Nadeau–Bengio correction declared in advance as a
sensitivity, it clears $0.05$ on Adult alone ($p = 0.0148$, against
0.0860 and 0.0764). We take the descriptive triple as
primary, as specified before the campaign ran, and report the corrected test because
it was promised — not because it agrees. A reader who weights the corrected test more
heavily than we do should read the geometry result as one clearly resolved dataset and
two suggestive ones.

### 7.3 Why hypervolume and IGD⁺ move while the non-dominated count does not

The advantage is specifically an improvement in convergence and coverage geometry as
measured by hypervolume and IGD⁺: median IGD⁺, NBI-S against WS-S, of 0.0511/0.0930 on
MAGIC, 0.0579/0.1688 on Adult and 0.0915/0.1948 on Bank Marketing. The joint non-dominated
fraction behaves differently — median differences of $+0.0208$, $+0.0714$ and $+0.0000$,
non-significant on all four datasets including Spambase.

This is not a contradiction. Hypervolume and IGD⁺ are positional: they measure proximity
to the reference and dominated space, so moving points closer or redistributing them along
the trade-off changes both. The joint non-dominated fraction is a mutual-dominance count,
and at $q = 2$ points at different positions along one trade-off band are mutually
non-dominated by construction — an arm can sit nearer the reference while almost none of
its points dominate the other arm's. The geometry result must not be read as NBI returning
more non-dominated solutions: it did not, on any dataset.

### 7.4 Anchor provenance: a negative result, not a caveat

Holding NBI geometry fixed and replacing surrogate-derived payoff information with the
pre-specified empirical-real anchor procedure did not improve the confirmatory fronts.
Paired medians are $-0.0741$ on MAGIC, $-0.0192$ on Adult and $-0.4093$ on Bank Marketing,
Holm-significant on MAGIC and Bank Marketing; Adult is a non-detection under CORE with an
interval spanning zero. This carries the same weight here as §7.2. Real anchors are the
expensive, better-grounded option, bought with real evaluations by NBI-R alone, and the
result went the other way. The sign is negative under CORE and AUGMENTED alike, with the
panel's one reference disagreement on Adult's anchor contrast, significant under AUGMENTED
($-0.0535$, Holm $p = 0.0174$) and not under CORE.

### 7.5 CHIM contraction is the associated mechanism, and only that

The deficit is not attributable to solver failure. Over the 240 arm-units the
certified fraction has median 1.000 with two exceptions (0.900, 0.950), the
per-unit maximum equality residual has median $6.6\times10^{-10}$ against a
campaign maximum of $6.9\times10^{-1}$ on one unit, and both arms return 20
distinct realized configurations throughout. Comparable, not identical.

This is an association across 30 replications within a dataset, not a demonstrated cause:
the within-dataset Spearman between contraction and deficit is $+0.683$ on Adult, $+0.678$
on Bank Marketing and $+0.320$ on Spambase — and $-0.111$ on MAGIC, where it is absent.
Nor does the injection control prove the negative is entirely geometric: the control set
is a superset of NBI-S's, so its effect cannot be negative.

### 7.6 Specification: a non-detection at $R = 30$

Replacing the historical observed-extrema normalization with the payoff-matrix reference
produced no detectable change in CORE-relative hypervolume on any primary dataset: medians
$+0.0000$, $-0.0006$ and $+0.0000$, exact ties in 3, 4 and 8 of 30 replications, Holm $p$
of 0.5165, 0.2087 and 0.3896. This is a non-detection at $R = 30$, not a demonstration
that normalization has no effect — and it is where a DoE-range-normalization explanation
of the historical pipeline's behaviour would have shown, had that been the principal
driver.

### 7.7 The grid baseline, and what it costs the argument

At the frozen comparator budget, the frozen-budget direct grid baseline attained higher median
CORE-relative hypervolume than every surrogate-assisted arm on three of the four datasets
— NBI-S against GRID, 0.9430/1.0068 on MAGIC, 0.9951/1.0476 on Adult, 0.9979/1.0625 on
Bank Marketing and 1.0812/1.0192 on Spambase, NBI-S winning 0/30, 5/30, 10/30 and 12/30
replications (Supplement S13). The budget asymmetry belongs in the same breath: the
comparator budget is the maximum over arms, 386 real evaluations set by NBI-R, while WS-S
and NBI-S cost 186 standalone, so comparators received roughly twice the real evaluations
those arms require. The contrast is outside the frozen primary family, uncorrected, and
significant on raw $p$ for MAGIC and Adult only.

This is stated plainly rather than deflected: at the budget this protocol froze before any
result existed, on this panel, surrogate assistance did not pay for itself against direct
search. The matched-to-the-most-expensive-arm rule bears on how large the deficit is; it
is not grounds for setting it aside, and is not revised now the result is known. The
geometry finding says which construction to use *given* that a front is built on a
surrogate — not that this family was the right instrument here.

### 7.8 The surrogate-gate regime

The advantage of NBI over weighted scalarization persisted even where the externally
audited surrogate failed the frozen quality criterion. On Adult and Bank Marketing the
composite-quality gate passed in 0 of 30 replications and the geometry effect is
Holm-significant on both, against 8 of 30 on MAGIC and 9 of 30 on Spambase
(Figure 6). The gate was diagnostic: every arm ran at every replication whatever it
said, so it annotates the evidence instead of selecting it. The inference is narrow:
absolute surrogate trustworthiness and relative front-construction geometry are different
questions and this study answers only the second. We do not conclude that the geometry
effect requires a reliable surrogate, still less that NBI compensates for an unreliable
one. The regime removes one alternative explanation — the effect is not confined to
replications where the surface was accurate.

### 7.9 The estimand is the returned front

Every result above is a property of the set an arm returns, after real revalidation of
every candidate and scoring against a finite method-independent reference — not the
out-of-sample performance of a configuration a practitioner would deploy. No selection
rule collapses the front to a point, and the holdout stage is an audit. A median advantage
of $+0.2549$ on Adult says NBI-S's front occupies more space relative to CORE than WS-S's;
it does not say by how much any chosen model improves.

### 7.10 What the decomposition does not establish

Spambase was prospectively separated from the primary family because screening showed
insufficient interior front geometry. Its geometry contrast returned a median of
$+0.0500$, numerically larger than MAGIC's significant $+0.0374$, with an interval
spanning zero, Holm $p = 0.1175$ and a standard deviation of 1.7706 against MAGIC's 0.0594
— too unstable to resolve the contrast. That is consistent with the boundary designation;
it does not confirm the boundary mechanism. Nor is Spambase globally null: its
anchor-provenance contrast is strongly negative and significant.

More broadly: four datasets, two objectives, one surrogate class, one budget rule. Within
that envelope geometry carries the effect, specification does not, and provenance is
unhelpful in a way associated with CHIM contraction. Whether those orderings survive three
objectives, another surrogate class, or a budget at which surrogate assistance is
competitive with direct search, this design was not built to answer.


## 8. Limitations

This study is a mechanistic decomposition executed under one frozen set of conditions. The
conditions are the limits.

### 8.1 Scope of the design

Four datasets, two objectives, one surrogate architecture, one budget regime, one learner
family. Dataset is the generalization unit and nothing is pooled across datasets, so the
geometry result rests on three datasets and the boundary result on one. The objective space
is two-dimensional throughout: at $q = 2$ the CHIM is a single segment and the quasi-normal
direction is one vector, so the geometry contrast studied here is the two-objective case of
NBI and not its general-$q$ behaviour. Seven XGBoost hyperparameters, an 88-run face-centred
central composite design and one fixed surrogate family define the decision space and the
response model; a different design, a richer surrogate or a third objective are separate
experiments, not extrapolations of this one.

The historical arm is likewise a single case. `HISTORICAL-WS-asrun` calls one archived
implementation unmodified; code-level reconstruction established that this implementation
built its front by normalized weighted scalarization while the accompanying historical text
used NBI terminology. That is a statement about one archived artifact, not about a class of
published pipelines, and supports no estimate of how common the conflation is.

### 8.2 The budget rule favours the comparators

The frozen comparator budget is the maximum over arms — `NBI-R`'s 386 evaluations — while
`WS-S` and `NBI-S` cost 186 standalone. The comparators therefore received roughly twice the
real evaluations those two arms require. The frozen-budget direct grid baseline nevertheless led: the median paired
difference favoured it over NBI-S on all four datasets, and its marginal median was the
higher of the two on three of the four — not on Spambase (Supplement S13). That asymmetry
belongs with the result: the rule was
fixed before any result existed and is not revised now, and the study can say neither what
the surrogate-assisted arms would do at parity nor that the grid is more efficient. A
budget-matched-per-arm comparison was not run.

### 8.3 The reference is finite

CORE is a finite, method-independent empirical set of 288 points before Pareto filtering. It
is not the true Pareto front, the primary endpoint is a CORE-relative hypervolume ratio rather
than a fraction of attainable hypervolume, and the ratio exceeds 1 in 27.8% of method-unit
rows. Ratios above 1 mean a candidate set improved on that finite reference; they are not
errors, but the endpoint's upper scale is set by an empirical artifact whose composition
would change under a different anchor search.

### 8.4 What the study did not resolve

**Spambase.** The boundary dataset did not resolve the geometry contrast. Its median
difference, $+0.0500$, is numerically larger than MAGIC's significant $+0.0374$, but its
dispersion is roughly thirty times larger ($sd = 1.7706$ against $0.0594$) and the interval
spans zero. This is consistent with the prospective boundary designation and does not confirm
the boundary mechanism.

**The CHIM association is not causal.** The anchor-injection control and the CHIM extent
ratios locate the `NBI-S` → `NBI-R` deficit with the relocated payoff matrix rather than with
anchor point-set composition, but the CHIM link is an association across 30 replications
within a dataset. It is positive on Adult, Bank Marketing and Spambase and **absent on MAGIC**
($\rho = -0.11$), which is the dataset where the injection control is itself nonzero. No
intervention manipulated CHIM extent directly, so the mechanism remains a hypothesis
(Figure 4).

### 8.5 Known weaknesses in the frozen protocol

**Screening criterion 1 is weaker than what it replaced.** Its original estimator was
withdrawn when the factor algebra producing it was retracted, and the replacement — raw
standardized quality badness against the raw leaf-count response — cannot fail a dataset whose
raw metrics trade off while its latent composite does not. The panel is screened for objective
conflict by a measurement that few candidates would fail. This is carried forward as a
limitation rather than repaired.

**The surrogate failed its own gate in most replications.** The composite-quality gate passed
in 8/30 replications on MAGIC and 9/30 on Spambase, and in 0/30 on Adult and Bank Marketing.
The gate is diagnostic, not adaptive, and the geometry effect is Holm-significant on both
zero-pass datasets — but the study measures relative front-construction geometry and says
nothing about absolute surrogate trustworthiness.

**Undefined spacing.** Schott spacing requires at least two gaps. It is undefined on
single-point fronts, giving 140 non-finite cells across the 70 method-by-reference blocks whose
front has one point. Those cells are counted and excluded from spacing summaries rather than
propagated, so spacing is summarized over a subset of blocks.

**Single machine.** All 120 units ran on one machine, 14 workers × 1 thread, 9h46m wall clock.
No cross-platform or cross-BLAS replication was attempted, so hardware-independent numerical
reproducibility is untested and timing figures are not machine-independent.

### 8.6 The process

The protocol reached its frozen state through 23 numbered amendments and seven adversarial
pre-campaign reviews; the first review found ten blocking defects and the last found none. Six
regressions were introduced while fixing earlier findings. Two amendments are recorded process
failures — a descoped stage that was not logged, and a decision rule committed alongside the
measurement it had to predate — and two published claims were corrected after an independent
verifier refuted the author's version. None of these occurred after any arm result was
observed, and all are recorded in the amendment ledger rather than summarized away. A protocol
requiring this much repair before execution is prospectively frozen, not pre-registered, and
should be read as such.


## 9. Reproducibility, data and code availability

### 9.1 Version tags

Five tags separate what was specified from what was measured (Supplement S1). `v0.1.0-dissertation`
(`67d9fe5`) is the historical implementation this study reconstructs; it is read and executed as
archived, never edited. `xgboost-hpo-protocol-v1` froze the specification before the
measurement-validation pilot, `-v2` incorporated the pilot, and `-v3` (`9b15ba7`) is the
confirmatory protocol under which every unit ran. These tags record what was specified before any
result existed; the 23 numbered amendments between them carry the cause of each change beside it.
`xgboost-hpo-confirmatory-results-v1` (`a4f20ed`) is the first verified analysis: the per-unit
indicator table (1,080 rows, one per method-unit), the primary and secondary analysis artifacts,
and the verified-claims file.

### 9.2 Seed derivation

A seed is a pure function of `(dataset, replication, method, stage)`. The four fields are joined
with the frozen namespace `xgboost-hpo-vrfnbi/v3`, hashed with BLAKE2b to eight bytes, and the
resulting 64-bit integer is passed to `numpy.random.SeedSequence`. BLAKE2b rather than Python's
`hash()`, which is salted per process: a campaign resumed in a new interpreter would otherwise draw
different candidates. All 3,840 campaign streams are distinct, stable across processes and
`PYTHONHASHSEED`, and independent of execution order and worker count. Because the replication index
remains a seed component, the paired design still pairs — every method sees the same outer partition
at a given replication. Seeds are narrowed to 32 bits only where a third-party API validates that
range. Unit seeds derive from `SEED_BASE = 20260914`.

### 9.3 Runner, ledger and cache

Each of the 120 units executes 20 stages, each checkpointed as a JSON payload written to a
temporary file and atomically replaced, so an interrupted run leaves no half-written stage and a
resumed run never recomputes a completed one. Request rows carry an attempt epoch and only the
latest attempt per `(method, stage)` is counted, which makes a resumed campaign equivalent to an
uninterrupted one in the accounting as well as in the science.

Two ledgers are kept separately and both reported. The **logical** ledger charges every evaluation a
method requests, hit or miss; it is the scientific budget and the only quantity entering a fairness
comparison. The campaign consumed **396,120 logical evaluations**, exactly the declared registry total, every
one of the 120 units reconciling individually. The **physical** ledger counts misses only:
**377,316 unique physical fits**, a **4.75% cache hit rate** — an engineering quantity that enters
no comparison between methods.

Memoization sits beneath method isolation. The cache key pins dataset identity, outer split
identity, inner fold definition, the configuration after canonical type normalization (integers
rounded, floats fixed to twelve decimals, keys sorted), the training seed and the evaluation
protocol version; a change to any of them is a different evaluation. Each method reaches the cache
only through its own view, which exposes its own history and nothing else, so reuse never becomes
cross-method information.

### 9.4 Environment and how to re-run

The campaign manifest stamps the environment: Python 3.11.15 on macOS 26.6.2 (arm64), numpy 2.4.6,
pandas 3.0.5, and in the same environment xgboost 3.2.0, scikit-learn 1.9.0 and scipy 1.17.1.
Execution used 14 process workers, one XGBoost thread per fit, `spawn` start method, completing
120/120 units with 0 failures in 9h46m. Two provenance scripts that
execute the archived dissertation code require an interpreter contemporary with it (numpy < 2,
pandas < 3); they refuse to run on pandas 3 and print that recipe rather than patch a historical
artifact.

The campaign is driven by `scripts/xgb_hpo_campaign.py` (`plan`, `dry-run`, `run`, `status`);
`run` is resumable and idempotent over completed stages. Analysis is reproduced in order by
`aggregate_campaign.py`, `primary_analysis.py` and `secondary_analysis.py`, and the claims map by
`build_claims_map.py`, which reads every headline figure from a committed artifact at generation
time. Audit artifacts rebuild under `--check`, comparing against the committed versions without
writing.

Raw datasets and heavy evaluation caches are deliberately unversioned. The four panel datasets are
UCI or OpenML mirrors, obtainable independently and checksum-verified on acquisition; the 88-run
face-centred central composite design is versioned with its SHA-256. The per-unit experiment tree,
including one SQLite evaluation cache per unit, is not committed. Committed instead are the frozen
code, the protocol, the amendment ledger, the per-unit indicator table and the analysis artifacts —
enough to recompute every reported quantity without re-running the campaign.


## 10. Conclusion

Surrogate-assisted multiobjective hyperparameter optimization pipelines change several
things at once. When such a pipeline is revised, the revision typically alters the
scalarization specification, the geometry by which a front is constructed, and the
information from which anchors and the payoff matrix are built — and a difference in
the returned front cannot then be attributed to any of them. This study separated the
three under a protocol frozen before any comparative result existed, and revalidated
every returned candidate on the real learner.

Two of the three mechanisms produced clear answers, in opposite directions. Correcting
the archived scalarization specification and its observed-extrema normalization changed
nothing detectable. Replacing weighted scalarization with canonical Normal Boundary
Intersection, holding everything else fixed, improved CORE-relative hypervolume on all
three geometry-positive datasets, with 26, 30 and 24 of 30 paired wins and Holm-adjusted
significance within each dataset. The third mechanism answered in the direction opposite
to the one the pipeline's design would suggest: substituting a pre-specified
empirical-real anchor procedure for surrogate-derived payoff information gave no
benefit, and degraded the front on two of the three datasets. The frozen anchor-injection control moves the
indicator by approximately zero on three of four datasets and by $+0.0133$ on MAGIC,
where it accounts for about 18% of that dataset's gap, so the deficit is associated
with the relocated payoff information rather than with which points enter the returned
set,
and the payoff matrices show a contraction of the convex hull of individual minima that
is associated with the deficit on three of four datasets. That association is reported
as an association.

Two results limit how far the geometry finding should be carried. The advantage is
specific to hypervolume and IGD⁺; the joint non-dominated fraction shows no significant
difference on any dataset in the panel. And under the frozen direct-search comparator
budget, the median paired difference favoured a coarse grid over NBI-S on all four
datasets, and the grid held the higher marginal median on three of the four — not on
Spambase — while receiving 386 real evaluations against the 186 the scalarization arms
require standalone. That asymmetry is a property of the frozen
comparator rule, not a defence of the arms: at the budget this protocol fixed in
advance, direct search led.

What the study establishes is therefore narrower than "NBI is better", and more useful.
Within this decision space, this surrogate architecture, two objectives and this budget,
front-construction geometry is the mechanism that moved the returned front;
specification did not; and anchor provenance moved it the wrong way. The prospectively
designated boundary dataset did not resolve, which is consistent with the reasoning
that separated it in advance without confirming that reasoning. Whether the geometry
advantage survives at larger budgets, at more than two objectives, or against direct
search given equal evaluations, this design cannot say, and we have tried not to say it.


## Declarations

**Prior dissemination and relationship to a dissertation.** This study extends work
first developed in the first author's master's dissertation (Ribeiro, 2026, UNIFEI,
unpublished), which applied a PCA/Varimax latent-objective construction to XGBoost
hyperparameter optimization. The present work reconstructs that pipeline from its
archived implementation, separates three mechanisms it changed together, and evaluates
them under a protocol frozen before any comparative result existed. The archived
implementation is reproduced by calling its frozen code unmodified rather than by
reimplementation, and is reported neutrally: a code-level reconstruction established
that it performs normalized weighted scalarization for front construction although the
historical text used NBI terminology. No claim is made about the correctness of the
dissertation's reported results, whose objective space differs from this study's
(§3.5).

**Relationship to companion work by the same authors.** Three instruments used here
were developed in a companion manuscript by the present authors (Ribeiro, Pereira and
de Paiva, 2026, unpublished, frozen and not submitted): the external surrogate
reliability gate, the comparator-budget rule — which matches direct-search
comparators to the most expensive arm rather than pairwise to each arm — and the
anchor-injection control. They are applied here, not introduced. The methodological
lineage of the latent-objective construction is set out in §2.7 and includes Pereira
et al. (2025) and de Azevedo et al. (2026), with which this work shares co-authors.

**Overlap statement.** The datasets, the frozen protocol, the confirmatory campaign and
every result reported here are specific to this study. No figure, table or result is
reproduced from the dissertation or from the companion manuscript.

**Data and code availability.** The protocol, the frozen factor models, every analysis
artifact and every script are version-controlled and tagged; see §9. Raw datasets are
public and are retrieved by checksum-verified loaders; evaluation caches are
deliberately unversioned.

**Competing interests.** The authors declare no competing interests.

**Funding.** The master's research underlying this study received support from the
Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG), project BPD-01045-22,
and from the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES).
Anderson Paulo de Paiva acknowledges research support from the Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq), process 312844/2023-9.

**Acknowledgements.** The authors acknowledge the Universidade Federal de Itajubá
(UNIFEI) for institutional support.

**Author contributions (CRediT).**

*Caio Tertuliano Ribeiro* — Conceptualization; Methodology; Software; Validation;
Formal analysis; Investigation; Data curation; Project administration; Writing –
original draft; Visualization.

*Matheus Costa Pereira* — Methodology; Writing – review & editing.

*Anderson Paulo de Paiva* — Conceptualization; Supervision; Funding acquisition;
Writing – review & editing.

The CRediT role *Resources* is not assigned. The taxonomy does not require every role to
be represented, and no author is credited with a contribution the work did not involve.

---

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Anthropic Claude and OpenAI ChatGPT in order to assist with manuscript drafting, language refinement, consistency checking and editorial review. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

This declaration concerns manuscript preparation only. The study's computational methods are described in the Methods section, and every reported result derives from the committed, version-controlled artifacts referenced there.

## References

Every entry below was verified against an authoritative source — Crossref, the
publisher's proceedings page, arXiv, or JMLR — at the DOI or identifier shown.
Machine-readable records are in `manuscript/references.bib`. Fields a source did not
state are omitted rather than inferred; two such omissions are noted explicitly.

Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization.
*Journal of Machine Learning Research*, 13, 281–305.

Costa, D. M. D., Paula, T. I., Silva, P. A. P. and Paiva, A. P. (2016). Normal boundary
intersection method based on principal components and Taguchi's signal-to-noise ratio
applied to the multiobjective optimization of 12L14 free machining steel turning
process. *The International Journal of Advanced Manufacturing Technology*, 87(1–4),
825–834. https://doi.org/10.1007/s00170-016-8478-7

Das, I. and Dennis, J. E. (1998). Normal-Boundary Intersection: a new method for
generating the Pareto surface in nonlinear multicriteria optimization problems.
*SIAM Journal on Optimization*, 8(3), 631–657.
https://doi.org/10.1137/S1052623496307510

de Azevedo, T. M., Pereira, M. C., Cesário, M. de C. and de Paiva, A. P. (2026).
Multiobjective and multivariate rationalization of CFD simulations for hydrodynamic
systems using the NBI-VRF method. *Thermal Science and Engineering Progress*, 74,
104722. https://doi.org/10.1016/j.tsep.2026.104722

Eggensperger, K., Müller, P., Mallik, N., Feurer, M., Sass, R., Klein, A., Awad, N.,
Lindauer, M. and Hutter, F. (2021). HPOBench: a collection of reproducible
multi-fidelity benchmark problems for HPO. Published at the NeurIPS Datasets and
Benchmarks Track, 2021. arXiv:2109.06716

Guerrero-Viu, J., Hauns, S., Izquierdo, S., Miotto, G., Schrodi, S., Biedenkapp, A.,
Elsken, T., Deng, D., Lindauer, M. and Hutter, F. (2021). Bag of baselines for
multi-objective joint neural architecture search and hyperparameter optimization.
arXiv:2105.01015. *(The arXiv record states no workshop venue, so none is asserted.)*

Ishibuchi, H., Masuda, H., Tanigaki, Y. and Nojima, Y. (2015). Modified distance
calculation in generational distance and inverted generational distance. In
*Evolutionary Multi-Criterion Optimization (EMO 2015)*, Lecture Notes in Computer
Science, Springer, 110–125. https://doi.org/10.1007/978-3-319-15892-1_8 *(The LNCS
series volume is omitted: sources do not state it consistently and it was not confirmed
by a second source. The DOI identifies the chapter unambiguously.)*

Karl, F., Pielok, T., Moosbauer, J., Pfisterer, F., Coors, S., Binder, M., Schneider,
L., Thomas, J., Richter, J., Lang, M., Garrido-Merchán, E. C., Branke, J. and Bischl,
B. (2023). Multi-objective hyperparameter optimization in machine learning — an
overview. *ACM Transactions on Evolutionary Learning and Optimization*, 3(4), Article
16, 1–50. https://doi.org/10.1145/3610536

Lujan-Moreno, G. A., Howard, P. R., Rojas, O. G. and Montgomery, D. C. (2018). Design
of experiments and response surface methodology to tune machine learning
hyperparameters, with a random forest case-study. *Expert Systems with Applications*,
109, 195–205. https://doi.org/10.1016/j.eswa.2018.05.024

Luz, E. R., Romão, E. L., Streitenberger, S. C., Mancilha, L. R., de Paiva, A. P. and
Balestrassi, P. P. (2021). A multiobjective optimization of the welding process in
aluminum alloy (AA) 6063 T4 tubes used in corona rings through normal boundary
intersection and multivariate techniques. *The International Journal of Advanced
Manufacturing Technology*, 117(5–6), 1517–1534.
https://doi.org/10.1007/s00170-021-07761-5

Morales-Hernández, A., Van Nieuwenhuyse, I. and Rojas Gonzalez, S. (2023). A survey on
multi-objective hyperparameter optimization algorithms for machine learning.
*Artificial Intelligence Review*, 56(8), 8043–8093.
https://doi.org/10.1007/s10462-022-10359-2

Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error.
*Machine Learning*, 52(3), 239–281. https://doi.org/10.1023/A:1024068626366

Pereira, M. C., Ribeiro, C. T., Mendes, R. R. A., Campos, P. H. da S. and de Paiva,
A. P. (2025). A hybrid multivariate normal boundary intersection approach with
post-optimization assisted by mixture design of experiments. *Engineering Applications
of Artificial Intelligence*, 162, 112510.
https://doi.org/10.1016/j.engappai.2025.112510

Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M. and Bischl, B. (2022). YAHPO
Gym — an efficient multi-objective multi-fidelity benchmark for hyperparameter
optimization. In *Proceedings of the First International Conference on Automated
Machine Learning*, PMLR 188, 3/1–39.
https://proceedings.mlr.press/v188/pfisterer22a.html

Ribeiro, C. T. (2026). *Multiobjective optimization of XGBoost hyperparameters using
design of experiments and latent objective construction*. MSc dissertation,
Universidade Federal de Itajubá (UNIFEI). Unpublished.

Ribeiro, C. T., Pereira, M. C. and de Paiva, A. P. (2026). *Surrogate reliability and
evaluation-matched comparison in multivariate response-surface optimization*.
Unpublished manuscript, frozen and not submitted.

Streitenberger, S. C., Romão, E. L., Paiva, A. P., Balestrassi, P. P., Freitas,
J. H. G. and Paes, V. C. (2022). Normal Boundary Intersection with factor analysis
approach for multiobjective stochastic optimization of a cladding process focusing on
reduction of energy consumption and rework. *Journal of Cleaner Production*, 333,
129915. https://doi.org/10.1016/j.jclepro.2021.129915

Vasquez-Ramos, J., Ruiz-Sandoval, M. G., Oliva, D., Ramos-Soto, O., Ramos-Frutos, J.,
Sharawi, M. and Pérez-Cisneros, M. (2025). Response surface-driven hyperparameter
optimization for XGBoost. *The Journal of Supercomputing*, 81(10), Article 1112.
https://doi.org/10.1007/s11227-025-07600-4


## Figure captions

**Figure 1. The arm lattice.** Each arrow is a single-factor contrast and the label
names the one thing it varies; everything else is held fixed by a run-time fingerprint
over the solver configuration, realizer, weight grid, surrogate identities and
reference. The two primary contrasts are marked.

**Figure 2. The geometry contrast, per replication.** Paired differences in
CORE-relative hypervolume ratio, NBI-S minus WS-S, sorted within each dataset, $R = 30$,
CORE reference. Blue favours NBI-S. The dashed line is the median. Note that the
vertical scales differ by dataset.

**Figure 3. Dispersion across the panel.** The same paired differences as Figure 2,
shown as distributions. The boundary dataset's spread is roughly thirty times MAGIC's,
which is why its contrast does not resolve at $R = 30$ despite a larger point estimate.

**Figure 4. Anchor provenance and the associated geometry.** Left: the full
NBI-S $\rightarrow$ NBI-R difference beside the anchor-injection control, which inserts
NBI-R's empirical anchors into NBI-S's set and changes nothing else. Right: the CHIM
extent under empirical anchors relative to surrogate-derived anchors, as a per-dataset
median. The juxtaposition is an association, not a demonstrated cause, and it does not
hold on MAGIC.

**Figure 5. Reference sensitivity.** Median paired difference under the CORE reference
(primary, horizontal) against the AUGMENTED reference (mandatory sensitivity,
vertical), for all twelve dataset-by-contrast cells. Red marks the single cell whose
significance differs between references; no cell changes direction.

**Figure 6. Surrogate-gate regime and baselines.** Left: per-dataset pass rates of the
frozen external reliability criterion over $R = 30$; the gate is diagnostic and filters
nothing. Right: median CORE-relative hypervolume ratio by method. The dashed rule marks
the CORE reference itself and is **not** an upper bound — a ratio above it means the
candidate set improved on the finite reference.
