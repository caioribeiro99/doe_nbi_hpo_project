# Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study

## Abstract

Surrogate-assisted multiobjective hyperparameter optimization pipelines routinely
change scalarization specification, Pareto-front construction geometry, and anchor
provenance together, so an observed difference cannot be attributed to any one of
them. We report a prospectively frozen, provenance-preserving reconstruction that
varies the three one at a time, with seven XGBoost hyperparameters on an 88-run face-centred
central composite design, four datasets, 30 replicated outer partitions and every returned
candidate revalidated on the real learner. Both minimized objectives are latent: a
Varimax-rotated composite quality factor and a leaf-count model-complexity factor, each a fixed
linear score of six quality responses and the log leaf count, fitted per dataset on pilot
design runs. The budget ledger counted every request, repeats included: 396,120 logical evaluations,
377,316 of them unique.

Replacing the archived observed-extrema normalization with a payoff-matrix reference
produced no detectable change. Holding the surrogate model, scaling, decision space,
candidate realization and budget fixed, canonical Normal Boundary Intersection (NBI) on the
surrogate payoff matrix (NBI-S) improved hypervolume over specification-matched weighted
scalarization on all three datasets that passed the pre-campaign front-geometry screen (a
curved front among the non-dominated design runs). Measured against CORE, a 288-point
empirical reference to which no compared method contributes, the median paired differences
were $+0.037$ on MAGIC (26/30 wins), $+0.255$ on Adult (30/30) and $+0.228$ on Bank Marketing
(24/30); all three bootstrap intervals exclude zero, and the secondary signed-rank test is
Holm-significant within each dataset. The composite-quality surrogate met its frozen external
reliability criterion in 8 of 30 replications on MAGIC, none on Adult or Bank Marketing and 9
of 30 on the fourth dataset, Spambase; the criterion was diagnostic and removed nothing, so
the Adult and Bank Marketing effects rest entirely on replications that failed it. Replacing
surrogate-derived payoff information with a pre-specified empirical-real anchor procedure,
costing 200 additional real evaluations, gave no benefit and degraded the front on two of
the three ($-0.074$ on MAGIC; $-0.019$ on Adult, a non-detection; $-0.409$ on Bank
Marketing). A post hoc analysis associates the deficit with contraction of the convex hull of
individual minima, clearly on Adult and Bank Marketing, weakly on Spambase and not on MAGIC.
Under the pre-declared Nadeau–Bengio corrected t-test, a sensitivity that allows for training
data shared across partitions, the geometry improvement reaches unadjusted $p < 0.05$ on
Adult only and the anchor deficit on MAGIC only. An AUGMENTED reference that adds to CORE every
point a compared method is scored on gives the same bootstrap-interval conclusion
in all 12 dataset-by-contrast cells and the same secondary-test conclusion in 11.
On every dataset these conclusions are conditional on the variance-weighted quality composite: rescoring each arm's revalidated candidates after the campaign under the pre-declared equal weighting changed the non-dominated sets in more than 20% of replications; under the equal weighting the WS-S $\rightarrow$ NBI-S and NBI-S $\rightarrow$ NBI-R median differences kept their direction (Supplement S4.5). Under the frozen
direct-search comparator budget the median paired difference favoured a coarse grid over
NBI-S on all four datasets, with the grid receiving 386 real evaluations against 186
standalone for NBI-S; the lead met the frozen 21-of-30 direction rule on MAGIC and Adult
only, its bootstrap interval also excluded zero on Bank Marketing (20/30 wins), and it was
unresolved on Spambase (18/30). On Spambase, which failed the screen and was retained as a boundary
control, the geometry contrast did not resolve (a numerically larger point estimate than the
smallest resolved effect, at roughly thirty times its dispersion), and its quality composite
is negatively rank-correlated with raw quality over uniform draws of the decision box (a post
hoc check), so this result cannot be read as evidence about front geometry.

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

This paper does not propose a new optimizer. It reports a prospectively frozen decomposition
of one such pipeline in which the three mechanisms are separated into single-factor contrasts,
executed under a budget ledger fixed in advance, and reported whether or not the result
favours the method under study. The pipeline is the one implemented in the first author's
master's dissertation (Ribeiro 2026) for surrogate-assisted multiobjective XGBoost tuning over
Varimax-rotated factor objectives. Its archived code normalizes the objectives by an
observed-extrema box and constructs the front by weighted scalarization, so the historical
method enters the arm set as a weighted sum (Section 3).

### 1.1 Research questions

Holding the decision space, design, surrogate family, weight grid, candidate realization and
evaluation protocol fixed, and changing exactly one mechanism per contrast (the budget is also
fixed in RQ1 and RQ2; in RQ3 the 200 real anchor evaluations are part of the change):

- **RQ1: specification and normalization.** Does replacing the historical observed-extrema
  normalization box with a payoff-matrix reference change the returned front?
- **RQ2: front construction geometry.** At a fixed reference and fixed surrogates, does
  canonical NBI construction differ from specification-matched weighted-sum construction?
- **RQ3: anchor and payoff provenance.** With the NBI formulation held fixed, what happens when the
  payoff matrix and anchors are obtained by direct search on the real objectives instead of
  from the fitted surrogates?

The study answers these on four binary-classification datasets, seven XGBoost hyperparameters,
an 88-run face-centred central composite design, two minimized latent objectives and $R = 30$
replicated outer partitions per dataset, with every returned candidate re-evaluated on the real
learner. The first objective is a Varimax-rotated composite quality factor, oriented on six
canonicalized quality responses; the second is a leaf-count model-complexity factor. Both are
fixed linear scores of all seven responses, fitted once per dataset on the 88 design rows of a
pre-campaign pilot partition (Section 5.5). The campaign is 396,120 logical
evaluations. The primary endpoint is the CORE-relative hypervolume ratio: hypervolume measured
against a finite, method-independent empirical reference of 288 points that no compared method
contributes to. CORE is not the true Pareto front, and a ratio above 1 simply means a candidate
set improved on that finite reference. Dataset is the generalization unit; nothing is pooled
across datasets.

RQ1, RQ2 and RQ3 form the primary inferential family. Section 4 adds two questions outside it:
the RQ2 contrast on Spambase, which failed the pre-campaign front-geometry screen and was
retained as a boundary control (RQ4), and direct search at the frozen comparator budget (RQ5).
Also declared before the campaign are a secondary question, whether the contrasts behave
differently where a surrogate fails its frozen external reliability gate, a diagnostic
covariate scored on 78 audit-only real evaluations (Section 6.7, Supplement S11); sensitivities
of the primary analysis, of which the AUGMENTED reference (CORE plus every point a compared
method is scored on) and the Nadeau–Bengio corrected test are reported in Sections 6.5 and
6.3.2; and a holdout audit of selection optimism that ranks no arm (Section 6.9). Declared
analyses that were computed only after the campaign, and those that were not executed, are
listed in Section 8.7.

### 1.2 What the study found

**Geometry separates.** On the three primary datasets, those that passed the pre-campaign
front-geometry screen on the design rows (Section 5.1), canonical NBI construction produced
higher CORE-relative hypervolume than specification-matched weighted scalarization: median
paired differences of $+0.0374$ on MAGIC (26/30 replications), $+0.2549$ on Adult (30/30) and
$+0.2281$ on Bank Marketing (24/30). Every bootstrap interval on the median difference excludes
zero, and the secondary signed-rank test is Holm-significant within every one of the three
datasets (Table 4); under the pre-declared Nadeau–Bengio sensitivity the contrast reaches
unadjusted $p < 0.05$ on Adult only (Section 6.3.2). The geometry effect is the main positive
result, and its scope is these datasets, this surrogate architecture, two objectives and this
budget. On Adult and Bank Marketing it was obtained with a composite-quality surrogate that
failed the frozen external reliability gate in every replication (Section 6.7). It is also
indicator-specific: the advantage appears in hypervolume and IGD$^{+}$, while the joint
non-dominated fraction shows no detectable difference under CORE on any of the four datasets
(Section 6.3.1).

**Anchor provenance did not help.** Replacing
surrogate-derived payoff information with the frozen empirical-real anchor procedure did not
improve the confirmatory NBI fronts: median differences of $-0.0741$ on MAGIC, $-0.0192$ on
Adult (a non-detection, interval spanning zero) and $-0.4093$ on Bank Marketing. Two of three
primary datasets degraded; under the Nadeau–Bengio sensitivity the deficit reaches unadjusted
$p < 0.05$ on MAGIC only. The paid anchor stage bought nothing measurable here.

**A frozen-budget direct grid baseline led NBI-S at the frozen comparator budget.** The
median paired difference favoured the direct grid baseline over NBI-S on all four datasets.
The grid's lead met the frozen 21-of-30 direction rule on MAGIC and Adult only (NBI-S won 0
and 5 of 30 replications); on Bank Marketing its bootstrap interval also excluded zero (NBI-S
won 10 of 30), and on Spambase it was unresolved (NBI-S won 12 of 30). The grid held the higher
marginal median on three of the four datasets, but not on Spambase, where NBI-S is the higher
of the two (Table 5). The paired statistic is the within-replication comparison and the
marginal medians are not, so the two need not agree. The budgets were not equal: the frozen
rule matched comparators to the most expensive arm (NBI-R, 386 real evaluations), while WS-S
and NBI-S cost 186 standalone, so the comparators received roughly twice the real evaluations
those arms require. The rule dates from protocol v1 (2026-09-13), before any arm was
implemented, and its value of 386 from protocol v2; both precede the confirmatory campaign and
were applied unchanged, and the asymmetry is not used to dismiss the result.

**Specification and normalization were undetectable.** The RQ1 contrast produced no detectable
change in CORE-relative hypervolume on any primary dataset, reported as a non-detection at
$R = 30$ rather than as an absence of effect.

### 1.3 Contributions

1. A single-factor decomposition of one surrogate-assisted multiobjective HPO pipeline into
   specification and normalization, front-construction geometry, and anchor and payoff
   provenance, with run-time enforcement that each contrast varies only its declared mechanism.
2. A protocol frozen before any confirmatory comparative result was inspected, with its
   dataset roles, endpoints, reference, statistical family and budget ledger fixed in advance,
   every pre-campaign change recorded as a numbered amendment, and the departures identified
   after the campaign listed as deviations (Section 8.7).
3. A confirmatory campaign of 396,120 logical evaluations over four datasets and $R = 30$
   paired replications, with every candidate revalidated on the real learner and every headline
   number traceable to a committed artifact.
4. Negative and scope-limiting results: the negative anchor-provenance contrast, the grid
   baseline that led NBI-S at the frozen comparator budget, and the indicator specificity of
   the geometry effect.
5. A mechanistic reading of the negative contrast (a prospectively frozen anchor-injection
   control together with a post hoc measurement of the contraction of the CHIM segment),
   reported as an association within datasets, not as a demonstrated cause (Figure 4).


## 2. Related work and methodological lineage

### 2.1 Normal Boundary Intersection

Das and Dennis (1998) introduced Normal Boundary Intersection (NBI) as a scalarization that
distributes subproblems geometrically rather than by weight. Each objective is minimized individually
to obtain an anchor; the anchors' objective vectors fill the columns of the payoff matrix, whose
diagonal is the utopia point $F^{*}$, and $\Phi$ denotes the payoff matrix with $F^{*}$ subtracted
from each column. The convex hull of individual minima (CHIM) is the simplex spanned by the anchors, a
quasi-normal direction $\hat{n} = -\Phi\mathbf{1}/\lVert\Phi\mathbf{1}\rVert$ points from the CHIM
toward the attainable set, and each subproblem maximizes the step $t$ along $\hat{n}$ from a CHIM
point under an equality constraint. At $q = 2$, as here, the CHIM is a segment whose extent
$\lVert\Phi_{:,1} - \Phi_{:,2}\rVert$ is the interval along which subproblem targets are placed.

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
requires a normalization box, and the choice of that box (observed extrema over the design rows, a
payoff-matrix utopia with a pseudo-nadir, or per-objective worst values over the box from
anti-optimization) is a modelling decision separate from the choice of geometry. Separating the
two is why a specification-matched
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
here. The comparators used below are a coarse grid, random search (Bergstra and Bengio 2012), Bayesian
optimization and the tree-structured Parzen estimator (TPE), both run on one objective at a time,
and NSGA-II. Multiobjective Bayesian optimization and multi-fidelity methods are outside the frozen
comparator set (Section 8.1).

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
test is carried here as a pre-declared sensitivity, not as the study's significance test, since a
set indicator computed on a returned front is not the generalization error it was derived for.

### 2.7 Methodological lineage

This work follows the first author's master's dissertation (Ribeiro 2026), which applied the
DoE–response-surface–NBI-VRF construction to XGBoost hyperparameter optimization, including on the
MAGIC Gamma Telescope dataset reused here (Section 5.1), and the group's published formulation in
Pereira et al. (2025), which formulates the method using canonical NBI. The archived dissertation
implementation builds its front by normalized weighted scalarization (Section 3.1), which is why
the arm set contains both a historical weighted-sum arm and a specification-matched one; no claim
is made about any implementation not examined here, including that of Pereira et al. (2025). A
bounded search located no work applying NBI or NBI-VRF to hyperparameter optimization;
non-location is weaker than absence, and the transfer is not offered as the contribution. NBI
(Section 2.1), Varimax-rotated factor objectives (Section 2.5), and design of experiments for
hyperparameter tuning (Section 2.3) are all prior art, and none of them is claimed here. What this
paper contributes is the prospectively frozen decomposition of that inherited construction into
the three mechanisms it bundles. Section 3 describes the historical pipeline and how it differs
from canonical NBI, and Section 4 states the three single-factor contrasts.


## 3. Historical method reconstruction

### 3.1 What the archived implementation computes

The historical pipeline (Ribeiro 2026) follows the sequence set out at the start of the
Introduction, with Varimax-rotated factor objectives as the reduced objectives. Its solver, frozen
at tag `v0.1.0-dissertation` (Section 9.1), exposes a single routine. For each weight pair
$(\beta_1,\beta_2)$ on a fixed
grid it maximizes $\sum_j \beta_j \bar f_j(x)$ by SLSQP from ten multistarts, where
$\bar f_j = (f_j - \text{nadir}_j)/(\text{utopia}_j - \text{nadir}_j)$. That is normalized weighted
scalarization. None of the structural elements of Normal Boundary Intersection is present: there are
no per-objective anchor minimizers, no payoff matrix and hence no $\Phi$, no convex hull of
individual minima, no quasi-normal direction $\hat n = -\Phi\mathbf{1}/\lVert\Phi\mathbf{1}\rVert$,
no $\max t \ \text{s.t.}\ \Phi\beta + t\hat n = F(x) - F^{*}$ subproblem, and no auxiliary
variable $t$ at all.
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
| Weight grid | asymmetric, 20 points at the solver's default step of 0.05 |
| Objective orientation | maximization, larger-is-better objectives |
| Surrogate fit | full quadratic response surfaces in uncoded natural units |
| Surrogate adequacy | no external validation and no gate |

**Table 2.** Properties of the archived implementation, reproduced rather than
repaired.

The normalization box is built from the component-wise maxima and minima of the design rows. These
are not the objective values attained at the other objective's minimizer, which is what a payoff
matrix supplies, so the historical method differs from canonical NBI in two independent respects —
geometry and reference — rather than one.

The weight grid is asymmetric. At the solver's default step it sweeps $b$ from $0.05$ to $1.00$
and forms $(1-b, b)$, so the
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
twice. HISTORICAL-WS applies the weighted sum under the shared specification (the solver,
symmetric weight grid with both vertices, minimization orientation and coded-unit surrogates that
WS-S itself uses) and differs from WS-S in the normalization reference alone (Section 5.8,
Figure 1). The contrast HISTORICAL-WS $\rightarrow$ WS-S therefore identifies normalization as a
single factor, isolated from geometry and from anchor provenance. Both historical entities have
every returned candidate revalidated on the real learner, as the archived protocol did.

### 3.5 What is not carried forward

What is carried forward is the archived *optimizer*, not its objective space. The
archived pipeline optimized four threshold quality metrics (accuracy, precision,
recall and specificity) against mean per-fold wall-clock time as its cost response,
and used no ranking or calibration metric. This study optimizes a Varimax-rotated
latent quality composite, oriented on six quality responses that add ROC AUC and log
loss to the archived four, against a latent complexity factor dominated by total leaf
count; both are fixed linear scores of all seven standardized responses (Section 5.5).

The cost substitution is deliberate and is a limitation of the comparison. A wall-clock
response is not reproducible from a seed: it varies with machine load, thread
scheduling and hardware, so a replicated design cannot hold it fixed across 30
partitions. Total leaf count is a deterministic model-complexity proxy that the same
configuration and seed reproduce exactly. The consequence is that HISTORICAL-WS-asrun
reproduces the archived *solver, normalization, weight-grid rule and surface
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

**RQ3: Anchor and payoff provenance.** Holding the NBI formulation fixed, what happens when payoff
and anchor information changes from surrogate-derived to the pre-specified empirical-real
procedure? Contrast: NBI-S $\rightarrow$ NBI-R. The quasi-normal is derived from each arm's own
payoff matrix and moves with the reference by construction: that is the intervention, not a
confound. A pre-declared anchor-injection control accompanies it; the CHIM-extent analysis of
Section 6.4.1 is post hoc.

**RQ4: The boundary case.** What happens on a dataset that failed the pre-campaign front-geometry
screen, which looks for the interior front geometry the RQ2 contrast acts on? The RQ2 contrast on
Spambase, which failed the screen and was retained as a boundary control (Section 5.1), analysed
outside the primary family.

**RQ5: Direct search.** How does the surrogate-assisted family compare with direct search (grid,
random, single-objective Bayesian optimization and TPE, and NSGA-II) at the frozen comparator
budget? That budget is matched to the most expensive arm, not to each arm, so a comparator
receives about twice what WS-S and NBI-S require standalone.

### 4.1 What was frozen

Fixed at protocol tag `xgboost-hpo-protocol-v3`, before any confirmatory arm-level result was
inspected.

*Panel and dataset roles.* From pre-campaign screening on the design rows of one pilot partition:
MAGIC, Adult and Bank Marketing passed the front-geometry screen (a curved front among the
non-dominated design runs) and form the primary panel; Spambase failed it, with a two-point
non-dominated set and curvature undefined, and was retained as a boundary geometry control by a
pre-campaign amendment adopted after this failure was known (amendment 22): the replacement rule
frozen at protocol v1, before the pilot, required a dataset failing criterion 1 or 2 to be
replaced from the registry, and retaining Spambase departs from that rule. Spambase executes in
full, excluded only from the primary inferential family (Section 5.1).

*Primary endpoint and reference.* The CORE-relative hypervolume ratio, CORE being a finite
method-independent empirical set no compared method contributes to. The AUGMENTED-relative ratio
is a mandatory sensitivity, never an alternate primary endpoint; a direction disagreement between
references is itself a reported result. IGD$^+$, generational distance, spacing and joint
non-dominated fraction are secondary: Supplement S10 gives their medians, the intervals declared
for them are not reported, and the one test on them is post hoc (Section 8.7).

*Family and multiplicity.* Within each primary-panel dataset, three contrasts $\times$ one
indicator, Holm-corrected within that dataset; the other arm pairs and all baseline comparisons
are descriptive and uncorrected. Primary evidence is the descriptive triple — median difference
with bootstrap interval, win fraction with Wilson interval, rank-biserial — and the test is
secondary.

*Boundary-control interpretation.* The protocol declared that a null geometry difference on
Spambase would not be evidence against the mechanism, because the pre-campaign screening, on a
single partition, had found a two-point design-row front with no interior point; a non-null
difference is reported and investigated after completion, without revising the protocol.
Section 6.6 reports how far that premise held across the confirmatory partitions, and
Section 8.5 why, beyond its fitting rows, the Spambase objective does not allow either outcome
to be read as evidence about front geometry.

*Budget.* A request-based logical ledger charges every evaluation, cache hit or not, and an arm's
standalone cost is what it would consume run alone. HISTORICAL-WS, WS-S and NBI-S each cost 186,
so RQ1 and RQ2 are budget-matched; NBI-R, the only arm buying anchors with real evaluations,
costs 386, and its 200 anchor evaluations are part of the RQ3 intervention. The comparator budget
is the maximum over arms, not matched to each arm, so that no comparator is handicapped: every
comparator receives 386 (matched NSGA-II, 384). A surrogate-assisted win over a comparator is
therefore conservative, and a comparator lead is not evidence of greater efficiency. The rule
dates from protocol v1 (2026-09-13) and its value of 386 from protocol v2, adopted after the
Stage A pilot revised two protocol parameters; both precede the confirmatory campaign, and
neither is revised now.

*Gate status.* The external surrogate reliability gate is diagnostic, not adaptive: every arm
runs at every replication regardless, and every primary comparison is also reported conditioned
on it (Supplement S11).

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

Roles were assigned by a screening rule frozen before the campaign, applied to the 88 design rows
of each dataset on one partition, which were evaluated before any arm ran; that partition came from a pre-campaign pilot,
is not one of the 30 confirmatory partitions, and its 88 evaluations are also the fitting sample of
the frozen factor model (Section 5.5). Two of its four criteria are structural: a raw-response
conflict between the six quality responses and the raw leaf-count response (criterion 1), and a
non-dominated set of design rows with detectable curvature (criterion 2, the front-geometry screen).
MAGIC, Adult and Bank Marketing met all four and form the *primary geometry panel*. Spambase failed
criterion 2 with a two-point non-dominated set, which the screening rule treated as leaving no
interior geometry for the front-construction contrast to act on. Whether that premise held in
execution is a result, reported in Section 6.6 together with the other three datasets' design-row
fronts on the 30 confirmatory partitions, not an assumption carried forward here. Spambase was
retained as a boundary geometry control by a pre-campaign amendment adopted after this failure was
known (amendment 22): the replacement rule frozen at protocol v1, before the pilot, required a
dataset failing criterion 1 or 2 to be replaced from the registry, and retaining Spambase departs
from that rule. No replacement dataset was selected, and Spambase runs every arm, budget, seed and
all $R = 30$ replications, excluded only from the primary inferential family. These are pre-campaign
screening measurements, not study results (Supplement S5). The dataset is the unit of
generalization; nothing is pooled.

### 5.2 Decision space

Seven XGBoost hyperparameters are optimized, with the bounds of the archived historical
configuration retained unchanged (Table 1). Every other learner setting is fixed and identical for
every method: `XGBClassifier` with the binary logistic objective, `tree_method="hist"`, one thread
per fit and no early stopping, so every fit runs all `n_estimators` boosting rounds; there is no
class reweighting, the training seed is the replication's seed (Section 5.13), the remaining
parameters that affect training keep their library defaults, and categorical features are one-hot
encoded.

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

The design is the historical pipeline's face-centred central composite in the seven factors, 88
runs, version-controlled, reused byte-identically across datasets and replications and checksummed.
It comprises 64 factorial points, 14 axial points and 10 centre points; face-centred means an axial
distance of $\alpha = 1$, so no run leaves the coded box. The factorial points are the resolution
VII half fraction of $2^7$ with generator $G = ABCDEF$ and defining relation $I = +ABCDEFG$, where
$A$ to $G$ are the factors in the order of Table 1 (so $G$ is `n_estimators`): the 64 corners whose
seven coded levels multiply to $+1$. In this fraction each main effect is aliased only with a
six-factor interaction and each two-factor interaction only with a five-factor interaction, so no
two linear or interaction terms of the full quadratic of Section 5.6 are aliased with each other,
and the axial runs make its squared terms separately estimable.

One *real evaluation* is one stratified, shuffled, seeded 5-fold cross-validation of one
configuration on the 80% training part of one outer partition, that is, five XGBoost fits; the only
exception is the holdout confirmation of Section 5.13, a single fit on that whole training part
scored on the held-out 20%. Outer partitions are resampled $R = 30$ times per dataset on an 80/20
split, and every method sees the same split at a given replication, so comparisons are paired by
partition.

A separate 78-point set, made of the design's complementary half fraction ($I = -ABCDEFG$) and 14
axial runs at half the design's axial distance, is evaluated on the real objectives and is
*audit-only*: disjoint from the 88 design rows by construction, it reaches the surrogate reliability
gate and nothing else. The gate passes when external $R^2 \ge 0.5$ and Spearman $\rho \ge 0.9$ per
objective. Both cut-offs are fixed at protocol level, not calibrated for this problem: they were
carried over unchanged from the companion manuscript in which the gate was first used (Declarations),
frozen before the measurement-validation pilot (Section 9.1) and never moved. The audit set, by
contrast, was adapted to this problem, because in that pilot uniformly sampled points carried too
little response spread for external $R^2$ to be informative. The gate is diagnostic, not adaptive:
every arm runs at every replication whatever it returns, and its pass rate is reported as a
covariate (Supplement S11), never used as a filter.

### 5.4 Responses, transforms and orientation

Seven responses are recorded per evaluation as the mean over folds. Six carry the `quality` role:
`Accuracy_Mean`, `Precision_Mean`, `Recall_Mean`, `Specificity_Mean`, `RocAuc_Mean` and
`LogLoss_Mean`. The first four are computed at a fixed decision threshold of 0.5, and log-loss
probabilities are clipped to $[10^{-7}, 1 - 10^{-7}]$. One response carries the `cost` role:
`Leaves_Mean`, the total leaf count of the fitted ensemble summed over all trees. Mean per-fold
wall-clock time, `Time_MeanFold`, is recorded on every evaluation and is a secondary audit variable
entering no objective. The learner is fitted without class weighting (Section 5.2), and precision is
set to 0 in a fold with no predicted positive. The positive class is 35.2% of MAGIC, 39.4% of
Spambase, 24.9% of Adult and 12.7% of Bank Marketing. Among the 88 design rows on which the factor
model of Section 5.5 is fitted, 13 on Adult, 16 on Bank Marketing, one on Spambase and none on MAGIC
predict no positive case at 0.5, all at the design's lowest learning rate (0.01) and tree count
(50). Their recall is 0, their precision 0 by that convention, their specificity 1 and their
accuracy the majority-class rate, although they still rank cases well above chance (ROC AUC 0.77 to
0.94 on Adult and Bank Marketing, 0.95 on the Spambase row; Section 8.5).

The six quality responses measure different properties. The four threshold metrics, kept for
continuity with the archived pipeline, describe one operating point at 0.5: accuracy is, fold by
fold, the prevalence-weighted mean of recall and specificity, so it adds no dimension of its own,
and on the 88 design rows recall and specificity are negatively rank-correlated on every dataset.
ROC AUC is threshold-free and measures ranking; log loss scores the predicted probabilities and so
reflects calibration as well as discrimination. The responses receive no hand-set weights: the
factor stage of Section 5.5 combines them, together with leaf count, through loadings fitted once on
the 88 design rows, so what the resulting quality composite tracks differs by dataset (Supplement
S4.2).

Transform and orientation are declared per response, not inferred from data. `Leaves_Mean` is
transformed by $\log(1 + \cdot)$; every other response is untransformed. The transform is not
cosmetic: leaf count varies by a factor of 591 to 4,356 over the design box, a quadratic fitted to
it raw reaches in-sample $R^2$ of only 0.49 to 0.65 against 0.95 transformed, and untransformed it
strips the cost surface to an intercept on two datasets. Responses are then canonicalized to
minimization by multiplying by $-1$ where higher is better, giving $M$, whose columns are
standardized to $Z$ using the mean and sample standard deviation (with the $n - 1$ divisor,
$n = 88$) computed on the design side only (the pilot design rows of Section 5.5), with
$\sigma_j \leftarrow 1$ wherever $\sigma_j = 0$.

### 5.5 Latent objective construction

**Extraction and rotation.** PCA of $Z$ retains $k = 3$ components. Loadings are the eigenvector
matrix scaled by the square roots of the eigenvalues, $\Lambda = V
\operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$; this is the matrix
that is rotated and reported, and Varimax gives $\Lambda_R = \Lambda R$. Scores are formed from
*standardized* component scores,

$$S = Z\,V\,\operatorname{diag}(1/\sqrt{\lambda})\,R, \tag{1}$$

then standardized column-wise to $S_z$. The $\operatorname{diag}(1/\sqrt{\lambda})$ term is
load-bearing: raw component scores have variances equal to the eigenvalues, so rotating them
orthogonally mixes axes of unequal scale and yields correlated factors. On this panel the maximum
off-diagonal correlation of the nominally orthogonal factors ranged from 0.379 to 0.698 when raw
scores were rotated, and is below $10^{-15}$ under the algebra above, which also makes $\Lambda_R$
the loading matrix of the scores actually in use, so the role and sign rules below are read off a
matrix that describes what is being oriented (Supplement S4.2).

**Role assignment and sign, both deterministic.** The *cost factor* is the component with the
largest absolute rotated loading on `Leaves_Mean`; the other two are *quality factors*. The cost
factor is oriented so that the rotated loading of `Leaves_Mean` on it is positive, each quality
factor so that the mean rotated loading over the six quality responses is positive: the role-block
mean rather than the single largest loading, because specificity anti-correlates with accuracy,
recall and AUC and dominates the leading quality factor on three of four datasets. After
orientation, a larger score is worse on average over the factor's role block (the six quality
responses, or `Leaves_Mean` for the cost factor), not on every response. Every oriented loading is
reported in Supplement S4.2; each is a correlation on the 88 pilot design rows on which the model is
fitted, where a positive loading marks a response that improves as the score decreases. On those
rows the leading quality factor of Spambase, Adult and Bank Marketing sets canonicalized specificity
($-0.92$, $-0.93$, $-0.89$) against canonicalized recall ($+0.92$, $+0.86$, $+0.87$), and the second
quality factor of MAGIC sets precision and specificity against recall and accuracy. On the same
rows, a lower value of the quality composite defined next goes with higher recall and lower
specificity on Spambase, Adult and Bank Marketing, and with better values of all six quality
responses on MAGIC.

**Objective 1, composite quality.** $f_{\text{quality}}(x) = \sum_{j \in Q} w_j\, s_j(x)$, where $Q$
is the index set of the two quality factors and $s_j(x)$ is the score on factor $j$ obtained by
applying the frozen mapping that produces $S_z$ to the seven responses measured at configuration $x$
(column $j$ of $S_z$ holds $s_j$ at the 88 design rows), with $w_j = h_j / \sum_{l \in Q} h_l$ and
$h_j = \sum_{i=1}^{7} (\Lambda_R)_{ij}^2$: the *rotated* sums of squared loadings, normalized over
the two quality factors. Varimax redistributes variance across components, so an unrotated
eigenvalue $\lambda_j$ and rotated component $j$ are not the same object and indexing one by the
other pairs unrelated quantities. On this panel the rotated shares are 0.556/0.444, 0.775/0.225,
0.547/0.453 and 0.587/0.413 against unrotated 0.706/0.294, 0.837/0.163, 0.872/0.128 and 0.845/0.155;
on Adult the second factor's weight differs by a factor of 3.5; both shares are persisted under
distinct names. Equal weighting, the historical choice, is the sensitivity declared in advance. The
two composites rank the 88 design rows at Spearman 0.987 to 0.991; that is rank agreement on the
fitting sample, not the declared estimand. The declared comparison of returned fronts, with its 20%
rule, was computed after the campaign from stored outputs by rescoring each arm's revalidated
candidates with the equal-weight composite (a post hoc operationalization; no arm was re-solved),
and is reported in Supplement S4.5.

**Objective 2, leaf-count complexity.** $f_{\text{cost}}(x) = s_c(x)$, where $c$ is the index of the
cost-role factor, whose rotated loading on `Leaves_Mean` is 0.89 to 0.98 on this panel: a latent
factor dominated by log leaf count. The `cost` role of Section 5.4 is named after the archived
pipeline's cost response, wall-clock time (Section 3.5); here the role designates model complexity,
measured as total leaf count. This objective is a deterministic model-complexity proxy, never
described as training time: before the campaign, total leaf count tracked measured per-fold time at
Spearman $\approx 0.86$ on 22 of the 88 MAGIC design configurations, an association and not an
identity, and measured time itself differed by a median 1.3% between two fits of one configuration.
The association over all 120 campaign units and the comparison of winning arms under the two cost
definitions, both declared in the frozen protocol, were computed after the campaign from stored
outputs and are reported in Supplement S4.6 (association) and S4.7 (winning arms). Both objectives
are minimized.

**What the two objectives contain.** Each factor score is computed from all of $Z$, so both
objectives are linear scores of all seven standardized responses (Supplement S4.2). The coefficient
of $f_{\text{quality}}$ on the standardized, log-transformed leaf count is $+0.01$ on MAGIC and
$+0.17$ to $+0.19$ on Spambase, Adult and Bank Marketing, so at equal quality responses the
composite prefers fewer leaves. On the 88-row fitting sample the two objectives are linearly
uncorrelated by construction (absolute Pearson correlation below $10^{-15}$), because the quality
factors and the cost factor are orthogonal there. Their rank correlation on that sample, $-0.20$,
$+0.32$, $+0.10$ and $-0.02$ on MAGIC, Spambase, Adult and Bank Marketing, is a nonlinearity
residual whose sign differs across datasets, not a measure of objective conflict; the objective
conflict the study reports is the raw one between the six quality responses and leaf count
(Supplement S5). How closely the composite tracks raw quality, on and beyond its fitting sample, is
reported in Section 8.5 and Supplement S4.3.

**$k = 3$ is fixed at protocol level, not criterion-selected.** It is inherited from the historical
construction, which forced at least three components so that its cost response, wall-clock time, had
a factor of its own beside the quality factors; keeping it lets this study decompose that
construction rather than re-choose its dimensionality. The Kaiser criterion retains 2, 2, 1 and 1
components on MAGIC, Spambase, Adult and Bank Marketing, with $\lambda_3/\lambda_4$ of 2.93, 2.10,
1.34 and 1.06, never three. Adopting those counts would leave a single component to serve both
objectives on Adult and Bank Marketing and would drop MAGIC's third component, which holds most of
the leaf count's variance. Three components carry 87.7% to 93.4% of the total variance of $Z$, and
two carry 78.8% to 80.7%. On Bank Marketing the third and fourth eigenvalues nearly tie, so the
third retained direction is poorly determined. The first four eigenvalues, cumulative variance and
Kaiser counts are reported in Supplement S4.1 as descriptive diagnostics; the component count is
chosen neither by them nor by which value optimizes better.

**One frozen model per dataset, fitted on the 88 design rows only.** The tuple
$(\mu, \sigma, V, \lambda, R, \text{roles}, \text{signs}, \mu_S, \sigma_S, w)$ is fitted once per
dataset on the 88 design rows as evaluated in the pre-campaign pilot, and nothing else: one
cross-validated evaluation of each run, as in Section 5.3, on the 80% training part of a single
stratified 80/20 partition of the dataset, the partition on which the screening of Section 5.1 was
measured. That partition is not one of the 30 confirmatory partitions, so no replication's
evaluations enter the fit; the model's standardization and factor orthogonality are exact on these
pilot rows only. Every replication re-evaluates the same 88 configurations on its own partition, and
the model is applied unchanged to those rows, on which the surfaces of Section 5.6 are fitted, and
to the 78 audit-only rows, every arm's revalidated candidates, every direct baseline, the reference
sets and the holdout confirmation (Supplement S4.1). Two constraints force this scope at once. A
per-replication refit makes the objective a different variable in every pair, so 30 paired indicator
values would not live in one objective space; for the same reason the four datasets, each with its
own fitted model, do not share an objective space, and every contrast is computed within a dataset
(Section 5.15). And an earlier specification that added the 78-point complement, for 166 fitting
points in all, was withdrawn, because that complement *is* the audit-only external construction (64
of its 78 coded points identical to the external set, the other 14 the same axial runs up to integer
rounding of `max_depth`), which would let the gate validate a surface against data that helped
define that surface's target; the intersection is verified as zero in coded space on all four
datasets. The per-replication refit, the model a replication's own design rows would have produced,
is computed only to measure its Tucker congruence with the pilot-fitted model and is never applied.
Its congruences, declared in the frozen protocol and computed after the campaign from stored
outputs, are reported in Supplement S4.4, both under the runner's greedy factor matching, which does
not resolve factor sign, and under a post hoc optimal-assignment matching.

### 5.6 Response surface models

Each objective is modelled by a full quadratic response surface in *coded* units, fitted by ordinary
least squares on the 88 design rows only, with backward elimination at $\alpha = 0.05$ and hierarchy
enforced. Evaluation is deterministic within a replication, so the 10 centre points return identical
responses and provide no estimate of pure error; the elimination's $t$-tests nonetheless count all
88 rows, 79 of them distinct, with $88 - p$ residual degrees of freedom for a $p$-term model. The
surfaces are fitted to the objective of Section 5.5, the factor score rather than a raw response, so
the $\log(1 + \cdot)$ transform of Section 5.4 sits inside the objective and the cost surface is
fitted on the transformed scale by construction. The same procedure, elimination rule and coded
parameterization are used identically by every arm that consumes a surrogate, which is what allows
the arm contrasts of Sections 5.8–5.11 to vary one mechanism at a time. Model order is fixed by this
rule and never revised after the fact. Two of its consequences are carried rather than smoothed
over. First, the rule is re-applied in every replication, so the retained terms vary across
replications. Second, a hyperparameter eliminated from a surface leaves that surface flat along it,
so that objective's surrogate anchor (Section 5.10) takes its value of that hyperparameter from the
optimizer's starting point rather than from either objective; the payoff matrix, pseudo-nadir and
CHIM (Sections 5.9–5.10) inherit that value, which in general does not minimize the other objective.
The external gate (Section 5.3) measures neither consequence.

One arm is the exception to this shared procedure: HISTORICAL-WS-asrun (Section 5.7), whose surfaces
this study's code fits in the archived uncoded natural-unit parameterization (Table 2), on this
study's objectives.

Everything specified here is scoped to these four datasets, this surrogate architecture, two
objectives and this budget.


### 5.7 HISTORICAL-WS-asrun — the archived solver, called unmodified

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

### 5.8 HISTORICAL-WS — the same weighted sum under the shared specification

HISTORICAL-WS applies the same weighted-sum scalarization, solver and shared symmetric weight grid
as the rest of the arm set, over the same surrogates fitted in coded units to the shared,
minimization-canonicalized objectives. It differs from WS-S in exactly one respect: it retains the
historical observed-extrema normalization box instead of the payoff-matrix reference, so the
HISTORICAL-WS $\to$ WS-S contrast isolates specification and normalization and nothing else.
Standalone cost 186 real evaluations: 88 design, 78 external validation, 20 revalidations.

### 5.9 WS-S — weighted sum over the surrogate payoff reference

WS-S minimizes a weighted sum of min–max normalized surrogate predictions over the shared
symmetric weight grid, normalized against the payoff reference rather than observed extremes. The
utopia end is the payoff matrix diagonal; the other end is the *pseudo-nadir*, its row-wise
maximum. The pseudo-nadir is used because it is what the CHIM construction itself induces, so WS-S
and NBI-S share one reference object and differ only in what they do with it. Each objective's worst
surrogate value over the box, found by anti-optimization, is also recorded per replication (the
frozen protocol calls it the true nadir, although it is not the Pareto nadir); the protocol declared
the contrast's sensitivity to using it as the far end of the box a secondary analysis, but that
analysis was not run and is not reported (Section 8.7). Standalone cost 186.

### 5.10 NBI-S — canonical Normal Boundary Intersection

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

### 5.11 NBI-R — NBI with empirical-real anchors

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
\;=\; \sqrt{8.5} \;=\; 2.9155 .$$

This is not an equicorrelation reading of $\rho = 0.25$, which would give 3.3166. It is a sensitivity
and never the study's significance test: the correction was derived for a
learner's generalization error under repeated resampling, and a Pareto quality indicator computed on
a returned set is not that quantity. The surrogate reliability gate is a covariate, not a filter:
every arm ran at every replication whatever the gate said, and every primary comparison is also
reported conditioned on gate status (Supplement S11). All conclusions are scoped to these four datasets,
this surrogate architecture, two objectives and this budget.


## 6. Results

### 6.1 Protocol integrity, accounting and the screened panel

The confirmatory campaign executed **120 of 120 units with 0 failures and 0 methodological
failures**, in 9 h 46 min of wall clock on 14 workers at one thread each. A unit is one
`(dataset, replication)` pair — 4 datasets $\times$ $R = 30$ replicated outer partitions — and every
arm, control and baseline ran at every unit.

Every request is charged to the requesting method whether or not it is served from cache
(Section 9.3). On that ledger the campaign charged 396,120 logical evaluations, exactly the
declared method-stage registry total, with every one of the 120 units reconciling individually
(Supplement S7).

Objective definitions were frozen before execution and applied, never refitted. One factor model
per dataset was fitted on that dataset's 88 face-centred central composite design rows, as evaluated
on the pre-campaign pilot partition (Section 5.5), and nothing else, then applied unchanged to all 30
replications; the per-replication refit never scores a front and enters only the Tucker-congruence
sensitivity declared in the frozen protocol, aggregated after the campaign from the stored per-unit
values (Supplement S4.4). Fitting once per dataset keeps the
30 paired indicator values inside one objective space, which a per-replication refit would not, and
the 78-point external validation set is disjoint from the fitting sample in coded space on all four
datasets, so the surrogate gate audits a surface against rows that never helped define its target.

The primary endpoint is the **CORE-relative hypervolume ratio**. The CORE reference is a finite,
method-independent empirical set — the 88 design rows plus the 200 anchor-search rows, **288 points
before Pareto filtering, present in every unit** — that no compared method contributes to. It is not
the true Pareto front, and a ratio above 1 is not an error: it means the returned set improved on
that finite reference.

Dataset roles were assigned by the pre-campaign front-geometry screen on the design rows and were
not revised once the confirmatory campaign began (Section 5.1; Supplement S5).
MAGIC, Adult and Bank Marketing passed it (a curved front among the non-dominated design runs) and
form the primary geometry panel; Spambase failed it and was retained as a boundary geometry control
by a pre-campaign amendment (Section 5.1), executed in full under identical arms, budgets and seeds,
and excluded only from the primary inferential family. Dataset is the generalization unit; nothing
is pooled.

Two integrity items are reported rather than smoothed. First, the campaign output contains **140
non-finite cells**, being Schott spacing and its coefficient of variation on the **70
method-by-reference blocks whose front has one point**. Schott spacing requires at least two gaps and
is undefined on a single-point front; `NaN` is the mathematically correct value there. Those blocks
are excluded from spacing summaries rather than propagated through a median. Second, five separate verification passes, carried out with the AI assistance stated in the generative-AI declaration, recomputed eleven headline claims from the raw per-unit artifacts, and the central contrast
was additionally rebuilt from `candidate_revalidation` rows through the frozen factor model with an
independent dominance filter and a two-dimensional hypervolume sweep, reproducing all 480 published
per-unit values with zero difference. Nine claims reproduced exactly; two were corrected, and
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
the frozen primary family, and carries no multiplicity correction. It compares the frozen
dissertation solver, called unmodified with its own asymmetric weight grid on uncoded surfaces and an
observed-extrema box that this study's code computes to the archived specification (Section 3.3),
against the same weighted sum under this campaign's shared coded-unit surrogates and symmetric grid: median paired difference in
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
scalarization on all three primary datasets, those that passed the pre-campaign
front-geometry screen on the design rows (Section 5.1). The medians are +0.0374 on MAGIC,
+0.2549 on Adult and +0.2281 on Bank Marketing; the win/tie/loss counts are
26/0/4, 30/0/0 and 24/0/6; the percentile bootstrap intervals on the median are
$[+0.0144, +0.0687]$, $[+0.1740, +0.5614]$ and $[+0.1670, +0.4741]$, none of them
spanning zero; and the matched-pairs rank-biserial correlation is $+0.92$, $+1.00$ and
$+0.88$. The secondary tests are Holm-significant within every primary dataset
($p = 9.42\times10^{-7}$, $5.59\times10^{-9}$ and $9.72\times10^{-6}$). The geometry effect is the
main positive result of the study, and its scope is exactly these datasets, this
surrogate architecture, two objectives and this budget. It is not a claim that NBI
recovers a true Pareto front: the endpoint is a ratio against CORE, a finite
method-independent empirical reference of 288 points before Pareto filtering, so a
ratio above 1 means only that the candidate set improved on that finite reference
(Table 4).

Two qualifications belong in the same breath as the result. First, the geometry
advantage did not require a surrogate that passed the study's own external reliability
criterion: on Adult and Bank Marketing the composite-quality gate passed in 0 of 30
replications, and the geometry effect is Holm-significant on both. The gate was
diagnostic rather than adaptive (every arm ran at every replication regardless of its
outcome), and absolute surrogate trustworthiness and relative front-construction
geometry are different questions, of which this study answers only the second. Second,
the coarse grid, given 386 real evaluations against NBI-S's 186 under the frozen
maximum-arm comparator rule, had the higher paired median on all four datasets (NBI-S
minus grid: $-0.0628$, $-0.0488$, $-0.1323$ and $-0.1286$ on MAGIC, Adult, Bank Marketing
and Spambase); its lead met the frozen 21-of-30 direction rule (Section 5.15) on MAGIC and
Adult only, its bootstrap interval also excluded zero on Bank Marketing (20/30), and it
was unresolved on Spambase (18/30). The grid's marginal median exceeded NBI-S's on three
of the four datasets (grid 1.0068 vs NBI-S 0.9430 on MAGIC; 1.0476 vs 0.9951 on Adult;
1.0625 vs 0.9979 on Bank Marketing) but not on Spambase, where NBI-S is the higher at
1.0812 against 1.0192. The two statistics answer different questions and only the paired
one is a within-replication comparison. The geometry contrast compares two front
constructions at a fixed surrogate budget; it is not a demonstration that the
surrogate-assisted family is preferable to direct search. Section 6.8 reports this
baseline, and the budget rule behind it, in full.

| Contrast | Dataset | Median diff. | Bootstrap interval | Win/tie/loss | Rank-biserial | Holm $p$ |
|---|---|---:|---:|---:|---:|---:|
| HISTORICAL-WS → WS-S | MAGIC | +0.0000 | [−0.0022, +0.0056] | 14/3/13 | +0.14 | 0.5165 |
| | Adult | −0.0006 | [−0.0033, +0.0000] | 9/4/17 | −0.28 | 0.2087 |
| | Bank Marketing | +0.0000 | [−0.0088, +0.0000] | 8/8/14 | −0.21 | 0.3896 |
| WS-S → NBI-S | MAGIC | **+0.0374** | [+0.0144, +0.0687] | 26/0/4 | +0.92 | 9.42e−07 |
| | Adult | **+0.2549** | [+0.1740, +0.5614] | 30/0/0 | +1.00 | 5.59e−09 |
| | Bank Marketing | **+0.2281** | [+0.1670, +0.4741] | 24/0/6 | +0.88 | 9.72e−06 |
| | Spambase (boundary) | +0.0500 | [−0.0043, +0.4369] | 18/2/10 | +0.41 | 0.1175 |
| NBI-S → NBI-R | MAGIC | **−0.0741** | [−0.0957, −0.0482] | 2/0/28 | −0.99 | 2.79e−08 |
| | Adult | −0.0192 | [−0.1610, +0.0299] | 11/0/19 | −0.37 | 0.1607 |
| | Bank Marketing | **−0.4093** | [−0.5821, −0.2059] | 7/0/23 | −0.82 | 4.73e−05 |

**Table 4.** The primary family: three contrasts per dataset on the CORE-relative hypervolume ratio, differences formed as (second − first), $R=30$; positive values favour the second arm. Bootstrap interval: 95% percentile bootstrap interval of the median difference; medians in bold are those whose interval excludes zero. Win/tie/loss: replications won, tied and lost by the second arm. Rank-biserial: matched-pairs rank-biserial correlation. Holm $p$: $p$-value of the secondary two-sided Wilcoxon signed-rank test, Holm-corrected within dataset over the three contrasts; the descriptive statistics are primary (Section 5.15). Rank-biserial and the test exclude zero differences.

Spambase is shown for completeness and is excluded from the primary inferential family,
having failed the screen and been retained as a boundary control (Section 5.1); it is treated in
Section 6.6.

#### 6.3.1 The geometry effect is indicator-specific

The advantage does not extend uniformly across the frozen secondary indicators, and
that limitation is reported here rather than deferred. On convergence and coverage
geometry the direction matches hypervolume: median IGD⁺ is 0.0511 for NBI-S against
0.0930 for WS-S on MAGIC, 0.0579 against 0.1688 on Adult, and 0.0915 against 0.1948 on
Bank Marketing, lower being better. On the joint non-dominated fraction it does not:
the median differences are +0.0208 on MAGIC, +0.0714 on Adult and +0.0000 on Bank
Marketing. The protocol declared secondary indicators descriptive and untested; a post hoc,
unadjusted signed-rank check against the CORE reference gives no $p$ below 0.05 on any of the four
datasets (raw $p$ = 0.160, 0.093, 0.627 and 0.586 on MAGIC, Adult, Bank Marketing and Spambase).
The defensible statement is therefore narrow: under the
frozen study conditions the advantage is specifically an improvement in convergence and
coverage geometry as measured by hypervolume and IGD⁺. The data do not support a
general better-coverage claim, and in particular do not support a claim that NBI
construction returned more non-dominated solutions (Figure 2).


#### 6.3.2 The pre-declared corrected test

Section 5.15 declared the Nadeau–Bengio corrected resampled $t$ as a sensitivity, with
an inflation of $\sqrt{8.5} = 2.9155$ at $R = 30$ and $n_{\text{test}}/n_{\text{train}} = 0.25$.
It is reported here whichever way it falls, as unadjusted $p$-values. Under the corrected test the geometry
contrast reaches $0.05$ on Adult ($p = 0.0148$) and on neither MAGIC
($p = 0.0860$) nor Bank Marketing ($p = 0.0764$). The
anchor-provenance contrast reaches it on MAGIC ($p = 0.0254$) and on neither
Adult ($p = 0.4161$) nor Bank Marketing ($p = 0.0791$).

This correction was derived for the generalization error of a learner under repeated
resampling, and a Pareto quality indicator computed on a returned set is not that
quantity; applying it here is a transfer, which is why it was declared a sensitivity
and not the study's significance test. The descriptive triple remains primary. But a sensitivity
fixed before the confirmatory campaign is reported whether or not it flatters the
finding, and under this one the geometry contrast clears $0.05$ on one of the three
primary datasets rather than three. Every corrected $p$-value appears in
Supplement S9.

### 6.4 Anchor and payoff provenance: NBI-S versus NBI-R

The third contrast holds the NBI formulation fixed and changes one declared input, the
provenance of the payoff matrix and anchors, from surrogate-derived (NBI-S) to the
pre-specified empirical-real procedure that obtains each anchor by direct search on the true
objectives (NBI-R). Because NBI builds its utopia point, CHIM and quasi-normal direction from
the objective values at its anchors, that one input moves all three, and with them every
subproblem target; that displacement is the intervention, not a confound. The same input
also replaces surrogate-predicted payoff values with real measurements while the subproblems
remain on the fitted surfaces (Section 5.11), and it costs 200 additional real evaluations,
so the contrast estimates the joint effect of these changes.

Replacing surrogate-derived payoff information with the frozen empirical-real anchor
procedure did not improve the confirmatory NBI fronts, and degraded CORE-relative
hypervolume on two of the three primary datasets. The medians are −0.0741 on MAGIC
(2/0/28, $[-0.0957, -0.0482]$, rank-biserial $-0.99$, Holm $p = 2.79\times10^{-8}$), −0.4093 on
Bank Marketing (7/0/23, $[-0.5821, -0.2059]$, rank-biserial $-0.82$, Holm $p = 4.73\times10^{-5}$)
and −0.0192 on Adult (11/0/19, $[-0.1610, +0.0299]$, rank-biserial $-0.37$, Holm $p$ = 0.1607), the
last a non-detection under CORE with an interval spanning zero. This negative result carries the
same evidentiary weight as the geometry result of Section 6.3: spending real
evaluations to relocate the payoff matrix onto empirically obtained anchors did not buy
better fronts under these conditions, and no alternative formulation under which it
does is offered here.

#### 6.4.1 What differs mechanically, stated as association

Two diagnostics narrow where the deficit comes from: a prospectively frozen control and a post hoc
geometric measure. The anchor-injection control inserts NBI-R's empirical anchors into NBI-S's
revalidated candidate set and changes nothing else; because the control set is a superset of
NBI-S's, its effect is non-negative by construction. Its median effect is +0.0133 on
MAGIC and +0.0000 on the other three datasets, against full NBI-S → NBI-R gaps of
−0.0741, −0.0192, −0.4093 and −0.6932. Because NBI-R returned none of its own empirical anchors
(Section 5.12), the control bounds what including those anchors adds to NBI-S's set rather than
decomposing the gap: the anchor points are not part of NBI-R's returned set, so the deficit lies in
the candidates NBI-R returns under its relocated frame, the joint change described at the start of
Section 6.4; the control does not establish which part of that change produces it. Solver
behaviour is comparable between the two arms: the certified fraction of the continuous subproblem
solutions (Section 5.10) has median 1.000 over the 240 arm-units, with two exceptions at 0.900 and
0.950; the per-unit maximum equality residual has median $6.6\times10^{-10}$, with a
campaign maximum of $6.9\times10^{-1}$ on a single MAGIC NBI-S unit; and both arms
return 20 distinct realized configurations in 239 of the 240 arm-units, with 19 in that same
unit. Solver behaviour is therefore comparable between the arms and does not account for the
deficit, but it is not identical and we do not state it as a universal.

The second diagnostic is post hoc: it was defined after the confirmatory results were known and
is computed from the matrix $\Phi$ recorded per replication. It is the CHIM extent
$\lVert\Phi_{:,1} - \Phi_{:,2}\rVert$, which at $q = 2$ is the segment along which the subproblems
distribute their targets; NBI-R's is built from real measurements and NBI-S's from surrogate
predictions (Section 5.11). Under empirical anchors it contracts to a median 20–81% of its
surrogate-derived extent: median ratio 0.811 on MAGIC, 0.204 on Adult, 0.272 on Bank Marketing and
0.370 on Spambase, with per-replication ratios from 0.022 to 1.297, so the extent grew in some
replications on MAGIC and Spambase. Within-dataset Spearman correlations between the
per-replication extent ratio and the per-replication hypervolume gap are +0.683 on Adult, +0.678 on
Bank Marketing, +0.320 on Spambase and −0.111 on MAGIC. This is a post hoc association measured
across 30 replications within a dataset, clear on Adult and Bank Marketing, weak on Spambase and
absent on MAGIC; it is not a demonstrated cause of the deficit, and it is not presented as one
(Figure 4).

### 6.5 Reference sensitivity

Every dataset-by-contrast conclusion was recomputed under both the primary
method-independent CORE reference and the mandatory AUGMENTED sensitivity reference.
Eleven of the twelve conclusions agree in direction and in significance, and no cell
anywhere flips direction (Figure 5). The single disagreement is Adult's anchor-provenance
contrast: under CORE the median is −0.0192 with Holm $p$ = 0.1607, non-significant;
under AUGMENTED the median is −0.0535 with Holm $p$ = 0.0174, significant. The sign is
negative under both references, so the disagreement is one of resolution, not of
direction. Because each reference also sets its own normalization box, and with it the
hypervolume reference point in objective units (Section 5.14), the disagreement cannot be
attributed to reference membership alone. Under AUGMENTED that cell is significant only on the
secondary test: the
primary descriptive triple remains equivocal, with the bootstrap interval on the median
spanning zero.

CORE remains primary because it was prospectively frozen as the method-independent primary
reference before the confirmatory campaign (amendment 21); AUGMENTED remains a mandatory sensitivity, not an alternative from
which a reference may be selected after the fact. The reported Adult conclusion is
accordingly the CORE non-detection, with the AUGMENTED result stated alongside it here
and in Supplement S9.


### 6.6 The boundary dataset did not resolve

Spambase was separated from the primary inferential family before the confirmatory campaign.
Screening on one pre-campaign pilot partition placed its non-dominated design-row set at two
points with undefined curvature, insufficient interior geometry for a front-construction
contrast to act on, and the protocol recorded before the campaign that a null result there
would be neither evidence for nor against the geometry mechanism. Spambase was retained as a
boundary geometry control by a pre-campaign amendment adopted after this failure was known
(amendment 22): the replacement rule frozen at protocol v1, before the pilot, required a
dataset failing criterion 1 or 2 to be replaced from the registry, and retaining Spambase
departs from that rule (Section 5.1). It was executed in full, with the same arms, budgets,
seeds, evaluation machinery and $R = 30$ replicated outer partitions, and excluded only
from the primary family.

On the geometry contrast the boundary dataset did not resolve. The median paired
difference in CORE-relative hypervolume ratio from WS-S to NBI-S is +0.0500, which
is numerically larger than MAGIC's Holm-significant +0.0374, with 18 wins, 2 ties and 10
losses over 30 replications, a percentile bootstrap interval of $[-0.0043, +0.4369]$
spanning zero, and Holm $p = 0.1175$. What distinguishes the two datasets is dispersion,
not location: the per-replication standard deviation is 1.7706 on Spambase against
0.0594 on MAGIC, roughly thirty times larger, over a range $-5.2874$ to $+5.7457$
(Figure 3).

The interpretation is therefore narrow. This is a failure to resolve the contrast at
$R = 30$, not a demonstration that the effect is absent. It is consistent with the
boundary designation stated before the campaign and does not confirm the boundary
mechanism: we neither read the non-significance as evidence that no geometry effect
exists on Spambase, nor treat it as corroboration of the reasoning behind the
designation. An earlier framing by the present authors asserted the stronger reading and
was refuted before publication by a post-campaign recomputation from the raw unit
artifacts (Section 8.6).

Three qualifications belong with it. First, the premise is less clean in execution than at
screening: the two-point front is a pre-campaign design-row measurement, whereas
across the 30 confirmatory replications the design-row front has median size 3, range 2
to 5, and is exactly 2 in 9 of them and three or more in the other 21. The degeneracy that
motivated the designation is not a fixed property of every replication. The three primary
datasets never showed it: their design-row fronts had at least three points in every
confirmatory replication (median 9, 6 and 5; minimum 6, 4 and 3 on MAGIC, Adult and Bank
Marketing; Supplement S5). Second, Spambase is not globally null: its
anchor-provenance contrast (Section 6.4) is strongly negative and Holm-significant at median
$-0.6932$, the largest such deficit in the panel, and its historical reconstruction gap
is also the panel's largest. Third, all of these results are measured in Spambase's frozen
objective space, whose quality composite agrees only weakly with equal-weight raw quality on
its fitting rows and is negatively rank-correlated with it over configurations drawn
uniformly from the decision box (post hoc; Supplement S4.3 and Section 8.5), so the Spambase
result cannot be read as evidence about front geometry.

The dispersion should be read with the scale of the endpoint (Section 6.1): the
CORE-relative hypervolume ratio is not bounded above by one, exceeds 1 in 35% of
Spambase method-unit rows, and reaches the campaign's overall maximum of 9.912 there,
against 2.349 outside Spambase. The large excursions concentrate on the dataset whose
finite reference set is least informative about the trade-off surface.

### 6.7 The geometry advantage persisted where the surrogate failed its own gate

The external reliability gate is an audit-only diagnostic. It scores the fitted
surrogates on a 78-point set (the design's complementary half fraction plus 14 axial
runs at half radius) that no fitting stage can reach, passing when external $R^2 \ge
0.5$ and Spearman $\rho \ge 0.9$ for the objective. It is diagnostic, not adaptive: every
arm ran at every replication whatever the gate said, so its pass rate is a result rather
than a filter, and the primary comparisons are additionally reported conditioned on gate
status (Supplement S11).

Over the confirmatory campaign the composite-quality gate passed in 8 of 30
replications on MAGIC and 9 of 30 on Spambase, and in 0 of 30 on both Adult and
Bank Marketing. The cost gate, on the complexity objective that leaf count dominates, passed
30/30, 30/30, 22/30 and 27/30 respectively (Supplement S11). Every failure, on either
objective, was on the rank criterion alone: external $R^2$ ranged from 0.574 to 0.962 and
met its cut-off in all 240 unit-by-objective cells, while the composite-quality Spearman
$\rho$ had medians of 0.854, 0.891, 0.867 and 0.855 on MAGIC, Spambase, Adult and Bank
Marketing.

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

On the CORE-relative hypervolume ratio, the coarse grid, given 386 real evaluations against
NBI-S's 186 under the frozen maximum-arm comparator rule, had the higher paired median on all
four datasets; its lead met the frozen 21-of-30 direction rule (Section 5.15) on MAGIC and
Adult only, its bootstrap interval also excluded zero on Bank Marketing (20/30), and it was
unresolved on Spambase (18/30). The grid's marginal median is the higher of the two on MAGIC,
Adult and Bank Marketing, but not on Spambase, where NBI-S is higher at 1.0812 against
1.0192. The paired statistic is the within-replication comparison; the marginal medians are
not, and the two do not have to agree (Table 5, Figure 6). Against NBI-S:

| dataset | NBI-S median | GRID median | median paired difference (NBI-S − GRID) | NBI-S wins | raw $p$ |
|---|---:|---:|---:|---:|---:|
| MAGIC | 0.9430 | 1.0068 | −0.0628 | 0/30 | 1.86e−09 |
| Adult | 0.9951 | 1.0476 | −0.0488 | 5/30 | 1.37e−04 |
| Bank Marketing | 0.9979 | 1.0625 | −0.1323 | 10/30 | 0.1642 |
| Spambase | 1.0812 | 1.0192 | −0.1286 | 12/30 | 0.8774 |

**Table 5.** NBI-S against the coarse grid: median CORE-relative hypervolume ratio of each method, median paired difference (NBI-S − GRID), NBI-S wins out of 30 replications, and the raw $p$ of a post hoc, unadjusted two-sided Wilcoxon signed-rank test (descriptive; outside the frozen primary family). Direct-search comparators operate under the frozen comparator budget of 386 evaluations; WS-S and NBI-S cost 186 standalone.

The raw $p$ is below 0.05 on MAGIC and Adult and not on Bank Marketing or Spambase. This
contrast is not in the frozen primary family, carries no Holm correction, and is
reported descriptively.

The budgets were not equal. The frozen comparator rule matched comparators to the *most
expensive arm*, not to each arm: the comparator budget is 386 real evaluations, NBI-R's
cost, while WS-S and NBI-S cost 186 standalone, so the comparators received roughly twice
the real evaluations those two arms require (Supplement S7.1). The rule dates from protocol
v1 (2026-09-13) and its value of 386 from protocol v2, adopted the same day after the Stage A
pilot revised two protocol parameters; neither is revised now. The grid result therefore
cannot be stated as an efficiency claim, nor can the asymmetry be used to dismiss it.

The other direct-search comparators, under the same frozen budget, sit below the grid: median
CORE-relative hypervolume ratios of 0.8907 (random) and 0.9580 (NSGA-II) on MAGIC, 0.4369 and
0.8650 on Adult, 0.3190 and 0.9674 on Bank Marketing, and 0.0000 and 0.4164 on Spambase.
Both also sit below NBI-S's marginal median on every dataset except NSGA-II on MAGIC
(Table 5). On the paired statistic only the grid had the higher median against NBI-S on all
four datasets, so that lead belongs to the grid, not to direct search as a class; against
random search NBI-S had the higher paired median on all four datasets, every bootstrap
interval excluding zero. Matched NSGA-II ran at $32 \times 12 = 384$ evaluations, two short
of the comparator budget, a shortfall stated rather than absorbed. The unmatched NSGA-II run,
ten times the matched budget on one replication per dataset, is reported as context only
(Supplement S13.2): no fairness claim attaches to it, it enters no budget-matched
comparison, and it contributes to neither reference set. The frozen protocol excludes the
single-objective Bayesian-optimization and TPE runs from the front-indicator table, because
each optimizes one objective rather than the trade-off; their single-objective endpoints are
reported in Supplement S13.1.

The grid's lead is also qualified by its construction: its 128-point mesh contains the
design's 64 factorial corners, so 64 of its evaluations re-measure points inside the
CORE reference it is scored against, an advantage on a CORE-relative indicator that the
surrogate-assisted arms do not have by construction, since their candidates come from
surrogate subproblems and coincide with a design corner, if at all, only in some units.
Like the budget rule, this construction was fixed before the confirmatory campaign, and
neither is offered as a reason to set the comparison aside: at the budget and construction
this protocol froze, the coarse grid had the higher paired median against NBI-S on all four
datasets.

### 6.9 Holdout confirmation

Each unit re-measured one candidate per arm: the revalidated candidate nearest, after min–max
normalization of the two objectives, to the utopia point of that arm's own revalidated set.
Each selected configuration was refitted on the replication's full training partition and
scored on a held-out partition whose labels were unavailable within the replication until the
confirmation stage (on its overlap with the pre-campaign pilot data, see Supplement S4.1), at
5 audit-only real evaluations per unit, charged to the study and to no arm. Across all four
datasets and all five arms, the median per-replication difference between internal and
holdout accuracy has magnitude below 0.012 in every one of the 20 dataset-by-arm cells, the
largest being 0.0117, and is negative (the held-out partition scoring better than the
internal resampling) in 13 of the 20 cells (Supplement S14). These are descriptive audit
figures read from `analysis/secondary_analysis.json`, not confirmatory claims.

The correct reading is an absence of gross optimism in the returned configurations at
this budget, and nothing more. The stage measures internal-versus-holdout optimism and was
never designed to rank arms, so the ordering of these differences is not interpreted and
a smaller drop is not evidence that an arm is better. Every primary claim in this study
is about the returned front under the frozen conditions, not about the
generalization of a single deployed configuration.

### 6.10 Declared analyses computed after the campaign

Seven analyses that the frozen protocol declared were computed after the campaign from stored
outputs, without new evaluations; Section 8.7 gives their status and the post hoc choices each
required, and none of them enters the primary family. Under the pre-declared equal weighting of the
two quality factors, the returned non-dominated sets differed from the variance-weighted ones in 9
to 23 of 30 replications, more than the 20% threshold of the frozen protocol on every dataset and
for every arm, so the conclusions are conditional on that weighting; every identifying contrast
kept its direction (Supplement S4.5). When measured mean per-fold time replaced the leaf-count cost
axis, the winning arm changed in 9, 17, 16 and 14 of 30 replications on MAGIC, Spambase, Adult and
Bank Marketing; the geometry contrast stayed positive on every dataset, though smaller on Adult and
Bank Marketing, and the anchor-provenance contrast changed sign on Adult (median $+0.0696$, NBI-R
ahead in 18 of 30), where under the leaf-count cost it was a non-detection (Supplement S4.7). Over
all 120 units, raw total leaf count tracked measured time at a median per-unit Spearman correlation
of 0.87 to 0.96 (Supplement S4.6). Under the post hoc optimal matching, factor models refitted on
each replication's own design rows matched the frozen models at a Tucker congruence of at least
0.895 on MAGIC, Spambase and Adult and of 0.751 at worst on Bank Marketing, where 10 of 30
replications fell below the conventional equivalence level of 0.95; the greedy matching recorded by
the runner, which does not resolve factor sign, gives lower values (Supplement S4.4). The
single-objective Bayesian-optimization and TPE endpoints and the unmatched NSGA-II run are context
only and enter no comparison (Supplement S13).


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

On the three primary datasets (those that passed the pre-campaign front-geometry screen on the
design rows, Section 5.1) and 30 replicated outer partitions, under the frozen study
conditions, canonical NBI front construction produced higher real-revalidated
CORE-relative hypervolume than specification-matched weighted scalarization: median paired
differences of $+0.0374$ on MAGIC, $+0.2549$ on Adult and $+0.2281$ on Bank Marketing,
favoured in 26/30, 30/30 and 24/30 replications with every bootstrap interval excluding zero,
rank-biserial $+0.92$, $+1.00$ and $+0.88$, Holm-significant within every primary dataset,
which is the generalization unit (Table 4).

The scope is part of the claim: two objectives, this surrogate architecture, this budget,
these three datasets. Nothing licenses "NBI is better than weighted sum" in general. The
endpoint is a CORE-relative hypervolume ratio against a finite method-independent set of
288 points, where a value above 1 is expected and means only that the returned set
improved on that finite reference.


The strength of the geometry evidence depends on which test is read. Under the
secondary Holm-corrected signed-rank test, the contrast is significant in all
three primary datasets. Under the Nadeau–Bengio correction declared in advance as a
sensitivity, reported unadjusted, it clears $0.05$ on Adult alone ($p = 0.0148$, against
0.0860 and 0.0764). We take the descriptive triple as
primary, as specified before the campaign ran, and report the corrected test because
it was promised, not because it agrees. A reader who weights the corrected test more
heavily than we do should read the geometry result as one clearly resolved dataset and
two suggestive ones.

### 7.3 Why hypervolume and IGD⁺ move while the non-dominated count does not

The advantage is specifically an improvement in convergence and coverage geometry as
measured by hypervolume and IGD⁺: median IGD⁺, NBI-S against WS-S, of 0.0511/0.0930 on
MAGIC, 0.0579/0.1688 on Adult and 0.0915/0.1948 on Bank Marketing. The joint non-dominated
fraction behaves differently: median differences of $+0.0208$, $+0.0714$ and $+0.0000$,
with no detectable difference under CORE on any of the four datasets including Spambase.

This is not a contradiction. Hypervolume and IGD⁺ are positional: they measure proximity
to the reference and dominated space, so moving points closer or redistributing them along
the trade-off changes both. The joint non-dominated fraction is a mutual-dominance count,
and at $q = 2$ points at different positions along one trade-off band are mutually
non-dominated by construction, so an arm can sit nearer the reference while almost none of
its points dominate the other arm's. The geometry result must not be read as NBI returning
more non-dominated solutions, which the joint non-dominated fraction does not show under
CORE on any dataset.

### 7.4 Anchor provenance: a negative result, not a caveat

Holding the NBI formulation fixed and replacing surrogate-derived payoff information with the
pre-specified empirical-real anchor procedure did not improve the confirmatory fronts.
Paired medians are $-0.0741$ on MAGIC, $-0.0192$ on Adult and $-0.4093$ on Bank Marketing,
Holm-significant on MAGIC and Bank Marketing; Adult is a non-detection under CORE with an
interval spanning zero. This carries the same weight here as Section 7.2. Real anchors are the
expensive, better-grounded option, bought with real evaluations by NBI-R alone, and the
result went the other way. The sign is negative under CORE and AUGMENTED alike, with the
panel's one reference disagreement on Adult's anchor contrast, Holm-significant under AUGMENTED
($-0.0535$, Holm $p = 0.0174$) and not under CORE, its bootstrap interval spanning zero under
both references.

### 7.5 CHIM contraction is a post hoc association, and only that

The deficit is not attributable to solver failure. Over the 240 arm-units the
certified fraction has median 1.000 with two exceptions (0.900, 0.950), the
per-unit maximum equality residual has median $6.6\times10^{-10}$ against a
campaign maximum of $6.9\times10^{-1}$ on one unit, and both arms return 20
distinct realized configurations in 239 of 240 arm-units. Comparable, not identical.

The CHIM contraction of Section 6.4.1 is a post hoc association across 30 replications
within a dataset, not a demonstrated cause: it is clear on Adult and Bank Marketing
(within-dataset Spearman between contraction and deficit $+0.683$ and $+0.678$), weak on
Spambase ($+0.320$) and absent on MAGIC ($-0.111$). Nor does the injection control prove the
negative is entirely geometric: the control set is a superset of NBI-S's, so its effect
cannot be negative, and because NBI-R returned none of its own empirical anchors
(Section 5.12) it bounds what adding those anchors to NBI-S's set does rather than
decomposing the gap.

### 7.6 Specification: a non-detection at $R = 30$

Replacing the historical observed-extrema normalization with the payoff-matrix reference
produced no detectable change in CORE-relative hypervolume on any primary dataset: medians
$+0.0000$, $-0.0006$ and $+0.0000$, exact ties in 3, 4 and 8 of 30 replications, Holm $p$
of 0.5165, 0.2087 and 0.3896. This is a non-detection at $R = 30$, not a demonstration
that normalization has no effect — and it is where a DoE-range-normalization explanation
of the historical pipeline's behaviour would have shown, had that been the principal
driver.

### 7.7 The grid baseline, and what it costs the argument

The coarse grid, given 386 real evaluations against NBI-S's 186 under the frozen maximum-arm
comparator rule, had the higher paired median on all four datasets; its lead met the frozen
21-of-30 direction rule on MAGIC and Adult only, its bootstrap interval also excluded zero on
Bank Marketing (20/30), and it was unresolved on Spambase (18/30; Section 6.8). Its marginal
median CORE-relative hypervolume ratio was higher than that of every surrogate-assisted arm on
three of the four datasets: NBI-S against GRID, 0.9430/1.0068 on MAGIC, 0.9951/1.0476 on
Adult, 0.9979/1.0625 on Bank Marketing and 1.0812/1.0192 on Spambase, NBI-S winning 0/30,
5/30, 10/30 and 12/30 replications (Supplement S13). The comparator budget is the maximum over
arms, set by NBI-R, so the comparators received roughly twice the real evaluations WS-S and
NBI-S require. The contrast is outside the frozen primary family and uncorrected, and its raw
$p$ is below 0.05 on MAGIC and Adult only.

This is stated plainly rather than deflected: at the budget this protocol froze before the
confirmatory campaign, on this panel, the coarse grid had the higher paired median against
NBI-S on all four datasets. The matched-to-the-most-expensive-arm rule bears on how large the
deficit is; it is not grounds for setting it aside, and is not revised now the result is
known. The geometry finding says which construction to use *given* that a front is built on a
surrogate, not that this family was the right instrument here.

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
one. The regime removes one alternative explanation: the effect is not confined to
replications whose surface met the frozen criterion.

### 7.9 The estimand is the returned front

Every result above is a property of the set an arm returns, after real revalidation of
every candidate and scoring against a finite method-independent reference — not the
out-of-sample performance of a configuration a practitioner would deploy. No selection
rule collapses the front to a point, and the holdout stage is an audit. A median advantage
of $+0.2549$ on Adult says NBI-S's front occupies more space relative to CORE than WS-S's;
it does not say by how much any chosen model improves.

### 7.10 What the decomposition does not establish

Spambase was separated from the primary family before the confirmatory campaign because
screening showed insufficient interior front geometry in its frozen objective space. Its
geometry contrast, a median of $+0.0500$ against MAGIC's Holm-significant $+0.0374$ but with
roughly thirty times MAGIC's dispersion and an interval spanning zero, did not resolve
(Section 6.6). That is consistent with the boundary designation; it does not confirm the
boundary mechanism. Because Spambase's quality composite is negatively rank-correlated with
raw quality over configurations drawn uniformly from the decision box (post hoc; Section 8.5,
Supplement S4.3), the result also cannot be read as evidence about front geometry. Nor is
Spambase globally null: its
anchor-provenance contrast is strongly negative and Holm-significant.

More broadly: four datasets, two objectives, one surrogate class, one budget rule. Within
that envelope geometry carries the effect on the three primary datasets and is unresolved on
the fourth, specification does not, and provenance is unhelpful in a way that a post hoc
analysis associates with CHIM contraction. Whether those orderings survive three
objectives, another surrogate class, or a budget at which surrogate assistance is
competitive with the coarse grid, this design was not built to answer.


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
experiments, not extrapolations of this one. So are comparisons with multiobjective Bayesian
optimization or multi-fidelity methods, none of which was run: Bayesian optimization and TPE each
optimized one objective, and the only comparator guided by both is one untuned NSGA-II
configuration. Multi-fidelity methods were scoped out before the campaign, because comparing them
under the frozen comparator budget would first require deciding what a partial-fidelity evaluation
costs; they are the obvious next comparison.

The historical arm is likewise a single case. HISTORICAL-WS-asrun calls the solver of one archived
implementation unmodified, on surfaces and an observed-extrema box that this study's code computes to
the archived specification for this study's objectives (Section 5.7); code-level examination
established that this implementation built its front by normalized weighted scalarization while the
accompanying historical text used NBI terminology. That is a statement about one archived artifact,
not about a class of published pipelines, and supports no estimate of how common the conflation is.

### 8.2 The budget rule favours the comparators

The frozen comparator budget is the maximum over arms, NBI-R's 386 evaluations, while WS-S and
NBI-S cost 186 standalone, so the comparators received roughly twice the real evaluations of those
two arms. The coarse grid had that advantage when it obtained the higher paired median against NBI-S
on all four datasets, a lead that met the frozen 21-of-30 direction rule on MAGIC and Adult only and whose
bootstrap interval also excluded zero on Bank Marketing (Section 6.8). The budget rule dates from
protocol v1 (2026-09-13), before any arm was implemented, and its value of 386 from protocol v2;
neither is revised now; the study can say neither what the surrogate-assisted arms would do at parity nor
that the grid is more efficient. A budget-matched-per-arm comparison was not run.

### 8.3 The reference is finite

CORE is a finite, method-independent empirical set of 288 points before Pareto filtering. It
is not the true Pareto front, the primary endpoint is a CORE-relative hypervolume ratio rather
than a fraction of attainable hypervolume, and the ratio exceeds 1 in between 13% (MAGIC) and 35%
(Spambase) of each dataset's method-unit rows (Supplement S15). Ratios above 1 mean a candidate
set improved on that finite reference; they are not errors, but the endpoint's upper scale is set
by an empirical artifact whose composition would change under a different anchor search.

### 8.4 What the study did not resolve

**Spambase.** The boundary dataset did not resolve the geometry contrast at $R = 30$
(Section 6.6). That is consistent with its boundary designation and does not confirm the boundary
mechanism, and Section 8.5 explains why it is not evidence about front geometry either.

**The CHIM association is post hoc and not causal.** The CHIM-extent diagnostic was defined after
the confirmatory results were known (Section 6.4.1), and its link to the NBI-S → NBI-R deficit is an
association across 30 replications within a dataset: clear on Adult and Bank Marketing, weak on
Spambase and absent on MAGIC ($\rho = -0.11$), where the injection control is itself nonzero. The
anchor-injection control cannot apportion the deficit either, because NBI-R never returned its own
empirical anchors (Section 5.12); it bounds only what adding them to NBI-S's set changes. No
intervention manipulated CHIM extent directly, so the mechanism remains a hypothesis (Figure 4).

### 8.5 Known weaknesses in the frozen protocol

**Screening criterion 1 is weaker than what it replaced.** Its original estimator was
withdrawn when the factor algebra producing it was retracted, and the replacement (raw
standardized quality badness against the raw leaf-count response) cannot fail a dataset whose
raw metrics trade off while its latent composite does not, which is the configuration the
corrected factor model produces on Spambase and Adult. The panel is screened for objective
conflict by a measurement that few candidates would fail. This is carried forward as a
limitation rather than repaired.

**The quality composite is a latent construct that depends on the 0.5 threshold.** It tracks
different things on different datasets (Sections 5.4 and 5.5; Supplement S4.2). On MAGIC it tracks
all six quality responses. On Spambase, Adult and Bank Marketing its leading quality factor is a
recall-versus-specificity axis at 0.5: across the fitting rows lower composite values go with
higher recall and lower specificity, a configuration at least as good on all six quality responses
can receive the worse composite value (Supplement S4.3), and on Adult and Bank Marketing the fitting
rows that predict no positive case receive the worst values, so the composite depends on where
predicted probabilities fall relative to 0.5, not only on how well cases are ranked. On Spambase it
agrees only weakly with equal-weight raw quality on the fitting rows (Spearman 0.243, against 0.835
to 0.929 elsewhere), and the protocol's orientation check used design configurations only; in a post
hoc check over the configurations random search drew uniformly from the box it is negatively aligned
in all 30 replications (median $-0.39$; Supplement S4.3), so the Spambase result cannot be read as
evidence about front geometry. Every method is scored by the same frozen composite, so the contrasts
remain valid comparisons within one objective space, with magnitudes in that space; a returned front
reads as a quality-versus-complexity trade-off on MAGIC, with qualification on Adult and Bank
Marketing, and not on Spambase. The arms were not run on a quality objective without the 0.5
threshold, which would be a separate experiment, and metrics at other thresholds cannot be
recomputed from the stored fold means, which keep no per-fold predicted probabilities.

**The surrogate failed its own gate in most replications.** The composite-quality gate passed
in 8/30 replications on MAGIC and 9/30 on Spambase, and in 0/30 on Adult and Bank Marketing.
The gate is diagnostic, not adaptive: no replication was removed, so the Adult and Bank Marketing
geometry effects come entirely from failing replications (Section 6.7), and the study measures
relative front-construction geometry, not absolute surrogate trustworthiness (Section 7.8).

**Undefined spacing.** Schott spacing requires at least two gaps. It is undefined on
single-point fronts, giving 140 non-finite cells across the 70 method-by-reference blocks whose
front has one point. Those cells are counted and excluded from spacing summaries rather than
propagated, so spacing is summarized over a subset of blocks.

**Single machine.** All 120 units ran on one machine, 14 workers × 1 thread, 9h46m wall clock.
No cross-platform or cross-BLAS replication was attempted, so hardware-independent numerical
reproducibility is untested and timing figures are not machine-independent.

### 8.6 The process

The protocol reached its frozen state through 23 numbered amendments and seven adversarial
pre-campaign reviews; the first review found ten blocking defects and the last none, and six
regressions introduced while fixing earlier findings were fixed before the freeze. Two amendments
are recorded process failures: a descoped stage that was not logged, and a decision rule committed
alongside the measurement it had to predate. The amendment ledger records that no arm result had
been observed when any amendment was made. One pre-freeze engineering smoke unit on Spambase
executed every arm and the metrics stage on 2026-09-14, before amendments 21 (primary reference) and
22 (Spambase's role) and before the v3 tag; the ledger records that only its execution and
accounting were inspected, and its outputs are not archived. After the campaign and before the
manuscript was drafted, two analysis claims were corrected when recomputation from the raw unit
artifacts refuted them; the corrections are recorded in the verified-claims file (Section 9.1). A
protocol requiring this much repair before execution is prospectively frozen, before the
confirmatory campaign, not pre-registered, and should be read as such.

### 8.7 Deviations from the frozen protocol

No amendment followed the freeze; departures in execution and reporting are listed here. Seven
analyses that the frozen protocol declared and the committed pipeline did not produce were computed
after the campaign from stored outputs, without new evaluations: the single-objective endpoints of
the Bayesian-optimization and TPE runs (Supplement S13.1), the unmatched NSGA-II table
(Supplement S13.2), the per-replication Tucker congruence (Supplement S4.4), the returned-front
weighting comparison with its 20% rule (Supplement S4.5), the leaf-count and wall-clock time
association over all 120 units (Supplement S4.6), the winning-arm comparison under the two cost
definitions (Supplement S4.7) and the gate-conditioned specification contrast (Supplement S11).
Choices the protocol did not fix are post hoc: optimal rather than greedy matching of refitted to
frozen factors, rescoring the returned candidates, instead of re-running the arms, for the weighting
and cost-definition comparisons, reading the 20% rule as met when any arm's fronts differ, the CORE
context beside the Bayesian-optimization and TPE endpoints, and the form of the secondary
significance test (Section 5.15). The eigenvalues and variance explained in Supplement S4.1 are
those of each dataset's frozen model, which every replication applies; the per-replication refits
stored none. Five analyses added after the confirmatory results are post hoc: the CHIM-extent
diagnostic (Section 6.4.1), the composite's behaviour beyond its fitting rows (Supplement S4.3),
the signed-rank check of the joint non-dominated fraction (Section 6.3.1), the signed-rank test of
the grid baseline against NBI-S (Table 5) and the breakdown of gate failures by criterion
(Section 6.7). Three declared items were
not executed: the sensitivity of the WS-S → NBI-S contrast to using each objective's worst value
over the box as the far end of the WS-S normalization box (Section 5.9), which would need new real
evaluations; the fixed-integer re-solve of each NBI subproblem required by section 9 of the frozen
protocol (Section 5.10), so the rounded points actually evaluated carry no equality certificate,
although their rounding displacement is summarized in Supplement S12.2; and the restricted-$t$
demonstration (item 3 of section 11 of the frozen protocol). Two declared summaries are not tabulated: the Wilson
intervals of the win fractions in the descriptive triple (section 10 of the frozen protocol), whose
role the 21-of-30 direction rule takes (Section 5.15), and the intervals declared for the secondary
indicators, reported only as medians in Supplement S10.


## 9. Reproducibility, data and code availability

### 9.1 Version tags

Five tags separate what was specified from what was measured (Supplement S1). `v0.1.0-dissertation`
(`67d9fe5`) is the historical implementation this study reconstructs; it is read and executed as
archived, never edited. `xgboost-hpo-protocol-v1` froze the specification before the
measurement-validation pilot, `-v2` incorporated the pilot, and `-v3` (`9b15ba7`) is the
confirmatory protocol under which every unit ran. These tags record what was specified before the
confirmatory campaign; the 23 numbered amendments between them carry the cause of each change beside it.
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

Two ledgers are kept separately and both reported. The logical ledger charges every evaluation a
method requests, hit or miss; it is the scientific budget and the only quantity entering a fairness
comparison. The campaign consumed 396,120 logical evaluations, exactly the declared registry total,
every one of the 120 units reconciling individually. The physical ledger counts misses only:
377,316 unique physical evaluations, a 4.75% cache hit rate. That is an engineering quantity that
enters no comparison between methods. The code's per-unit accounting stores this count under the
key `unique_physical_fits`.

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
writing. Parts of the campaign, analysis and audit code were written with AI coding assistance,
recorded as co-authorship in the commit history (Anthropic's Claude: Claude Opus 4.7 and
Claude Opus 5, used through Claude Code, named as co-author on all 66 commits made for this
study to its code, analysis scripts and tests).

Raw datasets and heavy evaluation caches are deliberately unversioned. The four panel datasets are
UCI or OpenML mirrors, obtainable independently and checksum-verified on acquisition; the 88-run
face-centred central composite design is versioned with its SHA-256. The per-unit experiment tree,
including one SQLite evaluation cache per unit, is not committed. Committed instead are the frozen
code, the protocol, the amendment ledger, the per-unit indicator and NBI mechanism tables and the
analysis artifacts, so every reported quantity traces to a committed artifact without re-running the
campaign. Regenerating the per-unit tables, and the post-campaign computations listed in
Section 8.7, needs the uncommitted per-unit tree.


## 10. Conclusion

Surrogate-assisted multiobjective hyperparameter optimization pipelines change several
things at once. When such a pipeline is revised, the revision typically alters the
scalarization specification, the geometry by which a front is constructed, and the
information from which anchors and the payoff matrix are built. A difference in the
returned front cannot then be attributed to any of them. This study separated the three
under a protocol frozen after documented measurement-validation pilots and before the
confirmatory campaign, and revalidated every returned candidate on the real learner.

Two of the three mechanisms produced clear answers, in opposite directions. Correcting
the archived scalarization specification and its observed-extrema normalization changed
nothing detectable. Replacing weighted scalarization with canonical Normal Boundary
Intersection, holding everything else fixed, improved CORE-relative hypervolume on all
three screened datasets, with 26, 30 and 24 of 30 paired wins, every bootstrap interval on the
median difference excluding zero and the secondary test Holm-significant within each dataset;
under the pre-declared Nadeau–Bengio sensitivity the improvement reached unadjusted $p < 0.05$
on Adult alone. The third mechanism answered in the direction opposite
to the one the pipeline's design would suggest: substituting a pre-specified
empirical-real anchor procedure for surrogate-derived payoff information gave no
benefit, and degraded the front on two of the three datasets, of which only MAGIC reached
unadjusted $p < 0.05$ under the Nadeau–Bengio sensitivity. The frozen anchor-injection
control, which adds the empirical anchors to NBI-S's candidates and changes nothing else,
moved the indicator by approximately zero on three of four datasets and by $+0.0133$ on
MAGIC; because NBI-R returned none of its empirical anchors, the control bounds what anchor
inclusion alone adds rather than decomposing the deficit. In a post hoc analysis the payoff
matrices show a contraction of the convex hull of individual minima that is associated with
the deficit, clearly on Adult and Bank Marketing, weakly on Spambase and not on MAGIC. Both
are reported as associations, not as causes.

Two results limit how far the geometry finding should be carried. The advantage is
specific to hypervolume and IGD$^{+}$; the joint non-dominated fraction shows no detectable
difference under CORE on any dataset in the panel. And under the frozen direct-search
comparator budget, the median paired difference favoured a coarse grid over NBI-S on all four
datasets: the grid's lead met the frozen 21-of-30 direction rule on MAGIC and Adult only, its
bootstrap interval also excluded zero on Bank Marketing (20/30 wins), and it was unresolved on
Spambase (18/30); the grid also held the higher marginal median on three of the four datasets,
but not on Spambase. The grid received 386 real evaluations against the 186 that WS-S and NBI-S
cost standalone. That asymmetry is a property of the frozen comparator rule, not a defence of
the arms: at the budget this protocol fixed in advance, the grid led.

What the study establishes is therefore narrower than "NBI is better".
Within this decision space, this surrogate architecture, two objectives and this budget, and on
the three screened datasets, front-construction geometry is the mechanism that moved the
returned front; specification did not; and anchor provenance moved it the wrong way. Spambase,
which failed the pre-campaign front-geometry screen and was retained as a boundary control
before the confirmatory campaign, did not resolve; because its quality composite is negatively
aligned with raw quality over configurations drawn uniformly from the decision box (post hoc; Section 8.5), that result is not
evidence about front geometry in either direction. Whether the geometry advantage survives at
larger budgets, at more than two objectives, or against direct search when the arms receive
the comparators' 386 evaluations, this design cannot say, and we have tried not to say it.


## Declarations

**Prior dissemination and relationship to a dissertation.** This study extends work
first developed in the first author's master's dissertation (Ribeiro 2026; available open access from the UNIFEI institutional repository), which applied a
PCA/Varimax latent-objective construction to XGBoost hyperparameter optimization. The present
work starts from that pipeline's archived implementation, separates three mechanisms it changed
together, and evaluates them under a protocol frozen before the confirmatory campaign. The
archived solver is reproduced by calling its frozen code unmodified rather than by
reimplementation, and is reported neutrally: a code-level examination established that it
performs normalized weighted scalarization for front construction although the dissertation
text uses NBI terminology. No claim is made about the correctness of the dissertation's reported
results, whose objective space differs from this study's (Section 3.5).

**Relationship to companion work by the same authors.** Three instruments in this study were
first used in a companion manuscript by the present authors (Ribeiro, Pereira and de Paiva 2026;
unpublished, frozen and not submitted): the external surrogate reliability gate, the rule of
matching direct-search comparators to the most expensive arm rather than pairwise to each arm,
and the anchor-injection control. They are applied here, not introduced. The methodological
lineage of the latent-objective construction is set out in Sections 2.5 and 2.7 and includes
Pereira et al. (2025) and de Azevedo et al. (2026), with which this work shares co-authors.

**Overlap statement.** Besides the archived implementation and the latent-objective
construction described above, the study takes three elements of its experimental set-up from
the dissertation, each identified where it is used: the MAGIC Gamma Telescope dataset
(Section 5.1), the seven hyperparameters and their bounds (Section 5.2, Table 1), and the
88-run face-centred central composite design (Section 5.3). The other three datasets were
selected for this study (Section 5.1); the frozen protocol, the confirmatory campaign and every
result reported here are specific to this study, and no dataset is shared with the companion
manuscript. Apart from the bounds restated in Table 1, no figure, table or result is reproduced
from the dissertation or from the companion manuscript.

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

During the preparation of this work, the authors used Anthropic's Claude (Claude Opus 4.7 and Claude Opus 5, through Claude Code) and OpenAI's ChatGPT in order to assist with manuscript drafting, restructuring, language refinement, consistency checking and editorial review; the collation and checking of references; writing and revising the research and analysis code; and the pre-campaign adversarial reviews of the protocol and the post-campaign verification of the results. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

The study's computational methods are described in Section 5, and every reported result derives from the committed, version-controlled artifacts listed in Section 9.

## References

DOIs and identifiers are given where they exist, and machine-readable records are in `manuscript/references.bib`. Fields a source did not state are omitted rather than inferred; two such omissions are noted explicitly.

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

Ribeiro, C. T. (2026). *Meta-otimização dos hiperparâmetros do algoritmo XGBoost para
classificação binária: uma integração entre planejamento de experimentos e o método da
interseção normal à fronteira* (in Portuguese). MSc dissertation, Universidade Federal de
Itajubá (UNIFEI). https://repositorio.unifei.edu.br/jspui/handle/123456789/4372

Ribeiro, C. T., Pereira, M. C. and de Paiva, A. P. (2026). *When is surrogate-assisted
multiobjective ensemble weighting trustworthy? A replicated study of mixture designs, Scheffé
surfaces and normal boundary intersection on the classifier-weight simplex*. Unpublished
manuscript, frozen and not submitted.

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
reference. The three primary contrasts are marked.

**Figure 2. The geometry contrast, per replication.** Paired differences in
CORE-relative hypervolume ratio, NBI-S minus WS-S, sorted within each dataset, $R = 30$,
CORE reference. Blue favours NBI-S. The dashed line is the median. Note that the
vertical scales differ by dataset.

**Figure 3. Dispersion across the panel.** The same paired differences as Figure 2,
shown as distributions. The boundary dataset's spread is roughly thirty times MAGIC's,
which is why its contrast does not resolve at $R = 30$ despite a larger point estimate.

**Figure 4. Anchor provenance and the associated geometry.** Left: the full
NBI-S $\rightarrow$ NBI-R difference beside the anchor-injection control, which inserts
NBI-R's empirical anchors into NBI-S's set and changes nothing else. Right (post hoc): the
CHIM extent under empirical anchors relative to surrogate-derived anchors, as a per-dataset
median. The juxtaposition is a post hoc association, not a demonstrated cause: clear on Adult
and Bank Marketing, weak on Spambase and absent on MAGIC.

**Figure 5. Reference sensitivity.** Median paired difference under the CORE reference
(primary, horizontal) against the AUGMENTED reference (mandatory sensitivity,
vertical), for all twelve dataset-by-contrast cells. Red marks the single cell whose
significance differs between references; no cell changes direction.

**Figure 6. Surrogate-gate regime and baselines.** Left: per-dataset pass rates of the
frozen external reliability criterion over $R = 30$; the gate is diagnostic and filters
nothing. Right: median CORE-relative hypervolume ratio by method. The dashed rule marks
the CORE reference itself and is **not** an upper bound — a ratio above it means the
candidate set improved on the finite reference.
