# Novelty matrix — seventeen elements of the present study

Companion to `literature_review.md`. Verdicts are drawn only from records verified in `lit/` (the seven cluster
reviews plus `predecessor_lineage.md`); every bibkey used exists in a cluster `.bib` file and in the manuscript's
merged `references.bib`. Three verdict labels are used:

- **KNOWN PRIOR WORK** — a verified study does substantially this thing.
- **PARTIAL PRIOR WORK** — the idea exists in a weaker, narrower or differently-motivated form; the specific
  instantiation used here does not.
- **NO DIRECT MATCH FOUND** — the searches described in the cluster reviews did not surface a study that does this.
  This phrasing is deliberate: it records the outcome of a literature search, not a proof of absence.

A fourth tag, **OWN PRIOR WORK — INHERITED**, is added wherever the element comes from Pereira, Tertuliano Ribeiro,
Mendes, Campos & de Paiva (2025), *EAAI* 162:112510 (`pereira2025hybrid`), from the first author's dissertation
(`ribeiro2026dissertation`), or from the wider de Paiva/Balestrassi group lineage on which both rest.

Where a cluster review reported that a question could not be settled — a blocked publisher, an unread full text, an
equation rendered as an image — that is stated in the entry rather than smoothed over.

---

## Element 1 — Classifier ensemble weights formally treated as mixture components on a simplex

**Verdict: KNOWN PRIOR WORK. Also OWN PRIOR WORK — INHERITED (group lineage).**

`kwon2024ensemble` treats the weights of five heterogeneous base classifiers explicitly as mixture components under
Σw_i = 1 and optimizes them in that space; it stops at a single classification metric (Accuracy or F1) with no
trade-off, no cost and no Pareto construction. `moreira2021ann_ensemble_mde` (group lineage) uses a mixture design to
determine ANN ensemble weights for PV forecasting, and stops at a single error objective (MAPE).
`bacci2019nbi_mixture` (group lineage) puts forecast-combination weights on a Simplex-Lattice and models the
combination's accuracy metrics over it, stopping short of classifiers and of any cost or anchor analysis.
`rocha2025ensemblestlf` states the formulation in exactly our words — a mixture design "on the probability simplex of
ensemble weights" — for three neural regressors, and stops at forecast-error factors. The non-negativity plus
sum-to-one restriction on combination coefficients is older still: `breiman1996stacked` imposes non-negativity as the
operational constraint and analyses the non-negativity-plus-sum-to-one (convex) case as the interpolating regime,
though it never treats that region as an experimental design space. **Caveat:** for `kwon2024ensemble` the sum-to-one
constraint is quotable from the retrieved publisher text but the non-negativity constraint is an image in that record
and is inferred from the mixture framing.

## Element 2 — Controlled statistical mixture design over classifier weights

**Verdict: KNOWN PRIOR WORK for the design itself; PARTIAL PRIOR WORK for the design as used here. Also OWN PRIOR WORK — INHERITED.**

`kwon2024ensemble` runs a mixture design of experiments over the weights of five classifiers and fits its model on the
design points; the retrieved record does not name the design family (lattice, centroid, axial) or the run count, so
the comparison to our 66-run lattice/centroid/axial hybrid cannot be made at that resolution.
`rocha2025ensemblestlf` uses a {3,5} simplex-lattice over ensemble weights (three components).
`bacci2019nbi_mixture` uses a Simplex-Lattice {m,q} over forecast-combination weights;
`leal2022portfolio_doptimal_mde` reduces a {5,10} lattice to 200 runs by D-optimal selection on five components — the
closest precedent for a *budgeted* design on a five-component simplex. `ribeiro2026dissertation` (**OWN PRIOR
WORK — INHERITED**) is the source of the 66-point Simplex-Lattice {3,10} arrangement reused here, although there it
schedules NBI scalarization weights rather than decision variables. What none of them does is reserve an independent
set of unseen compositions from the design for external validation of the fitted surface; every cited design is used
entirely for fitting.

## Element 3 — Scheffé/RSM surrogate of ensemble performance

**Verdict: KNOWN PRIOR WORK. Also OWN PRIOR WORK — INHERITED.**

`kwon2024ensemble` fits a Scheffé canonical polynomial (linear plus binary-interaction terms) to a classification
metric as a function of the classifier weights, reduces it by backward elimination at α = 0.05, and maximizes the
reduced model — this is the same modelling device applied to the same decision space, and it stops at one metric with
no external validation of the polynomial. `bacci2019nbi_mixture` fits Scheffé-type mixture models to factor scores of
forecast-combination metrics; `rocha2025ensemblestlf` fits mixture models over the ensemble-weight simplex (the
canonical Scheffé form is not named in its abstract); `leal2025mixtureweights` fits a quadratic response surface to
the sample log-likelihood over a weight simplex and stops at a single objective and two components.
`pereira2025hybrid` (**OWN PRIOR WORK — INHERITED**) fits Scheffé fourth-order polynomials over a simplex, but to
post-Pareto MCDM indicators (generalized distance, Shannon entropy) rather than to the objective functions; the same
is true of `rocha2021robustpoint`. The model family and its application to a weight simplex are therefore established;
what is not established is fitting it to ROC-AUC and log-loss with an explicit order-selection rule.

## Element 4 — Multiobjective optimization of those surrogate surfaces

**Verdict: KNOWN PRIOR WORK. Also OWN PRIOR WORK — INHERITED.**

`bacci2019nbi_mixture` optimizes mixture-model surfaces over combination weights multiobjectively and reports a Pareto
frontier of weight sets; `rocha2025ensemblestlf` does the same for ensemble mixing weights via NBI on factor surfaces;
`leal2022portfolio_doptimal_mde`, `mendes2016portfolio_mde`, `oliveira2011portfolio` and `monticeli2017portfolio`
optimize multiple responses modelled over a proportion simplex by desirability. `pereira2025hybrid` (**OWN PRIOR WORK
— INHERITED**) is the general statement of "fit surrogates, then run a multiobjective method on them". Outside the
mixture literature, multiobjective optimization of fitted surrogates is the entire subject of the surveys
`tabatabaei2015survey` and `chugh2019survey`. All of these stop at surrogate-space optimization: none re-scores the
resulting front on the true objectives against an independent reference, and none is a classifier-ensemble
instantiation with AUC, log-loss and cost.

## Element 5 — NBI over the ensemble-weight simplex

**Verdict: KNOWN PRIOR WORK. Also OWN PRIOR WORK — INHERITED.**

`rocha2025ensemblestlf` applies NBI to objectives defined over the probability simplex of ensemble weights, and lists
that construction among its own contributions; it stops at three neural regressors with error-metric factors and no
cost, gate, anchor variant or reference front. `bacci2019nbi_mixture` (group lineage) applies NBI to Scheffé mixture
models of forecast-combination weights, producing "different optimal weights set and the Pareto frontier
construction"; it stops at forecast residual metrics. `pereira2025hybrid` and `ribeiro2026dissertation` (**OWN PRIOR
WORK — INHERITED**), along with `rocha2017robustmcdm`, `rocha2021robustpoint` and `azevedo2026nbivrf`, establish the
NBI-plus-mixture-design combination inside the group, but on scalarization-weight simplices rather than on decision
variables. The method itself is `das1998nbi`. **No part of this element may be claimed.**

## Element 6 — Comparison of surrogate-derived anchors against anchors computed from real OOF objectives

**Verdict: NO DIRECT MATCH FOUND.**

The components exist separately and must be credited. `isermann1988payoff` demonstrates computationally that the
minima read off a payoff table built from individual optima "can often be large[ly]" wrong relative to the true minima
over the efficient set — but the error source is degenerate or alternative LP optima, not a metamodel.
`deb2010nadir` shows that the extreme values defining the normalization box are estimates that are hard to obtain
reliably and that better estimation changes downstream results; `he2021normalization` surveys ideal- and nadir-point
estimation across MOEAs and states that inaccurate estimation of the front range degrades performance — again, error
arising inside a search on the *true* objectives. `herrmann2026nonextreme` is the only located work that treats the
*choice* of individual minima as a design decision, replacing extreme minima with non-extreme ones to refine the
utopia–nadir hyperbox; all of its minima are computed on the true objectives. On the other side,
`abolghasemian2022haulage` runs NBI on metamodels with a metamodel-derived payoff matrix and never computes a true
anchor for comparison, while `gellerich2023doenbi` runs NBI on measured objectives, so its anchors are real, but never
contrasts them with surrogate anchors. The cluster D+G search log states the negative explicitly: no source consulted
yields a paper that computes NBI or NC anchors from a surrogate and separately from the true objectives and compares
the resulting frontiers.

## Element 7 — Comparison of surrogate NBI against metamodel-free / real-objective NBI

**Verdict: NO DIRECT MATCH FOUND.**

`gellerich2023doenbi` is the only verified metamodel-free NBI: it derives the experimental plan from the NBI
subproblems and evaluates each subproblem by physical test, precisely because no model mapping settings to quality
figures exists. It is the published precedent for the *idea* of NBI-C, and the manuscript should say so. It does not
run a surrogate NBI on the same problem, so it makes no comparison. `jin2002framework` supplies the mechanism that
makes the comparison worth running — "incorrect convergence will occur if the approximate model has false optima" —
and the remedy of periodic true-function evaluation, but for a single-objective evolution strategy.
`tabatabaei2015survey` names the sequential framework (fit once, then optimize) that this study uses and warns that it
is more exposed to surrogate error than the adaptive framework, without running the contrast.
`deb2019taxonomy` argues that per-objective metamodel errors accumulate and degrade the whole multiobjective
procedure. No verified source pairs a surrogate-objective NBI and a real-objective NBI on one problem and scores both.

## Element 8 — Real-objective OOF revalidation of every multiobjective candidate

**Verdict: PARTIAL PRIOR WORK. Also OWN PRIOR WORK — INHERITED (for the single-point form).**

Confirming *selected* solutions on the real system is standard practice. `lopes2016rpdmnbi` performs physical
confirmation runs with an L9 Taguchi array at the 10%, 50% and 90% weight settings of the frontier and checks them
against prediction intervals; `naves2017nbirfs` runs confirmation experiments and reports that observations fall
inside the prediction intervals; `karl2023svem` assesses candidate optimal formulations with explicit confirmation
runs; `kwon2024ensemble` reports verification of the optimized weights on test data; `ye2025mixinglaws` and
`liu2025regmix` confirm the surrogate-selected mixture with real training runs; and `ribeiro2026dissertation`
(**OWN PRIOR WORK — INHERITED**) states that "candidate solutions are re-evaluated on the real model". In
surrogate-assisted MOO the in-loop version is universal: `knowles2006parego`, `zhang2010moeadego`,
`ponweiser2008smsego`, `chugh2018krvea` and `jin2002framework` all evaluate their infill points on the true objectives
during the search. Where all of them stop: each confirms one or a handful of selected points, or evaluates
individually proposed infill points inside a loop. None re-evaluates *every* point of a completed surrogate-generated
Pareto set on the true objectives before any indicator is computed, so that the front's quality is measured on real
objectives rather than on the surrogate that produced it.

## Element 9 — Empirical Pareto-reference construction independent of the surrogate

**Verdict: PARTIAL PRIOR WORK.**

The indicator apparatus presupposes a reference set: `ishibuchi2015modified` defines IGD+ against "a pre-specified
reference point set" and is only as good as the set supplied; `audet2021performance` surveys the indicator families;
`zitzler2003performance` gives the theory. In benchmark multiobjective optimization the reference front is
analytically known, so the construction problem does not arise. The closest real-problem analogue found is
`borchert2022paretoselect`, which exhaustively benchmarks 13 forecasting methods on 44 datasets and releases all
evaluations, so measured ground truth exists for every candidate and a de facto reference front over a discrete
candidate set can be formed — but it is a benchmark artefact, not a purpose-built reference, and the front is over
model choices rather than over a continuous simplex. Within the authors' lineage, `pereira2026postpareto` and
`azevedo2026nbivrf` use hypervolume, IGD and spacing to rank optimizers head to head, never against a reference front.
The cluster D+G search log records that no consulted source builds an empirical reference by massive sampling plus an
ε-constraint sweep and then scores revalidated candidates against it with IGD+ and hypervolume.

## Element 10 — Inference cost as a multiobjective criterion

**Verdict: KNOWN PRIOR WORK.**

`maier2024hardware` integrates inference time into post-hoc ensemble selection and returns a Pareto front of accurate
and efficient ensembles over 83 tabular classification datasets; `maier2026hapens` extends the line, names the
objective "deployment cost", and identifies memory usage as the most effective cost metric — both are exactly
"inference cost as a Pareto objective for a post-hoc weighted ensemble over cached tabular predictions", and both stop
short of a mixture design, a Scheffé surrogate, NBI or a reference front. `borchert2022paretoselect` builds an
accuracy-versus-latency Pareto front for forecasting model selection using a learned metric surrogate, and stops at
selecting a single model. `zhao2018multiobjective` puts a sparsity ratio in the objective vector alongside false
positive and false negative rates — three objectives, two of them errors and one a parsimony term.
`qian2015pareto` optimizes validation error against the number of base learners with theoretical guarantees.
`gunasekaran2022cocktail` minimizes real dollar cost subject to accuracy and latency. **This element must not be
claimed.** What remains specific here is the *combination* of a ranking metric, a calibration metric and a per-model
inference cost in one three-objective formulation over a designed weight simplex.

## Element 11 — Support-based deployment cost compared against the continuous weighted-cost relaxation

**Verdict: NO DIRECT MATCH FOUND.**

Cluster F's explicit conclusion, after 22 verified entries plus read-but-excluded ones, is that no prior work contrasts
Σ w_i c_i against Σ c_i·1[w_i > ε] for the same weighted ensemble, and none shows the two disagreeing about which
method wins. The literature splits cleanly: papers that price a *deployed set* (`gunasekaran2022cocktail`,
`maier2024hardware`, `maier2026hapens`, `akkerman2026pace`, `ji2023pruning`, `gudipaty2025mel`, `moreira2026energy`,
and implicitly `zhou2002ensembling`); papers that price *per-example adaptive execution*
(`trapeznikov2013multistage`, `xu2013cstc`, `nan2017adaptive`, `chen2020frugalml`); and papers that put a continuous
cost or sparsity term into a scalarized training objective (`xu2012greedy`, `zhang2011sparse`). The three near misses
must be cited as such. `zhang2011sparse` uses a 1-norm penalty as the convex relaxation of a support count and never
audits whether the relaxation and the support agree. A companion of `nan2015budgeted` proves via total unimodularity
that a 0-1 support-cost programme is solved exactly by its LP relaxation — a result in the *opposite* direction, i.e.
the two accountings agree there. `wang2022committees` shows that switching cost accountings reorders which committee
looks best, but across ensemble-versus-cascade *architectures*, not across two cost functions applied to one and the
same weight vector.

## Element 12 — External reliability gate for surrogate adequacy

**Verdict: PARTIAL PRIOR WORK.**

"Validate the surrogate before trusting it" is established, in several forms, none of them a pass/fail admissibility
rule on unseen simplex compositions. `jin2001metamodelling` compares four metamodel families under multiple criteria on
fourteen problems and argues explicitly against judging a metamodel by a single measure of merit on a single test
problem. `abolghasemian2022haulage` validates its regression metamodels with PRESS and R² before running NBI on
them — the closest external analogue of a gate, applied to fit statistics rather than to held-out compositions, and
with no consequence attached other than proceeding. `deb2020surrogate` uses a statistical comparison of the
metamodeling accuracy of ten frameworks to decide which to use in each epoch. `jin2002framework` makes the amount of
true-function evaluation depend on estimated model fidelity. On the ML side, `chen2025aioli` is the sharpest
statement: it unifies data-mixing methods as each assuming a "mixing law" over the simplex and shows that measuring the
*fidelity* of that law explains method performance; `ye2025mixinglaws` fits its law on sampled mixtures and uses it to
predict unseen mixtures; `liu2025regmix` fits a regressor and confirms with real runs. Where they all stop: none
declares a pre-registered acceptance criterion — here, external R² ≥ 0.5 *and* Spearman ρ ≥ 0.9 on 100 unseen
Dirichlet compositions — whose failure disqualifies the surface from being optimized at all.

## Element 13 — Replicated outer-partition study of surrogate and optimizer stability

**Verdict: PARTIAL PRIOR WORK. Also OWN PRIOR WORK — INHERITED (weaker form).**

Replication exists in the neighbourhood but never in this form. `monticeli2017portfolio` runs computational replicas at
each mixture design point together with moving windows over the series — the lineage's closest analogue, capturing
series volatility rather than resampling variability. `leal2025mixtureweights` validates its DoE/RSM weight estimator
over 27 Monte Carlo scenarios plus two real datasets. On the ML side, `galvan2026simultaneous` uses nested
cross-validation over five benchmarks with Wilcoxon signed-rank tests and an ablation; `large2019probabilistic` uses
extensive resampled experiments on the UCI archive with external validation on a second archive;
`moradpour2026ensemble` applies Friedman and Nemenyi tests across methods; `xu2026pseo` reports average test rank over
80 datasets. `ribeiro2026dissertation` (**OWN PRIOR WORK — INHERITED**) replicates its pipeline on two additional
datasets with contrasting profiles. What none of them does is re-run an entire mixture-design → surrogate → NBI
pipeline over 30 outer stratified partitions per dataset with recorded seeds, pair every comparison by partition, and
correct the resulting tests for the overlap between partitions (`nadeau2003inference`); the entire mixture/NBI lineage
uses a single design and a single case study.

## Element 14 — Explicit decomposition of surrogate error vs anchor misplacement vs NBI geometry vs numerical solver failure

**Verdict: NO DIRECT MATCH FOUND.**

Each of the four causes has its own literature. Surrogate error: `jin2002framework` (false optima cause incorrect
convergence), `deb2019taxonomy` (per-objective metamodel errors accumulate and degrade the whole procedure),
`jin2001metamodelling`, `tabatabaei2015survey` (the sequential framework is the exposed one). Anchor misplacement:
`isermann1988payoff`, `deb2010nadir`, `he2021normalization`, `herrmann2026nonextreme`. NBI/CHIM geometry:
`messac2004nc` (NBI does not represent the complete frontier), `motta2012modified` (deficiencies beyond two
objectives), `wagner2025nbiplus` (coverage in many-objective space, 2025), `muellergritschneder2009bounded`
(boundary/trade-off limits and the interior). Solver behaviour: `siddiqui2012improving` reformulates the NBI sweep to
obtain the frontier in one optimization problem. No verified source attributes an observed loss of front quality on a
single problem to these four causes jointly, or designs the experiment so that they can be separated.

## Element 15 — The finding that a large Scheffé β_ij does not imply exploitable classifier synergy

**Verdict: PARTIAL PRIOR WORK for the interpretive principle; NO DIRECT MATCH FOUND for the empirical
classifier-ensemble demonstration.**

The principle is classical and must be credited, not claimed. `cornell2002mixtures` defines the sign of the quadratic
coefficient as synergistic versus antagonistic *blending*, and every located passage sets the baseline as the linear
blend — "a more desirable yield than would be expected by taking the average of the yields of the two pure blends" —
and never as the better pure component; the book's own answer key classifies blends by comparing the observed response
against the linear-blending prediction. `piepel1982component` exists precisely because mixture-model coefficients are
not readable as component effects under Σx_i = 1, and introduces a separate effect measure with a partial/total
distinction for "modifying and interpreting mixture response prediction equations". `scheffe1958mixtures` defines the
canonical form from which the edge condition β_ij > |β_i − β_j| follows algebraically. On the ML side,
`liu2025regmix` reports that mixture components "interact in complex ways often contradicting common sense".
**Two limitations must be reported.** First, the cluster C agent could not locate an explicit statement of the caveat
in the form needed here; Cornell's Chapter 2 pages are view-restricted in the only accessible copy, so absence of
evidence there is not evidence of absence. Second, what has no counterpart in any verified source is the empirical
result: fitted AUC surfaces that satisfy the correct classical criterion in every partition of four datasets while the
real out-of-fold blend of the top pair beats its better member on only one of them, with the disagreement
one-directional across all forty dataset-pair cells. Claim the ML demonstration; restate the principle with citations.

## Element 16 — Systematic R = 10 to R = 30 robustness analysis

**Verdict: NO DIRECT MATCH FOUND — but this is a property of the study design, not a methodological invention.**

No verified record in any of the seven clusters reports extending the replication count of a DoE/surrogate/NBI
pipeline and auditing which conclusions move. The statistical rationale is foundational rather than novel:
`nadeau2003inference` quantifies the variability of resampled performance estimates and supplies the correction used
here, and `bouckaert2004replicability` is the standard argument that machine-learning experiments must be replicable
across resamplings. `rojasgonzalez2020kriging` identifies surrogate MOO under noisy objective estimates — our
regime — as the field's under-served case, which is the reason such an audit is worth reporting. The manuscript
should present this as evidence of stability and as a transparency measure, not as a contribution; the only defensible
sentence is that no prior study of this pipeline family reports one.

## Element 17 — Holdout confirmation of OOF-selected Pareto/knee solutions

**Verdict: PARTIAL PRIOR WORK. Also OWN PRIOR WORK — INHERITED.**

Confirming a selected optimum on data or a system not used to fit the surrogate is standard. `lopes2016rpdmnbi`
confirms three frontier points with a physical L9 array against prediction intervals; `naves2017nbirfs` runs
confirmation experiments and checks them against 95% prediction intervals; `karl2023svem` assesses candidate
formulations with confirmation runs; `ye2025mixinglaws`, `liu2025regmix` and `chen2025aioli` confirm surrogate-chosen
mixtures with real training runs; `ribeiro2026dissertation` (**OWN PRIOR WORK — INHERITED**) re-evaluates candidates
on the real model. On the leakage-control side, `galvan2026simultaneous` uses nested cross-validation and
`large2019probabilistic` validates on a separate archive. Where they stop: none confirms the *out-of-fold-selected
knee of a Pareto front* on an untouched holdout across replicated partitions, and none reports both the level shift
and the ranking agreement between out-of-fold and holdout, so that a disagreement can be attributed to selection
optimism or to near-ties rather than assumed.

---

## (a) Elements that appear genuinely open

Stated as the outcome of a search, with the qualifier that the searches were bounded by the blocked publisher records
listed in `literature_review.md` §1.

1. **To the best of our literature search, we found no prior study that** computes the anchor points of an NBI (or
   Normal Constraint) construction both from a fitted surrogate and from the true objectives on the same problem and
   compares the resulting frontiers. (Element 6.)
2. **To the best of our literature search, we found no prior study that** runs the same scalarization-based front
   generator with surrogate objectives and metamodel-free on the true objectives, on the same problem, as a controlled
   contrast. (Element 7.)
3. **To the best of our literature search, we found no prior study that** re-evaluates *every* candidate of a
   completed surrogate-generated Pareto set on the true objectives before computing any front-quality indicator, as
   opposed to confirming one selected point or evaluating infill points inside a search loop. (Element 8, in
   combination with 9.)
4. **To the best of our literature search, we found no prior study that** constructs an empirical Pareto reference for
   a real machine-learning weighting problem by large-scale sampling plus an ε-constraint sweep, independently of the
   surrogate, and scores revalidated candidate sets against it with IGD+ and hypervolume. (Element 9.)
5. **To the best of our literature search, we found no prior study that** compares a support-based deployment cost
   Σ c_i·1[w_i > ε] against the continuous weighted relaxation Σ w_i c_i for the same weighted ensemble, or reports
   that the two accountings change which method wins. (Element 11.)
6. **To the best of our literature search, we found no prior study that** applies a pre-registered pass/fail
   admissibility criterion, evaluated on unseen compositions of the design simplex, to decide whether a fitted surface
   may be optimized at all. (Element 12; validation of surrogates is common, an admissibility gate is not.)
7. **To the best of our literature search, we found no prior study that** separates surrogate misspecification, anchor
   misplacement, front-generator geometry and numerical solver failure as distinct, separately evidenced causes of the
   same loss in front quality. (Element 14.)
8. **To the best of our literature search, we found no prior study that** demonstrates empirically, for classifier
   ensembles, that fitted mixture surfaces can satisfy the classical blending criterion while the real blends do not
   beat their better member — although the interpretive principle itself is classical and is credited to
   `cornell2002mixtures`, `piepel1982component` and `scheffe1958mixtures`. (Element 15.)
9. **To the best of our literature search, we found no prior study that** replicates a mixture-design → surrogate →
   NBI pipeline across resampled outer partitions with paired, overlap-corrected inference, or that reports how its
   conclusions move when the replication count is raised. (Elements 13 and 16; the second is a reporting practice, not
   a contribution.)

## (b) Warning list — claims in `claims_and_evidence.md` that the literature weakens or pre-empts

These are ordered by how much rewriting they force.

**W1 — The pipeline framing is fully pre-empted, and by more than the blacklist currently records.**
Blacklist item 9 correctly forbids claiming DoE + RSM + NBI (`pereira2025hybrid`). The literature adds three further
forbidden sentences, all supported by third-party or group records: *"we cast ensemble-weight selection as a
mixture-design problem"* and *"we are the first to run NBI over the ensemble-weight simplex"* are the stated
contributions of `rocha2025ensemblestlf`; *"we are the first to model ensemble performance over the simplex with
Scheffé polynomials"* is pre-empted by `kwon2024ensemble` for classifiers, `bacci2019nbi_mixture` for forecasts and
`moreira2021ann_ensemble_mde` for ANN ensembles; and *"we propose the use of mixture designs to optimize ensemble
weights"* is pre-empted by all four. **Recommend adding these to the blacklist.**

**W2 — Elements 1–3 are third-party prior art, not only lineage prior art.** The manuscript's internal framing
(`research_lineage.md`) treats the mixture formulation, the design over the weights and the Scheffé surrogate as
inherited from the group. `kwon2024ensemble` shows they are also independently published *for classifier ensembles*,
outside the group, in 2024. Any wording implying that transferring the construction to classifiers is itself the
novelty must go.

**W3 — Inference cost as a Pareto objective for post-hoc tabular ensembles is prior art (element 10).** Neither
`maier2024hardware` nor `maier2026hapens` currently appears in the claim structure, but a sentence such as "we
introduce deployment cost as a third objective for post-hoc ensembles" would be false. The defensible framing is the
*two cost accountings* (C8, element 11), not the presence of a cost objective.

**W4 — C8 ("cost definition changes the winning set") must be narrowed.** `wang2022committees` already shows that
changing the cost accounting reorders which committee looks best. The distinguishing feature here is that the two
accountings are applied to *one and the same weight vector* rather than to different architectures; the claim must be
stated at that resolution. This reinforces, rather than contradicts, the existing note that "BNP support-cost
bimodality as a mechanism" is blacklisted.

**W5 — C1 ("real anchors improve surrogate-assisted NBI") must credit the mechanism to prior work.** That anchor and
ideal/nadir misestimation degrades a multiobjective procedure is established (`isermann1988payoff`, `deb2010nadir`,
`he2021normalization`), and `herrmann2026nonextreme` already treats anchor choice as a lever. What is new is the
*surrogate provenance* of the error and the controlled A-versus-B contrast, not the discovery that anchors matter. The
existing limitations in C1 (BNP bimodality, Porto/UCI partition sensitivity, the UCI reversal under the support cost)
stand unchanged; the literature does not weaken them, but it does mean the effect cannot be presented as surprising.

**W6 — C2 ("metamodel-free NBI") must credit `gellerich2023doenbi`.** Running NBI directly on measured objectives,
without a metamodel and with real anchors, is published. The contribution here is the controlled comparison against
the surrogate arms on the same problem and the compute accounting, not the idea of a metamodel-free NBI. The existing
C2 limitations — that NBI-C is not evaluation-matched, that its candidates form 8–25% of the reference front it is
graded against, that the UCI gap is partly a subproblem-convergence effect of NBI-B, and that the UCI corrected
p ≈ 0.010 is knife-edge — must all be retained.

**W7 — C5 (the reliability gate) must credit the "validate before you trust" line.** `jin2001metamodelling`,
`abolghasemian2022haulage`, `deb2020surrogate` and especially `chen2025aioli` make surrogate-fidelity checking
standard practice. Only the *external-composition, pre-registered, pass/fail* form is distinctive. The existing,
correctly stated negative result — the gate identifies unusable surfaces but does not predict anchor misplacement — is
untouched by the literature and remains the most defensible sentence in C5.

**W8 — C3 (NBI-C vs random weighted scalarization) is weakened from two directions.** `knowles2006parego` is random
simplex-weight scalarization with a surrogate and true evaluation of the proposed points, so the comparator class is
canonical and cannot be presented as a straw man; and `chen2025aioli` reports that in data mixing no existing method
consistently beat a simple stratified-sampling baseline, which makes the honest reporting of the UCI tie and the
Santander split *more* important, not less. The existing note that the comparison is confounded (scalarization
optimizes the surrogates, NBI-C the exact cached-OOF objectives) must stay prominent.

**W9 — C6's attribution is already correct and must not be relaxed.** The added-after-review paragraph in
`claims_and_evidence.md` gets this right. The literature confirms it: Cornell's baseline is the linear blend, and the
edge condition follows algebraically. One addition — `piepel1982component` should be cited alongside
`cornell2002mixtures`, since it is the canonical statement that mixture coefficients are not directly interpretable as
effects. Also record that the search could not rule out the caveat appearing on a view-restricted page.

**W10 — C10 (R = 10 → R = 30) should not be sold as a contribution.** Nothing in the literature pre-empts it, but
nothing frames such an audit as a novelty either. The existing wording ("no directional conclusion reversed; R = 30
resolved the Porto/UCI anchor effect, quantified the BNP collapse rate and exposed two artifacts") is exactly right
and should not be strengthened.

**W11 — Two negatives rest on records that could not be read, and must be hedged in the manuscript.** (i) Whether the
Saha/Ekbal per-class vote weights are normalized is unknown (paywalled full texts), so "no prior multiobjective
ensemble-weighting study uses a probability simplex" must be phrased as "no verified record states a
simplex constraint". (ii) `galvan2026simultaneous` could not be read (MDPI 403), so neither its MOEA nor whether its
continuous voting weights are normalized may be asserted. (iii) The negative result "no NSGA-II/NSGA-III/MOEA-D
applied to simplex-constrained ensemble combination weights" is the outcome of Crossref, OpenAlex and four
zero-hit arXiv queries — report it as "no verified competitor was found", never as "none exists".

**W12 — Nothing in the literature weakens C7, C9, C11, C12, C13 or C14.** The AUC/log-loss degeneracy result, the
holdout-transfer analysis, the reference-quality diagnostics, the NBI outcome rates, the base-model context and the
execution provenance have no counterpart that pre-empts or contradicts them. C11's existing blacklist item ("true
Pareto front") is correct and should be retained verbatim.

## (c) Recommended central-contribution statement

> Mixture designs over combination weights, Scheffé surrogates of the resulting performance surface, and Normal
> Boundary Intersection run on those surrogates are established methodology — in the authors' own prior work
> [`pereira2025hybrid`; `ribeiro2026dissertation`] and, independently, in the wider literature on forecast, portfolio
> and neural-network ensembles [`bacci2019nbi_mixture`; `moreira2021ann_ensemble_mde`; `rocha2025ensemblestlf`] and on
> mixture-design weighting of classifier ensembles for a single accuracy objective [`kwon2024ensemble`]. This study
> proposes none of that pipeline; it audits it. Transferring the construction to classifier-ensemble weighting on
> cached out-of-fold probabilities — where one objective is a piecewise-constant rank statistic, a second is a convex
> calibration loss, and deployment cost is a step function of the weight vector's support rather than a smooth
> physical response — we run the identical Normal Boundary Intersection three ways: on the surrogate objectives with
> anchors taken from the surrogate, on the surrogate objectives with anchors recomputed from real out-of-fold
> single-objective optima, and metamodel-free on the real objectives; we re-evaluate every candidate returned by every
> method on the exact out-of-fold objectives; and we score all of them against an empirical Pareto reference that the
> surrogate cannot influence. Replicating this over four tabular binary datasets and thirty outer stratified
> partitions each, with an external reliability gate on unseen Dirichlet compositions and paired, overlap-corrected
> inference, lets us separate surrogate misspecification from anchor misplacement, show that the reliability gate
> identifies unusable surfaces without predicting anchor failure, and show that the choice between a weighted-cost
> relaxation and a support-based deployment cost changes which method wins. The contribution is therefore an
> evaluation architecture and the conditions it exposes — where surrogate-assisted NBI over a weight simplex can be
> trusted and where it fails silently — not a new optimizer, a new design family, or a new surrogate.
