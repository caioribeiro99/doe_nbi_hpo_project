## 8. Limitations

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
