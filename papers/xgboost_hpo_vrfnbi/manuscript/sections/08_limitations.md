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

The historical arm is likewise a single case. HISTORICAL-WS-asrun calls the solver of one archived
implementation unmodified, on surfaces and an observed-extrema box that this study's code computes to
the archived specification for this study's objectives (Section 5.7); code-level examination
established that this implementation built its front by normalized weighted scalarization while the
accompanying historical text used NBI terminology. That is a statement about one archived artifact,
not about a class of published pipelines, and supports no estimate of how common the conflation is.

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
indicators, which Supplement S10 reports as medians.
