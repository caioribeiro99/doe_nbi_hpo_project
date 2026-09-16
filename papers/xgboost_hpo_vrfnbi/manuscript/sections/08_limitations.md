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
real evaluations those two arms require. The evaluation-matched grid nevertheless attained
higher median CORE-relative hypervolume than every surrogate-assisted arm on every dataset in
the panel [TAB:baseline_vs_arms], and that asymmetry belongs with the result: the rule was
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
[FIG:chim_vs_gap].

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
