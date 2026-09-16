# 4. Confirmatory questions and prior commitments

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

**RQ5 — Direct search.** How does the surrogate-assisted family compare with evaluation-matched
direct search — grid, random, Bayesian optimization, TPE, NSGA-II — at the frozen comparator
budget? [TAB:rq_contrast_map] maps each question to its contrast and evidence.

## 4.1 What was frozen

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
