## 4. Confirmatory questions and prior commitments

Pipelines of this kind bundle three choices usually made silently: how the objective reference
is normalized, which scalarization geometry builds the front, and where the anchors and payoff
matrix come from. The design separates them by changing one factor at a time, everything else
held fixed and checked at run time by a contrast fingerprint.

**RQ1: Specification and normalization.** Does replacing the historical observed-extrema
normalization with the payoff-matrix reference change the returned front? Contrast:
HISTORICAL-WS $\rightarrow$ WS-S, differing only in that reference.

**RQ2: Front-construction geometry.** At a fixed reference, does canonical NBI construction
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
are descriptive and uncorrected. Primary evidence is the descriptive triple (median difference
with bootstrap interval, win fraction with Wilson interval, rank-biserial), and the test is
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

Throughout, the dataset is the unit of generalization and nothing is pooled: the primary
datasets are never combined into a 90-replication analysis, and no conclusion extends beyond
this panel, this surrogate architecture, two objectives and this budget.
