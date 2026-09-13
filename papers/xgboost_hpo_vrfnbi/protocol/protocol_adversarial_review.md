# Protocol adversarial review

Four roles, each reading `protocol/EXPERIMENT_PROTOCOL.md` and its supporting documents with a
different reason to reject them. Findings are **MUST FIX** (the protocol cannot freeze), **SHOULD
FIX** (fix or record why not), or **CONSIDER**.

Every MUST FIX must be resolved, and its resolution recorded here, before `xgboost-hpo-protocol-v1`
is applied.

---

# Role 1 — Multiobjective optimization methodologist

*Reads for: whether the geometry claims are well-posed and whether the arms are actually
distinguishable.*

### M1 (MUST FIX) — WS-S is under-specified: a weighted sum needs a nadir, and the payoff matrix gives two

`protocol/method_arms.md` defines WS-S as "weighted sum, reference from the payoff matrix, utopia is
its diagonal". A min–max normalized weighted sum needs both ends of the box. The payoff matrix
supplies the utopia as its diagonal, but the other end could be the **pseudo-nadir** (the row-wise
maximum of the payoff matrix, which is what `nbi_core.compute_anchors` calls `pseudo_nadir`) or the
**true nadir** (obtained by anti-optimizing each objective, which `compute_anchors` also computes and
calls `nadir`). These differ, and the choice changes every normalized objective value in the arm.

Without fixing it, the HISTORICAL-WS to WS-S contrast does not isolate one factor; it isolates an
unspecified one.

**Resolution.** WS-S uses the **pseudo-nadir**, the row-wise maximum of the payoff matrix. Reason:
the pseudo-nadir is the quantity the CHIM construction itself induces, so WS-S and NBI-S then share
*exactly* the same reference object and differ *only* in what they do with it, which is what the
contrast is for. The true nadir is recorded per replication and the sensitivity of the contrast to
that choice is a reported secondary analysis, not a free parameter. Written into
`protocol/method_arms.md` and `protocol/EXPERIMENT_PROTOCOL.md` §2.

### M2 (MUST FIX) — Sign conventions are not stated and the two code paths disagree

The dissertation's objectives are oriented so that **larger is better** (`Score_Cost` is a negated
z-score, `Score_Quality` is a mean of positively-oriented scores), and `run_nbi_weighted_sum`
maximizes their weighted sum. `nbi_core` canonicalizes everything to **minimization** before
building the payoff matrix. Four arms drawing on both code paths without a written convention will
silently flip an objective, and a flipped objective in a payoff matrix produces a CHIM that is
geometrically valid and scientifically meaningless.

**Resolution.** One convention, stated once and asserted in code: **every objective entering any arm
is canonicalized to minimization**, with the direction declared per objective rather than inferred
from a loading sign (`docs/METHODOLOGY_DECISIONS.md` D3 already requires this on the article track).
A test asserts that the canonicalized objectives of every arm agree elementwise on the design rows.
Written into `protocol/EXPERIMENT_PROTOCOL.md` §6.3.

### M3 (MUST FIX) — Integer hyperparameters break the NBI equality constraint

`max_depth` and `n_estimators` are integers, cast by `int(round(.))` at evaluation time. The
surrogate is fitted on the continuous relaxation and the NBI subproblem certifies the equality
constraint `F̂(x) = utopia + Φβ + t·n̂` at a **continuous** `x`. The point actually evaluated is a
rounded neighbour, which does not satisfy that constraint.

So a subproblem can report `success` with a residual below 1e-6 while the evaluated candidate sits
somewhere else on the surface. This affects NBI-S and NBI-R and not the weighted-sum arms, because
only NBI has an equality constraint to violate. It therefore **biases the very contrast the paper is
about**.

**Resolution.** Both of the following, not one:
1. Report, per subproblem, the objective displacement induced by rounding: `‖F̂(x_cont) − F̂(x_round)‖`
   alongside the residual. An arm's certification rate is reported as the fraction certified *and*
   the distribution of rounding displacement.
2. Re-solve each NBI subproblem with the integer dimensions **fixed at their rounded values** and the
   continuous dimensions re-optimized, and report the certified fraction of that restricted problem.
   This is cheap, uses no real evaluations, and gives a certificate for the point actually evaluated.

Written into `protocol/EXPERIMENT_PROTOCOL.md` §9.

### M4 (SHOULD FIX) — Surrogate anchors may sit on box corners and make Φ degenerate

Anchors for WS-S and NBI-S are minimizers of a backward-eliminated quadratic over a box. Such a
quadratic is frequently unbounded in the box's interior, so its minimizer lands on a corner. If two
objectives' surrogates are minimized at the same corner, the payoff matrix has two identical columns,
`Φ` is singular, and the quasi-normal is ill-defined.

**Resolution.** Detect and report. Each replication records the anchor decision vectors, the rank and
condition number of `Φ`, and whether any two anchors coincide. A replication with a rank-deficient
`Φ` is reported as such and excluded from the geometry contrasts, with the exclusion count published.
It is **not** silently repaired, because how often it happens is itself a result about the method.

### M5 (CONSIDER) — Two objectives make the geometry claim weak

With `q = 2` the CHIM is a line segment and the quasi-normal is a single direction. Much of what
distinguishes NBI from weighted sum appears only at `q ≥ 3`, where CHIM is a simplex and uniform
spread is a genuinely harder property. `docs/METHODOLOGY_DECISIONS.md` D2 already requires the core
to be dimension-agnostic, and `nbi_core` is.

Running the panel at `q = 3` (quality, calibration, cost) rather than `q = 2` would make the geometry
factor more informative, at the cost of a third surrogate and a larger weight grid. Recommended if
the pilot's budget allows; not a blocker.

---

# Role 2 — Automated machine learning empiricist

*Reads for: whether the result would be believed by the community that owns this problem.*

### A1 (MUST FIX) — The pilot as written cannot deliver the dataset screening

`protocol/EXPERIMENT_PROTOCOL.md` §12 defines the pilot as "one dataset, one replication", but
`protocol/dataset_selection.md` requires four screening measurements **for each candidate dataset**,
and §12 itself lists producing them as the pilot's purpose. These contradict.

**Resolution.** The pilot is two stages. **Stage A**, screening: the 88 design rows on one partition
for **every** candidate dataset, which is the only cost needed for all four screening measurements,
about 88 evaluations per candidate. **Stage B**, timing and certification: all four arms and all
comparators on one partition of one dataset. Written into `protocol/EXPERIMENT_PROTOCOL.md` §12.

### A2 (MUST FIX) — NSGA-II at 108 evaluations is below the threshold of a fair test

`baseline_gap_assessment.md` already flags this against interest, but flagging is not a design. At
`B_total_solution ≈ 108` and any reasonable population size, NSGA-II gets a handful of generations on
a seven-dimensional problem. A reviewer will read a weak NSGA-II as a budget artifact and will be
right.

**Resolution.** Fix the population size and generation count in advance, jointly, so that the product
matches the budget and the generation count is **at least ten**. That forces a population of about
ten, which is small but is a real evolutionary run rather than a truncated one. In addition, run
NSGA-II a second time at a budget **ten times** the matched one, reported separately and explicitly
labelled as unmatched, so the reader can see both the matched comparison and what the method does
when not starved. The extra cost is one comparator's worth of evaluations and buys the paper its
answer to the obvious objection. Written into `protocol/EXPERIMENT_PROTOCOL.md` §3.

### A3 (SHOULD FIX) — Four datasets, and the paper already knows what that buys

Paper 1 used four and had to concede a cross-dataset pattern rather than a tested claim.
`protocol/dataset_selection.md` repeats the number and the concession. Repeating a known limitation
by choice is weaker than repeating it by necessity.

**Resolution.** Add candidates from the registry up to whatever the pilot's measured cost allows,
and state the panel size as budget-determined with the measurement shown. The registry has eleven
usable binary datasets; the marginal cost of a light one (`spambase`, `german_credit`,
`breast_cancer`, `pima_diabetes` are all marked light) is small. If the pilot shows the panel can be
six or seven rather than four, take it.

### A4 (SHOULD FIX) — "Bayesian optimization, single objective per objective" is not a baseline, it is two baselines

Listing it in the comparator table as one row with "two fronts' worth of endpoints" hides that this
produces two single-objective optima, not a front. Compared by front indicators it will look
terrible for a reason that says nothing about Bayesian optimization.

**Resolution.** Report it as what it is: two single-objective runs, scored on the single-objective
endpoints only, never on front indicators. Alternatively replace it with a genuinely multiobjective
Bayesian method. Keep the dissertation's version for continuity, labelled, and do not put it in the
front-indicator table.

### A5 (CONSIDER) — No multi-fidelity comparator

`baseline_gap_assessment.md` scopes Hyperband and BOHB out, with a stated reason. The reason is
sound. It remains the first thing an automated-machine-learning reviewer will ask for. Ensure the
scoping sentence appears in the manuscript's limitations, not only in this workspace.

---

# Role 3 — Statistician

*Reads for: whether the inference supports what will be said.*

### S1 (MUST FIX) — The overlap correction is imported without its assumption

`protocol/EXPERIMENT_PROTOCOL.md` §10 specifies the Nadeau and Bengio corrected t-test "with the
overlap proportion stated". In Paper 1 the value ρ = 0.25 came from a stated resampling scheme. Here
the replications are outer stratified partitions of a fixed dataset, and ρ depends on the ratio of
test to training size in *that* scheme. Carrying 0.25 across would be an unexamined import.

**Resolution.** State ρ as a function of the actual partition scheme, derive it, and report every
significance statement as holding *at* that ρ, with the effect consistency that does not depend on ρ
reported beside it. If the scheme makes ρ ambiguous, report the descriptive triple (median with its
bootstrap interval, win fraction with its Wilson interval, rank-biserial correlation) as primary and
the test as secondary. Written into `protocol/EXPERIMENT_PROTOCOL.md` §10.

### S2 (MUST FIX) — Four primary arms give six pairwise contrasts, and only three are planned

`protocol/method_arms.md` names four contrasts and treats three as the identifying ones. But four
arms admit six pairwise comparisons, and a reader will compute the others. If only three are
corrected for multiplicity, the rest are uncontrolled comparisons that will be discussed anyway.

**Resolution.** Declare the three identifying contrasts as **primary** and pre-register them as the
family carrying the Holm correction within each dataset. The other three are declared **secondary
and descriptive** in advance, reported without tests, and never described as significant. Written
into `protocol/EXPERIMENT_PROTOCOL.md` §10.

### S3 (SHOULD FIX) — No power consideration at all

R = 30 is inherited from the dissertation and from Paper 1. Nothing states what effect size that
detects. If the three identifying contrasts are genuinely small, the campaign is designed to produce
inconclusive intervals.

**Resolution.** From the pilot, compute the paired standard deviation of each primary endpoint and
report the smallest effect the design detects at R = 30 with the planned interval. Do this **before**
the full campaign, and if the detectable effect is larger than any difference the pilot suggests
exists, say so in the protocol rather than discovering it in the results.

### S4 (SHOULD FIX) — The aggregation sensitivity analysis has no decision rule

`protocol/EXPERIMENT_PROTOCOL.md` §11.1 says to report how often the two weightings disagree about
the returned front. It does not say what follows from any particular frequency.

**Resolution.** Pre-register the reading: if the two weightings disagree about the returned front in
more than a stated fraction of replications, the paper's conclusions are reported as conditional on
the weighting, and the conditionality goes in the abstract. Below that fraction, it is a robustness
note. Fix the fraction before the campaign.

---

# Role 4 — Editor and research-integrity reviewer

*Reads for: whether this should be published at all, and whether the framing is honest.*

### E1 (MUST FIX) — The contribution is contingent and the protocol does not say what happens if it fails

`novelty_matrix.md` element 18 is "whatever the four-arm campaign measures", and element 14 concedes
that if all three contrasts measure zero the paper is a different paper. No document says what that
different paper is.

Deciding after the results is how a null result gets written up as a positive one.

**Resolution.** Write the outcome-contingent framing now, before the campaign, into
`novelty_matrix.md`: the statement to be made if the contrasts are large, the statement if they are
small but consistent, and the statement if they are indistinguishable from zero. The third is a
legitimate paper — a controlled demonstration that three choices practitioners agonize over do not
matter on this problem family — and drafting it in advance is what keeps it honest. Done below.

### E2 (MUST FIX) — Paper 1 is unpublished and this protocol depends on it in eight places

`research_lineage.md` marks the reliability gate, the evaluation-matched budgeting, the NSGA-II
harness and the anchor-injection control as inherited from Paper 1, and
`protocol/EXPERIMENT_PROTOCOL.md` relies on all four. Paper 1 is frozen at `paper-submission-v2` and
**has not been submitted**. It cannot be cited as published work, and if both are under review
simultaneously each must disclose the other.

**Resolution.** Two rules, written into the protocol. First, every inherited element is described
self-containedly in Paper 2's methodology, so that Paper 2 stands alone if Paper 1 is never
published. Second, the submission order is a decision for the authors, and whichever goes second
cites the first as under review or as published, as applicable. Neither paper may cite the other as
established. Tracked as open item 10.

### E3 (SHOULD FIX) — The audit framing risks reading as an attack on the first author's own thesis

Three audit documents, one of which finds a discrepancy between a dissertation's text and its code.
The dissertation is the first author's own, and `novelty_matrix.md` already forbids sensationalizing.
But the *volume* of audit material relative to the experimental content will, by itself, set a tone.

**Resolution.** In the manuscript, the audits occupy the methodology section as the justification for
the arm set, and nothing more. There is no "audit" section, no "discrepancy" in a heading, and the
weighted-sum finding is stated once, plainly, with the citation to the author's own
`METHODOLOGY_DECISIONS.md` D1 showing it was recorded before it was audited. The workspace documents
stay in the repository as evidence; they are not the paper.

### E4 (SHOULD FIX) — The claim blacklist has no enforcement

`novelty_matrix.md` lists ten forbidden claims. Paper 1 learned that a blacklisted phrase can be
reintroduced by a later editing pass and survive to the conclusion.

**Resolution.** Port Paper 1's mechanism: a script that scans the compiled manuscript for each
blacklisted phrasing and fails, run as part of the build, not as a checklist item.

### E5 (CONSIDER) — The venue is not chosen and it changes the paper

An automated-machine-learning venue wants the baselines of Role 2 and will not care about the
design-of-experiments lineage. An engineering-optimization venue wants the lineage and will accept a
smaller baseline set. The protocol currently tries to satisfy both, which is the more expensive
option.

Decide the venue before the campaign, since it determines whether A2's unmatched NSGA-II run and
A5's multi-fidelity comparator are required or optional.

---

# Resolutions applied

| ID | Status |
|---|---|
| M1 | RESOLVED — pseudo-nadir fixed as WS-S's second reference end, with sensitivity as a secondary analysis |
| M2 | RESOLVED — minimization canonicalization mandated, direction declared per objective, asserted by test |
| M3 | RESOLVED — rounding displacement reported per subproblem, plus a fixed-integer re-solve certificate |
| M4 | RESOLVED — anchor coincidence and `Φ` conditioning recorded; degenerate replications reported, not repaired |
| A1 | RESOLVED — pilot split into a screening stage over all candidates and a timing stage on one |
| A2 | RESOLVED — population and generations fixed jointly with at least ten generations, plus an unmatched ten-times run reported separately |
| A4 | RESOLVED — single-objective Bayesian optimization and Parzen estimator reported on endpoints only, never in the front-indicator table |
| S1 | RESOLVED — ρ derived from the actual partition scheme; descriptive triple primary if ambiguous |
| S2 | RESOLVED — three identifying contrasts pre-registered as the primary family; the other three descriptive only |
| E1 | RESOLVED — outcome-contingent framings written in advance, below |
| E2 | RESOLVED — inherited elements described self-containedly; neither paper cites the other as established |
| M5, A3, A5, S3, S4, E3, E4, E5 | recorded; A3, S3, S4 and E4 to be closed with numbers from the pilot, E3 and E4 at manuscript time, M5 and A5 as scoping decisions in the manuscript |

**Freeze status: READY.** All eleven MUST FIX items are resolved, and each resolution is written
into the protocol document it affects.

The seven freeze-blocking numbers are also decided, and decided from evidence that existed before any
campaign result rather than deferred to the pilot. Deferring them would have been circular: a
protocol whose parameters come from a pilot run under that protocol is not pre-registered.

| Blocker | Decided as | On what evidence |
|---|---|---|
| cost objective | total leaf count, primary; wall-clock time, secondary | eight candidates measured against the 88 measured times of the reproduced MAGIC design |
| quality objectives | four threshold metrics, plus ROC-AUC and log loss | the dissertation had no ranking or calibration metric |
| aggregation weighting | explained-variance, with equal weighting as the sensitivity | the audit measured the two ranking the design at Spearman 0.374 |
| gate thresholds | 100 Latin-hypercube points, R² ≥ 0.5 and Spearman ≥ 0.9 | inherited from Paper 1, and labelled as a transfer rather than a calibration |
| `B_anchor` | 100 real evaluations per objective | fixed in advance, identical across datasets, so it is a constant and not a knob |
| NSGA-II population and generations | 34 x 12 at q = 2, matching 408 exactly, plus an unmatched run at ten times | budget arithmetic, with the at-least-ten-generations floor of finding A2 |
| weighting-disagreement fraction | 0.20 | pre-registered so the reading cannot be chosen after the result |

The pilot's role is therefore **verification and screening**, not parameter selection: it confirms
the panel, measures per-evaluation cost, supplies the detectable effect size, and checks that every
NBI subproblem certifies on the real problem.

`xgboost-hpo-protocol-v1` may be applied.

---

# Outcome-contingent framing, written before the campaign (resolution of E1)

**If the contrasts are large.** The paper reports which of the three bundled choices the outcome
depends on, and by how much, and the contribution is the decomposition.

**If the contrasts are small but consistent in direction.** The paper reports the ordering with
intervals, declines to call it significant, and the contribution is that the choices are orderable
but practically minor on this problem family — which tells a practitioner not to spend effort there.

**If the contrasts are indistinguishable from zero.** The paper reports that three choices which the
literature treats as methodologically important are not distinguishable on this problem family at
this budget, with the intervals that bound how large an effect the design could have detected (S3).
The contribution is the controlled null, and the honest title says so. This is the outcome the
authors consider least likely and the one they are least free to avoid reporting, which is why it is
drafted here rather than later.

In all three cases the audits stay where E3 puts them, and the claim blacklist stays binding.
