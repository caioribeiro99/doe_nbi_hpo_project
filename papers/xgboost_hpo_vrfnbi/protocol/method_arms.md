# Method arms

**Why four and not two.** `audits/METHODOLOGICAL_IDENTITY_AUDIT.md` establishes that the historical
method differs from canonical NBI in **two** independent respects:

1. the **scalarization geometry** — a weighted sum of normalized objectives, versus a CHIM with a
   quasi-normal direction and a `max t` subproblem;
2. the **reference construction** — a normalization box from component-wise observed extremes of
   the design rows, versus a payoff matrix whose diagonal is the utopia point.

A two-arm comparison of HISTORICAL-WS against canonical NBI changes both at once and can attribute
any difference to neither. The arm set below is a two-by-two on those factors, with the fourth cell
dropped as uninformative and replaced by a real-anchor variant of NBI.

---

## The four primary arms

| Arm | Scalarization | Reference / anchors | Role |
|---|---|---|---|
| **HISTORICAL-WS** | weighted sum of min–max normalized surrogates | component-wise observed extremes of the design rows | reproduces the dissertation exactly; the historical baseline |
| **WS-S** | weighted sum of min–max normalized surrogates | payoff matrix from surrogate anchors: utopia is its diagonal, the other end is the **pseudo-nadir** (row-wise maximum of the payoff matrix) | isolates the **reference construction** against HISTORICAL-WS |
| **NBI-S** | canonical NBI: CHIM, quasi-normal, `max t` | payoff matrix from surrogate anchors | isolates the **scalarization geometry** against WS-S |
| **NBI-R** | canonical NBI | payoff matrix from **real** anchors, obtained by direct search on the true objectives | isolates **anchor provenance** against NBI-S |

### What each contrast identifies

| Contrast | Changes | Identifies |
|---|---|---|
| HISTORICAL-WS → WS-S | reference only | the effect of replacing observed extremes with a payoff matrix |
| WS-S → NBI-S | scalarization only | the effect of canonical NBI geometry, at a fixed reference |
| NBI-S → NBI-R | anchor provenance only | the cost of computing anchors from the surrogate rather than from the truth |
| HISTORICAL-WS → NBI-S | both | the total historical-to-canonical difference, decomposed by the two rows above |

The fourth cell of the two-by-two, weighted sum with real anchors, is not run as a primary arm: a
weighted sum uses only the normalization box, so real anchors would enter it in exactly the way
WS-S's surrogate anchors do, and the contrast would duplicate NBI-S → NBI-R without the geometry
that makes anchors matter. It is listed as a secondary arm below in case a reviewer asks.

### Why the pseudo-nadir and not the true nadir (resolution of review finding M1)

A min-max normalized weighted sum needs both ends of the box. The payoff matrix supplies the utopia
as its diagonal; the other end can be the **pseudo-nadir**, the row-wise maximum of the payoff
matrix, or the **true nadir**, obtained by anti-optimizing each objective. `nbi_core.compute_anchors`
computes both. They differ, and the choice rescales every normalized objective in the arm.

WS-S uses the **pseudo-nadir**, because it is the quantity the CHIM construction itself induces. WS-S
and NBI-S then share exactly the same reference object and differ only in what they do with it,
which is the whole point of that contrast. The true nadir is recorded per replication and the
sensitivity of the contrast to the choice is a reported secondary analysis, never a free parameter.

### Objective direction: one convention (resolution of review finding M2)

The dissertation's objectives are oriented so that **larger is better**, and its solver maximizes.
`nbi_core` canonicalizes to **minimization** before building the payoff matrix. Four arms drawing on
both code paths would silently flip an objective, and a flipped objective inside a payoff matrix
gives a CHIM that is geometrically valid and scientifically meaningless.

**Every objective entering any arm is canonicalized to minimization**, with its direction declared
per objective rather than inferred from a loading sign, as `docs/METHODOLOGY_DECISIONS.md` D3 already
requires on the article track. A test asserts that the canonicalized objectives of all four arms
agree elementwise on the design rows.

### The weight grid is symmetric, and the historical one is not

`audits/METHODOLOGICAL_IDENTITY_AUDIT.md` §2.3 found that the dissertation's grid runs `beta_1` from
0.95 down to 0.00: it contains the pure-cost vertex and not the pure-quality vertex, so the
historical returned set is systematically short at the quality end on every run.

The shared grid used by HISTORICAL-WS, WS-S, NBI-S and NBI-R is symmetric and contains both
vertices, with the same cardinality for every arm. `HISTORICAL-WS-asrun` keeps the asymmetric grid,
because its only job is to be faithful. The two are never mixed in one table, and the manuscript
states the asymmetry when it reports the historical arm.

### A caution that must survive into the manuscript

Paper 1 found, and independently reproduced, that a vertex value of the NBI weight vector returns
its own anchor directly, so an NBI arm with real anchors **contains** the real single-objective
optima as returned candidates. Part of any NBI-R advantage is therefore the injection of those
extreme points, not the relocated geometry.

**NBI-S → NBI-R must be decomposed the same way Paper 1 decomposed it:** rescore NBI-S's candidate
set augmented with the real anchors, and report how much of the gap that alone closes. Without that
control the anchor contrast is uninterpretable, and Paper 1 has already established this on a
different problem, so failing to run it here would be a knowing omission.

---

## What is held fixed across all four arms

Everything except the factor named in the contrast. Specifically:

| Held fixed | Value |
|---|---|
| Decision space and bounds | the seven XGBoost hyperparameters of `protocol/original_thesis_protocol.md` §2 |
| Design | the version-controlled 88-run face-centred central composite design |
| Evaluation | stratified 5-fold cross-validation, one seed per replication |
| Objective set | one specification, shared; see below |
| Objective reduction | one PCA/Varimax stage with an aggregation weighting declared in advance |
| Surrogate | quadratic response surface, backward elimination at α = 0.05, coded units |
| Weight grid | the same simplex lattice for every arm, same cardinality |
| Candidate validation | every returned candidate re-evaluated on the real objectives |
| Selection rule | reported as the full front; any scalar summary applied identically |
| Replications | R = 30, paired by partition |

Two of these are changes from the dissertation, and both are forced by the audits:

- **Coded units**, not uncoded (`docs/METHODOLOGY_DECISIONS.md` D6). Uncoded fitting on factors
  whose ranges span 0.29 to 650 is badly conditioned, and the dissertation's own tables report coded
  coefficients.
- **An aggregation weighting declared in advance** for the quality composite, with a stated sensitivity
  check. `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4 measured that this choice moves the quality
  ranking at Spearman 0.374 and changes which design row looks best, so leaving it to a default
  argument would put an uncontrolled factor inside every arm.

Neither change is applied to HISTORICAL-WS's own reproduction, which stays bit-faithful. HISTORICAL-WS
is run twice: once exactly as the dissertation ran it, for provenance, and once under the shared
specification, so that it is comparable to the other three. The protocol names these
`HISTORICAL-WS-asrun` and `HISTORICAL-WS` and never mixes them in one table.

---

## Objective specification, shared by all arms

Open until the dataset selection is fixed. What is already decided:

1. **The cost objective cannot be wall-clock training time.** `audits/provenance/README.md` shows
   that the dissertation's `Time_MeanFold` does not reproduce across environments, because it is a
   timing measurement rather than a function of the design. An objective that changes between runs
   of the same seed cannot anchor a paired comparison across 30 replications. Candidates: a
   machine-independent count (boosting rounds times depth-bounded node evaluations), model size, or
   inference latency measured under a fixed, reported protocol with repeated timing. Decided in
   `protocol/EXPERIMENT_PROTOCOL.md`.
2. **The quality objectives must include at least one threshold-free metric.** The dissertation used
   accuracy, precision, recall and specificity, all at a fixed 0.5 threshold. Adding a ranking
   metric and a calibration metric is needed for the result to speak to the machine-learning
   audience the venue has.
3. **The factor extraction is reported as scaled loadings,** and the text says so, per
   `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q1.

---

## Secondary arms, run only if the budget allows

| Arm | Purpose |
|---|---|
| WS-R | weighted sum with real anchors; completes the two-by-two if a reviewer asks |
| NBI-S with `restrict_t_nonnegative=True` | demonstrates on the real problem the defect that `audits/NBI_GEOMETRY_AUDIT.md` demonstrates on a synthetic one |
| Anchor-injection control | mandatory, not optional, per the caution above |

---

## Comparators

Separate from the arms, and not part of any contrast. They answer "is any of this worth it?" rather
than "which part of the change did what?". See `baseline_gap_assessment.md`.
