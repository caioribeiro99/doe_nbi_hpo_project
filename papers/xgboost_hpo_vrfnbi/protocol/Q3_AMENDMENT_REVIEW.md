# Adversarial review of the two-to-three objective amendment: REFUSED

**Decision: the amendment is refused. The confirmatory campaign runs at two objectives, as frozen in
`xgboost-hpo-protocol-v2`. No `xgboost-hpo-protocol-v3` is cut for this purpose.**

Four independent adversarial roles reviewed the proposal: a multiobjective methodologist, a
statistician, an automated-machine-learning empiricist, and an editor with a research-integrity
brief. They returned 2 for KEEP_Q2 and 2 for ADOPT_WITH_CHANGES, and an adjudicating pass recommended
ADOPT_WITH_CHANGES. **This document departs from that recommendation**, for reasons set out in §3.

Every load-bearing number below was reproduced independently before the decision was taken, and is
regenerable by `scripts/objective_count_evidence.py` into `audits/objective_count_evidence.json`. No
new real evaluations were used.

---

## 1. What the review found that the proposal had not measured

The proposal was taken without ever fitting the protocol's own surrogate to the objective it
proposed to promote. Every gate number in Stage A is for the **aggregated** composite. Fitted
per objective — the protocol's backward-eliminated surface, on the 88 design rows, scored on the
78-run complementary fraction:

| Dataset | objective | external R² | Spearman | terms | gate | dominant response | ρ with mean quality |
|---|---|---:|---:|---:|:--:|---|---:|
| MAGIC | quality 1 | 0.870 | 0.927 | 19 | PASS | ROC-AUC | +0.944 |
| MAGIC | **quality 2** | 0.896 | 0.951 | 16 | **PASS** | precision | −0.021 |
| MAGIC | cost | 0.917 | 0.940 | 18 | PASS | leaf count | −0.307 |
| Spambase | quality 1 | 0.969 | 0.861 | 14 | **FAIL** | specificity | +0.637 |
| Spambase | **quality 2** | **0.080** | **0.252** | **5** | **FAIL** | precision | +0.025 |
| Spambase | cost | 0.942 | 0.965 | 15 | PASS | leaf count | −0.796 |
| Adult | quality 1 | 0.905 | 0.912 | 8 | PASS | specificity | +0.614 |
| Adult | **quality 2** | 0.841 | 0.910 | 13 | **PASS** | log loss | +0.993 |
| Adult | cost | 0.934 | 0.955 | 17 | PASS | leaf count | −0.532 |
| Bank Marketing | quality 1 | 0.921 | 0.929 | 9 | PASS | specificity | +0.779 |
| Bank Marketing | **quality 2** | 0.720 | **0.874** | 12 | **FAIL** | log loss | +0.991 |
| Bank Marketing | cost | 0.942 | 0.958 | 17 | PASS | leaf count | −0.587 |

Three of twelve cells fail the gate the protocol froze at external R² ≥ 0.5 and Spearman ≥ 0.9.

## 2. The four reasons the amendment is refused

**2.1 The proposed objective fails the protocol's own reliability gate on half the panel.**
WS-S, NBI-S and NBI-R do not optimize the truth. They build the anchors, the payoff matrix, the
CHIM, the quasi-normal direction and every returned candidate from the **surrogate**. On Spambase
the third coordinate's surface retains five terms and scores external R² 0.080 with rank correlation
0.252. The third column of the payoff matrix would be the minimizer of a near-flat quadratic, the
quasi-normal `−Φ·1` would be rotated by it, and every candidate would inherit that. This is the same
failure mode amendment 2 diagnosed for untransformed leaf count — caught there, and uncaught here
until the review. §7 forbids moving the threshold after the fact.

The perversity is exact: **Spambase supplies the amendment's single strongest piece of evidence**,
a non-dominated count of 65 against a permutation null of 16.2, and that structure is measured on
real design evaluations. It is therefore precisely the structure no surrogate-driven arm can reach.

**2.2 The axes are not the same construct across the panel.** Correlation of each quality factor with
the unweighted mean of the six standardized canonicalized quality responses:

| | MAGIC | Spambase | Adult | Bank Marketing |
|---|---:|---:|---:|---:|
| quality 1 | **+0.944** | +0.637 | +0.614 | +0.779 |
| quality 2 | −0.021 | +0.025 | **+0.993** | **+0.991** |

On MAGIC and Spambase the *leading* axis is overall quality and the second is orthogonal to it. On
Adult and Bank Marketing the *second* axis is overall quality and the leading axis is a specificity
contrast. Dominant loadings: precision, precision, log loss, log loss. "Objective 2" does not name
the same quantity on any two datasets. A cross-dataset synthesis — which the study's own
question 13 requires — would be comparing a precision axis on one dataset against a log-loss axis on
another.

**2.3 HISTORICAL-WS cannot exist at three objectives.** It reproduces the dissertation by calling
the frozen `run_nbi_weighted_sum`, whose signature takes two models and a two-component weight grid,
and `method_arms.md` forbids reimplementing it. At three objectives one of the three primary
contrasts disappears.

**2.4 Two of the amendment's own arguments were wrong, and I made both errors.**

- The aggregation-weighting argument cited Spearman 0.374. That figure is from the **dissertation**
  pipeline, measured in `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4. Under the protocol's own factor
  stage the agreement is **0.865 to 0.964**. The free parameter the amendment offered to eliminate
  is about four times less awkward than claimed.
- The proposal stated that on Spambase "the suppressed axis is precision against specificity, which
  is the classical threshold trade-off". On the canonicalized Spambase design, precision and
  specificity correlate at **+0.863**: they agree. The −0.533 is a property of the rotation, not of
  those two metrics.

Both are corrected in `OBJECTIVE_COUNT_DECISION.md`.

## 3. Why this departs from the adjudication

The adjudicating pass recommended ADOPT_WITH_CHANGES, resting on one genuinely good observation: at
the measured throughput both objective counts fit together in about 1.3 days against a four-day
threshold, so the either/or need not be taken at all. Running both would dissolve §2.3, keep
objective count from becoming an unmeasured researcher degree of freedom, and cost nothing in wall
clock.

It is refused anyway, on three grounds the run-both proposal does not answer.

**Running an arm on an objective it cannot see does not become acceptable by labelling the result
secondary.** On Spambase, three of four arms would steer on a five-term near-flat surface. The
output would be noise, and publishing noise as a co-primary or as a sensitivity is the same act.

**§2.2 is not fixable by running more.** If objective 2 is precision on two datasets and log loss on
two others, no amount of compute makes the panel synthesizable. A per-dataset objective count, which
the editor role proposed, would leave a two-dataset q=3 result comparing a precision axis on MAGIC
against a log-loss axis on Adult.

**The amendment would add a large amount of unvalidated machinery for a secondary result.** The
review identified, and I confirmed, that hypervolume is a Monte-Carlo estimate at q ≥ 3 with a
front-dependent sampling box and a fixed seed; that the named IGD⁺ indicator is implemented as plain
IGD; that the spread measure sorts by the first objective and does not traverse a surface; that no
weight lattice exists for q ≥ 3; and that nothing filters dominated points, which NBI returns at
q ≥ 3 and weighted sum does not. Each is fixable. Fixing five pieces of indicator machinery so that
a secondary analysis can run is not a good trade when the primary analysis needs three of them fixed
anyway and the secondary is compromised by §2.1 and §2.2 regardless.

**Stated plainly, and against interest:** three objectives is where NBI's advantage over weighted-sum
scalarization is expected to be larger, because the convex hull of individual minima becomes a
simplex. Refusing the amendment is the choice **less** favourable to this paper's own hypothesis. It
is taken because the evidence does not support the objective, not because the objective is
inconvenient.

## 4. What the third objective is, and where it goes

Not discarded. Recorded as an evidenced future-work item with everything measured:

- It is load-bearing where it is reachable. On MAGIC and Adult it passes the gate, and on MAGIC,
  Spambase and Adult the three-objective non-dominated count clearly exceeds its permutation null
  (42 vs 21.8, 65 vs 16.2, 37 vs 18.1, all p = 0.000).
- On Bank Marketing it is **below** its own null (13 vs 22.7, p = 1.000) with inter-factor
  ρ = +0.759, so on that dataset the evidence runs against the amendment.
- The inter-factor correlations with bootstrap intervals: MAGIC −0.308 [−0.535, −0.053], Spambase
  −0.533 [−0.719, −0.303], Adult +0.649 [+0.443, +0.812], Bank Marketing +0.759 [+0.586, +0.871].
  The correct register is that redundancy was **not measured**, not that conflict was established.
- The review's strongest constructive suggestion, which two roles reached independently: the third
  objective a reviewer would actually want is **named** — quality, calibration, cost — rather than a
  latent rotated component. That formulation dissolves §2.2 entirely. It is the right shape for the
  follow-up study and is recorded as such.

## 5. What survives from the review and must be fixed before launch

Refusing the amendment does not dispose of the review. Most of its findings are defects in the
**two-objective** protocol, several of which the amendment merely made visible. They are tracked and
fixed before the campaign launches, in `PROTOCOL_AMENDMENTS.md` amendments 11 to 18 and in the
protocol sections named there. The most consequential:

| Finding | Why it matters at two objectives |
|---|---|
| §7 declares a pass/fail gate and never says what happens on failure | Spambase's quality composite **already fails** today at Spearman 0.847, and the campaign would proceed silently |
| The factor-model refit policy is unstated | if the model is refit per replication, the objective is not the same variable in every pair |
| The named IGD⁺ indicator is implemented as plain IGD | a primary endpoint does not compute what the protocol names |
| No dominance filter exists | NBI's returned set is certified feasible, not Pareto optimal |
| `k = 3` is not supported by any retention criterion | Kaiser retains 2, 2, 1, 1 components; never 3, on any dataset |
| The claim-blacklist scanner enforces none of the four terminology rules and currently scans nothing | a documented control that does not exist |
| Stage B was descoped from arms-and-comparators to a throughput benchmark, undocumented | the R = 30 detectable effect is unmeasured |
| The decision rule and the measurement it must predate were committed together | the ordering claim is not independently checkable |
