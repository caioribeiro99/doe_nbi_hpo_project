# Protocol amendment ledger

**What this document is for.** The confirmatory protocol was frozen, a measurement-validation pilot
then ran, and the pilot changed the protocol. This records every change, what caused it, and — the
question that actually matters — whether any comparative method-arm result had been observed when the
change was made.

**Terminology, used deliberately.** This protocol is **prospectively specified and pilot-amended
before the confirmatory campaign**. It is *not* described as pre-registered. Tag
`xgboost-hpo-protocol-v1` is the evidence that the original specification existed before Stage A ran;
tag `xgboost-hpo-protocol-v2` is the amended confirmatory protocol. Neither tag is moved.

**The fact that governs every row below.** At the time of every amendment, **no arm had been
executed**. Not HISTORICAL-WS, not WS-S, not NBI-S, not NBI-R, and no baseline. Stage A evaluated a
design matrix and three candidate external sets and computed screening statistics from them. It never
ran an optimizer, never produced a Pareto front, and never compared two methods. No amendment could
have been made to favour an arm, because no arm's behaviour was observable.

---

## Lineage

| Point | Commit | Date and time | What it is |
|---|---|---|---|
| Dissertation freeze | `67d9fe5` | 2025-12-28 18:28:45 −03:00 | `v0.1.0-dissertation`, the historical implementation |
| **Protocol v1** | `355c290` | 2026-09-13 12:05:18 −03:00 | `xgboost-hpo-protocol-v1`, frozen before the pilot |
| Stage A runner | `6972e49` | 2026-09-13 12:15:41 −03:00 | screening runner and factor-stage tests |
| Stage A corrections | `903a756` | 2026-09-13 12:40:30 −03:00 | amendments 1, 2, 3a, 4a, 9 |
| Stage A completion | `99cfdc8` | 2026-09-13 13:12:21 −03:00 | amendments 3b, 4b, 5, 6, 7, 8 |
| **Protocol v2** | `c50c380` | 2026-09-13 13:12:53 −03:00 | `xgboost-hpo-protocol-v2`, the confirmatory protocol |

Elapsed from v1 to v2: 67 minutes. Stage A performed 1,464 real evaluations in that window and zero
optimizer runs.

---

## Amendment 1 — the factor model is fitted on the design and applied to validation

| | |
|---|---|
| **v1 specification** | §6.3 fixed the extraction, rotation, component count, orientation and aggregation, but did not say on which data the model is fitted. |
| **Engineering problem** | The first screening implementation fitted the factor stage independently on the design set and on the held-out set, then scored a surface fitted against the first using targets defined by the second. Each fit carries its own standardization, rotation and sign orientation, so the two composites are in different coordinate systems. |
| **Evidence** | External R² for the quality surface of −1.116 on Adult and −1.523 on Bank Marketing, with rank correlations of 0.015 and −0.057. A surface has no relationship at all with the quantity it was fitted to predict only if the target is not that quantity. |
| **Revised specification** | The factor model is fitted on the design and **applied** to held-out points. This is also what the campaign requires: the method only ever sees the design. |
| **Commit** | `903a756`, 2026-09-13 12:40:30 −03:00 |
| **Arm results observed first?** | **No.** No arm existed. |
| **Could it favour an arm?** | **No.** All four arms consume the same factor stage; it is held fixed across arms by construction (`protocol/method_arms.md`). A change to a shared stage moves every arm identically. |

## Amendment 2 — the cost response is log-transformed

| | |
|---|---|
| **v1 specification** | §6.1 set the primary cost objective to total leaf count. §6.3 specified standardization but no per-response transform. |
| **Engineering problem** | The dissertation applies `log1p` to its own cost response (`time_transform="log1p"` is the frozen default in `factor_analysis.py`). Replacing wall-clock time with leaf count dropped that transform. Leaf count spans two to three orders of magnitude over the design box, measured range ratios 591 to 4,273, and a quadratic response surface cannot describe it raw. |
| **Evidence** | In-sample R² of the cost surface, raw against log-transformed: MAGIC 0.601 / 0.957, Spambase 0.645 / 0.947, Adult 0.499 / 0.948, Bank Marketing 0.487 / 0.949. Untransformed, backward elimination at α = 0.05 removed every term on Adult and Bank Marketing and left an intercept-only cost surface. |
| **Revised specification** | Each response declares a transform alongside its direction; neither is inferred. The cost response declares `log1p`. |
| **Commit** | `903a756`, 2026-09-13 12:40:30 −03:00 |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** Shared stage, identical for all four arms. It does change the objective all arms optimize, which is why it is an amendment to the protocol rather than an implementation detail. |

## Amendment 3 — factor sign orientation

Two steps, because the first attempt was itself wrong.

### 3a — an orientation rule is required at all

| | |
|---|---|
| **v1 specification** | §6.3: "Sign orientation: declared per objective, not inferred." That fixes the direction of each **response**. It does not fix the sign of each **factor**, which is a separate and necessary decision. |
| **Engineering problem** | A principal component's sign is arbitrary and Varimax does not fix it, so the quality composite's sign was arbitrary and its correlation with the cost factor — screening criterion 1 — could come out either way. |
| **Evidence** | On Bank Marketing, changing the log-loss transform, a change that does not alter which configurations are good, flipped the measured objective conflict from −0.662 to +0.688. |
| **Revised specification (first attempt)** | Orient each factor so the response loading most heavily on it has a positive loading. |
| **Commit** | `903a756`, 2026-09-13 12:40:30 −03:00 |

### 3b — the rule is the role-block mean, not the largest loading

| | |
|---|---|
| **Engineering problem** | Within the quality block, specificity trades off against accuracy, recall and the area under the curve across the decision threshold, so it loads with the opposite sign. On three of four candidate datasets specificity is the dominant loading on the leading quality factor. The rule therefore pointed the quality composite at specificity-badness. |
| **Evidence** | Dominant loading on the leading quality factor: Adult specificity +0.928 against a quality-block mean of −0.312; Spambase specificity +0.922 against −0.411. Measured objective conflict came out **positive** on Spambase (+0.319), Adult (+0.512) and Bank Marketing (+0.696), meaning "these objectives agree", on datasets whose raw metrics disagree: Spearman between log leaf count and accuracy is +0.29 to +0.41, and between log leaf count and log loss is −0.17 to −0.39, on every dataset. Screening would have rejected three of four datasets over a sign convention. |
| **Revised specification** | Orient each factor by the **mean loading over the responses in its own role block** — quality factors over the quality responses, the cost factor over the cost response. This is the rule the dissertation used. |
| **Commit** | `99cfdc8`, 2026-09-13 13:12:21 −03:00 |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** Shared stage. The rule is fixed by the response role assignment, which is itself fixed in §6.2, and cannot be re-derived per replication or per arm. |

## Amendment 4 — the external validation design

Two steps, because the first replacement did not work either.

### 4a — spanning set (tried, rejected)

| | |
|---|---|
| **v1 specification** | §7: 100 hyperparameter vectors per replication drawn by Latin hypercube over the box, with Paper 1's gate thresholds. |
| **Engineering problem** | In seven dimensions, uniform sampling places essentially no mass near the box corners, so the held-out set never probes the region where the surface must be trusted and carries far less response spread than the design. |
| **Evidence** | Quality-composite spread, design against uniform held-out set: MAGIC 0.654 / 0.330 (ratio 2.0), Spambase 0.894 / 0.091 (**9.8**), Adult 0.788 / 0.206 (3.8), Bank Marketing 0.957 / 0.357 (2.7). Since R² is `1 − SS_res/SS_tot`, a near-zero denominator makes any error large and negative: the measured −23.99 on Spambase is a property of the denominator, and rank correlation on the same data is 0.589. |
| **Attempted revision** | Half Latin hypercube, half arcsine-marginal (Beta(0.5, 0.5) coordinates concentrating at the ends of each range). |
| **Outcome** | **Rejected.** Spread ratios moved only to 2.00, 7.27, 2.91, 2.44. A design's response range comes from specific corner *combinations*, and with seven independent coordinates the chance of landing near the same end on all of them is about one in 128. No scheme with independent coordinates reproduces a factorial design's spread. |
| **Commit** | `903a756`, 2026-09-13 12:40:30 −03:00 |

### 4b — the design's complementary half fraction (adopted)

| | |
|---|---|
| **Revised specification** | The 88-run design is a face-centred central composite: a 64-run half fraction of the 2⁷ factorial with defining relation "product of all seven signs = +1", plus 14 axial and 10 centre runs. The external set is its **complementary half fraction**, the 64 corners with sign product −1, plus 14 axial runs at half the design's axial distance. Total 78 real evaluations. |
| **Why the axial runs** | On a two-level set every squared coordinate equals one, so the quadratic terms collapse into the intercept and corners alone cannot test curvature. |
| **Evidence** | The same surfaces, scored against each construction: MAGIC +0.461 / +0.468 / **+0.775**; Spambase −23.99 / −12.74 / **+0.953**; Adult −2.68 / −0.92 / **+0.911**; Bank Marketing −0.89 / −0.71 / **+0.917** (uniform / spanning / complement). |
| **Commit** | `99cfdc8`, 2026-09-13 13:12:21 −03:00 |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No, and this is worth stating carefully.** The external set feeds only the surrogate reliability gate, which is applied identically to WS-S, NBI-S and NBI-R and not at all to HISTORICAL-WS, whose historical pipeline has no gate. The gate's *pass rate* may change with the construction, and that is a reported quantity, not a comparison between arms. No optimizer sees a validation response (verified in `protocol/EXTERNAL_VALIDATION_VERIFICATION.md`). |

## Amendment 5 — external validation count, 100 → 78

| | |
|---|---|
| **Cause** | Mechanical consequence of amendment 4b: the complementary fraction has 64 corners and 14 axial runs. |
| **Evidence** | The construction is deterministic; 78 is its size, not a choice. |
| **Commit** | `99cfdc8` |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | It is 22 evaluations cheaper, which lowers `B_total_solution` for the three surrogate-gated arms and lowers the comparator budget with it. Because the comparator budget tracks the most expensive arm, every comparator is reduced by the same 22 evaluations. No arm gains relative budget. |

## Amendment 6 — budget consequences

| | |
|---|---|
| **v1 specification** | `B_total_solution` 108 / 208 / 208 / 408; comparator budget 408; NSGA-II population 34 for 12 generations (exact); unmatched NSGA-II at ten times the budget on every replication. |
| **Revised specification** | `B_total_solution` 108 / 186 / 186 / 386; comparator budget 386; NSGA-II **population 32 for 12 generations** (384, shortfall 2 evaluations, 0.5%, reported — 386 = 2 × 193 admits no exact factorization with ten or more generations); unmatched NSGA-II at ten times the budget on **one replication per dataset**. |
| **Evidence for the unmatched scoping** | At ten times the matched budget on all 30 replications and 4 datasets, the unmatched run alone costs 463,200 evaluations, about 180 hours. On one replication per dataset it is 15,440 evaluations, about 6 hours. |
| **Commit** | `99cfdc8` |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | The NSGA-II shortfall of 2 evaluations in 386 is a 0.5% handicap against NSGA-II and is reported. Scoping the unmatched run to one replication reduces the statistical weight of the comparison most likely to be *unfavourable* to the paper's own arms, so it is recorded here explicitly as a change that works against the paper's interest, taken for cost and not for outcome. |

## Amendment 7 — the campaign runs at two objectives

| | |
|---|---|
| **v1 specification** | Two objectives, with review finding M5 flagging three as scientifically preferable and open item 14 leaving the decision to the pilot's budget. |
| **Evidence** | Measured at 1.40 s per evaluation: two objectives give 285,120 evaluations, 111 hours, 4.6 days serial; three give 357,120 evaluations, 5.8 days, against a 5-day ceiling. |
| **Revised specification** | Two objectives. |
| **Commit** | `99cfdc8` |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | Possibly, and it is the one amendment where the answer is not a flat no. Objective count changes the geometry of the problem, and NBI's advantage over weighted-sum scalarization is generally larger at higher objective count, where the CHIM is a simplex rather than a segment. Choosing two objectives is therefore the **conservative** choice for the paper's own hypothesis. It is nonetheless revisited on scientific rather than budget grounds in `protocol/OBJECTIVE_COUNT_DECISION.md`, because a serial projection is not a sufficient reason to omit a load-bearing objective. |

## Amendment 8 — independent cross-check on the conflict measurement

| | |
|---|---|
| **v1 specification** | None. The measurement had no independent check. |
| **Engineering problem** | Amendment 3b showed that a derived quantity built from an extraction, a rotation and an orientation can silently invert. A more careful derivation is not a defence against that; an independent measurement is. |
| **Revised specification** | The screening computes objective conflict a second way, from the canonicalized responses directly with no factor stage involved, and **refuses to report** when the two disagree in sign. |
| **Commit** | `99cfdc8` |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** It is a guard that can only stop the run, never alter a result. |

## Amendment 9 — the screening surrogate uses backward elimination

| | |
|---|---|
| **v1 specification** | §7 specified backward elimination at α = 0.05 with hierarchy enforced for the campaign surrogate. The first screening implementation fitted a full 36-term quadratic instead, so the screening was not measuring the protocol's surrogate. |
| **Revised specification** | The screening fits the protocol's surrogate: backward elimination at α = 0.05, hierarchy enforced, coded units. |
| **Commit** | `903a756` |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** It made the screening faithful to a specification that already existed. |

## Amendment 10 — tag lineage recorded

| | |
|---|---|
| **Change** | `protocol/EXPERIMENT_PROTOCOL.md` gained a paragraph stating that v1 is not moved and why. |
| **Commit** | `c50c380`, 2026-09-13 13:12:53 −03:00 |
| **Could it favour an arm?** | No; documentation only. |

## Amendment 11 — Stage B was descoped, and that was not recorded

| | |
|---|---|
| **v2 specification** | §12 defines Stage B as "all four arms and all comparators on one partition of one dataset", producing the paired standard deviation of each primary endpoint, the smallest effect the design detects at R = 30, and confirmation that every NBI subproblem certifies on the real problem. |
| **What actually ran** | a throughput benchmark whose own header says "No arm is run", plus a calibration of the design and external set. Two of the three §12 deliverables do not exist. |
| **Why it matters** | the R = 30 detectable effect was unmeasured while a protocol amendment was being decided, which inverts the priority the decision rule itself insists on. |
| **Detected by** | the objective-count adversarial review, not by the author. |
| **Revised specification** | Stage B is completed before launch: `STAGE_B_STATISTICAL_SENSITIVITY.md` measures the paired endpoint standard deviations and the detectable effect from pre-campaign information only, and the arm-certification check runs on one real partition. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** The descoping is precisely that no arm ran. |
| **Could it favour an arm?** | **No**, but it could have concealed that the study is underpowered, which is worse than favouring an arm and is why it is recorded rather than quietly completed. |

## Amendment 12 — the objective-count decision rule is not independently dated

| | |
|---|---|
| **The claim the amendment rests on** | that `protocol/OBJECTIVE_COUNT_DECISION.md` §A–§E, including the four-day threshold for "comfortably under", was written **before** any throughput number was read. |
| **The problem** | the decision document and the throughput artifacts landed in the **same commit**, `064832c` (2026-09-14 11:22:31 −03:00), which carries no tag. Nothing a reviewer can check establishes the ordering, and file modification times are trivially writable. The v1-to-v2 lineage was handled correctly; this decision, the one that most needed the discipline, was not. |
| **Detected by** | the editor role of the adversarial review. |
| **What is done about it** | the history is **not** rewritten, because rewriting it to look better is the opposite of the remedy. Instead: (a) this entry records the defect in the same ledger that records everything else; (b) the decision was in any case **refused**, so the ordering claim no longer carries a conclusion; (c) from here on, any decision rule that must predate a measurement is committed and **tagged** on its own before the measurement runs. |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | The undated rule could have been written to fit the measurement. It was not, but that is an assertion rather than a check, which is the point of this entry. The refusal makes it moot for this decision and the tagging rule makes it checkable for the next. |

## Amendment 13 — per-objective surrogate gating, and the consequence of failure

| | |
|---|---|
| **v2 specification** | §7 gated the surrogate at external R² ≥ 0.5 and Spearman ≥ 0.9, measured for the aggregated quality composite and for cost, and **never said what failing does**. |
| **Engineering problem** | two things. The gate was never applied to each objective the campaign optimizes; and Spambase's quality composite **already fails today** at Spearman 0.847 while the campaign would have proceeded in silence. |
| **Evidence** | `audits/objective_count_evidence.json`: 3 of 12 per-objective cells fail. Spambase has no gate-passing quality objective at all — its leading factor scores Spearman 0.861 and its second scores R² 0.080 with rank correlation 0.252 on a five-term surface. |
| **Revised specification** | §7.1 fixes the consequence of failure before any arm runs: the gate never removes a dataset and never changes an arm; it is a reported per-replication covariate; every primary comparison is additionally reported conditioned on gate status; a dataset failing in the majority of replications is named in the results table and the abstract; and the thresholds never move. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** The rule is deliberately weak: a gate that changed the experiment would let surrogate quality select the evidence. HISTORICAL-WS is ungated because the historical pipeline has no gate, which is a property of that method and is reported as one. |

## Amendment 14 — one factor model per dataset, applied to every replication

| | |
|---|---|
| **v2 specification** | amendment 1 fixed "fitted on the design, applied to held-out points" and said nothing about refitting per replication. |
| **Engineering problem** | if the model is refit per replication, the objective is not the same variable in every pair, so 30 paired indicator values do not live in one objective space and no normalized indicator is invariant to that. |
| **Revised specification** | §7.2: one factor model per dataset, fitted on the 166-point reference set already evaluated and committed, applied to every replication; the per-replication refit reported as a sensitivity with Tucker congruence coefficients. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** Shared stage, identical across arms. |

## Amendment 15 — the objective count, decided and then refused

| | |
|---|---|
| **Proposal** | promote the second quality factor from a component of the aggregated composite to a third optimization objective. |
| **Outcome** | **refused**, after a four-role adversarial review. Full reasoning in `protocol/Q3_AMENDMENT_REVIEW.md`. |
| **Why** | the proposed objective fails §7's gate on two of four datasets; the two quality axes exchange roles across the panel, so "objective 2" does not name the same quantity on any two datasets; HISTORICAL-WS cannot exist at three objectives; and two of the proposal's own arguments were wrong. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | Refusing it is the choice **less** favourable to this paper's hypothesis, since NBI's advantage over weighted-sum scalarization is expected to grow with objective count. Recorded that way deliberately. |

## Amendment 16 — indicator and diagnostic corrections

| | |
|---|---|
| **v2 specification** | §9 names IGD⁺ among the indicators and §10 named no primary indicator. |
| **Engineering problems** | `reporting.igd` computes plain IGD, not the IGD⁺ the protocol names; nothing filtered dominated points, although a weighted-sum minimizer is weakly Pareto optimal by construction and an NBI subproblem solution need not be; and with five indicators and no primary, the realized comparison family sat between 3 and 15 tests per dataset. |
| **Revised specification** | `reporting.igd_plus` implements Ishibuchi et al.'s weakly Pareto compliant indicator; `dominance_filter` and `dominated_fraction` added, and every arm now reports the dominated share of its returned set so the asymmetry cannot be mistaken for approximation quality; §10 names the **hypervolume ratio** as the single primary indicator, with the other four secondary and descriptive; per-objective marginal comparisons are forbidden in §14. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | The dominance diagnostic is reported for **every** arm, not only the NBI ones, which is what keeps it neutral. Naming one primary indicator before any result is what keeps the family honest. |

## Amendment 17 — `k = 3` restated as an inherited assumption

| | |
|---|---|
| **v2 specification** | "Number of components: fixed at 3, with the Kaiser criterion reported alongside", which reads as though the criterion supports the choice. |
| **Evidence** | it does not. Kaiser retains **2, 2, 1 and 1** components on MAGIC, Spambase, Adult and Bank Marketing — never three, on any dataset. The third-to-fourth eigenvalue ratio is 2.93, 2.10, 1.34 and 1.06, so on Bank Marketing the retained subspace is effectively arbitrary. |
| **Revised specification** | §6.3 states that three components is an assumption inherited from the NBI-VRF construction the paper exists to decompose, not a criterion-supported choice, and that no standard retention criterion supports three on this panel. Eigenvalues and the ratio are reported per dataset per replication. |
| **Commit** | this one |
| **Arm results observed first?** | **No.** |
| **Could it favour an arm?** | **No.** Shared stage. |

## Amendment 18 — the claim-blacklist scanner made real

| | |
|---|---|
| **The problem** | `COST_OBJECTIVE_CLAIM_BOUNDARY.md` stated the cost-terminology rule was "enforced by `scripts/check_claim_blacklist.py`". It was not. The scanner read only `*.tex` and a compiled PDF, of which there were none, so it exited zero having scanned nothing; its patterns covered none of the four workspace terminology rules; and "pre-registered" appeared 15 times across live documents that `PROTOCOL_AMENDMENTS.md` declares must not use it. |
| **Revised specification** | the scanner reads `*.md` as well, gains twelve patterns covering the objective-count, surrogate-adequacy, Stage-A register, pre-registration and cost-terminology rules, exempts rule-stating documents **by name in an auditable list** rather than by a detectable phrase, and runs in the test suite on every commit rather than only at build. The "pre-registered" uses were swept: the term is now reserved for what tag v1 froze before any measurement. |
| **Commit** | this one |
| **Could it favour an arm?** | **No.** A control that can only refuse text. |

---

## Summary of exposure

| Question | Answer |
|---|---|
| Amendments made after observing any arm's result | **0 of 18** |
| Amendments made after observing any Pareto front | **0 of 18** |
| Amendments made after observing any method comparison | **0 of 18** |
| Amendments found by the author | 1–10 |
| Amendments found by adversarial review rather than by the author | **11–18** |
| Amendments affecting a stage shared identically by all arms | 1, 2, 3, 8, 9 |
| Amendments affecting budget symmetrically across arms and comparators | 5, 6 |
| Amendments affecting only the surrogate gate, which HISTORICAL-WS does not use | 4 |
| Amendments whose direction is conservative for the paper's own hypothesis | 6 (unmatched NSGA-II scoping), 7 (objective count) |

**Amendment 7, the objective count, was the one that deserved attention**, because it was the only
one justified by cost rather than correctness. It was reopened on scientific grounds, reviewed by
four adversarial roles, and **refused** (amendment 15). The evidence that refused it —
`audits/objective_count_evidence.json` — was produced by a script committed alongside it, because
evidence a reviewer has to re-derive from scratch is not provenance.

**Amendments 11 and 12 are process failures, and they are the author's.** Stage B was descoped
without being recorded, and the decision rule that had to predate a measurement was committed
together with it. Neither was found by the author. Both are recorded here in the same form as
everything else rather than corrected out of sight, because a ledger that only contains the
amendments its author noticed is not a ledger.

**How "no arm has been executed" can be checked** rather than taken on trust: no
`experiments/xgboost_hpo_vrfnbi/` directory exists at any commit up to and including this one, no
`EvaluationCache` database has been written, and `git log --diff-filter=A` shows no arm output
artifact. The campaign runner publishes its cache accounting at launch, which makes the claim
positively checkable from then on.

## What Stage A is, and is not

Stage A was an **engineering and measurement-validation pilot**. It is not confirmatory evidence and
no scientific conclusion rests on it.

Correct: *"all four candidate datasets passed the pre-campaign screening under the revised
measurement procedure."*

Incorrect, and not to be written: *"the surrogate is reliable on all four datasets."* The R = 30
confirmatory campaign determines the distribution and stability of surrogate fidelity. Stage A
measured one partition per dataset, which is one draw from that distribution.
