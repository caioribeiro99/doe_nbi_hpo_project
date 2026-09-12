# EAAI submission assessment

Written after the evaluation-matched NSGA-II baseline and the three-reviewer adversarial simulation. The simulation
returned **MAJOR from all three reviewers** (21 major comments: 17 valid, 4 partially valid, none invalid), and all
20 must-fix items have been applied. This document judges whether *Engineering Applications of Artificial
Intelligence* remains the right target.

**Recommendation: yes, submit to EAAI, with Applied Soft Computing as the fallback.** The reasoning, and the two
things that could change it, are below.

---

## 1. Scope fit

EAAI publishes applications of AI methods to engineering problems, including surrogate-assisted and multiobjective
optimization, and the journal already carries this exact methodological family — the authors' own predecessor is
EAAI 162:112510, and the group's related work appears in Applied Soft Computing, IJAMT and IJPE.

The honest tension: this is a **methodology-evaluation paper about machine-learning objectives**, not an engineering
application. Its datasets are credit, insurance and transaction risk, not a manufacturing process. Reviewer 3 raised
exactly this and it is the single largest scope risk.

The answer is that the paper's subject is the *optimization framework*, not the classifiers. What is diagnosed —
when a one-shot surrogate can be trusted, what anchor error does to a CHIM construction, why a step-function cost
breaks a continuous relaxation — is engineering-optimization content, and it transfers back to the process-
optimization setting the framework came from. The classifier ensemble is the test bed that makes the failure modes
observable, because a real objective evaluation costs a millisecond rather than a machining run.

**Verdict: in scope, but the framing must lead with the optimization question, not the ensembles.** The current title
and abstract do.

## 2. Novelty fit

The novelty review established that the construction is prior art four times over: Kwon et al. (2024) for classifier
ensembles specifically, Rocha et al. (2025) for a neural-network ensemble with NBI, Bacci et al. (2019) for forecast
combination, and the authors' own Pereira et al. (2025). The manuscript claims none of it, and says so in the
abstract, the introduction, a dedicated related-work subsection and the contribution list.

After the reviewer fixes, four further elements were also attributed to prior art rather than claimed:
metamodel-free NBI (Gellerich et al. 2023), that anchor error damages a scalarization (Isermann 1988; Deb et al.
2010; Herrmann et al. 2026), that a change of cost accounting reorders comparisons (Wang et al. 2022), and inference
cost as a Pareto objective for post-hoc weighted ensembles (Maier and Purucker 2024, 2026).

What remains, and what the paper must be judged on:

1. The controlled A/B/C decomposition, **with the injection control that separates set composition from CHIM
   relocation** — no prior work computes a CHIM method's anchors both from a surrogate and from the true objectives
   on the same problem and compares the resulting fronts.
2. Real-objective revalidation of every candidate against a reference the compared methods do not generate.
3. The external reliability gate, and the finding that it detects unusable surfaces but not misplaced anchors.
4. Replication over 30 partitions per dataset with overlap-corrected paired inference.
5. The disagreement between the cost that is optimized and the cost that is paid, for one weight vector.
6. The demonstration that the classical synergism criterion is satisfied by surfaces whose real blends contradict it.

**Verdict: sufficient for a full paper, but only because of the evaluation architecture.** If a reviewer insists the
contribution is "a negative result about someone else's method", the honest reply is that the method is substantially
the authors' own, and that reporting its failure modes with this much control is the useful thing to do with it.

## 3. Self-overlap with the predecessor

Highest-salience risk, since the predecessor is in the same journal and shares an author.

Mitigations already in place: a dimension-by-dimension comparison in `self_overlap_assessment.md` (13 dimensions, 2
overlapping, both labelled inherited); the competing-interest declaration names the predecessor and tells the editor
that NBI-A *is* that paper's construction and is reported here failing on two of four datasets; no prose, figure or
table is reused; the predecessor's full text was never accessible to the authors of this manuscript beyond its
abstract and first page.

The strongest argument is structural: the closest prior art is now **Kwon et al. (2024), an unrelated group**. A
single set of contributions separates this manuscript from Kwon, from Rocha and from the predecessor alike. The
self-overlap question is therefore no longer the binding one.

**Verdict: manageable, provided the cover letter raises it first rather than leaving it to be discovered.**

## 4. Expected reviewer concerns, and the current answer

| Concern | Status |
|---|---|
| "No evolutionary baseline" | **Closed.** NSGA-II at a matched real-objective budget, 120 runs, ratio 0.999996. |
| "Your own method loses to NSGA-II" | **Reported plainly.** It bounds one claim; the diagnosis is unaffected. It is also the paper's most useful practical finding. |
| "The anchor effect is a set-composition artifact" | **Found by simulated Reviewer 1, verified, corrected.** The injection control now decomposes it: 80/90/49/1% closed by injection alone. |
| "Only four datasets" | Open and acknowledged. The dataset is stated as the unit of generalization; nothing is pooled. |
| "Effects are practically negligible" | Acknowledged in the first paragraph of the Results. Ensembling buys 0.0007–0.0060 AUC; method differences are smaller. The paper is about trustworthiness, not leaderboard gain. |
| "Reference is self-graded" | Acknowledged; every primary comparison is repeated against a sample-core reference the five compared methods do not contribute to. |
| "Manuscript is too long" | **Open — see below.** |

## 5. Length

**This is now the main practical obstacle.** The manuscript is 43 pages and about 14,700 words of body prose in
single-column 11pt, against a compression target of 9,500–11,000.

Two compression passes ran. The first cut 16,369 → 13,676 words, but its audit had to restore 41 pieces of deleted
science, which is evidence that the remaining text is not padding. The second, deduplication-only pass reached 12,789
and its methodology agent reported it could not go further because "after removing everything with a quotable
duplicate, what remains is material the ownership map assigns to Methodology". The reviewer fixes then added back
roughly 1,900 words of required attribution, scoping and disclosure.

In EAAI's two-column production format 43 single-column pages is roughly 22 typeset pages, which is long but within
what the journal publishes. The options, in order of preference:

1. Move Methodology §3.5–3.6 (references, comparators, indicator definitions) to the supplement, keeping the
   equations — about 600 words.
2. Move the RQ2 edge-condition derivation to the supplement, keeping the result and Table 7 — about 400 words.
3. Move Related Work §2.1 and §2.2 wholesale to the supplementary literature review, keeping a single paragraph —
   about 700 words.

Together those reach about 13,000 words without losing a claim. Going below that requires cutting caveats, and the
first compression pass demonstrated what happens then.

**Recommendation: do option 3 before submission; hold 1 and 2 for the revision round if an editor asks.**

## 6. Data and code availability

Strong, and worth foregrounding: four annotated tags freezing the exact states, deterministic seeds, dataset
checksums with acquisition commands, a resumable checkpointed runner, 83 unit tests, and every figure and table
regenerable by one script. Raw data are not redistributed (Kaggle competition rules), which is stated with the
invariants needed to verify an independently obtained copy.

## 7. Ranking after the baseline and the review

| Rank | Venue | Change from the pre-baseline assessment |
|---|---|---|
| 1 | **Engineering Applications of Artificial Intelligence** | unchanged; the NSGA-II arm closes the concern that was most likely to be fatal here |
| 2 | **Applied Soft Computing** | strengthened — this readership most wanted the evolutionary baseline, and now it exists |
| 3 | Knowledge-Based Systems | unchanged |
| 4 | Expert Systems with Applications | unchanged; still needs a different, application-led framing |
| 5 | Information Sciences | weakened; still no theory |

**A genuine alternative worth considering:** the finding that a large Scheffé interaction coefficient can satisfy the
classical synergism criterion while the real blend contradicts it is a *mixture-design* result as much as a machine-
learning one. If EAAI and ASOC both decline, Quality and Reliability Engineering International or the Journal of
Quality Technology would find that specific contribution novel, at the cost of a much smaller audience.

## 8. Remaining blockers before a human can submit

1. **Authorship and funding are unresolved.** The author list is a placeholder. Whether the co-authors of Pereira et
   al. (2025) are co-authors here, and which grants supported this work, are decisions only the corresponding author
   can make. See `submission_eaai/author_contributions.md`.
2. **One lineage paper remains unread** (Rocha et al. 2020, *Engineering with Computers*), flagged in
   `research_lineage.md`.
3. **Length option 3** above, if the author agrees.
4. **The supplementary PDF has not been built** — `supplementary_plan.md` specifies S1–S13 but the document itself is
   not assembled.

None of these requires touching the frozen experiment.
