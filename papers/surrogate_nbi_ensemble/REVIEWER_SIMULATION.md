# Adversarial reviewer simulation and synthesis-editor verification

**Manuscript:** `/tmp/texout/main.pdf` (37 pages), source `papers/surrogate_nbi_ensemble/`
**Frozen evidence base:** `reports/pco213_postwork_benchmark/` (R = 30, 120 replications), `nsga2/`, `papers/surrogate_nbi_ensemble/{claims_and_evidence.md, research_lineage.md, novelty_matrix.md}`
**Simulation date:** 2026-09-12 · **Role of this document:** three adversarial reviews in full, followed by an independent verification of every major comment against the artifacts, a triage, and a submittability verdict.

Verification was done by reading the manuscript sections, the runner and analysis source (`scripts/pco213_run_postwork_benchmark.py`, `scripts/pco213_nsga2_analysis.py`, `src/mixens/{nbi,scheffe}.py`, `papers/surrogate_nbi_ensemble/build_assets.py`) and by recomputing from the frozen CSVs and the per-replication artifacts under `experiments/pco213_postwork_benchmark/`. One control the reviewers asked for but did not run — NBI-A's candidate set augmented with the three real anchors — was executed here; its per-partition output is in `anchor_control.csv` (session scratchpad) and is summarised under R1-M1.

---

## Summary of verdicts

| | VALID | PARTIALLY VALID | INVALID |
|---|---|---|---|
| Reviewer 1 (multiobjective optimization) | 5 | 2 | 0 |
| Reviewer 2 (ML evaluation / experimental validity) | 7 | 0 | 0 |
| Reviewer 3 (Associate Editor) | 5 | 2 | 0 |
| **Total (21 major comments)** | **17** | **4** | **0** |

No major comment was wholly unfounded. Four contain a specific sub-claim that is factually wrong; those are named in the verdicts (R1-M1's "internally inconsistent scalarization", R1-M6's "tautology", R3-M3's "instrumentation costs more than NBI-C", R3-M7's "RQ9 is sold as a contribution").

---

# PART I — THE THREE REVIEWS IN FULL

## Reviewer 1 — Multiobjective optimization (NBI, surrogate-assisted MOO, evolutionary MOO)

**Scores:** novelty 3 · technical 3 · experimental 3 · clarity 4 · **confidence 4** · **Recommendation: MAJOR**

### Summary

The paper transfers the established DoE–RSM–NBI pipeline (mixture design over the weight simplex, Scheffé canonical polynomials, Normal Boundary Intersection on those surfaces) from engineering and forecasting to classifier ensemble weighting, and asks not whether it produces a front but when that front can be believed. The construction is explicitly disclaimed as prior art (Bacci 2019, Moreira 2021, Rocha 2025, Kwon 2024, and the first author's own Pereira 2025 and dissertation, all cited with disclosure). The contribution is a three-arm decomposition — NBI-A (surrogate objectives, surrogate anchors), NBI-B (surrogate objectives, real anchors), NBI-C (metamodel-free) — plus four controls the lineage lacks: external surrogate validation with a pre-specified reliability gate, revalidation of all candidates on real objectives, an empirical Pareto reference with a sampled core of >=1e5 points, and an untouched holdout. 4 datasets x 30 partitions, paired by partition, Nadeau–Bengio corrected, nothing pooled. A pre-registered, evaluation-matched NSGA-II baseline is added and reported to beat the metamodel-free arm on all four datasets and both endpoints.

I checked roughly thirty-five quantitative claims against the CSVs and the numbers are accurate to a degree I rarely see: gate counts (0/5/16/30 and 30/3/12/30), win/tie/loss counts, Table 3 medians and their bootstrap intervals, the BNP NBI-A collapse values (seven at 0.036–0.120, two intermediate, twenty-one at 0.891–0.928), NBI-C level counts (>0.97 in 30/30/24/25; worst Porto partition 0.877), the cost-definition switch counts (10/24/8/20), the AUC–log-loss conflict deltas, the ensembling gains, the R10-vs-R30 inside-interval counts (9/12 and 7/12), and the Holm-corrected p-values in Table 8. Every one matched. The claims map, with its explicit "wording not allowed" rules, is honored in the manuscript text.

My objections are therefore not about arithmetic. They are about (i) whether the A-versus-B contrast isolates what the paper says it isolates, (ii) whether the cross-dataset mechanism statement survives being read in effect sizes rather than win counts, (iii) whether the compute accounting that drives the paper's practical recommendation is correct, (iv) whether the self-augmented reference is as harmless for the paired contrasts as the Limitations claim, (v) whether the comparator suite answers the budget question that matters for a surrogate-assisted paper, and (vi) whether the support-cost result is a finding or a tautology. Most of the required work is reanalysis of artifacts the authors already have, plus two or three short additional runs; the study underneath is sound and the diagnostic contribution is worth publishing once these are addressed.

### Major comments

**R1-M1. The A-versus-B contrast is not the clean single-factor experiment claimed, and on the dataset carrying the headline result it is largely a set-composition effect.**
Section 3.4 asserts that "A versus B isolates anchor misplacement with the surrogate objectives and the solver held fixed, and is a clean contrast"; Section 6.1 builds the whole mechanism story on it ("an anchor error is global rather than local: it relocates the CHIM, redirects every quasi-normal and rescales the objective space of all 66 subproblems"). Two problems.
(a) *Set composition.* Section 3.4 states that "For vertex values of beta the solution is the corresponding anchor, returned directly rather than re-derived." With m = 3 objectives, three of the 66 beta-lattice points are vertices, so NBI-B's returned candidate set literally contains the real single-objective optima, while NBI-A's contains the surrogate optima. Those same real optima are in the empirical reference (Section 3.5 revalidates "the references" and Section 3.6 unions all candidate sets into the reference), and `reference_diagnostics.csv` shows `src_refs` contributing a median of 4 points to the Santander core front. So NBI-B is handed points that are on the reference front by construction. This is not the CHIM mechanism; it is arithmetic on the candidate set, and hypervolume with extreme points is maximally sensitive to it.
(b) *Simultaneous change of the whole scalarization frame.* Moving from surrogate to real anchors changes the utopia point, both normalization scales (via the payoff-table pseudo-nadir), the CHIM simplex and the quasi-normal direction at once. Worse, NBI-B is internally inconsistent as a scalarization: Phi-bar is built from real objective values while the equality constraint Phi-bar*beta + t*n-hat = F-bar(w) evaluates the surrogate, so the two sides of the constraint come from different functions. "Anchor misplacement" is a reasonable label for the composite, but it is not an isolated factor, and the paper is scrupulous about exactly this issue for B versus C while asserting the opposite for A versus B. Related: the payoff-table pseudo-nadir is, for m = 3, precisely the estimator the nadir literature the paper cites (Deb et al. 2010) shows to be unreliable; the paper cites that work only to distance itself from it, never to note that its own normalization inherits the problem in all three arms.

*Evidence checked:* `sections/03_methodology.tex`; `sections/06_discussion.tex`; `tables/pareto_quality.csv`, `tables/reference_diagnostics.csv`. Per-partition computation (cost = weighted): the 7-point `single_objective_refs` set alone attains median HV ratio 0.971 on Santander against NBI-A's 0.789 and NBI-B's 0.981, beats NBI-A in 30/30 Santander partitions, and already covers a median 93% of the A-to-B hypervolume gap. Porto 31% (refs beat A in 23/30). BNP (refs 0.759 vs A 0.913, 9/30) and UCI credit (refs 0.337 vs A 0.622, 0/30) are worse than NBI-A, so there the mechanism is genuinely not set composition. Santander is the dataset with the largest reported effect (+0.179 HV) and the only one where the corrected test reaches significance (Holm p = 0.004).

*What would satisfy me:* Run the missing control — rescore NBI-A's candidate set augmented with the three real anchors, against the same reference, per partition; report Delta-HV and Delta-IGD+ for (NBI-A ∪ anchors) versus NBI-B. If most of the Santander gap disappears, the abstract, Section 5.3, Section 6.1 and the Conclusion must say that on Santander the effect is dominated by the returned extreme solutions and that the CHIM-relocation mechanism is evidenced by BNP and UCI credit only. Second, either add an arm that changes only the normalization, or drop "clean contrast" and state that A→B changes the utopia point, both normalization scales, the CHIM, the quasi-normal directions and three returned solutions simultaneously. Third, one sentence acknowledging that the payoff-table nadir at m = 3 is a known-biased estimator affecting all three arms.

**R1-M2. The cross-dataset mechanism claim is contradicted by the effect magnitudes read under the authors' own designated estimator; only win counts support it.**
The abstract, Section 5.3, Section 6.1 and the Conclusion all frame the anchor result as strongest where the surfaces fail; the Limitations name "real anchors matter most where the surrogate is unreliable" as the key cross-dataset statement. But 30/30 versus 24/30 is a statement about the variance of the paired difference, not its size. The paper's own pre-specified rule says that where the difference distribution is bimodal or heavy-tailed the median is the evidence, and it applies that rule to declare the BNP mean (+0.279) uninformative. Applied consistently, the median Delta-HV(B−A) ordering is Santander +0.196 (ROC-AUC gate 0/30), UCI credit +0.092 (gate 30/30), Porto +0.076 (16/30), BNP +0.059 (5/30). The dataset with perfect surfaces shows the second-largest anchor effect and the dataset with near-perfect failure the smallest. The ordering the paper implies is produced only by the BNP mean, which the paper itself discards. The Limitations flag that this is an association over four points; they do not flag that the four points do not show it.

*Evidence checked:* `statistics/paired_primary_effects.csv` (median Delta-HV +0.1960 / +0.0585 / +0.0756 / +0.0916; means +0.1792 / +0.2792 / +0.0853 / +0.0597). Recomputed under the method-independent sample-core reference from `nsga2/nsga2_pareto_quality.csv`: +0.1450 / +0.0701 / +0.0779 / +0.0892 — same inversion. Gate rates from `statistics/reliability_gate_r30.csv` and `tab02_gate.tex`. `reliability_gate_conditional_gain_r30.csv` shows no within-dataset gate contrast exists on Santander (0/30 AUC passes) and that the BNP contrast is the collapse partitions restated.

*What would satisfy me:* State median effect sizes next to every win count wherever this claim is made, and either drop the reliability ordering or replace it with what the data support: real anchors help on all four datasets, including the one whose surfaces are excellent; what is confined to unreliable surfaces is the catastrophic collapse mode (BNP, 7/30), not the typical-magnitude benefit.

**R1-M3. The compute accounting charges NBI-B nothing for the real anchors it depends on, which inflates the metamodel-free premium by roughly an order of magnitude and undercuts the paper's central practical recommendation.**
Table 5's caption and Section 3.4's variant table record NBI-B as consuming zero real objective evaluations ("0 (anchors reused)"); the Limitations repeat "NBI-A and NBI-B consume no real objective evaluations". But NBI-B's anchors are the real single-objective optima, and Section 3.5 says the ROC-AUC anchor comes from "a derivative-free ROC-AUC optimum over 4x10^4 sampled compositions with local refinement". The reproducibility section confirms 4.8x10^6 direct-AUC-search evaluations over 120 replications — about 10% of NBI-C's 4.2x10^5 per replication, and two orders of magnitude more than the 100 evaluations the paper carefully prices for the reliability gate. "Anchors reused" is true of the experiment's bookkeeping, not of the method. Recomputing from Table 5 with the single-objective reference stage charged to the arms that need it, the metamodel-free premium is 2528/(444+35) = 5.3x on Santander, 5.1x on BNP, 5.0x on Porto and 2.0x on UCI credit — not "73, 40, 45 and 3 times". Two load-bearing sentences fail: "a seventy-three-fold premium buys +0.008 hypervolume" (Section 6.5) and "on the convex log-loss the real anchors are one SLSQP call, so NBI-B costs essentially what NBI-A costs" (Section 6.1) — the latter ignores that there are three objectives and that the AUC anchor is not an SLSQP call; on the same table NBI-B-plus-references costs (444+35)/34 = 14x NBI-A on Santander. This matters because the paper's conditional recommendation is exactly where the unpriced anchor cost bites: in the expensive regime the framework exists for, obtaining real single-objective optima is the expensive part.

*Evidence checked:* `tables/tab05_compute.tex`; `tables/nbi_runs.csv`; `sections/03_methodology.tex`; `sections/09_reproducibility.tex`. Also internally inconsistent: "73, 40, 45 and 3 times" (5.5), "3 to 73 times" (6.5), "3 to 77 times" (7); FINAL_REPORT says 3–84x.

*What would satisfy me:* A compute table charging each arm for everything it needs to run standalone, in both wall clock and real objective evaluations, with the AUC anchor's 4x10^4 evaluations shown explicitly; restate the premium (about 2–5x); correct "NBI-B costs essentially what NBI-A costs"; reconcile the three wall-clock ranges.

**R1-M4. The self-augmented reference is not harmless for the paired contrasts, the authors have the fix in hand, and one reported finding does not survive it.**
The Limitations state that self-grading "inflates their absolute indicator values without affecting the paired contrasts on which every primary claim rests." That holds only when the two arms contribute equally, which is exactly what does not happen: NBI-A's revalidated front holds a median of 8–13 points on three datasets while NBI-B's holds 20–60 and NBI-C's 34–65. The authors already compute a method-independent sample-core reference — but only for the NSGA-II section. Recomputing the primary contrasts against it from `nsga2_pareto_quality.csv`: every headline direction holds, but magnitudes move materially (Santander Delta-HV(B−A) falls from +0.183 to +0.145, about 21%), and one stated finding reverses — Section 5.4's "On BNP Paribas the arms genuinely split: NBI-C covers more volume (24/30) while NBI-B is closer to the reference in IGD+ (24/30, Delta = −0.003)" becomes NBI-C better in IGD+ in 20/30 with median +0.0025 under the core reference. A claim the paper describes as a resolved genuine trade-off, and lists in Section 5.9 as one of the four things the extension settled, is an artifact of the reference definition. Separately, Section 5.5 describes the sample-core reference as "containing the search output of no optimizer", but Section 3.6 says the core includes a 40-cap epsilon-constraint sweep, and `reference_diagnostics.csv` shows that sweep contributing a median of 38 of ~135 Santander core-front points. The epsilon-constraint method is an optimizer; the sentence as written is false.

*Evidence checked:* `sections/07_limitations.tex`; `sections/05_results.tex` 5.4 and 5.5; `nsga2/nsga2_pareto_quality.csv` recomputed per partition under both reference definitions; `tables/reference_diagnostics.csv`.

*What would satisfy me:* Repeat Table 3 and the Section 5.4 comparisons against the sample-core reference, as primary or as a full sensitivity table; report the front-contribution share per arm per dataset; correct or withdraw the BNP HV/IGD+ split claim and note in Section 5.9 that it is reference-dependent; fix the "search output of no optimizer" sentence.

**R1-M5. The comparator suite answers the wrong budget question, and the paper's own framing says which question is right.**
NSGA-II is matched only to NBI-C's ~4.2x10^5 real evaluations. At population 66 that is roughly 6,400 generations on a five-variable problem: both methods are effectively run to convergence, and the result (+0.006 to +0.013 HV ratio) mostly says that at unlimited budget a modern EA edges out a scalarization-plus-local-solver scheme. That is not in dispute. The regime the whole framework exists for — the one Section 6.5 names as where "the surrogate is the only viable route" — is expensive evaluation, where NBI-A/NBI-B spend on the order of 166 real evaluations (66 design + 100 validation) plus anchors. The decisive missing experiment is NSGA-II, or plain random search, at *that* budget. It costs seconds given the cached probability matrices and the existing pymoo harness. The Limitations contemplate only the opposite direction. As it stands the paper establishes that the surrogate pipeline loses to an EA at a budget where nobody would use a surrogate, and says nothing about whether it beats an EA at the budget where one would.
Relatedly, Section 2.5 says the surrogate-assisted MOO field "answers primarily with model management", and cites ParEGO, MOEA/D-EGO and SMS-EGO. It then diagnoses a one-shot surrogate with no infill whatsoever. By the standards of the literature it cites, NBI-A is already known to be inadequate, and NBI-B is best understood as a single targeted model-management step. The manuscript never scopes its diagnosis this way. Finally, Messac et al. (2003) is cited precisely as having introduced the normalized normal constraint "partly to make anchor-based normalization better behaved" — the textbook remedy for the pathology this paper reports — and is never tried.

*Evidence checked:* `sections/05_results.tex` 5.5; `sections/07_limitations.tex`; `nsga2_preregistered_config.json`; `nsga2/nsga2_budget_runtime.csv`; `sections/02_related_work.tex` 2.5.

*What would satisfy me:* (1) NSGA-II and random Dirichlet search at the surrogate arms' real-evaluation budget (~166, plus 500 / 2,000 / 10,000), scored the same way. (2) Either one infill-based arm or an explicit scoping sentence in the abstract and Section 1 that the diagnosis applies to one-shot, no-infill surrogate pipelines of the DoE–RSM–NBI lineage. (3) A sentence on why NNC was not evaluated.

**R1-M6. The support-versus-weighted-cost result is close to a tautology as stated, and the experiment that would make it a contribution is cheap and not run.**
"The hypervolume-best method changes between the definitions in up to 24 of 30 partitions" is promoted to a headline contribution and to the abstract. But every method optimized the linear relaxation and none optimized the step cost, so scoring under a different objective and observing that the ranking changes is close to definitional; the size of the disagreement is a property of the cost range and the normalization box, not of the methods. The paper itself demonstrates this when it withdraws the BNP "bimodality" as a normalization-box artifact and retreats to "we therefore claim only the qualitative statement". The abstract and Conclusion do not retreat with it.
The informative version is answerable in seconds: with M = 5 there are only 31 non-empty supports and one evaluation is a matrix-vector product, so the support-cost problem can be solved essentially exactly by enumerating supports and running the continuous optimizer within each. NSGA-II, being derivative-free, can optimize the step cost directly with no modification. Neither is done. Consequently Section 6.6's "a step cost is not differentiable and cannot enter a gradient-based formulation" is an argument about NBI, not about the problem, and "optimize over supports explicitly where deployment cost is the real objective" is offered without evidence about what that buys. The related-work claim that "no prior work contrasts the step cost a deployed support pays with the continuous relaxation the optimizers minimize" also sits uncomfortably next to the ensemble-pruning literature the paper itself cites (Zhou 2002, Ji 2023) and the L0/L1 relaxation-gap literature.

*Evidence checked:* `sections/03_methodology.tex` 3.2; `sections/05_results.tex` 5.7; `sections/06_discussion.tex` 6.6; `sections/00_abstract.tex`; `statistics/cost_definition_sensitivity_r30.csv` (10/24/8/20 verified; pool excludes single-objective references — FINAL_REPORT says including them gives 14/30 and 25/30 on Santander and BNP, which the manuscript does not state); `nsga2_preregistered_config.json`.

*What would satisfy me:* A support-enumeration arm, or NSGA-II rerun with c_sup as f3. Absent that, downgrade the claim in the abstract and Conclusion to match the qualitative statement Section 5.7 defends, and state the comparison pool.

**R1-M7. The indicator machinery that carries every primary endpoint is under-specified, and one main-text table reports an interval that excludes its own point estimate.**
(a) Section 3.6 says approximation sets are scored "on min-max normalized objectives" with "the exact hypervolume with reference point 1.1 on each axis, expressed as a ratio to the reference hypervolume", but never says what defines the min-max box. This is load-bearing: Section 5.7 reports that a previously published finding was an artifact of that box (cost bound ~17 versus ~160–167 ms/1k across partitions of the same dataset), and Delta-HV is averaged across partitions whose boxes differ. Whether the box is at least common across arms within a partition (I believe it is, but the paper does not say) is the difference between a valid paired comparison and an invalid one. The tie tolerance of 1e-4 is stated but its interaction with the HV-ratio scale is not.
(b) Table 8 and supplementary Table S2 print the median paired difference next to the percentile-bootstrap interval of the *mean* while the column header reads "median [95% CI]". On Porto Seguro this produces rows where the point estimate lies outside its own stated interval: Delta-IGD+ "+0.0048 [+0.0051, +0.0139]" (true median interval [+0.0035, +0.0075]) and Delta-HV "+0.0115 [+0.0117, +0.0319]" (true [+0.0070, +0.0169]). Table 3 does this correctly, which is why the defect is a slip rather than a misunderstanding, but it is in the table that carries the paper's only external baseline.

*Evidence checked:* `sections/03_methodology.tex` 3.6; `sections/04_experimental_protocol.tex` 4.4; `build_assets.py` lines 793 and 836 pair `r['median']` with `r.ci95_mean_lo/hi`, whereas line 892 correctly uses `r.ci95_median_lo/hi`; cross-checked against `nsga2/nsga2_paired_effects.csv`. Holm p-values in Table 8 all verified correct.

*What would satisfy me:* (a) A short subsection in 3.6 stating exactly how the box is constructed, whether it is common across arms, whether it changes when a method is added, and how Delta-HV is interpretable across partitions with different boxes. (b) Fix Tables 8 and S2, and re-check whether any sentence in 5.5 was written off the wrong interval.

### Minor comments

1. Section 8 calls Porto Seguro and UCI credit "the two where they largely succeed", but Section 5.1 says Porto "sits on the threshold for both responses" with gate passes of 16/30 and 12/30. Reword the Conclusion.
2. The NBI-C median hypervolume ratios differ between Section 5.4 (0.989, 0.983, 0.982, 0.976) and Table 8 (0.990, 0.981, 0.981, 0.977) because the latter is scored against the NSGA-II-augmented reference. Explained in principle in 5.5 but not at the point of use; add a clause.
3. The wall-clock premium appears as "73, 40, 45 and 3 times" (5.5), "3 to 73 times" (6.5) and "3 to 77 times" (7). Pick one definition.
4. Section 5.6's "+254 ms per 1,000 rows" is the median over the 22/30 partitions in which the gap appears, not an overall estimate; the study-wide mean is +183 ms [142, 221]. Say which quantity is quoted, and reconcile with "250 ms" in Sections 7 and 8.
5. The UCI-credit ensembling gain is 0.000645; reporting it as "0.0007" rounds away 8% of a number used to argue that method differences are two orders of magnitude smaller. Use 0.0006 or three significant figures.
6. The UCI-credit NBI-C-versus-NBI-B significance (Holm p = 0.010) is knife-edge by the authors' own internal analysis (0.017–0.018 with rho = 0.30 or a 16-test family). Report the sensitivity, and justify rho = 0.25 given the paper's own statement that training parts share 75% of their rows.
7. The synergism test is operationalized at the 50/50 blend, but eq. (9) is about the interior optimum at t* = 1/2 + (beta_i − beta_j)/(2 beta_ij), which equals 1/2 only when beta_i = beta_j. The optimal-edge-point version should be the main-text analysis, and the "20 over-predictions, 0 under-predictions" tally recomputed at t*.
8. Section 5.2 reports vertex-quality gaps of "0.163, 0.145, 0.022 and 0.107" while `edge_condition_summary.csv` gives |beta_i − beta_j| as 0.130, 0.110, 0.025, 0.100. Say which is which at first use.
9. The Brier-score control ("R2_ext = 1.000 in all 120 replications") is presented as showing "design and fitting are sound". Brier is exactly quadratic in w, so this is an algebraic identity; phrase it as an implementation check.
10. Das and Dennis (1998) explicitly note that NBI subproblem solutions need not be Pareto optimal, and Messac et al. (2003) introduced a Pareto filter for that reason. The paper reports low non-dominated counts and attributes them to collapse or solver failure without citing the known property. Cite the classical caveat where these results are discussed.
11. Section 2.6's "no prior work contrasts the step cost..." is a strong negative sitting awkwardly beside the ensemble-pruning citations in the same paragraph. Soften to a statement about this specific comparison for weighted soft-voting ensembles.
12. Porto Seguro was subsampled to 200,000 rows by a single stratified draw, so all 30 Porto partitions are conditional on that one subsample. This belongs in Section 7.
13. Section 1 says the design lets the authors "run the same NBI three times per replication, changing only where objectives and anchors come from", which Section 3.4 then correctly contradicts.
14. Section 5.7's "best method changes" counts are computed over six comparison sets excluding the single-objective references; state the comparison pool.
15. The abstract packs eight gate counts, several win fractions and four qualifications into a single block and is very hard to read.
16. Terminology: c_sup is an L0-type cost and c_w its L1-type relaxation. Naming them that way would connect RQ7 to the relaxation-gap literature.

### Things done well

- Numerical discipline is exceptional; roughly thirty-five spot-checked claims all matched, including the awkward ones. The claim-to-evidence map encodes forbidden phrasings and the manuscript obeys them.
- The novelty positioning is honest in a way that is rare: four separate disclaimers, precise naming of the closest prior art, and disclosure of first-author overlap with Pereira 2025 and the dissertation.
- The paper reports the result that most damages it (NSGA-II beating the metamodel-free arm on all four datasets and both endpoints, plus better spacing).
- Pre-specification is real and verifiable: gate thresholds committed before the replicated run, NSGA-II configuration committed as JSON with an explicit no-tuning declaration, realized budget ratio 0.999996.
- The statistical architecture is right for the design: IGD+ over IGD, no pooling, dataset as unit of generalization, Nadeau–Bengio with Holm within dataset, median/win-fraction rule pre-specified.
- Section 5.9 is genuinely useful meta-evidence, and the authors undercut their own lenient criterion.
- The gate's blind spot is a real, well-diagnosed finding, and the authors correctly refuse the tempting reading of the gate-conditional gain.
- Section 7 anticipates and answers many objections honestly.

---

## Reviewer 2 — Machine learning evaluation / experimental validity

**Scores:** novelty 3 · technical 3 · experimental 3 · clarity 4 · **confidence 4** · **Recommendation: MAJOR**

### Summary

The paper takes an established DoE–RSM–NBI pipeline and asks not whether it produces a Pareto front for classifier-ensemble weights but when that front can be trusted. It decomposes the pipeline into three arms that differ only in the source of the objectives and the source of the NBI anchors, adds four controls the lineage lacks, and replicates over 4 public binary tabular datasets x 30 outer partitions with paired, overlap-corrected inference. An evaluation-matched NSGA-II baseline is added and reported as beating the authors' own best arm on all four datasets.

I checked roughly forty quantitative statements against the frozen CSVs and recomputed several analyses from the raw tables. Every number I traced matched, including the ones that hurt the paper's case, and the claims map with its explicit wording rules and blacklist is a model of practice. The findings that survive my checks are the anchor effect on Santander and BNP under the weighted cost, the gate/argmax divergence on BNP, the one-directional synergism-criterion result, and the cost-definition disagreement.

My objections are not that the experiments are wrong; they are that the claims are stronger than the instruments support, and that several decisive analyses are cheap given artifacts the authors already have. Specifically: (i) the Nadeau-Bengio rho = 0.25 is imported from a setting that does not describe this design; (ii) when the corrected test fails, the declared fallback is exactly the independence-assuming inference the correction exists to replace; (iii) the reliability gate is computed on the same 100 compositions used to select the polynomial order, and is reported only as R^2, a statistic the authors' own code documents as ill-posed here; (iv) the front-quality gains are never connected to a deployment decision, and at the one operating point computable from the shipped tables the two headline effects change sign; (v) the NSGA-II comparison is evaluation-matched but not cardinality-matched; (vi) the "non-smoothness is the mechanism" story is contradicted by the paper's own log-loss gate rates. All six are addressable without new experiments. I recommend major revision.

### Major comments

**R2-M1. The Nadeau-Bengio correction is applied with a rho borrowed from a design that is not this one, and every significant result is a knife-edge function of that constant.**
Eq. (NB) sets rho = n_test/n_train = 0.25. But NB's heuristic is derived for a statistic measured on the held-out test portion, where the correlation across resamples arises from training-set overlap. Here the primary endpoints (IGD+ and HV ratio on revalidated objectives) are computed entirely on the out-of-fold predictions of the 80% training part; the 20% holdout enters only Section 5.8. The quantity shared across replications is therefore the *evaluation* data, and two independent stratified 80% draws share 0.64N rows, i.e. 80% of each training part (verified by simulation: mean |A∩B|/|A| = 0.800, Jaccard 0.667 — Section 4.3's "75%" is not right, and errs toward understating the dependence). Applying NB's own ratio with the roles the two parts actually play gives rho ≈ 4, not 0.25. Recomputing the uncorrected two-sided p across rho: at rho = 1, Santander NBI-B vs NBI-A goes to p = 0.076 (IGD+) and 0.054 (HV); Santander NBI-C vs NBI-B to 0.043; UCI credit NBI-C vs NBI-B to 0.085 and 0.078; NSGA-II vs NBI-C to 0.224/0.175 (Santander) and 0.325/0.129 (UCI credit). These are before Holm. At rho = 4 nothing survives except the degenerate comparisons against the random Dirichlet sample. The claims map already records that the UCI result is knife-edge; the manuscript tells the reader none of this.

*Evidence checked:* `sections/04_experimental_protocol.tex` Eq. (NB) and 4.3; `statistics/analysis_config.json`; `statistics/paired_primary_tests.csv`; `nsga2/nsga2_paired_tests.csv`; recomputed p(rho) for rho in {0, 0.25, 0.5, 1, 4}; Monte-Carlo check of the 75% figure; `claims_and_evidence.md` C2.

*What would satisfy me:* (1) State explicitly in Section 4 that the primary endpoints are measured on the 80% training part's out-of-fold objectives and the holdout plays no role in them. (2) Justify rho for this layout, or replace the heuristic with a partition-level / row-block bootstrap or a mixed model. (3) Publish a rho-sensitivity table or curve for all primary tests and rewrite every significance claim to state the range of rho over which it holds. Correct 75% to 80%.

**R2-M2. When the corrected test fails, the paper falls back on precisely the uncorrected inference the correction exists to replace, and does so for claims that appear in the abstract.**
Section 4.4 correctly disclaims the Wilcoxon: "it is not corrected for overlap and cannot rescue a non-significant corrected result." But the declared substitute — "the median with its bootstrap interval, the win fraction with its Wilson interval and the rank-biserial correlation" — has exactly the same defect. The percentile bootstrap resamples the 30 partitions as if exchangeable and independent; the Wilson interval treats the 30 outcomes as independent Bernoulli trials; the rank-biserial has no overlap adjustment. This is not hypothetical: the BNP anchor claim returns Holm-corrected p = 0.66, the paper rests it on 30/30 with Wilson [0.89, 1.00], and that claim appears in the abstract, in Contribution 2, and in the Conclusion. Either the overlap matters — in which case the Wilson interval is too narrow by the same factor that inflated the t-test's standard error by 2.9x — or it does not, in which case the correction was unnecessary. The paper cannot have it both ways within two paragraphs.

*Evidence checked:* `sections/04_experimental_protocol.tex` 4.4; `sections/05_results.tex` 5.3; `statistics/paired_primary_tests.csv` rows bnp/weighted/"nbi_B vs nbi_A"; `claims_and_evidence.md` C1.

*What would satisfy me:* Either derive overlap-aware fallbacks (a bootstrap resampling the underlying rows), or drop the intervals on the fallback statistics and present unanimity as the qualitative statement it is. What is not acceptable is quoting Wilson [0.89, 1.00] as a valid 95% interval three sentences after explaining why such intervals are invalid here.

**R2-M3. The reliability gate is computed on the same 100 compositions used to select the polynomial order, and is reported only through a scale-free statistic that the authors' own code documents as ill-posed.**
In `src/mixens/scheffe.py::compare_orders` the parsimony rule is `best = min(est, key=lambda o: fits[o]["external"]["rmse"])` on the 100 Dirichlet compositions, and the gate then evaluates R^2_ext and Spearman on those same 100 points. There is no split. Two consequences. First, the pass rates in Table 2 are selection-optimistic. Second, and more seriously, it weakens the BNP mechanism the paper treats as a headline contribution: when selection and validation share the data, "the selected order has higher external R^2" is partly tautological. Separately, `external_validation()`'s own docstring states "on flat surfaces external R^2 alone is ill-posed, so the relative RMSE is the primary adequacy measure" — yet the gate is R^2_ext >= 0.5, and neither RMSE nor RMSE/range appears in Table 2, the Results narrative, or the Conclusion. The numbers matter: medians over the selected orders give Santander ROC-AUC external RMSE 0.0087 (validation SD 0.0073, hence R^2 = −0.33); BNP 0.0096 / 0.0103; Porto 0.0038 / 0.0057; UCI credit 0.0015 / 0.0166. So Santander's "complete failure" means predicting held-out ensemble AUC to about ±0.009, and UCI credit's "perfect" surface means ±0.0015 against a total ensembling gain of 0.00065. On three of four datasets the surrogate's external RMSE exceeds the entire signal it is being used to find. I also note the validation compositions are all interior: neither Dirichlet(1) nor Dirichlet(0.5) puts appreciable mass near a vertex, which is exactly where the anchors live and where the failure occurs.

*Evidence checked:* `src/mixens/scheffe.py` lines 168–193 and 245–278; `sections/03_methodology.tex`; `tables/tab02_gate.tex`; grep for "rmse" across `sections/*.tex` (one hit); `tables/scheffe_orders.csv`; `statistics/auc_logloss_conflict_r30.csv`.

*What would satisfy me:* (1) Split the 100 compositions into disjoint selection and validation sets and recompute Table 2. (2) Add external RMSE in native units and the validation-response SD or range to Table 2, with one sentence comparing that RMSE to the ensembling gain. (3) Add a near-vertex stratum (Dirichlet(0.1)) and report a vertex-region RMSE; if that error predicts the NBI-A collapses, Section 6.2 becomes "the gate was validated in the wrong region" — an actionable fix rather than a caution.

**R2-M4. On the one dataset where the paper recommends the inherited machinery, the optimization target itself is not distinguishable from zero on the untouched holdout.**
The Results open by framing everything against the ensembling gain — 0.0037, 0.0056, 0.0060 and 0.0007 — but those are out-of-fold, measured on the same data SLSQP optimized on. Recomputed on the untouched holdout, paired by partition, taking the best single model by out-of-fold AUC as a practitioner would: Santander +0.0076 (30/30), BNP +0.0038 (30/30), Porto +0.0053 (29/30) — all robust — but UCI credit +0.00050 with paired SD 0.00078, 23/30 wins, and a Nadeau-Bengio t of 1.19 at the paper's own rho = 0.25 (p ≈ 0.24). UCI credit is not peripheral: it is the 30/30 gate-pass dataset, it carries the RQ5 counterexample, and it carries Section 6.7's recommendation that "there the inherited machinery is the efficient choice." On that dataset the paper is comparing optimizers on a problem whose entire achievable gain over doing nothing is not distinguishable from zero on held-out data.

*Evidence checked:* `sections/05_results.tex` opening; `statistics/auc_logloss_conflict_r30.csv`; recomputed paired holdout gains from `tables/single_objective_refs.csv` against `tables/model_performance.csv`; `sections/06_discussion.tex` 6.7.

*What would satisfy me:* Report the holdout ensembling gain over the out-of-fold-best single model, paired, with an interval, for all four datasets, next to the out-of-fold figures. Then scope Section 6.7 and the Conclusion: if UCI credit's gain does not survive, "there the inherited machinery is the efficient choice" becomes "there no weighting method separates from the best single model, so the cheapest one is adequate".

**R2-M5. The paper never establishes that its front-quality differences correspond to a better decision, and at the one operating point computable from the shipped tables the two headline effects change sign.**
Every primary claim is an IGD+/HV statement about an approximation set. A practitioner deploys one weight vector. Section 5.8 asks only whether the out-of-fold-best of the four knee picks is also holdout-best *within* each method; it never compares methods at the selected point. Doing so from `mcdm_picks_holdout.csv` (knee rule, weighted cost, paired, holdout ROC-AUC): on Santander — where NBI-B beats NBI-A by +0.179 HV in 30/30 at Holm p = 0.004 — NBI-B's knee pick is *worse* than NBI-A's on the holdout by −0.0061 AUC, winning in only 15/30. On UCI credit — where NBI-C beats NBI-B by +0.277 HV in 30/30 at Holm p = 0.010 — NBI-C's knee pick is worse than NBI-B's by −0.0062 AUC, winning in 6/30. To be fair, these are not contradictions: the picks sit at different trade-offs (NBI-B's Santander knee is 0.96 cost units cheaper; NBI-C's UCI knee is 2.34 cheaper and better in log-loss on Santander), so a single-objective scoreboard is the wrong lens. But that is precisely the problem: nowhere does the paper show that the indicator differences it builds its case on buy the practitioner anything, under any selection rule, on held-out data.

*Evidence checked:* `sections/05_results.tex` 5.8; `statistics/holdout_transfer_r30.csv`; recomputed paired knee-pick differences from `tables/mcdm_picks_holdout.csv` across B−A, C−B, C−A, C−scalarization on holdout ROC-AUC, holdout log-loss and weighted cost; `statistics/paired_primary_tests.csv`.

*What would satisfy me:* A paired, cross-method holdout comparison of the *selected* solutions under at least two rules, reported over all three objectives — a per-partition dominance or attainment statement, or a paired comparison of the achieved triple. If better fronts do not translate into better picks, say so: that is a legitimate and publishable negative result entirely in keeping with this paper's character.

**R2-M6. The NSGA-II comparison is evaluation-matched but not cardinality-matched, and both primary indicators reward cardinality.**
NSGA-II returns 66 mutually non-dominated points in every one of the 120 runs; NBI-C's certified, revalidated, non-dominated front has median size 35 (Santander), 34 (BNP), 56 (Porto), 65 (UCI credit). On the two datasets where the margin is quoted most confidently, the winner carries roughly twice as many points. Hypervolume is monotone in set inclusion and IGD+ is improved by adding points near the reference, so part of the reported +0.006 to +0.013 gain is a set-size effect. Within-dataset, the correlation between the cardinality gap and the HV gain is +0.50, −0.37, +0.57, +0.68. The UCI credit case (65 vs 66, still +0.0121) argues that cardinality is not the whole story, and I credit that — but it does not dispose of the confound on Santander and BNP. The fix is already in the pipeline: `nsga2_pareto_quality.csv` carries `spacing_size_matched_percentile`, so the authors size-match for spacing and not for the two endpoints they draw conclusions from. I raise this not to defend NBI — I think the NSGA-II result is probably real, and reporting it at all is to the authors' great credit — but because the abstract and Conclusion state it as settled.

*Evidence checked:* `nsga2/nsga2_pareto_quality.csv` (n_front per set, recomputed within-dataset Pearson correlations); `nsga2/NSGA2_BASELINE_REPORT.md`; `nsga2/nsga2_paired_tests.csv`; `sections/05_results.tex` 5.6.

*What would satisfy me:* Repeat the comparison with NSGA-II's front subsampled to NBI-C's per-partition cardinality (crowding-based and uniform random, averaged over draws), or report attainment surfaces. If the conclusion holds, one sentence closes the objection.

**R2-M7. The causal mechanism the Discussion and Conclusion build on — smooth response means the surrogate works, rank statistic means it fails — is contradicted by the paper's own log-loss results, and the Brier control cannot carry the argument.**
Section 6.3 opens "Two of the three objectives behave as the framework's engineering origins assume", and the Conclusion reads as a clean contrast. The gate data do not support it. Log-loss surfaces pass in 30/30, 3/30, 12/30, 30/30; ROC-AUC surfaces in 0/30, 5/30, 16/30, 30/30. On BNP Paribas the smooth convex objective does *worse* than the rank statistic (3/30 vs 5/30; median R^2_ext −0.180 vs +0.118), and its external RMSE of 0.121 nats — 17.7% of the response range, the worst relative fit in the study — is a genuine failure, not an R^2 artifact. On Porto the two are effectively tied. The smoothness story survives on Santander alone, where it is admittedly striking. One dataset is an illustration, not a mechanism. The Brier result cannot fill the gap: R^2_ext = 1.000 for a response lying exactly in the model's span is an algebraic identity, as the claims map states, yet the Conclusion promotes it into evidence that "where the response is smooth in the weights, the surrogate is excellent."

*Evidence checked:* `statistics/reliability_gate_r30.csv` (all 16 rows); `tables/scheffe_orders.csv`; `sections/06_discussion.tex` 6.3; `sections/08_conclusion.tex`; `claims_and_evidence.md` C5.

*What would satisfy me:* Demote non-smoothness to one contributing factor, and either identify what actually distinguishes BNP and Porto log-loss failure from Santander and UCI success, or state plainly that the study can identify *that* adequacy varies by dataset and metric but cannot isolate why. Drop the Brier result from the Conclusion's evidentiary sentence.

### Minor comments

1. Section 4.3: "any two training parts share on average 75% of their rows". For two independent stratified 80/20 draws the expected shared fraction is 80% of each training part (Jaccard 2/3); confirmed by simulation (0.8003, 0.6670). The stated 75% appears nowhere in the artifacts and understates the dependence the correction exists to handle.
2. Section 7 states NBI-C costs "3 to 77 times" the wall clock of NBI-B. Table 5, Section 5.5 and Section 6.5 all say 73. The claims map instructs the mean-based 73x. Reconcile.
3. Section 4 opens "The protocol was fixed before the replicated study and was not changed after any result was observed." The NSGA-II arm was added after the R=30 analysis and after a manuscript draft was tagged (`paper-draft-v1-pre-nsga2`). Its own configuration was genuinely pre-committed — to the authors' credit — but the Section 4 sentence should be scoped to the original protocol, with Section 5.6 labelled a post-hoc addition carrying its own pre-specification.
4. Holm families have proliferated without justification: 8 weighted-cost tests per dataset, 8 for the support cost, 6 more per dataset x reference x cost for NSGA-II — 40 tests per dataset in 6 families. State the rationale for each boundary, and report the sensitivity of the two knife-edge results to a single 16-test family.
5. The abstract and Conclusion state the anchor result without the cost qualifier, while Section 5.3 discloses that under the support cost UCI credit reverses and BNP weakens to 19–20/30, and Porto's HV effect essentially vanishes. Under the cost model the paper itself argues deployment pays, the headline claim survives cleanly on one of four datasets. One clause in the abstract would fix the asymmetry.
6. Section 5.1 calls Santander's external rank correlation of 0.830 "respectable". The gate sets the Spearman threshold at 0.9, which that surface fails.
7. Table 5 reports NBI-A subproblem success of 0.39 on UCI credit and 0.58 on Porto. The convergence caveat is applied to the NBI-C vs NBI-B contrast on UCI credit but not to NBI-B vs NBI-A on the same dataset, where NBI-A converges even less. Spearman(dHV_{B−A}, n_certified_A) = +0.50 / −0.64 / +0.30 / −0.33 across the four datasets, so it is not a large confound — but report it symmetrically.
8. Section 5.5 reports size-matched spacing percentiles without stating which direction is better.
9. The most decision-relevant single number — the surrogate's external RMSE in native AUC units — appears only in supplementary Table S3.1. Promote it to Table 2.
10. Section 3.5's epsilon-constraint sweep is described as "40-cap" with no statement of what the cap is on or how it interacts with the 10^5 Dirichlet core.

### Things done well

- The claim-to-evidence map with per-claim wording rules, a forbidden-phrasing blacklist, and an automated audit against 136,940 source values is exceptional practice. Several sentences I set out to break had already been scoped by a rule. I found no invented or untraceable number.
- Every number I traced matched, including the ones that damage the paper.
- Adding an evaluation-matched external baseline that beats the authors' own best arm and reporting it in the abstract is rare. The configuration was pre-committed with a no-tuning declaration; realized budget ratio 0.999996; dual scoring under two reference definitions.
- The self-grading of the empirical reference is handled correctly and repeatedly.
- The R=10 versus R=30 analysis is more self-critical than the field's norm.
- Two earlier findings are retracted in the main text rather than quietly dropped, with the mechanism explained in each case.
- The limitations section is unusually complete and mostly accurate.
- The three-arm decomposition is the right experimental design for the question.

---

## Reviewer 3 — Associate Editor (novelty, positioning and scope)

**Scores:** novelty 2 · technical 4 · experimental 4 · clarity 4 · **confidence 4** · **Recommendation: MAJOR**

### Summary

The manuscript transfers an established DoE–RSM–NBI pipeline from process/forecasting optimization to classifier-ensemble weighting, and asks not whether it produces a Pareto front but when that front can be trusted. The pipeline is explicitly conceded to prior work — Kwon et al. (2024), Rocha et al. (2025) and Bacci et al. (2019), Pereira et al. (2025, EAAI) as the authors' own predecessor. What is offered as new is an evaluation architecture: three NBI arms, an externally validated reliability gate fixed before the benchmark, real-objective revalidation of every candidate, an empirical Pareto reference with a method-independent sampled core, 4 datasets x 30 partitions with paired overlap-corrected inference, a support-vs-weighted cost contrast, and a post-hoc NSGA-II arm at matched real-evaluation budget.

The execution is excellent and the framing is, on the whole, more honest than the norm. My concerns are about whether what remains supports a full paper here, and about three specific places where the follow-through does not match the internal analysis. (i) The headline novel finding — that anchor misplacement dominates interior surrogate error — is overstated relative to the authors' own numbers. (ii) Five references that the authors' own `novelty_matrix.md` mandates as required attributions sit in `references.bib` and are cited nowhere. (iii) The whole study is conducted in the cheap-evaluation regime, which the Discussion itself identifies as the regime in which the surrogate is unnecessary and NSGA-II is preferable; the regime that motivates the framework is never tested.

None of this is fatal. The paper's most durable results are well evidenced and genuinely useful. That is enough for a full paper, not a short communication, but it is not enough in the current 37-page, nine-research-question form, and the positioning needs to be redone in light of NSGA-II and the missing attributions. Recommendation: MAJOR revision.

### Major comments

**R3-M1. Novelty ledger and three missing attributions the authors' own review requires.**
After Kwon et al. (2024), Rocha et al. (2025), Bacci et al. (2019) and Pereira et al. (2025), the methodological delta of this manuscript is a set of controls and an ablation, not a method. The manuscript says so, and it deserves credit for saying so. What is self-serving is the follow-through. The authors' own `novelty_matrix.md` Section (b) lists specific attributions that must be made: W6 states that "running NBI directly on measured objectives, without a metamodel and with real anchors, is published" and must credit `gellerich2023doenbi`; W5 states that payoff-table/anchor unreliability is established and names `isermann1988payoff` and `herrmann2026nonextreme`, concluding that the anchor effect "cannot be presented as surprising"; W4 states that `wang2022committees` already shows that changing the cost accounting reorders which committee looks best, so the cost claim "must be narrowed"; W3 names `maier2026hapens` as already using the term "deployment cost" for post-hoc ensembles. All five bibliography entries exist in `references.bib`. None is cited anywhere in the manuscript. Each omission props up a sentence: Section 2.5's "We found no study that computes a CHIM method's anchors both from a surrogate and from the true objectives" reads as a discovery rather than a controlled instance of a forty-year-old known failure mode; Section 2.6's cost negative is broader than W4 permits; and Contribution 1 is true only because the metamodel-free precedent is not named. The net effect is that the manuscript's claimed novelty is measurably larger than the authors' own novelty analysis concluded it was.

*Evidence checked:* Grepped all `\cite` commands in `sections/*.tex` (89 unique keys) against `references.bib` (180 entries). Confirmed zero citations of `gellerich2023doenbi`, `isermann1988payoff`, `herrmann2026nonextreme`, `wang2022committees`, `maier2026hapens`. Cross-read `novelty_matrix.md` (b) W3–W6 and (a); `research_lineage.md` 2–3; `self_overlap_assessment.md` 1, 2, 4. Figure 1 marks NBI-C as "adapted" rather than "new", so the figure does not overclaim; the problem is in the prose and the missing citations.

*What would satisfy me:* Cite all five at the points the authors' own review specifies, and narrow the three affected sentences. Re-order the contributions so the two findings that are novel without qualification — the gate's blind spot and the edge-criterion disagreement — lead, and the anchor result follows as a quantification.

**R3-M2. The headline anchor finding is overstated against the authors' own numbers.**
Contribution 2, Section 6.1 and the Conclusion all assert that "removing the surrogate entirely, once anchors are correct, adds much less on three of the four datasets". That is not what the frozen statistics say. Paired hypervolume-ratio effects (weighted cost, 30 partitions): NBI-B vs NBI-A gives mean/median +0.179/+0.196 (Santander), +0.279/+0.059 (BNP), +0.085/+0.076 (Porto), +0.060/+0.092 (UCI); NBI-C vs NBI-B gives +0.008/+0.008, +0.009/+0.011, +0.108/+0.074, +0.277/+0.275. So the ordering holds decisively on Santander and BNP; on Porto Seguro the two effects are indistinguishable (medians 0.076 vs 0.074, and by the mean the surrogate-removal effect is larger); on UCI credit it reverses by a factor of three to five. Two clean datasets, one tie and one reversal — not three of four. The paper discloses the UCI reversal and attributes it to NBI-B's subproblem convergence, which is fair and well-evidenced; but that explanation cuts both ways, because it means the B-vs-C contrast is not a clean measurement of interior surrogate error on any dataset, so the quantity being compared against the anchor effect is itself confounded.

*Evidence checked:* `statistics/paired_primary_effects.csv` filtered to cost=weighted, endpoint=hv_ratio; `tab03_paired_primary.tex`; FINAL_REPORT §6; `claims_and_evidence.md` C1/C2 (which record the UCI convergence confound but not the Porto tie).

*What would satisfy me:* Replace "three of the four datasets" with the actual pattern in all four places. State plainly that with four datasets, one tie and one reversal, the ordering is an observation, not a tested claim. Add a paragraph acknowledging that B-vs-C changes the solver acceptance rule as well as the objective source.

**R3-M3. The pipeline is diagnosed in the one regime where nobody should use it, and the NSGA-II result makes that decisive.**
Every real objective evaluation here is a matrix-vector product plus a sort, 0.37–2.13 ms. Section 6.5 states the consequence itself. So the paper tests the trustworthiness of a surrogate pipeline exclusively outside the regime that motivates surrogates, and then transports the conclusions into that regime by assertion. NSGA-II makes this concrete: at a matched evaluation budget, with library-default operators and no tuning, a 2002 algorithm beats the metamodel-free arm on 16 of 16 cells, with better spacing on every dataset, and the paper concedes a practitioner whose evaluations are cheap "should prefer it". Two specific problems follow. First, the cost accounting: Table 5 and Section 7 state that "NBI-A and NBI-B consume no real objective evaluations", and Section 6.1 calls the anchor fix cheap because "the real anchors are one SLSQP call". NBI-B needs two real anchors, and the ROC-AUC anchor is a derivative-free search over 4x10^4 sampled compositions (4.8x10^6 across the study, roughly 10% of NBI-C's entire budget). Charging that to a separate stage does not make it free: that stage costs 444/243/441/54 s per partition against NBI-B's own 35/36/56/100 s. Second, the surrogate's economic case is never demonstrated, because the study's own instrumentation (10^5-point reference, 4x10^4-point AUC anchor, 100-point gate) consumes far more real evaluations than simply running NBI-C would.

*Evidence checked:* Sections 1.2, 3.4, 3.5, 5.5, 5.5b, 6.4, 6.5 and 7; `tables/tab05_compute.tex`; `sections/09_reproducibility.tex`; `nsga2/NSGA2_BASELINE_REPORT.md`.

*What would satisfy me:* (a) Report NBI-B's true real-evaluation cost as a line in Table 5, and remove "NBI-A and NBI-B consume no real objective evaluations" and "costs essentially what NBI-A costs". (b) Add an evaluation-budget frontier: indicator quality as a function of real evaluations consumed. (c) Restate the recommendation conditionally with an explicit budget threshold. (d) State in the abstract that the studied family is dominated by an untuned canonical MOEA at matched budget in the regime tested, and that the framework's case rests on a regime this study does not measure.

**R3-M4. The objective set is close to degenerate, which undercuts the multiobjective premise the paper is built on.**
Section 5.6 reports that the direct ROC-AUC optimum exceeds the SLSQP log-loss optimum by 0.00068, 0.00080, 0.00006 and 0.00034 AUC — 5 to 100 times smaller than the gain from ensembling at all, with bootstrap half-widths below 0.00004. Two of the three objectives are, in objective space, nearly the same objective; on Porto Seguro they practically coincide in weight space too. The paper reports this honestly and correctly refuses the stronger statement. But it never follows through, and the consequences are load-bearing. If f1 and f2 are near-collinear, the effective problem is two-dimensional (quality, cost), the Pareto front is thin, the hypervolume ratio saturates near its ceiling (all NBI-C medians 0.976–0.989), and paired differences of 0.006–0.013 are being computed on a nearly degenerate geometry. That plausibly explains both why every method scores so highly and why only 5 of the 16 primary corrected tests reach significance. It also raises the question the paper should have asked: is the anchor-misplacement result an artifact of a near-degenerate front where a displaced anchor relocates a very thin CHIM?

*Evidence checked:* `sections/05_results.tex` 5.6; FINAL_REPORT §7; `statistics/auc_logloss_conflict_r30.csv`; median indicator table in FINAL_REPORT §6; `paired_primary_effects.csv`.

*What would satisfy me:* A subsection quantifying the degeneracy directly (correlation between f1 and f2 over the reference front, effective dimensionality, fraction of reference-front points separated by more than measurement noise on the AUC axis) and stating what it implies for the indicator differences. Then either add a genuinely conflicting third objective on at least one dataset, or explicitly bound every anchor and comparator conclusion to near-degenerate objective sets in the abstract.

**R3-M5. The inferential fallback is not valid, and one headline table reports intervals that exclude their own point estimates.**
Section 4.4 sets out a careful protocol and correctly forbids the Wilcoxon as a rescue. But the paper then does rescue non-significant results, using win fractions with Wilson intervals and the rank-biserial correlation, which are subject to precisely the same defect. By the corrected test, 5 of the 16 primary NBI comparisons reach Holm-corrected significance; nearly all of the paper's directional claims — the BNP anchor effect (p = 0.66), Porto (0.79), UCI B-vs-A (0.45), the Porto NSGA-II result (0.58) — rest on the uncorrected fallback. Pre-specifying a fallback does not make the fallback valid. Separately, Table 8 labels its interval column "median [95% CI]" but pairs each median with the *mean's* bootstrap interval: for Porto Seguro it prints +0.0048 [+0.0051, +0.0139] and +0.0115 [+0.0117, +0.0319] — intervals that do not contain their own point estimates. The correct median intervals are present in the source CSV; the defect is in `build_assets.py`. The same bug affects supplementary Table S2. Table 3 uses the median intervals correctly, so this is localized — but it survived the authors' own 616-number audit, which is worth noting since that audit is offered as a quality guarantee.

*Evidence checked:* `sections/04_experimental_protocol.tex` 4.4; `statistics/paired_primary_tests.csv` Holm p-values; `nsga2/nsga2_paired_effects.csv` columns `ci95_mean_lo/hi` versus `ci95_median_lo/hi`; `tables/tab08_nsga2.tex` printed values; `build_assets.py` formatting; `tables/tab03_paired_primary.tex`.

*What would satisfy me:* (1) Regenerate Tables 8 and S2 with `ci95_median_lo/hi`, and add a regression check to `audit_numbers.py` that every printed interval contains its printed point estimate. (2) Either correct the win-fraction evidence for partition overlap, or state plainly in 4.4 and again in the Limitations that the win fractions, Wilson intervals and rank-biserial correlations carry the same overlap defect as the Wilcoxon test. (3) Add a table of how many primary tests reach corrected significance (5 of 16).

**R3-M6. Scope for EAAI: all four datasets are financial-risk classification and there is no engineering application.**
I raise this as an editorial question, not a rejection ground. The methodology is squarely in EAAI's tradition and the journal published the predecessor. But the predecessor was an engineering process — hard turning of AISI H13 steel — and this manuscript is an ML methodology evaluation on Santander Customer Transaction, BNP Paribas Cardif Claims, Porto Seguro Safe Driver and UCI default-of-credit-card-clients. All four are financial-services binary classification; there is no engineering problem anywhere in the paper. (The Limitations paragraph says "three of them from Kaggle credit- and insurance-risk competitions", which is literally true about provenance but leaves the impression that the fourth is a different domain; it is not.) The nearest engineering content is the inference-cost objective, which is not developed as such, and the manuscript never argues its own venue fit. Given that the paper's headline generalization is an association over four points from one application domain, this is simultaneously a scope problem and a generalization problem.

*Evidence checked:* `tables/tab01_datasets.tex`; `sections/07_limitations.tex` first paragraph; `sections/02_related_work.tex` 2.6 and `05_results.tex` 5.7; `venue_analysis.md` Part 2.

*What would satisfy me:* Add at least one engineering-domain binary classification problem with 30 partitions — industrial fault detection, quality inspection, condition monitoring or predictive maintenance. Short of that, an explicit paragraph in the introduction arguing engineering relevance, and the Limitations corrected to say all four datasets are financial-risk problems.

**R3-M7. Length and structure: nine research questions across 37 pages, with at least two that are not results.**
The manuscript runs 30 pages of body plus 7 of references, ~12,800 words, 8 figures and 8 tables, organized around nine research questions. The nine-RQ scaffold forces every finding to be reported at the same weight, so the two results that are novel without qualification get no more room than material that is not a contribution at all. RQ9 is meta-commentary on the authors' own project history; their own `novelty_matrix.md` W10 says it "should not be sold as a contribution", yet it occupies a full subsection plus Table 4 plus Figure 6. RQ8 resolves to a null result that is one paragraph of content. The abstract is roughly 440 words, well over the ~250 Elsevier expects. The title is two full lines. My concern is not page count as such but that the paper reads as three papers sharing a compute budget.

*Evidence checked:* `main.log` (37 pages); `wc -w` over `sections/*.tex` (13,347 words of source; 452 in the abstract); section structure of `05_results.tex`; `novelty_matrix.md` W10 and Element 16.

*What would satisfy me:* Cut RQ9 to a paragraph plus a supplementary table, cut RQ8 to a paragraph plus supplement, and reinvest the space in the edge-criterion result and the gate blind spot. Bring the abstract to 250 words and the title to one line. Target 24–26 pages of body. Alternatively split the mixture-design interpretation material into a companion paper for a quality-engineering venue.

### Minor comments

1. The NBI-C/NBI-B wall-clock ratio is stated three different ways. Pick one basis.
2. Section 5.5 calls NBI-C "best or tied-best among the surrogate-derived methods". NBI-C is metamodel-free by construction; this is a category error. Suggest "among the methods in the DoE–RSM–NBI family evaluated here".
3. Section 5.2 reports the "vertex-quality gap" as 0.163, 0.145, 0.022 and 0.107, while Table 7 reports |beta_i − beta_j| as 0.130, 0.110, 0.025 and 0.100. Two different quantities with confusingly similar values and no definition in the prose.
4. The abstract says the cost definition changes the winning method "in up to 24 of 30 partitions"; FINAL_REPORT records BNP as 23–24/30 because one partition's margin of 9e-5 falls inside the tie tolerance. Report the range.
5. Section 2.4 cites six same-group NBI-on-response-surface papers in a single undifferentiated block; none is discussed, and the block contains no work from outside the group. The de Paiva/Balestrassi orbit accounts for roughly 14 of 89 cited works.
6. The author block reads "Co-authors to be confirmed before submission" and the acknowledgements "Funding and institutional acknowledgements to be added". An editor cannot assess the self-overlap relationship until the author list is complete.
7. Section 3.6 and the Limitations correctly distinguish the sampled core from the augmented reference and never say "true Pareto front". Done well; a copy-editor must not smooth the distinction away.
8. Wilson and Jeffreys intervals are applied to frequencies over 30 partitions the paper itself says are not independent. Either drop them or note the assumption violation at the point of use.
9. `rocha2021robustpoint` is flagged in `research_lineage.md` §4 as "Still unread ... it should still be read before submission" and in `self_overlap_assessment.md` §4 item 3. It remains uncited.
10. Figure 1's caption should state explicitly that "new in this study" refers to the evaluation architecture and not to any element of the optimization pipeline.

### Things done well

- The novelty self-assessment is the most rigorous I have seen attached to a submission; `research_lineage.md` retracts the project's original novelty claim and lists verbatim the sentences that must not appear. The manuscript follows that list.
- The NSGA-II baseline is the decision that most damages the paper's standing, and the authors ran it, matched it properly, scored it under two reference definitions, and put the result in the abstract.
- The reliability gate thresholds were fixed in version control before the replicated benchmark, and surrogate optimization is run and reported regardless of the gate outcome, so the gate is a hypothesis under test.
- The edge-condition finding is genuinely novel, well evidenced, replicated on the holdout, and checked against optimal rather than 50/50 weighting.
- Several concerns a reviewer would normally raise are already anticipated and answered in the text.
- The reproducibility apparatus is far beyond what this venue typically sees, and the audit's own stated limitation is candidly noted.
- The paper reports three of its own earlier findings as artifacts it caught and corrected.

---

# PART II — SYNTHESIS EDITOR'S VERIFICATION OF EVERY MAJOR COMMENT

Each verdict states what was checked, what settles it, and where the manuscript already addresses the point.

## R1-M1 — "A versus B is not a clean single-factor contrast" — **PARTIALLY VALID** · **MUST FIX**

**Sub-claim (a), set composition — VALID, and quantified here for the first time.**
`src/mixens/nbi.py::run_nbi` confirms the mechanism: at a vertex beta, `x = anchors.x_star[j_vertex]` is returned directly with `t = 0`. NBI-B's candidate set therefore contains the three real optima verbatim. `references.json` for any replication shows the three anchors are `slsqp_direct_logloss`, `direct_auc_search` and `cheapest_vertex`, and `quality.json` shows `normalization_lo` is **exactly** those three values componentwise — i.e. NBI-B's anchors define the utopia corner of the normalization box and sit on the reference front by construction. The reviewer's `single_objective_refs` figures reproduce exactly (Santander refs 0.971 / A 0.789 / B 0.981; 30/30; median gap coverage 0.930; Porto 0.307, 23/30; BNP −2.57, 9/30; UCI −2.25, 0/30).

I ran the control the reviewer asked for — NBI-A's candidate set augmented with the three real anchors, scored against the same reference and the same box, all 120 partitions (`anchor_control.csv`):

| Dataset | median HV: A | A ∪ anchors | B | median Δ(B−A) | median Δ(B−(A∪anchors)) | residual share | B beats A∪anchors |
|---|---|---|---|---|---|---|---|
| Santander | 0.789 | 0.944 | 0.981 | +0.1960 | +0.0384 | 20% | 30/30 |
| BNP Paribas | 0.913 | 0.964 | 0.971 | +0.0585 | +0.0051 | 9% | 30/30 |
| Porto Seguro | 0.769 | 0.832 | 0.914 | +0.0756 | +0.0215 | 28% | 20/30 |
| UCI credit | 0.622 | 0.623 | 0.706 | +0.0916 | +0.0908 | 99% | 24/30 |

IGD+ behaves the same way (Santander median Δ falls from +0.1251 to +0.0163; UCI from +0.0499 to +0.0498).

**Reading.** The reviewer is right that on Santander — the headline result and the only Holm-significant one — **80% of the hypervolume gap is the three injected extreme solutions, not the relocated CHIM**; on BNP it is 91%. He is wrong that the CHIM effect disappears: the residual is positive in 30/30 Santander and BNP partitions, and on UCI credit the mechanism is essentially *all* CHIM relocation (99%). So the mechanism story survives but is much smaller than the reported numbers imply, and its cross-dataset evidence base is the opposite of what the paper says (see R1-M2).

**Sub-claim (b), simultaneous frame change — VALID.** §3.4's "A versus B ... is a clean contrast" is not defensible: A→B changes the utopia point, both normalization scales (`pseudo_nadir = F_star.max(axis=1)`), the CHIM simplex, all 66 quasi-normal directions, and three returned solutions at once. The paper is scrupulous about exactly this for B-vs-C and asserts the opposite here.

**Sub-claim (b'), "NBI-B is internally inconsistent as a scalarization" — INVALID.** `src/mixens/nbi.py::anchors_from_points` builds `F_star[i, j] = surrogates[i](x_star[j])` — the payoff matrix is the **surrogate** evaluated at the real argmax locations, not real objective values. Both sides of the equality constraint come from the same function. NBI-B is a consistent scalarization of the surrogates with relocated anchors. This sub-claim must not be conceded in the response.

**Sub-claim, payoff-table pseudo-nadir — VALID (minor).** The code confirms the estimator, `deb2010nadir` is cited only to distance the paper, and all three arms inherit the issue.

**Edit that resolves it (no re-run of the frozen experiment):** add the A∪anchors control (10 lines of rescoring over existing `nbi_*_candidates.csv` and `empirical_reference_front_weighted.csv`), rewrite the Santander mechanism sentences in the abstract, §5.3, §6.1 and §8 to say that the effect is dominated by the returned extreme solutions there and is genuinely CHIM-driven on UCI credit; delete "clean contrast" from §3.4 and enumerate what changes; add one sentence on the payoff-table nadir.

## R1-M2 — "Cross-dataset mechanism contradicted by effect magnitudes" — **VALID** · **MUST FIX**

Reproduced exactly. Median ΔHV(B−A) by gate pass rate: Santander +0.196 (ROC-AUC gate 0/30) > UCI credit +0.092 (30/30) > Porto +0.076 (16/30) > BNP +0.059 (5/30). Under the method-independent sample-core reference: +0.145 / +0.089 / +0.078 / +0.070 — the same inversion. The dataset whose surfaces are perfect shows the second-largest anchor effect; the dataset whose surfaces almost never pass shows the smallest. The implied ordering exists only in the BNP **mean** (+0.279), which §4.4's own pre-specified rule discards as uninformative for a bimodal distribution and which §5.3 explicitly declines to use.

The abstract's actual wording ("improve both primary endpoints in every partition of the two datasets whose surfaces fail, and in 23–24 of 30 ... and 24 of 30") is a *consistency* statement and is factually correct. What fails is the magnitude gloss: §7's "real anchors matter most where the surrogate is unreliable" and §6.1's framing.

**Edit:** report the median next to every win count; replace the reliability ordering with what the data support — real anchors help on all four datasets including the one with excellent surfaces; what is confined to unreliable surfaces is the **collapse mode** (BNP, 7/30), not the typical-magnitude benefit.

## R1-M3 — "Compute accounting charges NBI-B nothing for its anchors" — **VALID** · **MUST FIX**

Confirmed at source. `tables/nbi_runs.csv` records `n_real_objective_evals = 0` for variants A and B and ~4.2e5 for C. §3.5 defines the AUC anchor as a derivative-free search over 4x10^4 sampled compositions; §9 records 4.8x10^6 such evaluations over 120 replications, i.e. 4x10^4 per replication = **9.5% of NBI-C's entire real-evaluation budget**, and 400x the 100 evaluations the paper carefully prices for the gate. Table 5's own stage means make the point: single-objective references cost 444 / 243 / 441 / 54 s per partition against NBI-B's 35 / 36 / 56 / 100 s.

The reviewer's arithmetic is slightly off in the authors' favour of his own case: he charges the reference stage to NBI-B but not to NBI-C, which also needs the real anchors. Charging it to both gives a premium of **6.2x / 6.2x / 6.0x / 2.3x**; charging it only to B gives his 5.3 / 5.1 / 5.0 / 2.0. Either way the reported "73, 40, 45 and 3 times" is not the premium a practitioner faces, and §6.1's "the real anchors are one SLSQP call, so NBI-B costs essentially what NBI-A costs" is false — there are three anchors, and the AUC one is not an SLSQP call.

Also confirmed: three mutually inconsistent wall-clock statements in the manuscript (§5.5 "73, 40, 45 and 3"; §6.5 "3 to 73"; §7 "3 to 77") plus "3–84x" in FINAL_REPORT.

**Edit:** a standalone-cost table charging each arm for what it needs (design + fit + gate for A; plus the single-objective reference stage, with the 4x10^4 AUC-anchor evaluations shown explicitly, for B and C), in wall clock *and* real evaluations; delete "0 (anchors reused)" and "NBI-A and NBI-B consume no real objective evaluations"; restate the premium; reconcile to one wall-clock definition. All from `stage_times.csv` and `nbi_runs.csv`, no re-run.

## R1-M4 — "Self-augmented reference is not harmless; one finding does not survive" — **VALID** · **MUST FIX**

Three separate facts verified.

1. **Asymmetric contribution.** NBI-A's revalidated front median size is 13 / 8 / 11 / 11; NBI-B's 20 / 44 / 60 / 27; NBI-C's 35 / 34 / 56 / 65. The Limitations' "without affecting the paired contrasts" holds only under equal contribution, which does not obtain.
2. **Magnitudes move; one finding reverses.** Under the sample-core reference, Santander ΔHV(B−A) falls from +0.183 to +0.145 (−21%). §5.4's "On BNP Paribas the arms genuinely split: NBI-C covers more volume (24/30) while NBI-B is closer to the reference in IGD+ (24/30, Δ = −0.003)" becomes, under the sample-core reference, **NBI-C better in IGD+ in 20/30 with median +0.0025**. §5.9 lists this split as one of the four things the R=30 extension settled. It is a property of the reference definition, not of the arms.
3. **"Search output of no optimizer" is false.** `reference_sample.npz` sources for Santander rep 00: `dir03` 50,000, `dir1` 50,000, `edge` 190, `lattice6` 155, `validation` 100, `design` 66, **`eps_constraint` 40**, **`refs` 7**, `vertex` 5, `centroid` 1. The core therefore contains an ε-constraint optimizer's output *and* the seven single-objective references — two of which are NBI-B's and NBI-C's own anchors. `reference_diagnostics.csv` confirms the ε-sweep contributes a median 38 of ~135 Santander core-front points and `refs` a median 4. The claims-map wording rule C11 permits only "constructed independently of the surrogate and of every candidate method"; §5.5 goes beyond it.

**Edit:** repeat Table 3 and the §5.4 comparisons against the sample-core reference as a full sensitivity table (the scoring script and both references already exist); report per-arm front-contribution share; withdraw or reference-qualify the BNP HV/IGD+ split and correct §5.9; fix the §5.5 and Table S2 caption sentence.

## R1-M5 — "Comparator suite answers the wrong budget question" — **VALID** · **MUST FIX (scoping) / OUT OF SCOPE (the experiments)**

Facts confirmed: population 66 and `n_gen = round(target/66) − 1` gives ~6,350 generations on a 5-variable problem; realized eval ratio 0.999973–1.000000; the Limitations' budget paragraph contemplates only scaling the *cheap* methods up to 4x10^5 and then says "would answer a different, worthwhile question that we do not answer here" — a paragraph now stale, since §5.5b answers exactly that question at the high budget. The low-budget question (the surrogate arms use 66 design + 100 validation evaluations plus anchors) is genuinely unasked and is the one that matters for a surrogate paper. §2.5's framing of the field as model management, followed by a diagnosis of a no-infill one-shot surrogate, is also unscoped, and `messac2003normalized` is cited as the remedy for this exact pathology and never tried.

**Triage split.** The scoping sentence (abstract + §1: the diagnosis applies to one-shot, no-infill DoE–RSM–NBI pipelines, not to model-managed surrogate-assisted MOO), the sentence on why NNC was not evaluated, and the removal of the stale Limitations paragraph are **MUST FIX** and cost nothing. Low-budget NSGA-II / random-search ladders, a ParEGO or SMS-EGO arm, and an NNC arm are new experiments beyond the frozen protocol → **OUT OF SCOPE**, recorded below as reviewer-response material.

## R1-M6 — "Support-cost result is close to a tautology" — **PARTIALLY VALID** · **MUST FIX (claim scope) / OUT OF SCOPE (the arm)**

**The tautology charge itself does not hold.** If the disagreement were definitional it would be near-universal; it is 10/30 on Santander and 8/30 on Porto Seguro, and NBI-C remains the winner under both definitions on those two datasets (26/30, 20/30). The size of the disagreement is an empirical fact about where relaxation-optimal solutions land relative to cheap supports, and §5.7 supplies a mechanism (a linear cost gives no incentive to zero a component). Counts 10 / 24 / 8 / 20 verified against `cost_definition_sensitivity_r30.csv`.

**What does hold.** (i) §5.7 honestly retreats to "we therefore claim only the qualitative statement" after withdrawing the BNP normalization-box artifact, but the abstract, Contribution 5 and §8 do not retreat with it. (ii) The comparison pool is six sets with the single-objective references excluded; FINAL_REPORT records 14/30 and 25/30 including them, and the manuscript never names the pool. (iii) BNP is 23–24/30, not 24/30 (one partition's margin of 9e-5 is inside the tie tolerance). (iv) §6.6's "a step cost is not differentiable and cannot enter a gradient-based formulation" is an argument about NBI, not about the problem — NSGA-II could optimize it unchanged. (v) The §2.6 negative must be narrowed per the authors' own `novelty_matrix.md` W4 (see R3-M1).

**Edit:** align the abstract/Conclusion with §5.7's qualitative claim; name the pool; report 23–24/30; soften §6.6. The 31-support enumeration arm and an NSGA-II run with c_sup as f3 are new experiments → **OUT OF SCOPE**.

## R1-M7 — "Indicator machinery under-specified; Table 8 interval bug" — **VALID** · **MUST FIX**

**(a)** Settled from `scripts/pco213_run_postwork_benchmark.py::stage_quality`: `F_all = vstack([reference sample] + [every candidate set]); m_all = fast_pareto_mask(F_all); F_front = F_all[m_all]; lo = F_front.min(0); hi = F_front.max(0)`. So the box is the min/max of the **non-dominated union of the sampled core with every candidate set**. Consequences the manuscript never states: it *is* common to all arms within a partition (so the paired comparison is valid — the reviewer's worst case does not obtain); it *does* change when a method is added, which is exactly why Table 8's NBI-C medians (0.990 / 0.981 / 0.981 / 0.977) differ from §5.4's (0.989 / 0.983 / 0.982 / 0.976); and it differs between partitions, so ΔHV averaged across partitions is averaged across different scales — the very mechanism §5.7 invokes to withdraw the BNP bimodality finding. §3.6 says none of this.

**(b) Confirmed at source and in the compiled table.** `build_assets.py` `tab08_nsga2()` and its S2 companion format `f"{r['median']:+.4f} [{r.ci95_mean_lo:+.4f}, {r.ci95_mean_hi:+.4f}]"`, while `tab03_main_compact()` correctly uses `ci95_median_lo/hi`. `tab08_nsga2.tex` prints Porto Seguro as `+0.0048 [+0.0051, +0.0139]` and `+0.0115 [+0.0117, +0.0319]`; the true median intervals in `nsga2_paired_effects.csv` are [+0.0035, +0.0075] and [+0.0070, +0.0169]. Santander and UCI rows are also mislabelled (mean intervals under a median header) even where they happen to bracket the median. Holm p-values in Table 8 are correct.

**Edit:** a `\subsection` in §3.6 defining the box and its three consequences; a two-token change in `build_assets.py` plus rebuild; a regression assertion in `audit_numbers.py` that every printed interval contains its point estimate.

## R2-M1 — "Nadeau-Bengio rho imported from the wrong design" — **VALID** · **MUST FIX**

Three components, all verified.

1. **Where the endpoints live.** §4.2: out-of-fold probabilities "are the only quantities the design, the surfaces, the optimizers and the selection rules ever see"; holdout probabilities are "used exclusively for the confirmation analysis of §5.8". So the primary endpoints are measured entirely on the 80% training part. NB's heuristic rho = n_test/n_train is derived for a statistic measured on the *held-out* portion. The analogy is at best unargued and at worst inverted; the reviewer's rho ≈ 4 is itself heuristic, but the direction (rho = 0.25 is too small here) is well motivated.
2. **The 75% figure is wrong.** For two independent stratified 80/20 draws the expected shared fraction is 0.64N = 80% of each training part; Jaccard = 2/3. Neither is 75%, and the stated figure errs toward *understating* the dependence the correction exists to handle.
3. **Sensitivity.** Recomputed p(rho) from the published `t_nadeau_bengio` reproduces the reviewer's numbers to three decimals: at rho = 1, Santander B-vs-A p = 0.076 / 0.054, Santander C-vs-B HV p = 0.043, UCI C-vs-B p = 0.085 / 0.078, NSGA-II vs NBI-C p = 0.224 / 0.175 (Santander) and 0.325 / 0.129 (UCI). At rho = 4 nothing survives except the random-Dirichlet floor checks. All pre-Holm.

**Edit (no re-run):** state in §4.2/§4.3 where the endpoints are measured; correct 75% → 80% (or give the Jaccard); add a rho-sensitivity column or supplementary table for all primary tests; rewrite significance claims to state the rho range over which they hold. Replacing the heuristic with a row-block bootstrap or a mixed model is **NICE TO FIX**.

## R2-M2 — "The declared fallback is the inference the correction replaces" — **VALID** · **MUST FIX**

Verified verbatim. §4.4 forbids the Wilcoxon as a rescue, then names "the median with its bootstrap interval, the win fraction with its Wilson interval and the rank-biserial correlation" as the evidence. All three treat the 30 partitions as exchangeable/independent. §5.3 then quotes "the Wilson interval for a 30/30 win fraction is [0.89, 1.00]" for a comparison whose Holm-corrected p is 0.66, and that claim appears in the abstract, Contribution 2 and the Conclusion. §4.4's closing line "Frequencies over replications are reported with Wilson and Jeffreys 95% intervals" carries no overlap caveat, although §4.3 establishes the overlap two paragraphs earlier. The internal inconsistency is real and self-inflicted.

**Edit (cheapest honest fix):** present unanimity as the qualitative statement it is — "positive in all 30 partitions; no interval is quoted because the partitions are not independent" — or attach the overlap caveat at each point of use. Also report the corrected-significance tally (5 of 16 primary weighted-cost tests; verified).

## R2-M3 — "Gate computed on the selection set; R² ill-posed; no near-vertex validation" — **VALID** · **MUST FIX (disclosure + Table 2) / OUT OF SCOPE (the re-runs)**

Verified at source. `compare_orders(W_design, y_design, W_val, y_val, ...)` selects the order by `fits[o]["external"]["rmse"]` on `W_val`, and the gate is then evaluated on the same `W_val`. No split. `external_validation`'s docstring says in terms: "on flat surfaces external R² alone is ill-posed, so the relative RMSE is the primary adequacy measure" — and the gate is R²_ext ≥ 0.5 with RMSE appearing exactly once in the whole manuscript (the selection rule) and nowhere in Table 2, the Results narrative or the Conclusion.

Medians over the selected order, recomputed from `scheffe_orders.csv`, against the ensembling gain from `auc_logloss_conflict_r30.csv`:

| Dataset | external RMSE (ROC-AUC) | rel. to range | ensembling gain | RMSE > gain? |
|---|---|---|---|---|
| Santander | 0.0087 | 0.202 | 0.0037 | yes |
| BNP Paribas | 0.0096 | 0.176 | 0.0056 | yes |
| Porto Seguro | 0.0037 | 0.130 | 0.0060 | no |
| UCI credit | 0.0015 | 0.021 | 0.00065 | yes |

On three of four datasets the surrogate's external error exceeds the entire signal it is deployed to find. This is a more informative statement of RQ1 than any pass count, and it is currently in a supplementary table.

The validation draw is Dirichlet(1) (60) + Dirichlet(0.5) (40) — both interior-concentrated, so the gate is not validated where the anchors live, which is where the failure the paper diagnoses occurs.

**Triage split.** Disclosing the shared selection/validation set, and promoting external RMSE plus the response range into Table 2 with the ensembling-gain comparison, are **MUST FIX** and use existing data. A 50/50 selection/validation split and a near-vertex Dirichlet(0.1) stratum would change a pre-committed instrument and re-run a frozen stage → **OUT OF SCOPE**; both are the strongest reviewer-response experiments available and should be offered as such.

## R2-M4 — "UCI ensembling gain does not survive to the holdout" — **VALID** · **MUST FIX**

Reproduced exactly, paired by partition, taking the out-of-fold-best single model as the baseline:

| Dataset | OOF gain | Holdout gain (mean) | sd | wins | t_NB (rho=0.25) | p |
|---|---|---|---|---|---|---|
| Santander | +0.00368 | **+0.00759** | 0.00053 | 30/30 | 26.8 | <0.001 |
| BNP Paribas | +0.00563 | **+0.00377** | 0.00061 | 30/30 | 11.5 | <0.001 |
| Porto Seguro | +0.00596 | **+0.00532** | 0.00237 | 29/30 | 4.2 | <0.001 |
| UCI credit | +0.00065 | **+0.00050** | 0.00079 | 23/30 | 1.19 | **0.242** |

UCI credit is the dataset that carries the 30/30 gate pass, the RQ5 counterexample and §6.7's "there the inherited machinery is the efficient choice". On it, the total achievable gain of *any* weighting method over the best single model is not distinguishable from zero on held-out data at the paper's own rho.

**Edit (reanalysis only):** report the holdout gain next to the out-of-fold figures in the §5 framing paragraph, and rescope §6.7 to "there no weighting method separates from the best single model, so the cheapest one is adequate" — a defensible and arguably more interesting conclusion.

## R2-M5 — "Front quality never connected to a deployment decision; two headline effects change sign at the knee" — **VALID** · **MUST FIX**

Reproduced from `mcdm_picks_holdout.csv` (knee rule, weighted cost, paired, holdout):

| Contrast | Dataset | Δ holdout AUC (median) | wins | Δ holdout log-loss | Δ weighted cost |
|---|---|---|---|---|---|
| NBI-B − NBI-A | Santander | **−0.00466** | 15/30 | −0.00207 (14/30) | B is 0.97 cheaper |
| NBI-B − NBI-A | BNP | +0.00294 | 28/30 | +0.00227 (28/30) | ~equal |
| NBI-C − NBI-B | UCI credit | **−0.01189** | 6/30 | −0.00713 (6/30) | C is 2.84 cheaper |
| NBI-C − NBI-B | Santander | +0.00163 | 26/30 | +0.00155 (26/30) | ~equal |

The two contrasts carrying the paper's only Holm-significant results (Santander B-vs-A at p = 0.004; UCI C-vs-B at p = 0.010) have *worse* single-objective holdout AUC at the selected point. The reviewer's own caveat is correct and must be preserved in any response: the picks sit at materially different cost trade-offs, so this is not a contradiction. It is, however, an unanswered question that goes to the paper's own framing ("when can this front be trusted"), and §5.8 answers only the within-method version.

**Edit (reanalysis only):** add a paired cross-method holdout comparison of the selected solutions under at least two rules, reported over all three objectives (dominance/attainment or the achieved triple), and state plainly whether better fronts buy better picks.

## R2-M6 — "NSGA-II evaluation-matched but not cardinality-matched" — **VALID** · **MUST FIX (disclosure) / NICE TO FIX (size-matched repetition)**

Verified: NSGA-II returns exactly 66 mutually non-dominated points in all 120 runs (min = max = 66); NBI-C's median front sizes are 35 / 34 / 56 / 65. Within-dataset Pearson correlation between the cardinality gap and the HV-ratio gap: **+0.50 / −0.37 / +0.57 / +0.68**. HV is monotone in set inclusion and IGD+ improves with points near the reference, so a set-size component is present on Santander and Porto. The UCI case (65 vs 66, still +0.0114) and BNP's negative correlation both argue the effect is not purely cardinality, and the reviewer says so. The pipeline already size-matches for *spacing* (`spacing_size_matched_percentile`) and not for the two endpoints that carry the conclusions.

**Edit:** state the cardinality difference in §5.5b as a bound on the margin (**MUST FIX**, one sentence). Subsampling NSGA-II's front to NBI-C's per-partition cardinality, or attainment surfaces, is a rescoring of existing artifacts (**NICE TO FIX**).

## R2-M7 — "Non-smoothness as the mechanism is contradicted by the paper's own log-loss results" — **VALID** · **MUST FIX**

Verified against `reliability_gate_r30.csv` and `scheffe_orders.csv`. Log-loss gate passes 30 / 3 / 12 / 30 against ROC-AUC 0 / 5 / 16 / 30. On **BNP Paribas the smooth convex objective does worse than the rank statistic** (3/30 vs 5/30; median R²_ext −0.180 vs +0.118) and its external RMSE of 0.1214 nats is 17.7% of the response range — the worst relative external fit anywhere in the study. Porto is effectively tied (12 vs 16). The smoothness story survives on Santander alone (log-loss R²_ext 0.971 vs ROC-AUC −0.333). One dataset is an illustration, not a mechanism.

The Brier control is an algebraic identity — the claims map says so in terms ("algebraic identity — quadratic in w — used only as a sanity check") — yet §8's *Model class* paragraph promotes it into evidence that "where the response is smooth in the weights, the surrogate is excellent". A sanity check on the fitting code cannot be evidence about a mechanism.

**Edit:** demote non-smoothness to one contributing factor in §6.3 and §8; either name what distinguishes BNP/Porto log-loss failure from Santander/UCI success, or state that the study can identify *that* adequacy varies but not why; drop Brier from the Conclusion's evidentiary sentence and keep it in §5.1 as the implementation check §5.1 already calls it.

## R3-M1 — "Five mandated attributions are missing" — **VALID** · **MUST FIX (highest priority)**

Confirmed by exhaustive grep: 89 unique `\cite` keys across `sections/*.tex` and `main.tex`; 180 entries in `references.bib`; **zero occurrences** of `gellerich2023doenbi`, `isermann1988payoff`, `herrmann2026nonextreme`, `wang2022committees`, `maier2026hapens` anywhere outside the bibliography. `novelty_matrix.md` (b) mandates each:

- **W6** — "Running NBI directly on measured objectives, without a metamodel and with real anchors, is published" → must credit `gellerich2023doenbi`. Contribution 1 and §2.5 currently read as discovery.
- **W5** — anchor/ideal/nadir misestimation is established (`isermann1988payoff`, `deb2010nadir`, `he2021normalization`; `herrmann2026nonextreme` treats anchor choice as a lever) → the effect "cannot be presented as surprising". §6.1 presents it as the paper's central mechanism without this framing.
- **W4** — `wang2022committees` already shows cost accounting reorders which committee looks best → §2.6's "no prior work contrasts the step cost..." is broader than permitted.
- **W3** — `maier2026hapens` already uses "deployment cost" for post-hoc ensembles.

This is the single most consequential item for an editor, because the omission is not an oversight the authors could not have known about: their own novelty review names the citations and the sentences they constrain. Fixing it does not destroy the contribution — it converts "we discovered that anchors matter" into "we quantified a known failure mode when the anchor error has surrogate provenance, with a controlled contrast", which is defensible and which the open items in `novelty_matrix.md` (a) 3, 5, 6, 7, 8 survive.

**Edit:** cite all five at the mandated points; narrow §2.5, §2.6 and Contribution 1; re-order contributions so the gate blind spot and the edge-criterion result lead.

## R3-M2 — "'Three of the four datasets' is overstated" — **VALID** · **MUST FIX**

Verified from `paired_primary_effects.csv` (weighted, hv_ratio): B−A mean/median vs C−B mean/median = Santander +0.179/+0.196 vs +0.008/+0.008; BNP +0.279/+0.059 vs +0.009/+0.011; **Porto +0.085/+0.076 vs +0.108/+0.074** (a tie on the median and a reversal on the mean); **UCI +0.060/+0.092 vs +0.277/+0.275** (a 3–5x reversal). Two clean, one tie, one reversal.

Contribution 2 is already the most accurate of the three statements — it carves out UCI credit explicitly — but still misses Porto. §6.1 ("adds much less on three of the four datasets") and §8 ("adds far less on three datasets") are wrong as written. The reviewer's further point is also correct and already half-conceded in §3.4: because B-vs-C changes the solver acceptance rule as well as the objective source, the comparator magnitude is itself confounded, so the *ratio* of the two effects is weaker evidence than either effect alone.

**Edit:** replace the phrase in §6.1, §8, Contribution 2 and the abstract with the actual pattern; add one sentence that with four datasets, one tie and one reversal, the ordering is an observation rather than a tested claim.

## R3-M3 — "Diagnosed only in the cheap regime; anchors unpriced" — **PARTIALLY VALID** · **MUST FIX (the pricing) / OUT OF SCOPE (the frontier)**

**The regime argument — VALID**, and it duplicates R1-M5 from the editorial side: the paper measures a surrogate pipeline exclusively in the regime where §6.5 itself says a surrogate is unnecessary, then transports the conclusion to the expensive regime by assertion.

**The unpriced-anchor argument — VALID**, identical to R1-M3.

**The instrumentation argument — INVALID.** "the study's own instrumentation ... consumes far more real evaluations than simply running NBI-C would" does not hold. Per replication: reference ≈ 1.29x10^5 points + 2.0x10^4 check points, AUC anchor 4x10^4, gate 100, design 66 ≈ **1.9x10^5** against NBI-C's **4.2x10^5** (`summary.json`: 15.5 M reference + 2.4 M check + 4.8 M AUC-search over 120 replications; 50.2 M for NBI-C). The instrumentation is roughly 0.45x NBI-C, not "far more". This sub-claim must be rebutted, not conceded.

**Edit:** price the anchors in Table 5 (see R1-M3); add the "regime this study does not measure" clause to the abstract; remove the stale Limitations paragraph that says the budget-matched question is not answered. The evaluation-budget frontier across arms is a new set of runs → **OUT OF SCOPE**.

## R3-M4 — "The objective set is close to degenerate" — **VALID, and stronger than stated** · **MUST FIX (quantify and bound) / OUT OF SCOPE (a new objective)**

The reviewer argues from the optimum-to-optimum gap. I checked the stronger version directly on the empirical reference fronts (all 120 replications, `empirical_reference_front_weighted.csv`): median Spearman(−AUC, log-loss) over the front is **+0.64 (Santander), +0.83 (BNP), +0.98 (Porto), +0.97 (UCI credit)**, with median Pearson ≈ **+0.99 on all four**. The two "conflicting" objectives are positively associated along the entire front. The effective problem is close to two-dimensional (quality, cost), which is consistent with every NBI-C hypervolume ratio sitting at 0.976–0.990 and with only 5 of 16 corrected tests reaching significance.

The manuscript reports the degeneracy at the optima (§5.6) and correctly refuses the stronger "the problems are the same". It never draws the consequence for the indicator geometry that carries every primary claim.

**Edit (computable from shipped artifacts):** a short subsection quantifying the front-level correlation and effective dimensionality, and a sentence bounding the anchor and comparator conclusions to a near-degenerate objective set. Adding a genuinely conflicting objective (fairness, subgroup calibration, tail metric) is a new experiment → **OUT OF SCOPE**.

## R3-M5 — "Invalid fallback; Table 8 intervals exclude their point estimates" — **VALID** · **MUST FIX**

Duplicates R2-M2 and R1-M7(b); both verified above at source. The additional element is the corrected-significance tally, which I confirm: of the 16 primary weighted-cost NBI tests (2 comparisons x 2 endpoints x 4 datasets), **5 reach Holm-corrected p < 0.05** — Santander B-vs-A IGD+ (0.0074) and HV (0.0037), Santander C-vs-B HV (0.0025), UCI C-vs-B IGD+ (0.0098) and HV (0.0095). The reviewer's observation that the Table 8 defect survived the 616-number audit is fair and worth acting on, because the audit is offered in §9 as a quality guarantee.

**Edit:** two-token fix in `build_assets.py` + rebuild; attach the overlap caveat to the fallback statistics; add the 5-of-16 tally; add the interval-contains-point-estimate assertion to `audit_numbers.py`.

## R3-M6 — "All four datasets are financial risk; no engineering application" — **VALID** · **MUST FIX (framing) / OUT OF SCOPE (a fifth dataset)**

Confirmed from `tab01_datasets.tex`: Santander Customer Transaction (retail banking), BNP Paribas Cardif Claims (insurance), Porto Seguro Safe Driver (auto insurance), UCI default of credit card clients (consumer credit). All four are financial-services binary classification. §7's "three of them from Kaggle credit- and insurance-risk competitions" is literally true about provenance and misleading about domain coverage — the fourth is credit risk too. The paper never argues its own venue fit, and the cross-dataset generalization is an association over four points from one domain with similar feature types and signal-to-noise.

**Edit:** correct the Limitations sentence to say all four are financial-risk problems; add an introduction paragraph arguing engineering relevance (the measured inference-cost objective is the natural hook). A fifth, engineering-domain dataset with 30 partitions is **OUT OF SCOPE** and should be offered as future work / reviewer-response.

## R3-M7 — "Length, nine RQs, two non-results" — **PARTIALLY VALID** · **MUST FIX (abstract/title) / NICE TO FIX (restructure)**

Facts confirmed: 37 pages (`main.log`), 13,172 words of section source including markup, **452 words in the abstract** against Elsevier's ~250, a two-line title, nine RQs, RQ9 occupying §5.9 + Table 4 + Figure 6, RQ8 resolving to a null result.

**The "sold as a contribution" charge is overstated.** `novelty_matrix.md` W10 says RQ9 "should not be sold as a contribution" and the manuscript complies: §5.9 frames it as meta-evidence about how much replication such a study needs, Contribution 4 claims the *evaluation architecture* and the three overturned findings rather than the R=10/R=30 audit, and §5.9 undercuts its own criterion ("itself lenient ... only 6 of 12 hypervolume and 5 of 12 IGD+ cells agree" against the twenty new partitions). The compliance is real; the space allocation is nonetheless disproportionate.

**Edit:** abstract to ~250 words and title to one line are **MUST FIX** (venue requirement). Compressing RQ8/RQ9 and reinvesting in the edge-criterion and gate-blind-spot results is **NICE TO FIX**.

---

# PART III — TRIAGE

## MUST FIX BEFORE SUBMISSION (20 items)

Every one is a wording, attribution, table-generation or reanalysis change. **None requires re-running the frozen benchmark**; the two that need computation (items 3 and 15) run over `experiments/pco213_postwork_benchmark/` artifacts that already exist.

| # | Item | Source | Touches frozen experiment? |
|---|---|---|---|
| 1 | Cite `gellerich2023doenbi`, `isermann1988payoff`, `herrmann2026nonextreme`, `wang2022committees`, `maier2026hapens` at the points `novelty_matrix.md` W3–W6 specify; narrow §2.5, §2.6 and Contribution 1 accordingly | R3-M1 | No |
| 2 | Regenerate Table 8 and supplementary Table S2 with `ci95_median_lo/hi`; add an `audit_numbers.py` assertion that every printed interval contains its point estimate | R1-M7b, R3-M5 | No |
| 3 | Add the NBI-A ∪ real-anchors control and rewrite the Santander anchor mechanism: ~80% of that gap is the three returned extreme solutions, ~20% CHIM relocation; UCI credit is 99% CHIM | R1-M1 | No (rescoring) |
| 4 | Delete "clean contrast" from §3.4; state that A→B changes the utopia point, both normalization scales, the CHIM, all 66 quasi-normals and three returned solutions at once | R1-M1 | No |
| 5 | Replace "three of the four datasets" in §6.1, §8, Contribution 2 and the abstract with: two clean, Porto a tie, UCI a reversal | R3-M2 | No |
| 6 | Report the median effect size next to every win count for the anchor claim; drop or restate "real anchors matter most where the surrogate is unreliable" (§5.3, §6.1, §7, §8) | R1-M2 | No |
| 7 | Charge each arm for everything it needs standalone in Table 5 — including the 4x10^4-evaluation AUC anchor for B and C — restate the premium (~2–6x), delete "NBI-B costs essentially what NBI-A costs" and "NBI-A and NBI-B consume no real objective evaluations" | R1-M3, R3-M3 | No |
| 8 | Reconcile the three wall-clock statements (73/40/45/3 vs "3 to 73" vs "3 to 77") to one stated definition | R1-M3 | No |
| 9 | Repeat Table 3 and the §5.4 comparisons against the sample-core reference as a full sensitivity table; report per-arm front-contribution share | R1-M4 | No |
| 10 | Withdraw or reference-qualify the BNP hypervolume/IGD+ "genuine split"; correct §5.9's list of what the extension settled | R1-M4 | No |
| 11 | Fix "sample-core reference containing the search output of no optimizer" — the core carries 40 ε-constraint points and the 7 single-objective references, two of which are NBI-B/C's own anchors | R1-M4 | No |
| 12 | Add a §3.6 subsection defining the normalization box: min/max of the non-dominated union of core and all candidate sets; common across arms within a partition; changes when a method is added (why Table 8 ≠ §5.4); differs between partitions | R1-M7a | No |
| 13 | State in §4 that primary endpoints are measured on the 80% training part's out-of-fold objectives; correct "75% of their rows" to 80%; publish a rho-sensitivity table for all primary tests | R2-M1 | No |
| 14 | Drop the intervals on the fallback statistics or attach the overlap caveat at each point of use; report that 5 of 16 primary tests reach corrected significance | R2-M2, R3-M5 | No |
| 15 | Disclose that the gate is scored on the same 100 compositions that select the order; add external RMSE in native units and the response range to Table 2, with the comparison to the ensembling gain | R2-M3 | No |
| 16 | Report the holdout ensembling gain for all four datasets next to the out-of-fold figures; rescope §6.7 for UCI credit (+0.00050, 23/30, p ≈ 0.24) | R2-M4 | No (reanalysis) |
| 17 | Add the paired cross-method holdout comparison of the selected knee solutions over all three objectives, and say plainly whether better fronts buy better picks | R2-M5 | No (reanalysis) |
| 18 | State NSGA-II's returned cardinality (66 in 120/120) against NBI-C's front sizes (35/34/56/65) in §5.5b as a bound on the margin | R2-M6 | No |
| 19 | Demote non-smoothness to a contributing factor in §6.3 and §8; drop the Brier identity from the Conclusion's evidentiary sentence | R2-M7 | No |
| 20 | Scope the diagnosis to one-shot, no-infill DoE–RSM–NBI pipelines in the abstract and §1; quantify the AUC/log-loss collinearity on the front and bound the conclusions; align the cost claim in the abstract/§8 with §5.7's qualitative statement, naming the six-set pool and reporting BNP as 23–24/30; correct the Limitations to say all four datasets are financial-risk; fix §8's "the two where they largely succeed"; align §1's "changing only where objectives and anchors come from" with §3.4; scope §4's pre-specification sentence and label §5.5b post hoc; abstract to ~250 words and title to one line | R1-M5, R1-M6, R3-M3/4/6/7, minors | No |

## NICE TO FIX

- Replace the Nadeau-Bengio heuristic with a row-block bootstrap or a partition random-effects model (the rho-sensitivity table is the MUST FIX minimum). *(R2-M1)*
- Size-matched or attainment-surface repetition of the NSGA-II comparison — a rescoring of existing artifacts. *(R2-M6)*
- Move the t*-based synergism test into the main text and recompute the 20/0 tally at t* rather than at 1/2. *(R1 minor 7)*
- Cite Das & Dennis's own caveat that NBI subproblem solutions need not be Pareto optimal, and Messac's Pareto filter, where low non-dominated counts and the spacing result are discussed. *(R1 minor 10)*
- Report Spearman(ΔHV_{B−A}, n_certified_A) symmetrically with the C-vs-B convergence caveat. *(R2 minor 7)*
- Say which direction is better for the size-matched spacing percentile. *(R2 minor 8)*
- Report the UCI C-vs-B Holm sensitivity (0.017–0.018 at rho = 0.30 or a 16-test family) and justify the six Holm family boundaries. *(R1 minor 6, R2 minor 4)*
- Disambiguate "vertex-quality gap" (0.163/0.145/0.022/0.107) from |beta_i − beta_j| (0.130/0.110/0.025/0.100) at first use. *(R1 minor 8, R3 minor 3)*
- Note the Porto single-stratified-subsample caveat in §7. *(R1 minor 12)*
- Rename c_sup / c_w as L0-type / L1-type to connect RQ7 to the relaxation-gap literature. *(R1 minor 16)*
- Trim the six-paper same-group NBI-RSM block in §2.4 and add non-group examples; read and cite (or record as checked) `rocha2021robustpoint`, still flagged unread in `research_lineage.md` §4. *(R3 minors 5, 9)*
- Fix "best or tied-best among the surrogate-derived methods" for the metamodel-free NBI-C. *(R3 minor 2)*
- Complete the author block and acknowledgements; state the Pereira et al. (2025) relationship in the cover letter. *(R3 minor 6)*
- Compress RQ8 and RQ9 to a paragraph each plus supplement; reinvest in the edge-criterion result and the gate blind spot; target 24–26 pages. *(R3-M7)*
- Report the UCI ensembling gain as 0.0006 or three significant figures rather than 0.0007. *(R1 minor 5)*
- Say that "+254 ms" is the median over the 22/30 partitions where the gap appears, and reconcile with "250 ms" elsewhere. *(R1 minor 4)*
- Add the Figure 1 caption clause that "new in this study" means the evaluation architecture, not the pipeline. *(R3 minor 10)*

## OUT OF SCOPE — record as reviewer-response material

These require new experiments beyond the frozen protocol, a fifth dataset, a larger R, or a second evolutionary optimizer. None should be attempted before submission; each should be named in the response letter as the experiment that would settle the point.

1. **NSGA-II and random Dirichlet search at the surrogate arms' real-evaluation budget** (~166), with a ladder at 500 / 2,000 / 10,000, scored identically. *(R1-M5 — the single highest-value future experiment; it is what would tell a reader at which budget the surrogate pipeline stops being worth anything.)*
2. **An infill-based surrogate arm** (ParEGO or SMS-EGO on the same 66-point initial design). *(R1-M5)*
3. **A normalized-normal-constraint arm**, the textbook remedy for the anchor-normalization pathology this paper reports. *(R1-M5)*
4. **A support-enumeration arm** over the 31 non-empty supports, and/or NSGA-II rerun with c_sup as f3. *(R1-M6 — would convert the cost caveat into a result.)*
5. **A 50/50 selection/validation split of the 100 compositions, plus a near-vertex Dirichlet(0.1) validation stratum**, with Table 2 recomputed. *(R2-M3 — cheap, but it changes a pre-committed instrument; offer it as the response experiment. If the vertex-region error predicts the NBI-A collapses, §6.2 upgrades from "surface quality and argmax quality are different properties" to "the gate was validated in the wrong region", which is actionable.)*
6. **An arm changing only the normalization** (real payoff matrix, surrogate optima still returned at the vertices), to separate normalization from anchor substitution. *(R1-M1)*
7. **A fifth, engineering-domain dataset** (fault detection, quality inspection, condition monitoring, predictive maintenance) with 30 partitions. *(R3-M6)*
8. **A genuinely conflicting third objective** (fairness, subgroup calibration, a tail metric) on at least one dataset, to test whether the anchor result survives a non-degenerate front. *(R3-M4)*

---

# PART IV — CLOSING ASSESSMENT

**Recommendations:** Reviewer 1 MAJOR · Reviewer 2 MAJOR · Reviewer 3 (AE) MAJOR.

**Is the paper submittable once the MUST FIX items are addressed? Yes.**

Three things support that verdict. First, no reviewer found an error in the frozen experiment, the statistics pipeline or the reported numbers. I independently re-derived roughly thirty quantities across all three reviews — gate counts, paired medians and win/tie/loss counts, the rho-sensitivity curve, the holdout ensembling gains, the knee-pick contrasts, the cardinality correlations, the front-level objective correlations, the reference-source decomposition, the compute stages — and every figure the manuscript prints was reproducible from the artifacts. The one arithmetic defect found anywhere is a formatting bug in one table generator (mean intervals under a median header), and it is two tokens.

Second, all twenty MUST FIX items are wording, attribution, table-generation or reanalysis work. The two that require computation — the NBI-A ∪ anchors control and the sample-core repetition of Table 3 — run over `experiments/pco213_postwork_benchmark/` and `nsga2/` artifacts that already exist, using scoring code already in the repository. The frozen protocol does not need to be touched, and the 81 hours of stage time do not need to be spent again.

Third — and this is what makes the revision worth doing rather than merely survivable — the two items that change what the paper *claims* both make it more defensible. The A∪anchors control shows that the Santander anchor effect is ~80% set composition and ~20% CHIM relocation, while UCI credit is ~99% CHIM: that is a sharper, better-evidenced mechanism statement than the current one, and it happens to fix R1-M2's inversion at the same time, because the CHIM residual really is largest where the surfaces are best. The sample-core repetition costs the paper one finding (the BNP hypervolume/IGD+ "genuine split", which does not survive) and confirms every other direction under a reference the methods influence far less. A paper whose self-corrections improve it is a paper in good health.

Two cautions for the revision. (i) **R3-M1 is not optional and is not a citation-hygiene matter.** The authors' own `novelty_matrix.md` names five works, the sentences they constrain, and the narrower claims that must replace them; an editor who reads the supplementary material and then the manuscript will see the gap immediately, and the appearance is worse than the substance. It should be the first edit made. (ii) **Do not over-concede.** Three sub-claims across the three reviews are factually wrong and should be rebutted in the response letter, politely and with the evidence: NBI-B is *not* an internally inconsistent scalarization (`anchors_from_points` evaluates the surrogates at the real anchor locations, so both sides of the equality constraint come from the same function); the support-cost result is *not* a tautology (the winner is unchanged in 20/30 and 22/30 partitions on Santander and Porto, which a definitional artifact could not produce); and the study's instrumentation consumes roughly **0.45x**, not "far more than", NBI-C's real-evaluation budget.

After the twenty MUST FIX items the paper is an honest, unusually well-controlled negative-and-diagnostic result about when an inherited optimization pipeline can be believed, with its novelty stated at its true size and its strongest claims scoped to what the four datasets actually show. That is publishable at EAAI. It will not be a high-novelty paper — Reviewer 3's novelty score of 2 is the correct one and will not move — but the technical and experimental scores of 4 are earned, and the revision makes the positioning match them.

---

*Verification artifacts: `anchor_control.csv` (per-partition NBI-A ∪ anchors control, 120 rows) in the session scratchpad. All other computations are re-derivable from `reports/pco213_postwork_benchmark/` and `experiments/pco213_postwork_benchmark/` with the repository's own `src/mixens/pareto_tools.py`.*
