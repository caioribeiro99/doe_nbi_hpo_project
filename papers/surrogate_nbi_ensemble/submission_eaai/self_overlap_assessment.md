# Self-overlap and incrementality assessment

**Question.** Is the present manuscript sufficiently different from the authors' own prior work — in particular from
Pereira, Tertuliano Ribeiro, Mendes, Campos and de Paiva (2025), *Engineering Applications of Artificial Intelligence*
162:112510 — to stand as an independent contribution?

**Answer.** Yes, but the difference is not where the project originally assumed it was. The pipeline is inherited; the
*evaluation architecture and the findings* are new. The manuscript must therefore be framed as a transfer-and-stress-
test study, and it must cite three own-prior-work items prominently rather than one. Framed that way the overlap is
defensible. Framed as "we introduce mixture design + RSM + NBI for ensemble weighting" it would be self-plagiarism of
the idea and would be correctly rejected.

---

## 1. The comparison, dimension by dimension

Predecessor = Pereira et al. (2025). Also relevant, and often *closer* than the predecessor on the dimensions that
matter here: Bacci et al. (2019), *Int. J. Prod. Econ.* 212:186–211, which applies mixture DoE + Scheffé + NBI to
time-series forecast-combination weights; Moreira et al. (2021), *Renew. Sustain. Energy Rev.* 135:110450, which uses
mixture DoE for ANN-ensemble weights; and the first author's UNIFEI master's dissertation (2026) on DoE + RSM + NBI
for XGBoost hyperparameter optimization.

| Dimension | Predecessor (2025 EAAI) | Present manuscript | Overlap |
|---|---|---|---|
| **Research question** | How can post-Pareto selection be improved by modelling MCDM metrics over the weight simplex? | Under which conditions is surrogate-assisted NBI trustworthy when the decision variables are ensemble weights? | **Low.** One asks how to choose on the front; the other asks whether the front is right. |
| **Application domain** | Hard turning of AISI H13 steel (manufacturing process optimization) | Binary classification on four public tabular datasets | **None.** |
| **Decision variables** | 3 machining parameters (cutting speed, feed, depth of cut) in a box domain | 5 ensemble weights on the probability simplex | **None with the predecessor.** *High with Bacci 2019 and Moreira 2021*, where combination weights are also the mixture components. |
| **Mathematical formulation** | Objectives are RSM surfaces in process factors; mixture polynomials model MCDM metrics in the *scalarization* weights | Objectives are Scheffé surfaces in the *decision* variables, which are themselves the mixture | **Low with the predecessor** (the simplex plays a different role). **High with Bacci 2019.** |
| **Experimental design** | 19-run central composite design; Simplex-Lattice over NBI weights | 66-run mixture design over the decision simplex + 100 unseen Dirichlet validation points | **Moderate** (Simplex-Lattice is shared); the reserved external validation set is new. |
| **Response models** | Full quadratic RSM on Varimax-rotated factor scores with FMSE agglutination; Scheffé 4th-order for GD and entropy | Scheffé linear/quadratic/special-cubic fitted directly to ROC-AUC, log-loss and Brier; parsimony order selection | **Moderate** on the polynomial family; **none** on what is modelled. No factor analysis or FMSE is used here, deliberately, because it would hide the individual objectives the anchor analysis needs. |
| **Objectives** | 3 latent factor scores from 8 correlated manufacturing responses (reliability, quality, economic) | ROC-AUC (non-smooth rank statistic), log-loss (convex), inference cost (two definitions) | **None.** The non-smoothness of ROC-AUC is the technical reason the transfer is not trivial. |
| **Optimization algorithm** | NBI on surrogate surfaces; second NBI on Scheffé models of GD/entropy (MBPA) | NBI on the simplex in three variants (A, B, C); MBPA deliberately not used | **Moderate** (Das & Dennis NBI, shared and cited as inherited). The A/B/C decomposition has no counterpart. |
| **Anchor strategy** | Payoff matrix from individual optima of the surrogate surfaces | Surrogate anchors (A) *as the control condition*, real out-of-fold anchors (B), and metamodel-free (C) | **The predecessor's construction is our control arm.** This is the sharpest scientific difference: what the predecessor does is what we show can fail. |
| **Validation** | Comparison with other optimizers and benchmark functions; RF monitoring; no visible confirmation runs | Every candidate revalidated on real out-of-fold objectives; empirical Pareto reference with convergence check; untouched-holdout confirmation; external reliability gate | **None.** |
| **Datasets / problems** | One case study | 4 datasets × 30 partitions = 120 replications | **None.** |
| **Statistical analysis** | None beyond descriptive comparison | Paired by partition; percentile bootstrap; Wilson/Jeffreys; Nadeau–Bengio corrected tests with Holm within dataset; rank-biserial effect sizes | **None.** |
| **Empirical findings** | A selected optimum for the turning process; MBPA yields interpolated weights | Real anchors repair surrogate NBI; the gate detects bad surfaces but not bad anchors; β_ij marks weak vertices, not synergy; cost definition changes the winner; R10→R30 stability | **None.** |
| **Claimed contribution** | A hybrid multivariate NBI with mixture-design post-optimization | A validation architecture and a set of conditions under which the inherited pipeline is and is not trustworthy | **Low, if framed correctly.** |

**Net assessment.** Two of thirteen dimensions overlap substantially with the predecessor (the polynomial family and
the NBI core), both of which are explicitly labelled inherited. The dimension where overlap is genuinely *high* is
with Bacci et al. (2019), not with the predecessor — and it is the one the original project framing would have claimed
as its novelty.

---

## 2. Wording and methodology that could read as derivative

| Risk | Where | Mitigation |
|---|---|---|
| "We propose to treat ensemble weights as mixture components" | Introduction | Delete. Attribute to Bacci et al. (2019) and Moreira et al. (2021); state that the *correspondence* between simplex constraints and mixture experiments is standard and that we adopt it. |
| "We introduce a DoE–RSM–NBI framework for …" | Abstract, introduction, contributions | Delete everywhere. Replace with "building on our prior DoE–RSM–NBI framework". |
| Scheffé canonical polynomial exposition | Methodology §3.3 | Standard textbook material (Scheffé 1958, 1963; Cornell 2002). Present as background with citations, written afresh; do not reuse the predecessor's phrasing. |
| NBI exposition (payoff matrix, CHIM, quasi-normal) | Methodology §3.4 | Cite Das & Dennis (1998) directly, not the predecessor's rendering of it. Our added detail (free-variable parameterization, projection, feasible-iterate acceptance) is specific to the simplex and the non-smooth objective, and is described as such. |
| Simplex-Lattice design over weights | Methodology §3.3 | Cite Scheffé and the group precedents; our 66-run design differs in composition and is described in full, with the reserved validation set flagged as new. |
| Reuse of code | Reproducibility | The repository is the dissertation's; state this. The mathematical modules (simplex utilities, Scheffé model, NBI core) are reused with attribution and extended; the factor-analysis and post-optimization modules are not used. |
| Figures | All | No figure is reused from any prior publication. All eight are generated by `build_assets.py` from this study's artifacts. |
| Prose | All | No prose is copied from the predecessor or the dissertation. The predecessor's full text was not accessible in any case; only its abstract and first page were read. |

---

## 3. How to cite the prior work

**In the abstract** (one clause, early):

> Building on a DoE–RSM–NBI framework previously developed for multiobjective engineering optimization, we transfer it
> to classifier ensemble weighting and evaluate when it can be trusted.

**In the introduction** (a short paragraph, before the contributions):

> The combination of mixture designs, response-surface metamodels and Normal Boundary Intersection is not new, and we
> do not claim it. It has been used to weight portfolio components [Mendes 2016; Leal 2022], to weight the members of
> a time-series forecast combination, where a simplex-lattice over the combination weights feeds Scheffé models that
> are then optimized by NBI [Bacci 2019], and to weight neural-network ensembles [Moreira 2021]; our own recent work
> extends the framework with a mixture-design post-Pareto stage [Pereira 2025]. What has not been established is
> whether the pipeline remains trustworthy when the responses are classifier performance metrics — one of which is a
> non-smooth rank statistic — and when the decision variables are ensemble weights whose deployment cost is a step
> function. That is the question of this paper.

**In related work** (§2.4), a dedicated subsection on the framework's origins, with an explicit statement that the
present study does *not* claim the combination as new and that NBI-A reproduces the predecessor's anchor construction
as a control condition.

**In the cover letter**, first paragraph: name the predecessor, state the shared framework, and state the four things
this manuscript adds that it does not contain (compositional decision variables with non-smooth ML objectives; the
anchor decomposition; revalidation against an independent empirical reference; 120 replications with corrected paired
inference).

---

## 4. Residual risks

1. **Same journal.** Submitting to EAAI, where the predecessor appeared, puts the two in front of overlapping
   reviewers. This is a manageable risk and arguably an advantage — a critical follow-up in the venue of the original
   is a recognizable pattern — but it makes the framing non-negotiable: any residual "we introduce" phrasing will be
   caught.

2. **Author overlap is partial.** The present first author is the second author of the predecessor. The contribution
   statement should make the relationship explicit rather than leaving it to be inferred from the author lists.

3. **Resolved: the previously unread precedent does not pre-empt us.** de Paula, Gomes, Gomes and Paiva (2019),
   *A Mixture Design of Experiments Approach for Genetic Algorithm Tuning Applied to Multi-objective Optimization*
   (Springer AISC, pp. 600–610, DOI 10.1007/978-3-030-21803-4_60), has now been read. Its mixture components are the
   **weights of the objective functions**, crossed with three genetic-algorithm hyperparameters as process variables.
   It is a mixture design over scalarization weights, not over model-combination weights, and does not pre-empt this
   work. Rocha et al. (2020), *Engineering with Computers*, DOI 10.1007/s00366-020-00973-5, remains unread; it concerns
   post-Pareto point selection, which this study deliberately does not address, so the residual risk is low. Read it
   before submission.

4. **A closer, non-own precedent appeared in the novelty review, and it changes the framing.** Rocha, Rotella,
   Balestrassi, Melgani and Zambroni de Souza (2025), *IEEE Access* 13:207903–207915, apply a {3,5} simplex-lattice
   mixture design over the probability simplex of a three-network ensemble, reduce correlated error metrics by factor
   analysis, run NBI on the resulting surrogate objectives and select by entropy — the present pipeline, applied to a
   regression ensemble, published before this manuscript. Balestrassi is a UNIFEI collaborator of the predecessor's
   senior author, so this is adjacent-group work, but it is a separate author team and it is prior art. Consequences:
   (i) the contribution list must not contain any variant of "we cast ensemble weighting as a mixture-design problem",
   which is that paper's own stated contribution; (ii) it must be cited in the introduction, in related work and
   alongside the contributions; (iii) the self-overlap question is now partly an *adjacent-group* overlap question, and
   the manuscript is safer for it, because what distinguishes this study from Rocha et al. (2025) — non-smooth
   classifier objectives, the anchor decomposition, revalidation, the empirical reference, the gate, replication, and
   the deployment-cost contrast — is the same list that distinguishes it from the authors' own prior work.

4. **Dissertation overlap.** The master's dissertation is open access in the UNIFEI repository and shares the
   repository, the objective triple and part of the code. It optimizes XGBoost hyperparameters in a box domain rather
   than ensemble weights on a simplex, so the scientific overlap is small, but it must be cited and the shared code
   base disclosed in the reproducibility statement, which it is.

5. **The closest prior art is external, which strengthens the incrementality case rather than weakening it.**
   Kwon, Lee and Lee (2024), *J. Soc. Korea Ind. Syst. Eng.* 47(4):161–170, already fit a Scheffé polynomial to the
   accuracy of a five-classifier ensemble over the weight simplex, obtained from a mixture design, and maximize it
   under the sum-to-one constraint on eleven tabular binary datasets. That is elements 1–3 of this pipeline, applied to
   classifiers, by an unrelated group. It removes any remaining temptation to present the mixture formulation as a
   contribution — but it also means the self-overlap question is no longer the binding one: what separates this study
   from the authors' own prior work (multiobjective treatment, the anchor decomposition, revalidation, the empirical
   reference, the gate, replication, the deployment-cost contrast) is exactly what separates it from Kwon et al. and
   from Rocha et al. (2025). A single set of contributions answers all three comparisons, which is the strongest
   position the manuscript could be in.

**Verdict: sufficiently different, conditional on the framing above.** The overlap that matters is no longer
with the 2025 EAAI predecessor but with Rocha et al. (2025); the manuscript's distinguishing contributions separate it
from both, and they are the same contributions in each case.
