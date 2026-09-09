# Research lineage: what this study inherits, adapts and adds

This document reconstructs the methodological lineage of the present study so that the manuscript can state precisely
what is inherited from the authors' own prior work, what is adapted when the decision variables become classifier
ensemble weights, and what is genuinely new. It is the reference for the Related Work section and for
`self_overlap_assessment.md`. Bibliographic records were verified against Crossref, OpenAlex, Semantic Scholar, the
publisher first-page PDFs and the UNIFEI institutional repository; the verification trail is in
`../../../scratchpad/lit/predecessor_lineage.md` (session artifact) and the entries are in `references.bib`.

**Headline correction to the framing of this project.** The combination *mixture design of experiments over
combination weights + polynomial surrogate of performance over the weight simplex + Normal Boundary Intersection on
those surrogates* is **not new and is not ours to claim**. It is established methodology in the authors' own research
group, published at least since Bacci et al. (2019) for time-series forecast combination, and applied to neural-network
ensemble weights by Moreira et al. (2021). The 2025 EAAI paper of which the present first author is a co-author extends
the same machinery to post-Pareto decision making. The present study is a **transfer and stress test** of that
framework in a different problem class, plus a set of controls and validation layers that the lineage does not contain.

---

## 1. The four stages of the lineage

### Stage 1 — Group methodology: mixture designs over combination weights (2016–2022)

| Work | What it does | Weights are | Objectives | Optimizer |
|---|---|---|---|---|
| Mendes et al. (2016), *Comput. Oper. Res.* 66:434–444 | portfolio proportions modelled by mixture DoE, ARMA–GARCH returns | asset proportions on the simplex | return, risk, entropy | desirability |
| **Bacci et al. (2019), *Int. J. Prod. Econ.* 212:186–211** | **Simplex-Lattice over forecast-combination weights → PCFA factor scores → Scheffé mixture models → NBI on those models → entropy/GPE selection** | **forecast-combination weights on the simplex** | **rotated factor scores of residual accuracy metrics** | **NBI on mixture surrogates** |
| Moreira et al. (2021), *Renew. Sustain. Energy Rev.* 135:110450 | DoE + clustering select ANNs, then "a mixture (MDE) is employed to determine the ideal weights for the ensemble formation" | ANN ensemble weights on the simplex | forecast error (MAPE), single objective | mixture-DoE optimization |
| Leal et al. (2022), *Production* 32:e20210119 | Simplex-Lattice {5,10} reduced by D-optimal selection to 200 runs; FA/FMSE; desirability | energy-asset portfolio weights | return, variance, entropy | desirability |

Bacci et al. (2019) is the structural precedent of the present pipeline and Moreira et al. (2021) is the precedent for
applying it to a *model ensemble*. Any claim of the form "we are the first to treat combination weights as mixture
components and fit a Scheffé model over them" is false and must not appear.

### Stage 2 — The 2025 EAAI predecessor (own prior work)

> Pereira, M. C., Tertuliano Ribeiro, C., Mendes, R. R. A., Campos, P. H. S., de Paiva, A. P. (2025).
> A hybrid multivariate normal boundary intersection approach with post-optimization assisted by mixture design of
> experiments. *Engineering Applications of Artificial Intelligence*, **162**, 112510.
> DOI [10.1016/j.engappai.2025.112510](https://doi.org/10.1016/j.engappai.2025.112510). ISSN 0952-1976.
> Received 24 Sep 2024, revised 4 Jun 2025, accepted 23 Sep 2025, online 4 Oct 2025, issue Dec 2025.
> Keywords: Multiobjective optimization; Normal boundary intersection; Post-pareto optimization; Multi-criteria
> decision-making; Mixture design of experiments; Random forest.

What it does, from the abstract, the first-page PDF and the rendered section outline (full text paywalled; see the
"could not be seen" list below):

- **Domain:** hard turning of AISI H13 tempered steel bars with PCBN 7025 inserts. A 19-run central composite design
  acquires eight correlated responses grouped into Reliability (tool life, mean time to failure, wear rate), Quality
  (Ra, Rt) and Economic (process cost, ROI, overall equipment effectiveness) dimensions.
- **Decision variables:** three continuous process factors — cutting speed, feed rate, depth of cut. A box domain, not
  a simplex.
- **Objectives:** three latent surfaces (VRF1–VRF3), full quadratic response surfaces fitted to Varimax-rotated
  principal-component factor scores, with a Factor Mean Square Error agglutination so each factor is minimized
  regardless of the sign of its loadings. The objectives are entirely surrogate.
- **NBI:** standard Das & Dennis construction on the three surrogate surfaces. Individual optima of the surfaces form
  the payoff matrix whose diagonal is the utopia point — **anchors are surrogate optima**. The NBI weight vectors
  themselves are scheduled by a Simplex-Lattice {q, m} design.
- **Mixture-design post-optimization ("Mixture Based Performance Assessment"):** during the NBI sweep, post-Pareto
  metrics of each solution — a Khuri–Conlon generalized distance and the Shannon entropy of the weights — are recorded
  and modelled by **Scheffé fourth-order polynomials in the scalarization weights**; NBI is then re-applied to those
  polynomials to obtain interpolated, "non-assignable" weight vectors and select the final solution.
- **Validation:** comparison against other optimizers and on benchmark functions; a Random Forest online
  quality-monitoring implementation. Physical confirmation runs at the selected point are not visible in the accessible
  text.
- **Replication:** a single 19-run design, one case study, one selected optimum. No resampling replication.

Not visible (paywalled): full Sections 2 and 4, the Simplex-Lattice degree and the number of NBI subproblems, Scheffé
fit statistics, the identity of the comparator methods and benchmark functions, the Random Forest set-up, any
confirmation-run table, the Highlights block and the reference list.

**Crucially, in the predecessor the mixture/Scheffé machinery operates on the *scalarization-weight* simplex for
post-Pareto selection, while the objectives themselves are RSM surfaces in the process factors.** In the present study
the mixture/Scheffé machinery operates on the *decision-variable* simplex, because the decision variables are
themselves a composition. That is a genuine structural difference from the predecessor — but not from Bacci et al.
(2019), where the combination weights are likewise the mixture components.

### Stage 3 — The first author's master's dissertation (own prior work)

> Ribeiro, C. T. (2026). *Meta-otimização dos hiperparâmetros do algoritmo XGBoost para classificação binária: uma
> integração entre planejamento de experimentos e o método da interseção normal à fronteira.* Master's dissertation,
> 153 pp., Programa de Pós-Graduação em Engenharia de Produção, Universidade Federal de Itajubá. Advisor: Anderson
> Paulo de Paiva. Defended February 2026; open access,
> handle [123456789/4372](https://repositorio.unifei.edu.br/jspui/handle/123456789/4372). No DOI; not indexed in
> Crossref/OpenAlex/Semantic Scholar.

DoE + factor analysis + response surfaces + NBI applied to **XGBoost hyperparameter** optimization for binary
classification. It supplies the repository, the runner conventions and part of the mathematical code base used here,
and it is the reason the objectives of the present study are ROC-AUC, log-loss and a cost term. Its decision variables
are hyperparameters in a box domain, not a composition, and it optimizes one model rather than weighting an ensemble.

### Stage 4 — The present study

Classifier ensemble weighting: `p(w) = P w` with `w` on the probability simplex over five heterogeneous base
classifiers, evaluated on cached out-of-fold probabilities; a 66-run mixture design; Scheffé linear/quadratic/special-
cubic surfaces with external validation on 100 unseen Dirichlet compositions and a pre-registered reliability gate;
three NBI variants that separate surrogate objectives with surrogate anchors (A), surrogate objectives with real
anchors (B) and metamodel-free NBI on the real out-of-fold objectives (C); real-objective revalidation of every
candidate against an empirical Pareto reference built independently of the surrogate; ROC-AUC, log-loss and inference
cost as objectives, with the continuous weighted cost contrasted against a support-based deployment cost; four datasets
× 30 outer partitions with paired, overlap-corrected statistics and untouched-holdout confirmation.

---

## 2. Component-by-component classification

Legend: **INHERITED** — used essentially as in the lineage; **ADAPTED** — the lineage supplies the idea, but the
compositional decision variables or the machine-learning objectives force a substantive change; **NEW** — no
counterpart in the lineage.

| # | Component | Class | Lineage source | What changes here |
|---|---|---|---|---|
| 1 | Mixture formulation of combination weights (non-negativity + sum-to-one; simplex as the experimental region) | **INHERITED** | Bacci 2019; Moreira 2021; Mendes 2016; Leal 2022 | Components are classifiers rather than forecasts, assets or ANNs. The formulation itself is unchanged. |
| 2 | Mixture design over the weights | **ADAPTED** | Simplex-Lattice in Bacci 2019 and the predecessor; D-optimal reduction in Leal 2022 | A fixed 66-run design combining the {5,3} and {5,2} lattices, the overall centroid, 5 axial points, 5 quaternary centroids and 10 centroid–ternary midpoints, chosen so the pure vertices (single models) and the uniform blend are design points, and augmented by 100 *unseen* Dirichlet compositions reserved for external validation. The lineage designs do not reserve a validation set. |
| 3 | Scheffé canonical polynomials as the performance surrogate | **ADAPTED** | Bacci 2019 (factor scores); predecessor (post-Pareto metrics) | Fitted directly to ROC-AUC, log-loss and Brier rather than to rotated factor scores, and the order is selected by an explicit parsimony rule (lowest order within 10% of the best external RMSE) rather than fixed a priori. No factor analysis or FMSE agglutination is used: the three objectives are kept explicit and interpretable, which is what makes the anchor analysis possible. |
| 4 | External validation of the surrogate + reliability gate | **NEW** | — | 100 held-out Dirichlet points per replication; a pre-registered gate (external R² ≥ 0.5 **and** Spearman ρ ≥ 0.9) decides whether a surface is usable. The lineage reports in-sample fit statistics; it does not validate the mixture surrogate on unseen compositions, and it has no admissibility criterion. |
| 5 | NBI construction (payoff matrix, CHIM, quasi-normal, β lattice) | **INHERITED** | Das & Dennis 1998 via the predecessor, Bacci 2019, Azevedo 2026, Pereira 2026 | Unchanged mathematics on M−1 free variables, with a projection step and a feasible-iterate acceptance rule added for the non-smooth case (see 7). |
| 6 | Anchors from surrogate optima (NBI-A) | **INHERITED** | The predecessor's payoff matrix; Bacci 2019 | This is exactly the lineage's construction; here it is the *control condition* rather than the method. |
| 7 | Anchors from real objectives (NBI-B) and metamodel-free NBI (NBI-C) | **NEW** | — | The lineage always anchors on, and always optimizes, the surrogate. Running the identical NBI with anchors recomputed from real out-of-fold single-objective optima, and again with the real objectives replacing the surrogate entirely, is what turns the pipeline into a controlled experiment about surrogate fidelity. NBI-C additionally requires handling a piecewise-constant ROC-AUC: finite-difference steps, multistart from the simplex, and acceptance of a feasible iterate when SLSQP cannot certify optimality. |
| 8 | Real-objective revalidation of every candidate | **NEW** | — | Every point returned by any method is re-evaluated on the exact out-of-fold objectives before any indicator is computed. The lineage evaluates candidates on the surrogate and confirms at most the single selected solution (and, in the manufacturing papers, by physical experiment). |
| 9 | Empirical Pareto reference independent of the surrogate | **NEW** | — | ≥ 100,000 Dirichlet(1) and Dirichlet(0.3) samples plus lattice points and an ε-constraint sweep, with an independent 20,000-point displacement check and up to three enlargement rounds. The lineage has no reference front: front quality is judged by comparing optimizers to each other (Pereira 2026 compares NBI against NSGA-II, MOEA/D, weighted sum and MOLA using hypervolume, IGD and spacing). |
| 10 | Pareto quality indicators (GD, IGD, IGD⁺, spacing, hypervolume, joint non-dominated fraction) | **ADAPTED** | Pereira 2026 uses HV, IGD and spacing to rank optimizers | Here the same family of indicators scores *revalidated* candidate sets against the empirical reference of the same partition, so an indicator measures approximation error rather than a head-to-head ranking. IGD⁺ and the hypervolume ratio are pre-registered as the primary endpoints. |
| 11 | Post-Pareto selection (entropy, generalized distance, MBPA) | **NOT USED** | Predecessor §2.6; Bacci 2019; Rocha 2017 | Deliberately excluded. The present study asks whether the *front* is trustworthy, not how to pick a point on it; a knee rule and TOPSIS appear only in the holdout-transfer analysis as fixed selection rules. This is a scope boundary, not an omission to hide. |
| 12 | Objectives: predictive quality and calibration | **ADAPTED** | The dissertation optimizes AUC-type and loss-type criteria for XGBoost | Here ROC-AUC (rank-based, piecewise constant in `w`) and log-loss (convex in `w`) are optimized jointly over a composition, which is what exposes the misspecification of a low-order polynomial on some datasets. |
| 13 | Cost objective | **ADAPTED** | Predecessor's economic dimension (process cost, ROI, OEE); Bacci/Leal use entropy as a diversification term | Cost here is *inference* cost, measured per model as median wall-clock milliseconds per 1,000 predictions, and it enters in two mathematically different forms: the continuous relaxation Σᵢ wᵢ cᵢ that the optimizers see, and the support cost Σᵢ cᵢ·1[wᵢ > ε] that deployment actually pays. |
| 14 | Support-based deployment cost contrasted with the weighted relaxation | **NEW** | — | No counterpart in the lineage, where cost is a smooth physical or economic response. |
| 15 | Replication over resampled partitions | **NEW** | — | The lineage uses a single design and a single case study. Here the entire pipeline is re-run on 30 outer stratified 80/20 partitions per dataset with recorded seeds, and every comparison is paired by partition and tested with a Nadeau–Bengio correction for the overlap between partitions. |
| 16 | Untouched-holdout confirmation | **NEW** | — | 20% of each partition is never used for fitting, weighting, thresholding or selection, and the out-of-fold-selected solutions are confirmed on it. |
| 17 | Failure decomposition (surrogate error vs anchor misplacement vs CHIM geometry vs solver failure) | **NEW** | — | Made possible by 4, 7, 8 and 15 together. |
| 18 | Interpretation of β_ij | **ADAPTED** | Standard mixture-DoE reading: β_ij > 0 is "synergistic blending" | Tested here against the real 50/50 blend of the pair, and found to track the *vertex-quality gap* rather than exploitable complementarity on three of four datasets. |
| 19 | Software base (simplex utilities, Scheffé model, NBI core) | **INHERITED** | The dissertation's repository (`doe_nbi_hpo_project`) | Reused with attribution and extended; the factor-analysis and post-optimization modules of the dissertation are not used. |

Summary counts: **INHERITED 4**, **ADAPTED 7**, **NEW 7**, plus one component deliberately excluded.

---

## 3. How the manuscript must phrase the lineage

Acceptable, and supported by the record:

> Mixture designs of experiments have been used in our research group to weight the components of a combination:
> for portfolio proportions [Mendes 2016; Leal 2022], for time-series forecast combination, where a Simplex-Lattice
> over the combination weights feeds Scheffé models that are then optimized by Normal Boundary Intersection
> [Bacci 2019], and for neural-network ensemble weights [Moreira 2021]. The same DoE–RSM–NBI framework was recently
> extended with a mixture-design post-Pareto stage for multiobjective engineering optimization [Pereira 2025]. The
> present study transfers that framework to classifier ensemble weighting and asks a different question: not whether
> the pipeline produces a front, but under which conditions the front it produces can be trusted.

Not acceptable:

- "We propose the use of mixture designs to optimize ensemble weights." (Moreira 2021, Bacci 2019.)
- "We introduce the combination of mixture design, response surfaces and NBI." (Bacci 2019; Pereira 2025.)
- "We are the first to model ensemble performance over the simplex with Scheffé polynomials." (Bacci 2019 for
  forecasts; Moreira 2021 for ANN ensembles.)
- Any phrasing that presents the predecessor as an external competitor rather than as own prior work.

The defensible statement of what is new is the *evaluation architecture*, not the pipeline: real-versus-surrogate
anchors as a controlled contrast, metamodel-free NBI as the reference arm, revalidation of every candidate on the true
objectives, an empirical Pareto reference the surrogate cannot influence, an external reliability gate, replication
over partitions with paired corrected inference, and a deployment-cost definition that changes the answer.

---

## 4. Open item

Two group papers relevant to this lineage could not be read and are therefore excluded from `references.bib`:
Rocha et al. (2020), *Engineering with Computers*, DOI 10.1007/s00366-020-00973-5 (robust optimal point selection by
MCDM on RSM), and de Paula et al. (2019), Springer AISC, DOI 10.1007/978-3-030-21803-4_60 (a mixture design of
experiments for genetic-algorithm tuning). Both landing pages returned access challenges and neither Crossref nor
Semantic Scholar carries an abstract. If either is cited, its abstract must be read first; de Paula et al. (2019) in
particular could be a further precedent for "mixture DoE over algorithmic weights" and should be checked before the
novelty claims are finalized.
