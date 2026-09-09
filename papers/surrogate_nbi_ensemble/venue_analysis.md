# Title exploration and venue analysis

## Part 1 — Title candidates

Twelve serious candidates, scored on precision (does it say what the paper actually shows?), novelty signalling
(does it avoid claiming the inherited DoE–RSM–NBI combination?), discoverability (will the intended reader's search
find it?), methodological clarity, and venue fit. Scores are 1–5; the total is out of 25.

| # | Title | Prec. | Nov. | Disc. | Clar. | Fit | Total |
|---|---|---|---|---|---|---|---|
| 1 | **When Is Surrogate-Assisted Multiobjective Ensemble Weighting Trustworthy? A Replicated Study of Mixture Designs, Scheffé Surfaces and Normal Boundary Intersection on the Classifier-Weight Simplex** | 5 | 5 | 4 | 5 | 5 | **24** |
| 2 | Surrogate Fidelity and Anchor Construction in Multiobjective Ensemble Weight Optimization | 5 | 5 | 4 | 4 | 5 | 23 |
| 3 | Real Anchors Matter More Than Real Surfaces: Diagnosing Surrogate-Assisted Normal Boundary Intersection for Classifier Ensemble Weighting | 5 | 5 | 4 | 4 | 4 | 22 |
| 4 | From Mixture Design to Ensemble Weighting: Stress-Testing Response-Surface-Assisted Normal Boundary Intersection | 4 | 5 | 4 | 4 | 5 | 22 |
| 5 | Mixture Designs on the Classifier-Weight Simplex: When Response Surfaces Describe but Do Not Optimize | 5 | 4 | 4 | 5 | 4 | 22 |
| 6 | A Replicated Diagnostic Study of Surrogate-Assisted Pareto Optimization for Weighted Classifier Ensembles | 4 | 4 | 4 | 4 | 5 | 21 |
| 7 | Anchors, Surfaces and Deployment Cost: What Governs Surrogate-Assisted Multiobjective Ensemble Weighting | 4 | 4 | 4 | 4 | 4 | 20 |
| 8 | Response-Surface Metamodels of Ensemble Performance: Reproducible Description, Unreliable Optimization | 4 | 4 | 3 | 5 | 4 | 20 |
| 9 | Decomposing Failure in Surrogate-Assisted Normal Boundary Intersection: Surface Error, Anchor Misplacement and Front Geometry | 5 | 4 | 3 | 4 | 4 | 20 |
| 10 | Do Scheffé Interactions Identify Complementary Classifiers? A Replicated Mixture-Design Study of Ensemble Weighting | 3 | 4 | 4 | 4 | 4 | 19 |
| 11 | Cost-Aware Multiobjective Ensemble Weighting on the Simplex: Weighted Relaxation versus Deployment Cost | 3 | 4 | 4 | 4 | 4 | 19 |
| 12 | Transferring a DoE–RSM–NBI Framework from Process Optimization to Classifier Ensemble Weighting: A Replicated Evaluation | 4 | 5 | 2 | 4 | 4 | 19 |

**Recommended: candidate 1.** It states the question rather than a result, names all three methodological keywords a
searcher would use (mixture design, response surface / Scheffé, normal boundary intersection) plus the application
(ensemble weighting), signals replication, and — critically — it claims a *study*, not a method. That is the honest
framing given the lineage: the pipeline is inherited, the evaluation is ours.

**Runner-up: candidate 2** if the venue caps title length. It keeps the two causal factors the paper actually
isolates (surrogate fidelity, anchor construction) but loses the NBI and mixture-design keywords, which costs
discoverability among the engineering-optimization readership most likely to cite it.

**Avoid:** candidate 10 leads with a finding stated as a question the paper answers negatively, which reads as a
narrow note rather than a study; candidate 12 is accurate but its keyword soup ("DoE–RSM–NBI") is not what anyone
outside the group searches for.

Titles to reject outright, for the record: anything containing "novel", "efficient", "outperforms", or "framework"
as the head noun. The paper does not propose a framework and does not outperform anything by design.

---

## Part 2 — Venue analysis

Assessed against what the manuscript actually is: a diagnostic, replicated evaluation study that transfers an
existing DoE–RSM–NBI framework to a new problem class and reports mostly negative or conditional findings about it,
with four datasets and a strong statistical protocol.

### Ranked recommendation

| Rank | Venue | Fit | Main risk |
|---|---|---|---|
| 1 | **Engineering Applications of Artificial Intelligence** | Very good | Self-overlap with the 2025 predecessor in the same journal |
| 2 | **Applied Soft Computing** | Good | May want an evolutionary baseline |
| 3 | **Knowledge-Based Systems** | Good | Prefers a proposed method over a diagnosis |
| 4 | **Expert Systems with Applications** | Moderate | Strong application/novelty expectation; four datasets thin |
| 5 | **Information Sciences** | Moderate | Wants theory or a new algorithm |
| — | *Alternatives worth considering* | | see below |

### 1. Engineering Applications of Artificial Intelligence (Elsevier, IF ≈ 8)

- **Scope.** Explicitly covers applications of AI methods to engineering problems, including surrogate-assisted and
  multiobjective optimization. The 2025 predecessor is in this journal, so the editorial board already accepts this
  methodological family.
- **Novelty threshold.** Moderate-to-high, but EAAI does publish careful evaluation and methodology-transfer studies,
  not only new algorithms.
- **Experimental expectations.** Multiple case studies or datasets; statistical comparison; usually a comparison
  against established optimizers.
- **Likely reviewer concerns.** (i) *Incrementality relative to the authors' own 2025 paper in the same journal* —
  this is the dominant risk and must be addressed head-on, in the cover letter and in Section 2; (ii) absence of
  NSGA-II/MOEA/D, which is a normal expectation for this readership and which the group's own sibling paper
  (Pereira et al. 2026, IJAMT) does provide; (iii) whether four datasets support the mechanism claims.
- **Are four datasets enough?** For EAAI, yes, provided the paper is explicit that the dataset is the unit of
  generalization and does not over-generalize. The 30-partition replication and the corrected paired statistics are
  well beyond what this venue typically sees and should be foregrounded.
- **What to emphasize.** The controlled decomposition (A/B/C), the real-objective revalidation, the empirical
  reference, the replication protocol, and the honest negative findings. Frame the predecessor as the origin of the
  framework and this paper as the stress test that establishes its conditions of validity — a natural, defensible
  sequence for the same journal.
- **Verdict.** Best fit. The self-overlap risk is real but manageable and is lower here than it looks, because
  publishing the critical follow-up in the same venue as the original is a recognizable and respectable pattern.

### 2. Applied Soft Computing (Elsevier, IF ≈ 7–8)

- **Scope.** Soft computing methods including surrogate-assisted MOO, ensembles and metaheuristics. The de Paiva
  group publishes here.
- **Novelty threshold.** Moderate. Comparative and diagnostic studies appear regularly.
- **Reviewer concerns.** Almost certainly "why no NSGA-II / MOEA/D?", since this readership is evolutionary-computation
  centred. Also likely: why fixed hyperparameters, and why only five base models.
- **Four datasets?** Acceptable with the replication protocol, though this readership often expects more problems
  (a benchmark suite mentality carried over from EMO papers).
- **What to emphasize.** The failure decomposition and the reliability gate as transferable diagnostics; the
  non-smoothness of ROC-AUC as the reason classical surrogate machinery struggles.
- **Verdict.** Strong second choice. Adding a canonical evolutionary baseline would materially raise acceptance odds
  here, more than at any other venue on this list.

### 3. Knowledge-Based Systems (Elsevier, IF ≈ 7–8)

- **Scope.** Broad ML/decision-support; ensemble methods are well within scope.
- **Novelty threshold.** Moderate-to-high, and oriented to *proposed methods*. A paper whose contribution is
  "here is when the existing pipeline fails" fits less naturally than at EAAI.
- **Reviewer concerns.** "What do you propose?" The answer must be crisp: a validation architecture (external gate +
  real anchors + real-objective revalidation + empirical reference) that is itself a reusable recommendation, plus
  NBI-B as a concrete, cheap fix that repairs the pipeline in 30/30 and 24/30 partitions.
- **Four datasets?** Borderline; this readership is used to 10–20 UCI-scale datasets. Mitigate by stressing that each
  dataset costs 30 full replications of a ten-stage pipeline and 20 hours of compute, and that adding shallow datasets
  would not strengthen the mechanism claims.
- **Verdict.** Viable, particularly if the abstract leads with NBI-B as the actionable fix.

### 4. Expert Systems with Applications (Elsevier, IF ≈ 7–8)

- **Scope.** Applied ML and decision systems; very high volume.
- **Novelty threshold.** High and application-oriented; reviewers frequently ask for a demonstrated practical gain.
- **Reviewer concerns.** The headline practical gains here are small in absolute terms (ensembling buys 0.0007–0.0060
  AUC), the paper's conclusions are largely cautionary, and the four datasets are all credit/insurance risk. The
  cost-definition finding is the most "applied" result and would have to carry the paper.
- **Four datasets?** Probably not, for this venue, without a stronger applied narrative.
- **Verdict.** Third-tier fit. Only if reframed around cost-aware deployment of ensembles, which would be a different
  paper.

### 5. Information Sciences (Elsevier, IF ≈ 8)

- **Scope.** Broad, with a theoretical lean.
- **Novelty threshold.** High; typically expects a new algorithm, a theoretical result, or a large-scale study.
- **Reviewer concerns.** No new algorithm and no theory. The non-smoothness argument for why low-order polynomials
  misrepresent ROC-AUC over the simplex is the closest thing to a theoretical contribution, and it is currently
  empirical rather than formal.
- **Verdict.** Weak fit unless the misspecification argument is developed formally.

### Alternatives worth considering

- **Computers & Industrial Engineering** or **Computers & Operations Research** — natural homes for the DoE/NBI
  lineage (Mendes et al. 2016 is in the latter). Would receive the methodology sympathetically but may find the
  classifier-ensemble application peripheral.
- **Journal of Machine Learning Research / Machine Learning (Springer)** — the replication protocol and the corrected
  statistics would be appreciated, but the mixture-design framing would need heavy translation and the four-dataset
  scope is too small.
- **Data Mining and Knowledge Discovery** — good fit for the "when does this actually work" genre, though ensemble
  weighting may read as a solved problem there.
- **Quality and Reliability Engineering International / Journal of Quality Technology** — the natural home of the
  Scheffé/mixture-design readership, who would find the β_ij interpretive finding genuinely interesting. Smaller
  audience and lower impact factor, but the finding that a large interaction coefficient does not imply an exploitable
  blend is arguably a *mixture-design* contribution as much as an ML one.

### Cross-cutting reviewer concerns and prepared responses

| Concern | Response |
|---|---|
| "This is incremental over your 2025 EAAI paper." | Different decision variables (compositional, not process factors), different objectives (non-smooth rank statistic and calibration, not rotated factor scores), a controlled anchor experiment the predecessor does not contain, real-objective revalidation, an empirical reference, a reliability gate, and 120 replications against one case study. See `self_overlap_assessment.md`. |
| "Where is NSGA-II / MOEA/D?" | See `baseline_gap_assessment.md`. The study decomposes one pipeline rather than ranking optimizers; an evolutionary arm answers a different question. If required, it can reuse cached OOF predictions and cost roughly one NBI-C run. |
| "Only four datasets." | The dataset is stated as the unit of generalization and no pooled inference is made. Each dataset carries 30 full replications; the constraint is compute, not convenience. |
| "The gains are tiny." | Acknowledged explicitly in the first paragraph of the results. The paper is about the trustworthiness of the front, not about maximizing AUC. |
| "Your reference front is not the true Pareto front." | Never claimed. Construction, convergence check and self-grading caveat are stated in the methods and limitations. |
| "Fixed hyperparameters / no HPO." | Deliberate: the study isolates the weighting layer. Stated in the limitations. |
| "NBI-C is much more expensive." | Reported as measured ratios (3× to 77×), and the recommendation is explicitly conditional on where the surfaces fail. |

### Submission recommendation

Submit to **Engineering Applications of Artificial Intelligence** with candidate title 1, a cover letter that states
the relationship to the 2025 predecessor in the first paragraph, and the baseline-gap assessment appended as a
supplementary note in case a reviewer raises the evolutionary-baseline question. If EAAI declines on incrementality
grounds, move to **Applied Soft Computing** with an added NSGA-II arm.

**Do not submit yet.** See the final section of the manuscript checklist for what remains.
