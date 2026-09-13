# Research lineage

Who did what, in order, and what Paper 2 inherits rather than invents. The three-way distinction is
the one the Paper-1 mandate imposed and it carries over: **INHERITED**, **ADAPTED**, **NEW**.

---

## The line

| # | Work | Contribution to this line | Relation to the present authors |
|---|---|---|---|
| 1 | Das and Dennis (1998) | Normal Boundary Intersection: anchors, payoff matrix, CHIM, quasi-normal, `max t` | none |
| 2 | Costa, Paula, Silva and Paiva (2016), IJAMT 87(1–4) 825–834 | NBI over principal components with Taguchi signal-to-noise ratios | shared senior author |
| 3 | Luz et al. (2021), IJAMT 117(5–6) 1517–1534 | NBI with multivariate techniques, welding | shared senior author |
| 4 | Streitenberger et al. (2022), J. Cleaner Production 333, 129915 | NBI with a factor-analysis approach, stochastic objectives | shared senior author |
| 5 | **Pereira, Tertuliano Ribeiro, Mendes, Campos and de Paiva (2025)**, EAAI 162, 112510 | hybrid multivariate NBI with mixture-design post-optimization | **two co-authors here; the present first author is a co-author there** |
| 6 | **Ribeiro (2026)**, master's dissertation, UNIFEI, unpublished | the construction applied to XGBoost hyperparameter optimization | **the present first author** |
| 7 | de Azevedo, Pereira, Cesário and de Paiva (2026), Thermal Sci. Eng. Progress 74, 104722 | the construction named **NBI-VRF**, applied to computational fluid dynamics | shared co-authors |
| 8 | **Ribeiro, Pereira and de Paiva (2026)**, frozen at `paper-submission-v2`, not submitted | the same construction on the classifier-weight simplex, with a diagnosis of when its surrogate can be trusted | **the present authors** |
| 9 | **This paper** | — | — |

## What Paper 2 inherits, and from whom

**INHERITED. Not claimable, and cited on first use.**

| Element | Owner |
|---|---|
| Normal Boundary Intersection | Das and Dennis (1998) |
| Principal components or factor analysis to reduce correlated responses before optimization | line items 2–4 |
| Varimax rotation of the extracted components, and factor scores as optimization objectives | line items 2–4, named NBI-VRF at item 7 |
| The combination of design of experiments, response surfaces and NBI | item 5, and independently established elsewhere |
| Design of experiments and response surfaces for hyperparameter tuning | Lujan-Moreno, Howard, Rojas and Montgomery (2018) |
| Response surfaces for XGBoost hyperparameters specifically | Vasquez-Ramos et al. (2025), an unrelated group |
| Multiobjective hyperparameter optimization with a cost objective | the field; see `literature_review.md` Cluster D |
| The 88-run face-centred central composite design on seven XGBoost hyperparameters | item 6 |
| Re-evaluating every returned candidate on the real objectives | item 6; a genuine strength of the original protocol |
| The external reliability gate on surrogate adequacy | item 8 |
| Evaluation-matched budgeting, and the NSGA-II harness | item 8 |
| The anchor-injection control that separates set composition from geometry | item 8 |

**ADAPTED. Inherited in substance, changed in a way that must be stated.**

| Element | Change | Forced by |
|---|---|---|
| Response-surface fitting | uncoded → coded units | `METHODOLOGY_DECISIONS.md` D6; conditioning |
| Objective count | three extracted and two optimized, undeclared → one declared specification | `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q3 |
| Quality aggregation | a default argument → a pre-registered weighting with a sensitivity check | `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` Q4 |
| Selection rule | `max(Accuracy_Mean)` → the full front, with the legacy column kept and labelled | `METHODOLOGY_DECISIONS.md` D7 |
| Cost objective | wall-clock training time → an objective that is reproducible from seeds | `audits/provenance/README.md` |
| NBI subproblem | `t ≥ 0` → `t` free | `audits/NBI_GEOMETRY_AUDIT.md` |
| Comparator set | four single-objective methods → the set in `baseline_gap_assessment.md` | `literature_review.md` Cluster D |

**NEW. What Paper 2 may actually claim, subject to `novelty_matrix.md`.**

Not a combination of methods. The candidates are the controlled four-arm decomposition, the
identity audits that make the historical arm reproducible at all, and whatever the campaign
measures. None is claimable until measured.

## Self-overlap, stated up front

Two of the three authors of item 5 are authors here, and the present first author is a co-author of
item 5. Item 6 is the present first author's own dissertation, and this paper is its follow-up. Item
8 is the present authors' immediately preceding manuscript, frozen and unsubmitted.

Item 8 is a particular problem and must be handled deliberately. It shares the senior authors, the
methodological family, several verified references, and the NSGA-II and anchor-injection machinery.
It is **not yet published**, so it cannot be cited as published work, and if both are under review
at once each must disclose the other to the editor. What separates them is not subtle: item 8
optimizes ensemble weights on a simplex; this paper optimizes hyperparameters in a box. The decision
space, the surrogate family (Scheffé canonical polynomials against ordinary quadratic response
surfaces) and the objectives all differ.

A dimension-by-dimension self-overlap assessment against item 8 and item 5 is required before
submission, on the model of `papers/surrogate_nbi_ensemble/self_overlap_assessment.md`. It is not
required before the protocol freeze and is tracked as an open item.

## A statement that is permitted, and one that is not

Permitted: **"Pereira et al. (2025) formulate the method using canonical NBI."** This is a statement
about the published formulation.

Not permitted: **"their code implements canonical NBI."** The EAAI 2025 implementation has not been
audited and no claim is made about it. The audits here cover the dissertation code and the
article-track code only.
