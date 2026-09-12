# Supplementary material plan

The main manuscript carries 8 figures and 6 tables. Everything else produced by the R = 30 study goes here. All
supplementary items are generated from the frozen artifacts at tag `pco213-postwork-r30` (commit `b3ed050`) and are
reproducible with `papers/surrogate_nbi_ensemble/build_assets.py` and the two analysis scripts; no supplementary item
introduces a number that is not already in `reports/pco213_postwork_benchmark/`.

## What stays in the main text

| Item | Content | Why it must be in the main text |
|---|---|---|
| Fig. 1 | Lineage and pipeline, colour-coded inherited / adapted / new | The paper's central framing claim is the lineage boundary |
| Fig. 2 | Surrogate predicted vs observed on unseen compositions, 4 regimes | Evidence for RQ1 |
| Fig. 3 | NBI-A/B/C revalidated fronts vs the empirical reference | Shows the anchor failure geometrically |
| Fig. 4 | Paired Δ distributions, both endpoints, all datasets | The primary result |
| Fig. 5 | Gate pass frequencies and external R² | Evidence for RQ1 and for the gate's limits |
| Fig. 6 | R = 10 vs R = 30 stability | Evidence for RQ9 |
| Fig. 7 | β_ij vs real 50/50 blend gain | The interpretive finding |
| Fig. 8 | Cost-definition winner counts and a worked example | Evidence for RQ7 |
| Table 1 | Datasets and partition sizes | Protocol |
| Table 2 | Surrogate reliability by dataset × response | RQ1 |
| Table 3 | Primary paired comparisons, weighted cost | The primary result |
| Table 4 | R10 → R30 robustness | RQ9 |
| Table 5 | Compute per stage and per variant | Bounds the NBI-C claim |
| Table 6 | Holdout ranking agreement and cost sensitivity | RQ7, RQ8 |

## Supplementary sections

### S1. Extended protocol
- S1.1 The 66 design points in full (composition table), and the 100 validation compositions' generating parameters.
- S1.2 Base-model hyperparameters and the exact preprocessing pipeline per model (`bench_models.py` extract).
- S1.3 Seed map: dataset × replication → outer seed, inner seed, component offsets (`manifests/benchmark_manifest.json`).
- S1.4 Dataset provenance: source, acquisition command, expected invariants, SHA-256 (`manifests/*_dataset_manifest.json`).
- S1.5 Statistical analysis plan as pre-specified (`STATISTICAL_ANALYSIS_R30.md` in full).

### S2. Base models and single-objective references
- Table S2.1 — per-model OOF AUC, holdout AUC, OOF log-loss, Brier, inference cost, per dataset, mean ± sd over 30 partitions. Source `tables/model_performance.csv`.
- Table S2.2 — all single-objective references (best single, uniform, logistic stacking, SLSQP log-loss, direct AUC, both Scheffé optima, cheapest vertex) with OOF and holdout metrics and both costs. Source `tables/single_objective_refs.csv`.
- Table S2.3 — weight-vector stability of the real optima over 30 partitions (mean ± sd per component).
- Figure S2.1 — `fig13_weight_composition.png`, `fig14_neff.png` (effective number of models on the front).

### S3. Complete Scheffé results
- Table S3.1 — order selection counts, R²_train, R²_ext, external RMSE, Spearman, condition number, extrapolation excess, per dataset × response × order. Source `tables/scheffe_orders.csv` (all 120 replications).
- Table S3.2 — all 15 quadratic coefficients (5 linear + 10 interaction) per dataset × response: mean, sd, CV, median, IQR, bootstrap CI, sign frequency with Wilson interval, rank statistics, top-1/top-3 frequency. Source `statistics/coefficient_stability_r30.csv`.
- Table S3.3 — for every pair: vertex-quality gap, real 50/50 blend minus better member, frequency the blend wins, Pareto participation. This is the full version of Fig. 7.
- Figure S3.1 — `fig11_coef_stability_roc_auc.png`, `fig11_coef_stability_log_loss.png`.
- Figure S3.2 — `r30_fig05_beta_ij_stability.png` (coefficient intervals across partitions).
- Figure S3.3 — `fig15_diversity_vs_beta.png` and the β_ij vs error-correlation Spearman values per dataset.
- Figure S3.4 — `fig16_surrogate_validation.png` (all dataset × response panels, extending Fig. 2).

### S4. Complete Pareto indicators
- Table S4.1 — every indicator (GD, IGD, IGD⁺, spacing, spacing CV, size-matched spacing percentile, hypervolume, hypervolume ratio, coverage both directions, joint non-dominated fraction, front size, extreme-point gaps, mean weights) for all 7 candidate sets × 2 cost definitions × 4 datasets, median and IQR over 30 partitions. Source `tables/pareto_quality.csv` (1,680 rows).
- Table S4.2 — the same, per replication, as a machine-readable CSV attachment.
- Figure S4.1 — `fig04_igd_plus.png`, `fig05_hv_ratio.png`, `fig06_joint_nd.png`, `fig07_gd.png`, `fig08_igd.png`, `fig09_spacing_cv.png` (distributions over 120 replications).
- Figure S4.2 — `fig02_fronts_*.png` for all four datasets (extending Fig. 3).
- Figure S4.3 — `fig12_active_support.png` (support frequency on the reference front).

### S5. Full paired statistics
- Table S5.1 — `statistics/paired_primary_effects.csv` in full: 64 rows (4 datasets × 4 comparisons × 2 endpoints × 2 costs) with n, mean, median, sd, quartiles, min, max, both bootstrap intervals, W/T/L, win fraction with Wilson interval, rank-biserial, Nadeau–Bengio t and p, Wilcoxon p.
- Table S5.2 — `statistics/paired_primary_tests.csv`: the Holm-corrected families.
- Table S5.3 — `statistics/win_tie_loss.csv`: all pairwise set comparisons (241 rows), not only the primary four.
- Table S5.4 — `statistics/proportion_intervals.csv`: Wilson and Jeffreys intervals for all reported frequencies.
- **Table S1 in the main numbering** — the support-cost repetition of Table 3 (`tabS01_paired_support.tex`, already generated). This one is referenced from the main text and should be the first supplementary table.
- Figure S5.1 — `r30_fig01_paired_delta_igd_plus.png`, `r30_fig02_paired_delta_hv.png`, `r30_fig06_win_tie_loss.png`.

### S6. Reliability gate and regime analysis
- Table S6.1 — `statistics/reliability_gate_r30.csv` including the Brier and PR-AUC responses omitted from Table 2.
- Table S6.2 — `statistics/reliability_gate_conditional_gain_r30.csv`: B − A gain split by gate outcome, with the explicit warning that on BNP Paribas the split restates NBI-A's collapse.
- Table S6.3 — `statistics/bimodality_regimes_r30.csv`: Sarle's bimodality coefficient and GMM ΔBIC per set × metric × cost, flagged as exploratory diagnostics with no formal multimodality test.
- S6.4 — the BNP Paribas collapse case study: the seven collapse partitions listed by index, their selected order, external R², gate outcome, anchor composition and anchor cost; the Spearman associations; and the statement that anchor cost is a marker, not the mechanism.
- Figure S6.1 — `r30_fig04_gate_pass.png`, `r30_fig09_anchor_costs.png`.

### S7. NBI solver diagnostics
- Table S7.1 — `tables/nbi_runs.csv`: per replication and variant, subproblems attempted, certified/feasible, front sizes under both costs, total function evaluations, real objective evaluations, seconds, anchor metrics and costs, surrogate reliability flags.
- S7.2 — the differing success semantics across arms (SLSQP certification for A/B versus residual feasibility for C) and the filtered-versus-unfiltered version of the UCI credit NBI-C vs NBI-B comparison.
- S7.3 — the Porto Seguro NBI-B low-hypervolume regime: 6 partitions with 100% certification, evidence that the failure is CHIM geometry rather than solver failure.
- Figure S7.1 — `r30_fig10_nbi_success_ecdf.png`, `fig17_nbi_variants_paired.png`.

### S8. Empirical reference convergence
- Table S8.1 — `tables/reference_diagnostics.csv`: points generated, rounds used, displacement by the independent check, front sizes under both costs, per replication.
- S8.2 — the composition of the final reference by source (sample, lattice, ε-constraint, candidate sets), and the resulting self-grading caveat with the fraction contributed by NBI-B and NBI-C.
- S8.3 — the four replications that stopped at the three-round cap and the bound on the objective-space effect.

### S9. Cost definition
- Table S9.1 — `statistics/cost_definition_sensitivity_r30.csv` in full, both endpoints.
- S9.2 — the BNP Paribas normalization-box analysis: the 13 narrow-box versus 17 wide-box partitions, the cost bounds, and the demonstration that under a common bound the apparent bimodality disappears. Stated explicitly as a correction of the earlier R = 10 reading.
- Table S9.3 — inference cost per model per dataset with the measurement protocol (`tables/inference_costs.csv`).
- Figure S9.1 — `fig03_cost_definitions_*.png` (all four datasets), `r30_fig08_cost_definition_ranking.png`.

### S10. Holdout transfer
- Table S10.1 — `statistics/holdout_transfer_r30.csv`: OOF and holdout AUC and log-loss per set, paired differences with intervals, fraction of partitions with |shift| > 0.005, ranking agreement, under both cost definitions.
- Table S10.2 — `tables/mcdm_picks_holdout.csv`: knee and TOPSIS picks with their OOF and holdout metrics.
- S10.3 — the near-tie analysis for BNP Paribas: OOF gaps between the best and second-best pick against holdout noise; the regret distribution when the two disagree.
- Figure S10.1 — `r30_fig07_holdout_transfer.png`.

### S11. R = 10 to R = 30 expanded
- Table S11.1 — `statistics/r10_vs_r30_stability.csv` in full (353 rows across 16 metric families), extending Table 4 beyond the hypervolume and IGD⁺ cells.
- S11.2 — the batch-comparison analysis: partitions 0–9 against 10–29 alone, the 6/12 and 5/12 agreement, the Santander IGD⁺ level shift and the UCI credit Fisher test, presented as the measurement of partition sensitivity.
- Figure S11.1 — `r30_fig03_r10_vs_r30_stability.png`.

### S12. Corrections log
A short, explicit list of readings that appeared in the earlier R = 10 analysis and did not survive the R = 30 study or
adversarial verification, with the evidence that overturned each: (i) the BNP Paribas support-cost bimodality of NBI-C,
now shown to be a normalization-box artifact; (ii) the Santander deployment-cost gap between the AUC and log-loss
optima, now shown to be a near-threshold kNN weight; (iii) the description of the random Dirichlet comparator as
budget-matched; (iv) the reading of the gate-conditional anchor gain on BNP Paribas; (v) uniform spacing as an
advantage of NBI after revalidation. Including this list is a deliberate choice: each correction is traceable in the
repository history and reviewers should be able to see what changed.

## Format and delivery

- Supplementary tables that exceed one page are supplied as CSV attachments as well as typeset tables, with the exact
  source path from `reports/pco213_postwork_benchmark/` named in each caption.
- All 35 benchmark figures are included at publication resolution; the 8 main-text figures are regenerated at
  manuscript width rather than reused from the benchmark set.
- A single `supplementary.pdf` carries S1–S12; the CSV attachments accompany it in one archive whose manifest lists
  each file's source path and the commit it was generated from.
