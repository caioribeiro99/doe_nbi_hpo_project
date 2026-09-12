# Claims and evidence map (R = 30, frozen at tag `pco213-postwork-r30`, commit `b3ed050`)

Every quantitative sentence in the Results section must trace to a row of this document. Paths are relative to
`reports/pco213_postwork_benchmark/` (tables, statistics) or `figures/pco213_postwork_benchmark/` (benchmark figures);
manuscript figures are in `papers/surrogate_nbi_ensemble/figures/` and are regenerated from the same artifacts by
`build_assets.py`. Δ > 0 always means the second-named (new) method is better (ΔHV = HV_new − HV_ref;
ΔIGD+ = IGD+_ref − IGD+_new). "30/30" style counts are wins over the 30 outer partitions of one dataset; nothing is
pooled over the 120 partitions. Statistical unit: one outer partition of one dataset; the dataset is the unit of
generalization. Tier: PRIMARY (pre-specified endpoint, paired, corrected test), SECONDARY (pre-specified descriptive),
EXPLORATORY (post hoc diagnosis).

Sign/level conventions: HV ratio = hypervolume of the set / hypervolume of the empirical reference (same partition,
reference point 1.1 on each normalized axis); IGD+ on min–max normalized objectives; weighted cost = Σ w_i c_i;
support cost = Σ c_i 1[w_i > 10⁻³], c_i in ms per 1,000 rows.

---

## C1 — Real anchors improve surrogate-assisted NBI (NBI-B vs NBI-A), weighted cost — PRIMARY

| Dataset | ΔHV mean [95% CI] · median | ΔIGD+ mean [95% CI] · median | W/T/L (HV · IGD+) | r_rb | NB Holm p (HV · IGD+) |
|---|---|---|---|---|---|
| Santander | +0.179 [0.148, 0.210] · 0.196 | +0.120 [0.098, 0.143] · 0.125 | 30/0/0 · 30/0/0 | 1.00 | 0.004 · 0.007 |
| BNP | +0.279 [0.154, 0.415] · 0.059 | +0.239 [0.111, 0.377] · 0.015 | 30/0/0 · 30/0/0 | 1.00 | 0.66 · 0.66 |
| Porto | +0.085 [0.037, 0.131] · 0.076 | +0.038 [0.017, 0.058] · 0.030 | 24/0/6 · **23/0/7** | 0.66 · 0.65 | 0.79 · 0.79 |
| UCI | +0.060 [0.034, 0.083] · 0.092 | +0.036 [0.021, 0.050] · 0.050 | **24/1/5** · 24/0/6 | 0.75 | 0.45 · 0.45 |

**Wording rule (added 2026-09-11, verified against `paired_primary_effects.csv`).** The two endpoints do NOT share a
win count on Porto Seguro or UCI credit. Forbidden: "improves both primary endpoints in 24/30 on Porto Seguro".
Permitted compact form: "unanimous on Santander and BNP Paribas; 23–24 of 30 on Porto Seguro and 24 of 30 on UCI
credit". Exact per-endpoint counts: Porto IGD+ 23/0/7, Porto HV 24/0/6, UCI IGD+ 24/0/6, UCI HV 24/1/5.

- Source: `statistics/paired_primary_effects.csv` (cost = weighted, comparison = "nbi_B vs nbi_A"); `statistics/paired_primary_tests.csv`.
- Figures: manuscript Fig. 4 (paired deltas), Fig. 3 (example fronts); benchmark `r30_fig02_paired_delta_hv.png`, `r30_fig06_win_tie_loss.png`.
- Wilson 95% for 30/30: [0.886, 1.000]; for 24/30: [0.627, 0.905].
- Limitations: (i) the BNP mean is driven by 7/30 NBI-A collapse partitions (bimodal; mean +0.28 vs median +0.06), so the mean-based corrected test is uninformative and the median/win fraction is the evidence; (ii) Porto and UCI win fractions are dominated by partitions 10–29 (18/20 and 19/20 vs 6/10 and 5/10+1 tie in partitions 0–9; Fisher p = 0.009 on UCI) — partition sensitivity, not a sharpened fixed rate; (iii) under the support cost the UCI effect reverses (NBI-B worse in 26–27/30) and BNP shrinks to 19–20/30 (C1-S below).
- Wording allowed: "real anchors improved NBI-B over NBI-A in every partition of the two datasets whose AUC surfaces almost never pass the gate, and in 24 of 30 partitions on the other two (weighted cost)".
- Wording NOT allowed: "the gate predicts when real anchors are needed" (see C5).

### C1-S — same comparison under the support cost — SECONDARY (sensitivity)

| Dataset | ΔHV mean · median · W/T/L | ΔIGD+ mean · median · W/T/L |
|---|---|---|
| Santander | +0.126 · 0.130 · 30/0/0 | +0.060 · 0.059 · 30/0/0 |
| BNP | +0.151 · 0.074 · 20/0/10 | +0.225 · 0.019 · 19/0/11 |
| Porto | +0.004 · 0.005 · 19/0/11 | +0.080 · 0.063 · 29/0/1 |
| UCI | −0.043 · −0.016 · 4/0/26 | −0.024 · −0.010 · 3/0/27 |

- Source: `statistics/paired_primary_effects.csv` (cost = support). Manuscript Table S1.

---

## C2 — Metamodel-free NBI-C approximates the empirical reference closely and is best or tied-best (weighted cost) — PRIMARY (C vs B) / SECONDARY (levels)

Levels (median over 30 partitions; `README.md` median table, `tables/pareto_quality.csv`):

| Dataset | HV ratio A / B / C | IGD+ A / B / C | scalarization HV · IGD+ | n front A / B / C |
|---|---|---|---|---|
| Santander | 0.789 / 0.981 / 0.989 | 0.136 / 0.011 / 0.010 | 0.978 · 0.007 | 13 / 20 / 35 |
| BNP | 0.913 / 0.971 / 0.983 | 0.027 / 0.012 / 0.014 | 0.474 · 0.308 | 8 / 44 / 34 |
| Porto | 0.769 / 0.914 / 0.982 | 0.097 / 0.035 / 0.007 | 0.762 · 0.104 | 11 / 60 / 56 |
| UCI | 0.622 / 0.706 / 0.976 | 0.217 / 0.154 / 0.010 | 0.978 · 0.009 | 11 / 27 / 65 |

Paired C vs B (weighted; `statistics/paired_primary_effects.csv`):

| Dataset | ΔHV mean [CI] · W/T/L | ΔIGD+ mean [CI] · W/T/L | NB Holm p (HV · IGD+) |
|---|---|---|---|
| Santander | +0.0077 [0.0064, 0.0089] · 28/0/2 | +0.0016 [0.0011, 0.0021] · 28/0/2 | 0.003 · 0.099 |
| BNP | +0.0086 [0.0043, 0.0132] · 24/0/6 | −0.0029 [−0.0044, −0.0011] · 6/0/24 | 0.66 · 0.66 |
| Porto | +0.108 [0.064, 0.162] · 28/0/2 | +0.042 [0.024, 0.063] · 28/0/2 | 0.79 · 0.79 |
| UCI | +0.277 [0.226, 0.332] · 30/0/0 | +0.158 [0.129, 0.191] · 29/0/1 | 0.010 · 0.010 |

- **CORRECTED (verified 2026-09-09 against `tables/pareto_quality.csv`).** The earlier phrasing "HV ≥ 0.97 in 30/30 partitions of each dataset" is FALSE. Per-partition counts of HV ratio ≥ 0.97: Santander 30/30, BNP 30/30, Porto 24/30, UCI 25/30. At ≥ 0.95: 30/30, 30/30, 26/30, 30/30. Minima: 0.971, 0.975, 0.877, 0.961. The defensible statements are (a) the **median** HV ratio is 0.989 / 0.983 / 0.982 / 0.976, and (b) HV ratio ≥ 0.95 in all 30 partitions of three datasets and 26/30 on Porto. The FINAL_REPORT §11-A sentence refers to medians and must not be read as a per-partition count.
- Figures: manuscript Fig. 3, Fig. 4; benchmark `fig05_hv_ratio.png`, `fig04_igd_plus.png`.
- Limitations: (i) NBI-C is not evaluation-matched to anything (≈ 4.2 × 10⁵ real OOF evaluations per partition; 0 for A/B); (ii) NBI-C candidates form 8–25% of the reference front it is graded against (direction unaffected, magnitude partly self-graded); (iii) the UCI gap C − B is a subproblem-convergence effect of NBI-B (median 29/66 certified; Spearman(ΔHV, n_valid_B) = −0.92; retaining B's unconverged iterates cuts the gap to +0.108 HV while keeping 30/30); (iv) the UCI Holm p ≈ 0.010 is knife-edge (0.017–0.018 with ρ = 0.30 or a 16-test family); (v) on BNP the IGD+/HV split is genuine: B is closer to the reference (IGD+), C covers more volume.
- Compute premium (`tables/stage_times.csv`, mean s per partition A / B / C, verified): Santander 34.0 / 34.6 / 2528.4; BNP 23.1 / 35.6 / 1409.8; Porto 84.8 / 55.8 / 2504.5; UCI 124.8 / 99.7 / 306.6; **C/B = 73.1×, 39.5×, 44.9×, 3.1×**. The manuscript quotes 73 / 40 / 45 / 3 and the range "3 to 73 times". FINAL_REPORT quotes 77×/40×/46×/3× from medians; use the mean-based values above in the manuscript for consistency with Table 5.

---

## C3 — NBI-C vs random weighted scalarization on the surfaces (weighted cost) — PRIMARY, confounded

| Dataset | ΔHV mean [CI] · W/T/L | ΔIGD+ mean [CI] · W/T/L |
|---|---|---|
| Santander | +0.011 [0.008, 0.013] · 25/0/5 | −0.003 [−0.005, −0.001] · 10/0/20 |
| BNP | +0.539 [0.473, 0.606] · 30/0/0 | +0.398 [0.293, 0.527] · 30/0/0 |
| Porto | +0.270 [0.189, 0.369] · 30/0/0 | +1.64 [0.08, 4.69] · 30/0/0 (median +0.091) |
| UCI | +0.0015 [−0.003, 0.007] · 16/0/14 | +0.0008 [−0.001, 0.003] · 18/0/12 |

- Source: `statistics/paired_primary_effects.csv` (comparison "nbi_C vs ws_random_scalarization").
- Limitation: scalarization optimizes the surrogates, NBI-C the exact cached-OOF objectives; the BNP/Porto advantage cannot be attributed to the NBI construction rather than the objective source (no scalarization-on-cached-OOF comparator). On identical surfaces NBI-B beats scalarization 30/30 (BNP), 25/30 (Porto) and loses 29/30 (UCI) (`statistics/win_tie_loss.csv`).
- Under the support cost UCI reverses to 0/30 (scalarization better) and Santander IGD+ reverses to 30/30 in C's favour.
- Wording allowed: "random scalarization on externally validated surfaces is a cheap alternative that ties NBI-C on UCI and matches it on Santander hypervolume".

## C4 — NBI-C vs 66-point random Dirichlet(1) search — floor check only — SECONDARY

- 30/30 wins on every dataset and endpoint (ΔHV +0.71 / +0.94 / +0.93 / +0.25); the comparator's HV ratio is exactly 0 in 10/30 (Santander), 21/30 (BNP), 25/30 (Porto) partitions because uniform blends cost 8–20× the reference front (`tables/pareto_quality.csv`, set = random_dirichlet_budget).
- Every other set, including the unoptimized 66-run design, also beats it 30/30 (`statistics/win_tie_loss.csv`).
- Wording NOT allowed: "budget-matched", "search efficiency". It is candidate-count-matched only.

---

## C5 — Surrogate reliability is dataset- and metric-dependent; the gate separates the extremes — PRIMARY (frequencies)

| Dataset | AUC pass k/30 [Wilson] · median R²ext [CI] | log-loss pass k/30 [Wilson] · median R²ext [CI] | selected AUC order |
|---|---|---|---|
| Santander | 0/30 [0.00, 0.11] · −0.333 [−0.559, −0.199] | 30/30 [0.89, 1.00] · 0.971 [0.967, 0.973] | quadratic 26, linear 4 |
| BNP | 5/30 [0.07, 0.34] · 0.118 [−0.034, 0.244] | 3/30 [0.03, 0.26] · −0.180 [−0.246, 0.042] | linear 22, quadratic 7, sp.cubic 1 |
| Porto | 16/30 [0.36, 0.70] · 0.519 [0.482, 0.581] | 12/30 [0.25, 0.58] · 0.506 [0.492, 0.580] | quadratic 26, sp.cubic 2, linear 2 |
| UCI | 30/30 [0.89, 1.00] · 0.992 [0.989, 0.992] | 30/30 [0.89, 1.00] · 0.993 [0.991, 0.993] | quadratic 20, sp.cubic 10 |

- Source: `statistics/reliability_gate_r30.csv`; manuscript Table 2, Fig. 5, Fig. 2; benchmark `r30_fig04_gate_pass.png`, `fig16_surrogate_validation.png`.
- Extrapolation excess ≈ 0 everywhere (`tables/scheffe_orders.csv`, column extrapolation_excess): failures are polynomial misspecification, not range extrapolation.
- Brier surface: R²ext = 1.000 in 120/120 (algebraic identity — quadratic in w — used only as a sanity check).
- Within-dataset gate does NOT predict anchor failure (EXPLORATORY, `FINAL_REPORT.md` §7): on BNP the 7 NBI-A collapse partitions (7, 9, 18, 20, 23, 26, 27) are exactly those where the quadratic AUC order is selected (7/7 vs 0/22 linear-selected), have *higher* external R² (median 0.46 vs 0.08) and pass the gate more often (3/7 vs 2/23). On Porto gate pass does not change the B − A gain (12/16 vs 12/14); only on UCI does NBI-A's HV correlate with R²ext (+0.59 AUC, +0.57 log-loss). Source: `statistics/reliability_gate_conditional_gain_r30.csv`, `statistics/bimodality_regimes_r30.csv`.
- Wording allowed: "the gate identifies unusable surfaces; it does not predict anchor misplacement".

---

## C6 — Scheffé coefficients are partition-stable; the classical synergism criterion is satisfied yet contradicted by the real objectives — SECONDARY (stability) / SECONDARY (blend test)

**Correct attribution (added after the literature review).** The classical mixture-DoE reading of β_ij is *departure
from linear blending*, defined against the chord and not against the better pure component (Scheffé 1958; Cornell
2002 §2.2, verified in the source text; cf. Piepel 1982 on the non-interpretability of mixture coefficients as
component effects). The exact criterion for the fitted quadratic to have an edge optimum beating both pure components
is **β_ij > |β_i − β_j|**, derived and verified algebraically:
on the edge w_i = t, ŷ(t) = β_j + (β_i−β_j)t + β_ij t(1−t), so t* = (β_i−β_j+β_ij)/(2β_ij) ∈ (0,1) iff β_ij > |β_i−β_j|,
and there ŷ(t*) = β_j + (β_i−β_j+β_ij)²/(4β_ij) > max(β_i, β_j). **We therefore do NOT claim the interpretive
principle as new.** The claim is the empirical one below.

| Dataset | top pair | β_ij | mean \|β_i − β_j\| | criterion met (k/30) | real blend beats better member (k/30) | agree over 10 pairs | surface over-predicts | under-predicts |
|---|---|---|---|---|---|---|---|---|
| Santander | gnb·knn | 0.243 | 0.130 | 30/30 | 0/30 | 6/10 | 4 | 0 |
| BNP | gnb·xgb | 0.148 | 0.110 | 30/30 | 0/30 | 4/10 | 6 | 0 |
| Porto | gnb·knn | 0.071 | 0.025 | 30/30 | 30/30 | 7/10 | 3 | 0 |
| UCI | gnb·xgb | 0.130 | 0.100 | 30/30 | 0/30 | 3/10 | 7 | 0 |

- Source: `papers/surrogate_nbi_ensemble/tables/edge_condition_summary.csv` and `edge_condition_per_replication.csv`, generated by `build_assets.py::tab07_edge_condition` from `experiments/.../scheffe.json` (quadratic coefficients, all 120 replications) and `statistics/coefficient_stability_r30.csv`.
- **The defensible claim:** the fitted surfaces satisfy the correct classical criterion in 30/30 partitions on all four datasets, yet the real out-of-fold blend beats its better member on only one of them; over all 40 dataset-pair cells the disagreement is one-directional (20 over-predictions, 0 under-predictions). This is a *surrogate misspecification* result about classifier-performance surfaces, not a correction of mixture-design theory.
- Tier: SECONDARY. Limitation: conditional on the heterogeneous five-model zoo; a zoo of near-equal vertices would produce smaller β_ij and a smaller gap to test.

- Sign stability: top-4 AUC interactions have sign frequency exactly 1.0 in all 4 datasets; top-1 term identical at R = 10 and R = 30 in all 8 cells; top-1 per-partition frequency 1.00 (Santander gnb·knn, BNP gnb·xgb, UCI gnb·xgb) and 0.63 (Porto gnb·knn vs knn·xgb 0.37). Between-partition CV 0.011–0.17 for the top pair (`statistics/coefficient_stability_r30.csv`).
- Largest AUC β_ij (mean ± sd): Santander gnb·knn +0.243 ± 0.003; BNP gnb·xgb +0.148 ± 0.026; Porto gnb·knn +0.071 ± 0.008; UCI gnb·xgb +0.130 ± 0.008.
- Vertex-quality gap of the top pair: 0.163 / 0.145 / 0.022 / 0.107 (Santander / BNP / Porto / UCI).
- Real 50/50 blend of the top-β pair minus its better member (OOF AUC, mean of 30; freq. blend beats better member): Santander −0.0050 (0/30), BNP −0.0180 (0/30), Porto +0.0098 (30/30), UCI −0.0201 (0/30). Holdout: 0/30, 0/30, 29/30, 0/30 (FINAL_REPORT §4).
- Pairs whose 50/50 blend beats the better member in ≥ 80% of partitions carry the largest β only on Porto (FINAL_REPORT §4 list).
- Log-loss interactions are negative by convexity (Jensen); only AUC signs are informative.
- Figures: manuscript Fig. 7; benchmark `r30_fig05_beta_ij_stability.png`, `fig10_beta_ij_roc_auc.png`, `fig15_diversity_vs_beta.png`.
- Limitation: depends on the fixed five-model zoo (a very weak component inflates the interactions of every pair containing it).
- Wording NOT allowed: "large β_ij identifies complementary classifiers".

---

## C7 — AUC-vs-log-loss conflict is objective-space degenerate; weight-space and cost differences remain — SECONDARY

| Dataset | ΔAUC direct-AUC − SLSQP (mean [CI]) | Δlog-loss cost | holdout ΔAUC | weight L1 distance | support Jaccard | ensembling gain over best single (AUC) |
|---|---|---|---|---|---|---|
| Santander | +0.00068 [0.00066, 0.00071] | +0.0032 | +0.0020 (30/30 > 0) | 0.35 | 0.50 | +0.0037 |
| BNP | +0.00080 [0.00078, 0.00083] | +0.0009 | +0.0003 | 0.38 | 0.76 | +0.0056 |
| Porto | +0.00006 [0.00004, 0.00007] | +0.000004 | −0.00008 | 0.06 | 0.97 | +0.0060 |
| UCI | +0.00034 [0.00031, 0.00038] | +0.0005 | +0.0003 | 0.20 | 0.87 | +0.0007 |

- Source: `statistics/auc_logloss_conflict_r30.csv`.
- The Santander support-cost gap between the two optima (median +254 ms/1k in 22/30) is an activity-threshold artifact: the AUC optimum retains kNN weight 0.0015–0.062 above 10⁻³ that is AUC-inert (zeroing it changes OOF AUC by 3 × 10⁻⁶ on average); an AUC-equivalent kNN-free blend costs ≈ +4 ms/1k (FINAL_REPORT §7, EXPLORATORY).
- Wording NOT allowed: "the AUC optimum is much more expensive to deploy".

---

## C8 — Cost definition changes the winning set — SECONDARY (sensitivity)

| Dataset | HV-best set differs W vs S (k/30 [Wilson]) | best under W (freq.) | best under S (freq.) | median rank ρ (W vs S) |
|---|---|---|---|---|
| Santander | 10/30 [0.19, 0.51] | C 24, scal 5, B 1 | C 26, design 4 | 0.83 |
| BNP | 24/30 [0.63, 0.90] (one margin 9 × 10⁻⁵ inside tie tolerance → 23) | C 24, B 6 | design 13, B 13, C 4 | 0.89 |
| Porto | 8/30 [0.14, 0.44] | C 28, B 2 | C 20, B 6, scal 4 | 0.94 |
| UCI | 20/30 [0.49, 0.81] | C 16, scal 14 | scal 22, design 8 | 0.66 |

- Source: `statistics/cost_definition_sensitivity_r30.csv` (six comparison sets, single-objective references excluded). Manuscript Fig. 8, Table 6; benchmark `r30_fig08_cost_definition_ranking.png`, `fig03_cost_definitions_*.png`.
- BNP support-cost "bimodality" of NBI-C (13/30 below 0.8) is a per-replication normalization-box artifact (cost bound ≈ 17 ms/1k in 13 partitions vs 160–167 in 17); under a common bound NBI-C's support HV is 0.968–0.990 in 30/30 (kNN region inside) or the 66-run design wins 30/30 (kNN region excluded) (FINAL_REPORT §7, EXPLORATORY).
- Robust qualitative statement: NBI sets optimized under the linear cost omit the cheapest single-model and two-model supports that the raw design contains; the size of the penalty depends on the cost range.
- Wording NOT allowed: "BNP shows a bimodal support-cost regime".

---

## C9 — Holdout transfer: level yes; ranking dataset-dependent — SECONDARY

- Ranking agreement (OOF-best of the four knee picks = holdout-best, weighted cost, AUC): 30/30, 15/30, 23/30, 30/30 (Santander, BNP, Porto, UCI); partitions 10–29 alone: 20/20, 10/20, 15/20, 20/20 (R = 10: 10/10, 5/10, 8/10, 10/10). Source: `statistics/holdout_transfer_r30.csv` (metric = ranking_agreement_auc). Manuscript Table 6, benchmark `r30_fig07_holdout_transfer.png`.
- Regret when they differ ≤ 0.0022 holdout AUC (BNP mean 0.0010, Porto 0.0006); on BNP the OOF gap between best and second-best pick (median 0.0005) is below holdout noise (sd ≈ 0.0012).
- Level shift holdout − OOF AUC of every weighted-cost knee pick within ±0.005 as point estimate (largest −0.0046, Santander scalarization); intervals exclude 0 on Santander (negative) and BNP (positive); the shift equals the weight-averaged base-model shift on the same partition (r = 0.97–1.00) → partition/refit effect, not selection optimism. Source: `statistics/holdout_transfer_r30.csv` (metric = roc_auc, delta_mean, delta_ci95_*).
- Rates are configuration-specific (support cost: 28/30, 28/30, 17/30, 16/30; TOPSIS rule: 29, 28, 20, 29).
- Wording allowed: "on BNP the four picks are within holdout noise of each other, so ranking agreement of 15/30 measures near-tie instability rather than selection optimism".

---

## C10 — R = 10 → R = 30 robustness — SECONDARY

| Quantity | cells with R10 inside the R30 95% interval | largest move |
|---|---|---|
| HV ratio, NBI-A/B/C (12) | 9/12 (outside: Santander A 0.846 → 0.800, Santander C 0.991 → 0.987, BNP B 0.966 → 0.974) | Santander A −0.046; Porto B 0.820 → 0.866 (inside) |
| IGD+, NBI-A/B/C (12) | 7/12 (Santander A, B, C upward; BNP B 0.0138 → 0.0114; Porto A 0.0751 → 0.0911) | Santander C 0.0069 → 0.0105 (+52%) |
| Joint-ND fraction (12) | 12/12 | — |
| Gate pass (8) | 8/8 | BNP log-loss 0.20 → 0.10 |
| External R² (8) | 7/8 | BNP log-loss 0.09 → −0.10 |
| Top β means (24) / sign freq (24) | 24/24 / 24/24 | ≤ 6% |
| NBI success rates (12) | 11/12 | UCI A 0.46 → 0.39 |
| Reference sizes/displacement (12) | 9/12 | Porto displacement −21% |
| Holdout − OOF AUC (16) | 16/16 | ≤ 0.003 |
| Runtime (4) | 3/4 | ≤ 4% |
| Primary effects: median sign (32) | 32/32 identical | Porto/UCI B − A from ≈ 50% to 80% wins |

- Source: `statistics/r10_vs_r30_stability.csv`; manuscript Table 4, Fig. 6; benchmark `r30_fig03_r10_vs_r30_stability.png`.
- Caveat: nested criterion (partitions 0–9 are one third of the R = 30 sample); against partitions 10–29 alone only 6/12 HV and 5/12 IGD+ cells pass; the Santander IGD+ level and the Porto/UCI anchor win rates differ between batches under an identical protocol — the partition sensitivity the extension was designed to expose.
- Wording allowed: "no directional conclusion reversed; R = 30 resolved the Porto/UCI anchor effect, quantified the BNP collapse rate (7/30) and exposed two artifacts".

---

## C11 — Empirical Pareto reference quality and its self-grading — SECONDARY (infrastructure)

**Wording rule (added 2026-09-11).** The reference has two layers and the manuscript must never conflate them.
(a) The **sampled core** — ≥ 10⁵ Dirichlet points, lattice, vertices, edges and the ε-constraint sweep — is
constructed independently of the surrogate and of every candidate method. (b) The **final augmented reference** used
for scoring is the non-dominated union of that core with *every* candidate set. Permitted: "an empirical reference
whose sampled core is constructed independently of the surrogate and is subsequently augmented with all candidate
sets". **Forbidden:** "an independent reference", "a reference the methods cannot influence", "independently
constructed reference", or any phrasing implying the *final* reference is method-independent. The self-grading
caveat (NBI-B and NBI-C contribute 30–40% of the final front) must accompany any absolute indicator value.


- Median 100,564 points per partition; median displacement by an independent 20k check 2.2–3.9%; 116/120 partitions within the 5% tolerance (max 6.0% at the 3-round cap). Median sample-front sizes: weighted 135 / 120 / 94 / 481, support 40 / 39 / 14 / 87. Final reference = non-dominated union of sample and every candidate set (NBI-B/C contribute 30–40% of points). HV effect of residual displacement ≤ 0.2% (HV ratio ≥ 0.998). Source: `tables/reference_diagnostics.csv`, FINAL_REPORT §5.
- Wording NOT allowed: "true Pareto front".

## C12 — NBI outcome rates and failure modes — SECONDARY / EXPLORATORY

- Certified/feasible fraction (mean over 30): A 0.98 / 0.95 / 0.58 / 0.39; B 0.95 / 1.00 / 1.00 / 0.48; C 0.83 / 0.83 / 0.98 / 0.98 (Santander / BNP / Porto / UCI). Source: `tables/nbi_runs.csv`.
- Porto NBI-B low-HV regime (6/30 with HV < 0.8) with 100% subproblem certification → CHIM geometry (anchor configuration), not solver failure (EXPLORATORY, `statistics/bimodality_regimes_r30.csv`).
- UCI A/B certification bimodal (0.26–0.52 in most, 0.95–1.00 in 3) — vertex-β subproblems infeasible with the simplex parameterization.
- Semantics differ: A/B = SLSQP-certified on smooth surrogates; C = equality-feasible (residual < 10⁻³) under the lenient rule with a reduced budget (2 starts, maxiter 120, fd-eps 10⁻³) on a piecewise-constant AUC.

## C13 — Base models and single-objective references — SECONDARY (context)

- Mean OOF AUC per model (`tables/model_performance.csv`): Santander LR 0.859, GNB 0.888, kNN 0.725, RF 0.845, XGB 0.881; BNP 0.735 / 0.603 / 0.678 / 0.743 / 0.748; Porto 0.625 / 0.588 / 0.567 / 0.621 / 0.592; UCI 0.724 / 0.658 / 0.757 / 0.783 / 0.765. kNN costs 42–320× the cheapest model (24–246 ms/1k).
- SLSQP log-loss optimum OOF AUC / log-loss: 0.8917 / 0.2088; 0.7539 / 0.4689; 0.6311 / 0.1525; 0.7836 / 0.4268. Best single: 0.8880; 0.7482; 0.6251; 0.7829. Uniform: 0.8893; 0.7358; 0.6151; 0.7671. Stacking (logistic): 0.8868; 0.7540; 0.6239; 0.7826 (`tables/single_objective_refs.csv`).
- Scheffé-surface optima are below the direct optima on every dataset (e.g., Santander AUC 0.8902 vs 0.8923) and 30–50× more expensive on Santander (kNN-bearing).

## C14 — Execution provenance — SECONDARY

- 120/120 partitions; 3,600 model fits; 19,920 design evaluations; 23,760 NBI subproblems; 50.2 M real-objective evaluations (NBI-C); 15.5 M reference points + 2.4 M check points; 4.8 M direct-AUC-search evaluations; 81.1 h cumulative stage time (R = 10: 27.5 h; extension 53.6 h); 2 retries total (Porto rep 0 comparators, UCI rep 0 reference), 0 in partitions 10–29; 68 tests. Source: `summary.json` (counts, runtime), `manifests/benchmark_manifest.json`.
- Environment: Python 3.11.15, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0, statsmodels 0.15.0, xgboost 3.2.0, macOS arm64 (Apple M4 Max), 8 worker threads.

---

## Numerical claim audit (run 2026-09-12 on the compiled PDF)

`audit_numbers.py` extracts every quantitative sentence from the compiled manuscript and checks each number against
the union of values in the frozen R = 30 tables, the statistics outputs, the NSGA-II tables, `summary.json`, the
manifests and the NSGA-II configuration — 136,940 distinct source values.

| Item | Result |
|---|---|
| Non-trivial numbers checked | 616 |
| x/y count claims checked | 150 |
| Flags after excluding citation years, documented seed offsets and W/T/L triples | **12** |
| Flags that are invented or untraceable numbers | **0** |

All twelve resolved:

- Nine are table cells that the PDF text extractor splits across columns — Table 5's NBI success triples
  ("0.98 / 0.95 / 0.83") read as "98/0", "95/1" and so on, and Table 1's train/holdout counts. They are not prose
  claims and each value is present in `tables/nbi_runs.csv`.
- `800` is derived in prose: 80 pending replications × 10 stages, stated in the reproducibility section.
- `829` is the megabyte size of the unversioned raw artifact tree, stated in the reproducibility section.
- `0/22` is the tail of "7/7 against 0/22 of the linear-selected ones", a genuine count verified in
  `tables/scheffe_orders.csv` (7 quadratic-selected collapse partitions, 22 linear-selected non-collapse).

The audit is a screen for transcription drift and invented figures. It cannot detect a real number attached to the
wrong claim; that is what this document's per-claim mapping is for.

## Claim blacklist (must not appear as positive claims)

1. NBI produces more uniformly spaced real fronts (spacing percentiles: NBI-C 0.99 / 0.54 / 0.96 / 0.41 vs random 1.00 — not supported).
2. The reliability gate predicts surrogate-NBI failure (within-dataset it does not; C5).
3. Large β_ij identifies classifier complementarity (C6).
4. NBI-C beats budget-matched random search (C4).
5. Any inference over 120 independent experiments.
6. BNP support-cost bimodality as a mechanism (C8).
7. Santander deployment-cost gap attributed to the AUC optimum (C7).
8. NBI replaces SLSQP for the mono-objective log-loss problem (SLSQP is the direct reference; C13).
9. The present paper introduced DoE + RSM + NBI (inherited from Pereira et al., 2025; see research_lineage.md).
