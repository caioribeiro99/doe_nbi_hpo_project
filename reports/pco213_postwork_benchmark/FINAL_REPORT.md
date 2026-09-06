# DoE + Scheffé RSM + NBI for multiobjective ensemble-weight optimization — multi-dataset replicated benchmark

**Final research report** · run 2026-09-04 21:38 → 2026-09-06 01:10 (America/Sao_Paulo) · repository `caioribeiro99/doe_nbi_hpo_project`, branch `pco213-classification-mixture-ensemble`

Machine-readable basis: `summary.json`, `tables/*.csv`, `manifests/*.json` (this folder); figures in `figures/pco213_postwork_benchmark/`. Raw per-replication artifacts (OOF matrices, reference samples, candidate tables, stage status) are under `experiments/pco213_postwork_benchmark/` and are not versioned.

---

## 1. Provenance

| Item | Value |
|---|---|
| Runner commit (recorded in the manifest) | `6fa7877` |
| Commits created in this work | `c2e37c4` benchmark runner + modules + tests · `f1249ff` tmux launch/resume script · `46af91a` aggregation/figures script · `6fa7877` fix: `minimize_on_simplex` returns the best feasible point at vertex optima · `2a8961b` aggregated results, figures, manifests (+ this report's commit) |
| Base | `a771cff` (failed NBI subproblems reported instead of crashing), on top of `3791683` |
| Tests | 67 passed (`tests/mixens`, 10 new) |
| Environment | Python 3.11.15; numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0, statsmodels 0.15.0, xgboost 3.2.0; macOS arm64, Apple M4 Max, 14 cores, 36 GB |
| Git state at the end | working tree clean; all commits pushed to `origin/pco213-classification-mixture-ensemble` |

### Datasets (all invariant-validated at load; Kaggle test labels never used; raw files not committed)

| Dataset | Source / acquisition | SHA-256 of the raw train file | Rows used × features | Prevalence | Missing |
|---|---|---|---|---|---|
| santander | Kaggle competition, public mirror (existing loader cascade) | `704545ba…6596e` | 200,000 × 200 numeric | 0.1005 | 0 |
| bnp | Kaggle competition; competition download endpoint returned HTTP 403 for this account, so the file was taken from the public dataset mirror `hjimbean/kaggle-classification-autofe-benchmark` (same 114,321 × 133 layout, string categoricals, v22 with 18,210 levels) | `d3e0b615…d37c5` | 114,321 × 131 (112 numeric, 19 categorical) | 0.7612 | 0.34 |
| porto | Kaggle competition; public mirror `pushero/porto-seguros-safe-driver-prediction-train-data` (train.csv byte size identical to the official file, 115,852,544 B) | `7115d4f9…edf13` | 200,000 of 595,212 (stratified cap, seed 20260904) × 57 (26 numeric, 14 categorical, 17 binary; −1 → NaN for numeric) | 0.0365 | 0.004 |
| uci_credit | UCI ML repository #350 (xls) | `30c6be3a…0933` | 30,000 × 23 (20 numeric, 3 categorical) | 0.2212 | 0 |

Full manifests (columns, preprocessing decisions, model parameters, seeds, timings, status per stage) are in `manifests/`.

### Design of the experiment

- Outer replication: R = 10 independent stratified 80/20 partitions per dataset (seed = 20260904 + r); inner 5-fold stratified OOF on the training side; holdout used only for confirmation.
- Model zoo (fixed, no HPO): lr, gnb, knn(k=100, brute), rf(200 trees), xgb(400 trees); fold-safe `ColumnTransformer` preprocessing (median imputation + indicator, scaling for lr/knn, one-hot with 0.5% minimum frequency for lr/gnb/knn, ordinal codes for rf/xgb).
- Inference cost per model: median of 5 timed `predict_proba` calls on a fixed 10k-row holdout batch (preprocessing included), ms per 1k predictions.
- Mixture DoE: 66-run controlled design ({5,3} lattice ∪ {5,2} midpoints ∪ centroid ∪ 5 axial ∪ 5 quaternary centroids ∪ 10 interior centroid–ternary midpoints) + 100 independent Dirichlet validation points (60 Dir(1), 40 Dir(0.5)) never used in fitting.
- Response surfaces: Scheffé linear, quadratic, special cubic for ROC-AUC, log-loss, Brier, PR-AUC; selection = lowest order whose external RMSE is within 10% of the best; reliability gate = external R² ≥ 0.5 and Spearman ≥ 0.9 on the unseen points.
- Single-objective references: best single model, uniform voting, logistic stacking, direct SLSQP log-loss optimum (exact, convex), direct AUC search (40k Dirichlet(1)/(0.3) points + design + Nelder–Mead polish), Scheffé optima, real anchors.
- Empirical Pareto reference: ≥ 100k Dirichlet points (50% α = 1, 50% α = 0.3, shuffled) + vertices, centroid, 190 edge points, {5,6} lattice interior points, design, validation, references, and an exact ε-constraint sweep (40 caps) for log-loss × weighted cost; another 100k added while an independent 20k check displaced > 5% of the front (max 3 rounds). Final empirical reference = non-dominated union of the sample and every candidate set. Both cost definitions: weighted Σwᵢcᵢ and support Σ_{wᵢ>10⁻³} cᵢ.
- NBI: 66-β lattice ({3,10}), objectives (−AUC, log-loss, weighted cost), normalized by anchor payoff. A: Scheffé surfaces + surrogate anchors; B: same surfaces + real anchors (direct AUC optimum, SLSQP log-loss optimum, cheapest vertex); C: metamodel-free on the cached OOF (rank-based AUC, exact log-loss, SLSQP with finite-difference step 10⁻³, feasible-iterate acceptance). Vertex β's return their anchor; other subproblems warm-start at the CHIM pre-image.
- Comparators: random weighted scalarization on the surfaces with real anchors (66 λ), budget-matched random Dirichlet search (66 points), the 66 DoE runs, the single-objective references.
- Every candidate revalidated on the real cached OOF (AUC, log-loss, Brier, PR-AUC, both costs, N_eff, entropy, support), Pareto-filtered again on real objectives; quality (GD, IGD, IGD+, spacing, spacing-CV, hypervolume ratio with reference point 1.1³, joint non-dominated fraction over all valid candidates, coverage, extreme recovery, size-matched spacing percentile) computed only on real objectives with a common normalization from the empirical reference front.

---

## 2. Execution summary

| Item | Value |
|---|---|
| Completed replications | 40 / 40 (10 per dataset); no reduction of R was necessary (pilot-pass projection 1.16 days for R = 10, ceiling 5 days) |
| Total stage time | 27.53 h (wall clock 27.5 h) |
| Per dataset | santander 10.57 h · porto 9.64 h · bnp 5.50 h · uci_credit 1.82 h |
| Per stage | metamodel-free NBI-C 19.01 h · single-objective references 3.30 h · empirical reference 2.20 h · OOF 1.39 h · NBI-A 0.73 h · NBI-B 0.52 h · quality 0.29 h · rest < 0.1 h |
| Model fits | 1,200 (25 fold fits + 5 refits per replication) |
| DoE evaluations | 6,640 (66 design + 100 validation per replication) |
| NBI subproblems | 7,920 (66 × 3 variants × 40) |
| Real-objective evaluations in NBI-C | 17,004,670 (≈ 425k per replication) |
| Direct AUC-search evaluations | 1,614,624 |
| Reference points evaluated | 5,822,560 (100,564 per replication per round; 1.3–1.6 rounds on average) plus 800,000 independent check-set evaluations |
| Failed / retried stages | 2 retries, 0 unresolved: `porto/rep_00/comparators` (a solver-robustness defect, fixed in `6fa7877` and retried on resume) and `uci_credit/rep_00/reference` (interrupted by the restart, recomputed) |

---

## 3. Model-level performance (mean over 10 replications; OOF AUC / holdout AUC / OOF log-loss / cost ms per 1k)

| Dataset | lr | gnb | knn | rf | xgb |
|---|---|---|---|---|---|
| santander | 0.8592 / 0.8600 / 0.2324 / 0.72 | **0.8878** / 0.8812 / 0.2112 / 0.78 | 0.7253 / 0.7262 / 0.3512 / 240.3 | 0.8448 / 0.8463 / 0.2752 / 3.75 | 0.8808 / 0.8839 / 0.2202 / 1.05 |
| bnp | 0.7345 / 0.7348 / 0.4844 / 3.35 | 0.5989 / 0.6007 / 3.829 / 3.72 | 0.6780 / 0.6796 / 0.5126 / 150.2 | 0.7425 / 0.7453 / 0.4821 / 6.18 | **0.7481** / 0.7533 / 0.4740 / 3.49 |
| porto | **0.6244** / 0.6297 / 0.1529 / 1.38 | 0.5896 / 0.5922 / 1.884 / 1.58 | 0.5668 / 0.5632 / 0.1610 / 232.3 | 0.6194 / 0.6241 / 0.1533 / 4.32 | 0.5906 / 0.6008 / 0.1589 / 1.64 |
| uci_credit | 0.7235 / 0.7248 / 0.4656 / 0.51 | 0.6602 / 0.6705 / 1.134 / 0.57 | 0.7568 / 0.7605 / 0.4451 / 23.9 | **0.7830** / 0.7831 / 0.4274 / 5.76 | 0.7655 / 0.7652 / 0.4553 / 0.85 |

kNN costs 36–330× the cheapest model on every dataset; GNB is severely over-confident on BNP, Porto and UCI (log-loss 1.1–3.8).

---

## 4. Scheffé response surfaces (selected order, unseen Dirichlet points; medians over replications)

| Dataset | Response | Selected order (freq.) | Reliable | R²ext median (min) | rel-RMSE | Spearman | R²ext by order lin / quad / sp.cubic |
|---|---|---|---|---|---|---|---|
| santander | ROC-AUC | quadratic 8, linear 2 | 0 / 10 | −0.40 (−0.63) | 0.185 | 0.818 | −0.94 / −0.30 / −0.34 |
| santander | log-loss | quadratic 9, sp.cubic 1 | 10 / 10 | 0.973 (0.962) | 0.033 | 0.994 | 0.82 / 0.97 / 0.97 |
| bnp | ROC-AUC | linear 7, quadratic 2, sp.cubic 1 | 2 / 10 | 0.03 (−1.27) | 0.198 | 0.908 | −0.02 / −0.27 / −0.35 |
| bnp | log-loss | linear 7, sp.cubic 2, quadratic 1 | 2 / 10 | 0.08 (−0.47) | 0.155 | 0.963 | −0.04 / −0.44 / −0.18 |
| porto | ROC-AUC | quadratic 10 | 6 / 10 | 0.52 (0.35) | 0.127 | 0.912 | 0.24 / 0.52 / 0.57 |
| porto | log-loss | linear 7, sp.cubic 2, quadratic 1 | 5 / 10 | 0.62 (0.43) | 0.106 | 0.996 | 0.53 / 0.46 / 0.50 |
| uci_credit | ROC-AUC | quadratic 5, sp.cubic 5 | 10 / 10 | 0.994 (0.982) | 0.016 | 0.996 | 0.89 / 0.99 / 0.99 |
| uci_credit | log-loss | quadratic 8, sp.cubic 2 | 10 / 10 | 0.995 (0.985) | 0.014 | 0.995 | 0.85 / 0.99 / 0.995 |
| all | Brier | quadratic 10 | 10 / 10 | 1.000 | 0.000 | 1.000 | exact (algebraic identity, sanity check only) |
| santander / bnp / porto / uci | PR-AUC | — | 0 / 3 / 8 / 10 | −0.38 / 0.44 / 0.77 / 0.95 | | | |

Reading: the surrogate is dataset- and metric-dependent, tracking the heterogeneity of the model pool. The extremes are unambiguous: on UCI every response is captured to R²ext ≈ 0.99; on Santander log-loss is captured (0.97) but AUC is not (negative R²ext at every order; R² < 0 arises because the surface is worse than the mean on a response whose range across the simplex is only ≈ 0.04 AUC, absolute RMSE ≈ 0.008; the surface is qualitatively wrong along the kNN direction, as diagnosed in the first post-work run); on BNP neither AUC nor log-loss is captured by any order (higher orders are worse externally, over-fitting a surface with an exponential-like blow-up near the over-confident GNB vertex, whose OOF log-loss is 3.97). The intermediate cells are borderline and validation-draw dependent: Porto AUC values straddle the 0.5 threshold (0.35–0.70), and on Porto/BNP log-loss the positive R²ext of the (mostly linear) selected surface is carried by the few validation points nearest the GNB vertex; three of Porto's five log-loss failures pass R² and fail only the Spearman gate. Extrapolation excess on the validation points is ≈ 0 everywhere, so failures are misspecification/interior error, not range extrapolation. The Brier row is a fitting-code sanity check, not evidence of surrogate quality.

### Interaction coefficients (quadratic fit, all replications)

Sign frequency is 1.0 or 0.0 for the four largest interactions in every dataset × response, and the top pair is the same in 10/10 replications (9/10 cases; Porto AUC 7/10). The dominant pairs differ across datasets:

| Dataset | Largest AUC interactions β_ij (mean ± sd) | Largest log-loss interactions |
|---|---|---|
| santander | gnb·knn +0.242±0.003, knn·xgb +0.231, knn·rf +0.199, lr·knn +0.196 | gnb·knn −0.174±0.001, lr·knn −0.147, knn·xgb −0.136 |
| bnp | gnb·xgb +0.157±0.023, gnb·rf +0.147, lr·gnb +0.141, knn·xgb +0.113 | gnb·xgb −4.02±0.24, gnb·rf −3.99, gnb·knn −3.99, lr·gnb −3.97 |
| porto | gnb·knn +0.071±0.008, knn·xgb +0.068, lr·knn +0.062, knn·rf +0.052 | gnb·knn −1.93±0.09, gnb·xgb −1.92, gnb·rf −1.91 |
| uci_credit | gnb·xgb +0.129±0.008, gnb·rf +0.101, lr·xgb +0.088, lr·gnb +0.077 | gnb·xgb −0.73±0.12, gnb·knn −0.70, gnb·rf −0.68 |

In every dataset the largest positive AUC interactions involve the weakest classifier by OOF AUC (kNN on Santander/Porto, GNB on BNP/UCI; on Porto the single largest term is gnb·knn in 7/10 replications and knn·xgb in 3/10), and the largest negative log-loss interactions involve GNB (the over-confident one). Between-replication CV of these coefficients ranges from 0.006 (Santander) to 0.18 (UCI log-loss, BNP AUC), median 0.06; three near-zero terms outside the top four flip sign. The coefficients are taken from the quadratic fit in every replication irrespective of the selected order, and the design is identical across replications, so this is stability under data resampling and does not imply surface adequacy. The interpretation must be the Scheffé one: β_ij is the edge-midpoint gain over the linear average of the two vertices, so a weak-strong pair whose blend tracks the stronger member has β_ij ≈ 2 × (vertex gap); Spearman(β_ij, vertex AUC gap) is 0.82 / 0.94 / 0.81 on Santander / BNP / UCI (0.10 on Porto, where all models lie within 0.06 AUC). On the real OOF predictions the 50/50 blend of the top-β pair is below its stronger member on Santander (−0.005), BNP (−0.017) and UCI (−0.020) in 10/10 replications; only Porto's gnb+knn blend is genuinely synergistic (+0.009). The pairs whose 50/50 blend beats both members are lr+rf (Santander), rf+xgb (BNP), gnb+xgb (Porto) and knn+xgb (UCI), none of which carries the largest β_ij.

---

## 5. Single-objective references (mean over replications; OOF AUC / OOF log-loss / holdout AUC / holdout log-loss / weighted cost)

| Dataset | best single | uniform | stacking | SLSQP log-loss | direct AUC | Scheffé LL optimum | Scheffé AUC optimum |
|---|---|---|---|---|---|---|---|
| santander | 0.8878/0.2112/0.8812/0.2169/0.78 | 0.8890/0.2269/0.8899/0.2260/49.3 | 0.8866/0.2154/0.8872/0.2156 | **0.8914/0.2090**/0.8889/0.2107/0.87 | 0.8921/0.2125/0.8909/0.2129/6.20 | 0.8900/0.2112/0.8859/0.2135/25.2 | 0.8897/0.2153/0.8857/0.2171/41.1 |
| bnp | 0.7481/0.4740/0.7533/0.4693/3.49 | 0.7364/0.5205/0.7384/0.5198/33.4 | 0.7539/0.4720/0.7565/0.4700 | **0.7538/0.4689**/0.7571/0.4667/3.87 | 0.7546/0.4699/0.7574/0.4681/4.40 | 0.7456/0.4873/0.7504/0.4836/3.50 | 0.7491/0.4739/0.7536/0.4703/7.84 |
| porto | 0.6244/0.1529/0.6297/0.1526/1.38 | 0.6149/0.1971/0.6192/0.1970/48.2 | 0.6228/0.1532/0.6277/0.1530 | **0.6302/0.1525**/0.6349/0.1523/14.4 | 0.6303/0.1525/0.6348/0.1523/17.7 | 0.6184/0.1647/0.6230/0.1644/3.87 | 0.6294/0.1526/0.6342/0.1523/2.43 |
| uci_credit | 0.7830/0.4274/0.7831/0.4274/5.76 | 0.7673/0.4644/0.7679/0.4636/6.33 | 0.7828/0.4284/0.7825/0.4288 | **0.7837/0.4267**/0.7834/0.4269/6.03 | 0.7840/0.4271/0.7840/0.4270/6.85 | 0.7833/0.4274/0.7832/0.4273/6.12 | 0.7839/0.4271/0.7841/0.4271/7.54 |

Weight stability of the real optima (mean ± sd over replications): Santander SLSQP = gnb 0.68±0.01, xgb 0.32±0.01; BNP SLSQP = xgb 0.63±0.02, lr 0.21±0.02, rf 0.15±0.01; Porto SLSQP = lr 0.54±0.04, rf 0.32±0.03, xgb 0.09±0.02, knn 0.05±0.02; UCI SLSQP = rf 0.78±0.04, xgb 0.16±0.02, knn 0.06±0.03. The direct-AUC optima have the same support with small shifts. The Scheffé-derived optima are worse than the direct optima on every dataset (AUC −0.001 to −0.012; log-loss +0.0003 to +0.018) and, on Santander, 30–50× more expensive because they carry kNN weight.

---

## 6. Empirical Pareto reference: convergence

| Dataset | Points (mean) | Rounds (mean, max) | Front displaced by the independent 20k check (mean, max) | Front size weighted / support (median) |
|---|---|---|---|---|
| santander | 150,564 | 1.5, 3 | 0.025, 0.060 | 142 / 46 |
| bnp | 130,564 | 1.3, 2 | 0.023, 0.042 | 122 / 41 |
| porto | 160,564 | 1.6, 3 | 0.034, 0.049 | 100 / 14 |
| uci_credit | 140,564 | 1.4, 2 | 0.036, 0.050 | 457 / 80 |

The references were extended until an independently drawn 20k-point check displaced no more than 5% of the weighted-cost sample front: 25 replications needed one 100k round, 12 two, 3 three; 39/40 met the tolerance and Santander replication 1 stopped at the 3-round cap at 5.97%. Three qualifications. The check applies to the weighted-cost front only; applying the same 20k check to the support-cost fronts displaces 0.3–9.1% on average (16/40 replications above 5%, max 20.5%). The 20k check is one fifth of the sample density and doubles as the stopping rule, so it understates how far a further 100k points move the front (about 2×: 5–23% of the weighted front, up to 36% of the support front in the multi-round replications). In objective space these displacements are immaterial for the indicators: the hypervolume ratio of a sample front to its union with the displacing points is ≥ 0.998 and the normalized IGD ≤ 0.011. The final reference is the non-dominated union with every candidate set (NBI-B and NBI-C together contribute 30–40% of its points, NBI-C alone 8–25%), so convergence and coverage indicators are partly self-graded; the 800,000 check-set evaluations are not included in the 5.8 M count. Contributors to the weighted-cost sample fronts: Dirichlet(0.3) points, the exact ε-constraint sweep and edge points.

---

## 7. Pareto quality vs the empirical reference (medians over 10 replications; IQR in `summary.json`)

### 7.1 Weighted (linear) cost

| Dataset | Set | n front | GD | IGD | IGD+ | HV ratio | joint-ND (all valid) | spacing CV | size-matched spacing pct |
|---|---|---|---|---|---|---|---|---|---|
| santander | NBI-A | 13 | 0.250 | 0.117 | 0.083 | 0.852 | 0.000 | 1.50 | 0.99 |
| santander | NBI-B | 19 | 0.011 | 0.053 | 0.007 | 0.983 | 0.047 | 1.41 | 0.78 |
| santander | NBI-C | 37 | 0.005 | 0.043 | 0.005 | **0.993** | 0.459 | 2.78 | 0.99 |
| santander | random scalarization | 54 | 0.001 | 0.215 | 0.007 | 0.978 | 0.720 | 2.23 | 0.00 |
| santander | random Dirichlet (66) | 5 | 2.38 | 0.433 | 0.361 | 0.562 | 0.000 | 0.93 | 1.00 |
| santander | DoE runs (66) | 7 | 0.001 | 0.093 | 0.007 | 0.982 | 0.061 | 1.07 | 0.64 |
| bnp | NBI-A | 8 | 0.002 | 0.190 | 0.028 | 0.917 | 0.083 | 0.81 | 0.49 |
| bnp | NBI-B | 44 | 0.000 | 0.060 | 0.014 | 0.968 | 0.583 | 3.06 | 0.99 |
| bnp | NBI-C | 34 | 0.009 | 0.027 | 0.014 | **0.983** | 0.336 | 0.92 | 0.54 |
| bnp | random scalarization | 2 | 0.084 | 0.445 | 0.295 | 0.493 | 0.000 | — | — |
| bnp | random Dirichlet (66) | 4 | 5.57 | 1.55 | 1.53 | 0.000 | 0.000 | 1.27 | 1.00 |
| bnp | DoE runs (66) | 6 | 0.084 | 0.170 | 0.031 | 0.916 | 0.030 | 1.04 | 0.84 |
| porto | NBI-A | 12 | 0.033 | 0.219 | 0.065 | 0.823 | 0.015 | 0.59 | 0.10 |
| porto | NBI-B | 56 | 0.015 | 0.071 | 0.048 | 0.883 | 0.432 | 2.61 | 0.99 |
| porto | NBI-C | 56 | 0.003 | 0.031 | 0.008 | **0.982** | 0.727 | 1.56 | 0.98 |
| porto | random scalarization | 14 | 0.035 | 0.312 | 0.083 | 0.786 | 0.076 | 2.20 | 0.87 |
| porto | random Dirichlet (66) | 5 | 56.1 | 1.76 | 1.76 | 0.000 | 0.000 | 1.37 | 1.00 |
| porto | DoE runs (66) | 3 | 0.082 | 0.355 | 0.159 | 0.640 | 0.008 | 0.79 | 0.39 |
| uci_credit | NBI-A | 16 | 0.009 | 0.199 | 0.170 | 0.669 | 0.292 | 1.64 | 0.46 |
| uci_credit | NBI-B | 28 | 0.003 | 0.180 | 0.157 | 0.688 | 0.491 | 2.06 | 0.40 |
| uci_credit | NBI-C | 63 | 0.001 | 0.031 | 0.009 | **0.977** | 0.785 | 2.19 | 0.45 |
| uci_credit | random scalarization | 55 | 0.001 | 0.133 | 0.008 | 0.980 | 0.720 | 1.51 | 0.01 |
| uci_credit | random Dirichlet (66) | 10 | 0.398 | 0.226 | 0.141 | 0.750 | 0.000 | 1.02 | 0.97 |
| uci_credit | DoE runs (66) | 8 | 0.012 | 0.083 | 0.033 | 0.920 | 0.045 | 1.06 | 0.78 |

IQR of the NBI variants (IGD+ · HV ratio · joint-ND): Santander A 0.062–0.135 · 0.80–0.89 · 0.00–0.01; B 0.006–0.010 · 0.981–0.984 · 0.04–0.05; C 0.004–0.009 · 0.988–0.994 · 0.37–0.52. BNP A 0.026–0.244 · 0.64–0.92 · 0.03–0.09; B 0.012–0.015 · 0.962–0.971 · 0.50–0.60; C 0.012–0.015 · 0.980–0.984 · 0.33–0.35. Porto A 0.036–0.109 · 0.72–0.90 · 0.00–0.04; B 0.032–0.053 · 0.86–0.92 · 0.30–0.53; C 0.007–0.010 · 0.980–0.983 · 0.46–0.84. UCI A 0.117–0.244 · 0.54–0.79 · 0.22–0.40; B 0.125–0.234 · 0.57–0.78 · 0.40–0.60; C 0.007–0.015 · 0.967–0.983 · 0.75–0.83.

### 7.2 Support (deployment) cost

| Dataset | NBI-A IGD+ / HV / jND | NBI-B | NBI-C | random scalarization | random Dirichlet | DoE runs |
|---|---|---|---|---|---|---|
| santander | 0.063 / 0.860 / 0.00 | 0.009 / 0.985 / 0.03 | **0.002 / 0.992 / 0.46** | 0.010 / 0.985 / 0.00 | 0.608 / 0.078 / 0 | 0.008 / 0.988 / 0.03 |
| bnp | 0.041 / 0.929 / 0.00 | 0.021 / 0.843 / 0.11 | 0.059 / 0.663 / 0.20 | 0.353 / 0.566 / 0.00 | 7.82 / 0.000 / 0 | **0.038 / 0.965** / 0.03 |
| porto | 0.066 / 0.870 / 0.00 | 0.018 / 0.845 / 0.02 | **0.008 / 0.932 / 0.14** | 0.081 / 0.852 / 0.02 | 1.36 / 0.000 / 0 | 0.172 / 0.700 / 0.00 |
| uci_credit | 0.042 / 0.917 / 0.05 | 0.051 / 0.882 / 0.25 | 0.042 / 0.938 / 0.17 | **0.006 / 0.982** / 0.12 | 0.546 / 0.098 / 0 | 0.012 / 0.974 / 0.02 |

### 7.3 Paired differences across replications (weighted cost; mean, fraction of replications with the sign shown)

| Dataset | B − A IGD+ (frac < 0) | B − A HV (frac > 0) | C − B IGD+ (frac < 0) | C − B HV (frac > 0) | C − B joint-ND (frac > 0) | B − random Dirichlet HV | B − random scalarization HV |
|---|---|---|---|---|---|---|---|
| santander | −0.087 (10/10) | +0.136 (10/10) | −0.002 (10/10) | +0.009 (10/10) | +0.39 (10/10) | +0.53 (10/10) | +0.004 (8/10) |
| bnp | −0.207 (10/10) | +0.249 (10/10) | +0.0005 (4/10) | +0.015 (9/10) | −0.17 (1/10) | +0.94 (10/10) | +0.46 (10/10) |
| porto | −0.006 (6/10) | +0.017 (6/10) | −0.058 (10/10) | +0.152 (10/10) | +0.27 (8/10) | +0.82 (10/10) | +0.02 (6/10) |
| uci_credit | −0.005 (5/10) | +0.009 (5/10) | −0.170 (9/10) | +0.305 (10/10) | +0.29 (10/10) | −0.07 (5/10) | −0.31 (1/10) |

### 7.4 NBI subproblem outcome and cost

| Dataset | A certified | B certified | C feasible | A / B / C seconds per replication | C / B compute ratio |
|---|---|---|---|---|---|
| santander | 0.96 | 0.93 | 0.83 | 38 / 38 / 2,613 | 69× |
| bnp | 0.96 | 0.99 | 0.83 | 25 / 17 / 1,432 | 84× |
| porto | 0.56 | 1.00 | 0.97 | 86 / 33 / 2,486 | 76× |
| uci_credit | 0.46 | 0.51 | 0.97 | 111 / 94 / 311 | 3.3× |

The two columns do not measure the same thing. For A and B a subproblem counts when SLSQP certifies convergence and the NBI equality residual is < 10⁻³; for C, whose real AUC objective is piecewise constant so SLSQP essentially never certifies (on UCI replication 0 all 61 non-vertex successes exit on the iteration limit), a subproblem counts when the equality is satisfied to < 10⁻³ within the iteration budget (`accept_feasible`); under the A/B rule C would score 4/66 there. C also ran at a reduced optimizer budget (2 starts, 120 iterations, finite-difference gradients, versus 10 starts and 300 iterations for A/B), so the compute ratios are configuration-dependent. Three of the 66 β's are anchor vertices counted without solver work. The UCI rates are bimodal (26–52% in eight replications, 95–100% in two; medians 35% / 42%): on the Scheffé surfaces the NBI equality is infeasible for interior β's with β_cost ≈ 0.3–0.6 (global minimum residual 0.01–0.07 even allowing t < 0), a CHIM/front-geometry effect that the reliability gate does not capture. Surface reliability is therefore neither sufficient (UCI) nor necessary (Santander AUC unreliable in 10/10, A/B 93–96%) for NBI subproblem success, and anchor placement matters independently (Porto A 56% vs B 100% on identical surfaces).

### 7.5 Holdout confirmation of the knee picks (weighted cost; mean over replications)

Mean holdout-minus-OOF AUC for the knee point of NBI-A/B/C and random scalarization is within ±0.005 on every dataset (Santander −0.001 to −0.003, BNP +0.002 to +0.003, Porto up to +0.0051, UCI ±0.001; between-replication sd ≤ 0.011, individual replications up to +0.023 on Porto) and mean log-loss within ±0.003. The shift is shared with the individual base models on the same partitions (correlation 0.87–0.96 across replications), so it reflects the partition and the full-training refit rather than selection optimism: the OOF-selected weights transfer to the holdout. This is level agreement, not ranking agreement; holdout and OOF name the same best of the four knee picks in 5/10 (BNP), 8/10 (Porto) and 10/10 (Santander, UCI) replications, with between-pick gaps of the order of the replication noise. Holdout labels enter only these confirmation metrics; 10k unlabelled holdout feature rows were also used as the inference-cost timing batch. The knee picks themselves: Santander NBI-C AUC 0.8919 / log-loss 0.2098 at 1.2 ms per 1k (holdout 0.8905 / 0.2106); BNP NBI-B 0.7509 / 0.4716 at 3.4 ms; Porto NBI-C 0.6297 / 0.1526 at 2.4 ms; UCI NBI-B 0.7766 / 0.4324 at 3.5 ms.

---

## 8. Fronts: composition, faces and costs

Active-model frequency on the weighted-cost empirical reference fronts (fraction of front points with wᵢ > 10⁻³): Santander gnb 0.97, xgb 0.70, lr 0.40, rf 0.24, knn 0.12; BNP lr 1.00, xgb 0.96, rf 0.54, gnb 0.49, knn 0.003; Porto lr 1.00, xgb 0.94, rf 0.81, knn 0.39, gnb 0.21; UCI lr 0.99, xgb 0.97, rf 0.70, gnb 0.60, knn 0.23.

Support-cost fronts (dominant active sets, share of front points): Santander gnb+knn+rf+xgb 0.35, gnb+xgb 0.24, gnb+rf+xgb 0.23; BNP lr+rf+xgb 0.35, rf+xgb 0.17, lr+gnb+rf+xgb 0.15; Porto lr+knn+rf+xgb 0.35, lr 0.15, lr+rf 0.09, lr+xgb 0.09; UCI lr+knn+rf+xgb 0.17, lr+rf 0.14, lr+xgb 0.12. kNN is active on 38%, 1%, 44% and 34% of the support-front points (Santander, BNP, Porto, UCI): it survives only as the last, expensive AUC increment, never in the cheap or the log-loss-optimal region.

AUC-vs-log-loss conflict (direct-AUC optimum minus SLSQP optimum, OOF, mean over replications): ΔAUC +0.00068 (Santander, sd 0.00005), +0.00082 (BNP), +0.00006 (Porto), +0.00030 (UCI) against Δlog-loss +0.0035, +0.0009, +0.00001, +0.0004 and Δweighted-cost +5.3, +0.5, +3.3, +0.8 ms per 1k.

Diversity vs interaction strength (pair-level means, n = 10 pairs per dataset; replications are near-identical): Spearman(β_ij^AUC, pairwise correlation of OOF soft errors |p − y|) = −0.52 (Santander, p = 0.13), −0.83 (BNP, p = 0.005), −0.13 (Porto), −0.79 (UCI, p = 0.01); Spearman(β_ij^log-loss, error correlation) = 0.65 / 0.93 / 0.90 / 0.89 (all p ≤ 0.05). The AUC association is not driven by the weakest-classifier pairs: it is stronger among the six pairs of stronger models (−0.83 to −0.94 on Santander/BNP/UCI) and weak or reversed among the four pairs containing the weakest model, and it is largely explained by the pair's vertex-AUC gap (partial Spearman given the gap −0.1 to −0.4). The log-loss association is a two-cluster contrast between GNB pairs (soft-error correlation near 0 or negative because 75–93% of GNB's probabilities lie beyond 0.01/0.99; β −0.7 to −4.0) and all other pairs; soft-error correlation saturates near 1 on imbalanced data for any pair of low-probability predictors (Porto non-GNB pairs 0.985–0.997), so it measures calibration and vertex-quality asymmetry more than instance-level diversity.

---

## 9. Cross-dataset answers

**Q1. Are Scheffé pair interactions reproducible across datasets?** Within a dataset, yes: the four largest interactions have sign frequency exactly 1.0 or 0.0 in all 16 dataset × response cells, and the top pair is identical in 77 of 80 dataset × response × replication cases (Porto AUC alternates between two kNN pairs). Across datasets, the magnitudes and the dominant pairs are not reproducible: the top AUC pair is gnb·knn on Santander and Porto and gnb·xgb on BNP and UCI, and the log-loss interaction scale spans 0.17 (Santander) to 4.0 (BNP), driven by GNB's over-confidence. Stability here is under data resampling with a fixed design; it does not imply that the quadratic surface is adequate (it fails the gate for AUC in 10/10 Santander and 5/10 BNP replications).

**Q2. Do the same classifier pairs exhibit complementarity repeatedly?** The pairs that repeat are those containing the dataset's weakest (AUC) or most over-confident (log-loss) classifier, kNN or GNB, and their large β_ij are the mechanical Scheffé consequence of a poor vertex (β_ij ≈ 2 × vertex gap), not complementarity that survives in the interior: the 50/50 blend of the top-β pair is worse than its stronger member on Santander, BNP and UCI in every replication, kNN carries ≤ 0.07 mean weight in every real optimum, and GNB carries 0 outside Santander. The pairs whose blend genuinely beats both members differ by dataset (lr+rf, rf+xgb, gnb+xgb, knn+xgb on Santander, BNP, Porto, UCI) and are not the largest-β pairs, so the cross-dataset difference reflects which classifier is weakest rather than a reproducible synergy structure.

**Q3. Do Pareto solutions collapse onto a small simplex face or remain structurally diverse?** They concentrate on 2–4-model faces: {gnb, xgb} (+rf) on Santander, {lr, rf, xgb} on BNP, {lr, rf, xgb} (+knn at the expensive end) on Porto, {lr, rf, xgb} on UCI. Structural diversity exists only along the cost axis (cheap single-model corners, e.g. pure lr on Porto/UCI), not on the quality plateau.

**Q4. Is the AUC-vs-log-loss conflict practically meaningful?** Along the AUC axis, no: the direct-AUC optimum (verified to be the maximum-AUC point of the ≥ 100k reference sample and of every candidate set, to 4 × 10⁻⁵) exceeds the exact SLSQP log-loss optimum by only +0.00006 to +0.00082 AUC (sd across replications ≤ 0.0001), below the between-replication sd of the AUC level itself (0.0006–0.003) and 5–100× smaller than the ensembling gain over the best single model; the whole bi-objective (AUC, log-loss) front of the reference spans exactly this extent. Two qualifications. On Santander the log-loss side is not negligible: the +0.0035 sacrifice is 1.7% of log-loss, 15% of the simplex log-loss range, and exceeds the entire ensembling gain in log-loss over the best single model (0.0022), so the AUC optimum there is worse calibrated than GNB alone. And the degeneracy is in objective space, not in weight or cost space: the two optima are distinct weight vectors (mean L1 distance 0.36 on Santander/BNP) and the AUC optimum is markedly more expensive (Santander weighted cost 6.2 vs 0.9 ms/1k). The three-objective problem is therefore effectively quality-vs-cost, with the AUC axis adding a cost, not a benefit.

**Q5. Does the cost objective create genuinely different solutions?** Yes. The cost range on the fronts spans 0.5–240 ms per 1k, the cheap end is single- or two-model (lr, lr+gnb, lr+xgb) and the quality end adds rf/xgb (and, at the extreme, kNN), so the fronts contain solutions that differ in support, not only in proportions. The weighted-cost fronts contain 100–457 points precisely because cost separates otherwise quality-equivalent blends.

**Q6. Does support-based cost change conclusions relative to weighted cost?** On BNP yes, elsewhere little. None of the NBI variants optimized the step cost (all minimize the linear weighted cost), so the support-cost tables measure how weighted-cost solutions transfer. By construction the step cost collapses each face's front to two dimensions: sample-front medians fall from 100–457 to 14–80 points (union fronts 178–542 → 29–107), each 66-point candidate set retains only 7–13 own-non-dominated points, and the all-valid joint-ND fraction is capped near 0.10–0.15, so NBI-B's fall from 0.43–0.58 to 0.02–0.11 on Porto/BNP must be read against that ceiling (on Porto NBI-B's HV and IGD+ do not worsen and NBI-C stays best in 8/10 replications). On BNP the method ranking genuinely flips: NBI-C's HV ratio drops from 0.98 to 0.55–0.72 in 6/10 replications (0.97 in the other 4) and the DoE runs (6/10) or the single-objective references (3/10) become the best set; the direction is robust to raising the activity threshold to 10⁻² (NBI-C median 0.56), although the median 0.66 at 10⁻³ is inflated in 4/10 replications by a single near-threshold kNN point that stretches the cost normalization ~10×. Santander and Porto conclusions are unchanged and UCI shifts weakly. kNN activity is not reduced by the step cost (1–44% of support-front points vs 0.3–39% on the weighted fronts). Deployment conclusions must be drawn on the support-cost front, and the support-cost hypervolume is fragile to the activity threshold; the linear cost is the smooth relaxation the optimizers need.

**Q7. How accurately does NBI approximate the empirical Pareto reference?** Metamodel-free NBI-C: median IGD+ 0.005–0.014, HV ratio 0.977–0.993, joint-ND 0.34–0.79, GD ≤ 0.009 on every dataset (weighted cost). Surrogate NBI with real anchors (B): IGD+ 0.007–0.014 and HV 0.97–0.98 where the log-loss surface is reliable (Santander, BNP), 0.048 / 0.88 on Porto and 0.157 / 0.69 on UCI. Surrogate NBI with surrogate anchors (A): HV 0.67–0.92, joint-ND ≤ 0.29.

**Q8. Does real-anchor NBI consistently outperform surrogate-anchor NBI?** Under weighted cost it never hurts on IGD+/HV and it helps in 10/10 replications on the two datasets whose surrogate anchors are misplaced, which coincide with the datasets failing the reliability gate: Santander median ΔIGD+ −0.076 and ΔHV +0.13 (means −0.087 / +0.14); BNP median −0.015 / +0.05, with means of −0.21 / +0.25 pulled by three replications (7, 8, 9) in which the surrogate anchors collapse (ΔIGD+ −0.30 to −0.89). The proximate driver is the surrogate AUC anchor landing at a far more expensive mixture (Santander 31–55 vs 2–17 ms/1k; BNP replications 7 and 9, 18–31 vs 4.4). On Porto and UCI, IGD+ and HV show no consistent direction (better in 4–5 of 10), although real anchors still improve subproblem success (Porto B 66/66 vs A 5–64), GD and joint-ND in 9–10 of 10 replications. Within BNP and Porto, per-replication surface reliability does not predict the gain, so this is a between-dataset association over four datasets. Under support cost the BNP HV gain holds in only 5/10 and UCI-B is worse in 8/10. Anchor construction is the first-order fix for surrogate NBI, and it is sufficient only when the surface itself is adequate.

**Q9. Does metamodel-free optimization materially outperform surrogate NBI?** Under weighted cost NBI-C is the best or statistically indistinguishable-from-best candidate set on convergence/coverage indicators on every dataset (median IGD+ 0.005–0.014, HV ratio 0.977–0.993). It is clearly best on Santander and Porto; on BNP, NBI-B is nominally ahead on IGD+ (0.0136 vs 0.0140, C wins 4/10) while C is ahead on HV (9/10); on UCI the surface-based random scalarization, which costs seconds, is nominally ahead on both IGD+ (0.0084 vs 0.0094) and HV (0.980 vs 0.977), C winning only 4–5 of 10. NBI-C beats NBI-B in HV ratio in ≥ 9/10 replications on all four datasets (Santander +0.009, BNP +0.015, Porto +0.15, UCI +0.31), robust to retaining B's unconverged candidates, and in IGD+ in 10/10, 4/10, 10/10 and 9/10. It has the worst spacing CV of the three variants on 3/4 datasets. The compute price per replication is 69× (Santander), 84× (BNP), 76× (Porto) and 3.3× (UCI) that of NBI-B, at ≈ 425k real evaluations, with C run at a reduced optimizer budget.

**Q10. When Scheffé fails, is it external R², extrapolation, or local misspecification?** Not extrapolation (validation points lie inside the design's predicted range, excess ≈ 0). It is misspecification of the polynomial form: on Santander AUC no order reaches positive R²ext (−0.94 / −0.30 / −0.34) and on BNP higher orders are worse than linear for both AUC and log-loss (over-fitting a surface with an exponential-like blow-up near the GNB vertex). The reliability gate caught these cases in 10/10 (Santander AUC) and 8/10 (BNP) replications, and they are exactly the cases where surrogate anchors fail (Q8).

**Q11. Does classifier error diversity correlate with β_ij strength?** In the expected direction, with the caveats of §8: with n = 10 pairs per dataset the AUC association is significant only on BNP (−0.83) and UCI (−0.79), null on Porto and non-significant on Santander, while the log-loss association is 0.65–0.93 everywhere. It is not a weakest-classifier confound (it is stronger among pairs of stronger models); it is largely explained by the vertex-quality gap of the pair, because β_ij is numerically the midpoint gain over the linear average of the two vertices, and for log-loss it is a GNB-versus-rest contrast. Diversity measured as soft-error correlation predicts interaction magnitude but not whether the interaction is exploitable, and on BNP and Porto the quadratic log-loss surface is unreliable in 10/10 replications, so the non-GNB β_ij there are not interpretable.

**Q12. Is NBI more uniform than budget-matched random search after real-objective validation?** Two separate answers. Quality: "budget-matched" means 66 candidate points (66 real evaluations for the random baseline versus ~230 for NBI-A, ~40k more for NBI-B's real-AUC anchor and ~425k for NBI-C). Under weighted cost, 66-point Dirichlet(1) search is far behind NBI-B and NBI-C on Santander, BNP and Porto (HV, IGD+, IGD, GD all 10/10; median HV 0.56 / 0.00 / 0.00 vs 0.97–0.99) and behind NBI-C on UCI (10/10), but not behind NBI-A/B on UCI, where random search has the higher median HV (0.75 vs 0.67/0.69) and better IGD+ because A/B return only 17–34 valid candidates; random search's joint-ND is 0 in 39/40 replications. The exact 0.00 hypervolumes on BNP and Porto are a reference-box effect of Dirichlet(1) with a 150–250 ms/1k kNN in the pool (every point carries ≈ 22% kNN weight); Dirichlet(0.1) at the same 66 evaluations reaches HV 0.91 / 0.73 / 0.53 / 0.88, at which point NBI-A no longer beats random search on Santander, BNP or UCI while NBI-B/C still do on three datasets and NBI-C on UCI. Random scalarization on the surfaces with real anchors matches NBI-B/C on hypervolume on Santander (0.978) with worse coverage (IGD 0.215 vs 0.043–0.053), ties NBI-C on UCI, is indistinguishable from NBI-A/B on Porto (HV 0.79, 6–47 front points), and collapses on BNP (1–3 front points in 9/10 replications). Uniformity: the classical uniform-spacing claim is not supported by the benchmark's size-matched test (Schott spacing of the validated NBI front vs 200 random equal-size subsets of the empirical reference front, of which 30–40% are NBI's own candidates): median percentiles 0.78 / 0.99 / 0.995 / 0.40 (NBI-B) and 0.995 / 0.54 / 0.985 / 0.45 (NBI-C). The indicator is dominated by the isolation of the 2–3 anchor points (80–98% of the nearest-neighbour-distance variance) and by absolute scale; with the relative (CV) form UCI becomes the least even and BNP NBI-C becomes significantly more even (0.065), after trimming the three most isolated points the outcome is mixed, and every comparator including random Dirichlet search scores above 0.5. The defensible statement is that the even β-spacing designed on the surrogate CHIM does not translate into a measurable evenness advantage on the real front under any tested definition.

---

## 10. Conclusion

**Does the evidence support DoE + response-surface metamodeling + NBI as a reproducible and interpretable framework for multiobjective ensemble-weight optimization?**

Partly, and the parts must be separated:

- **DoE/RSM component — reproducible and interpretable, but not reliably predictive.** The 66-run design, the fold-safe OOF protocol, the coefficient signs and rankings, and the reliability gate reproduce across 10 outer replications on all four datasets (coefficient sd 1–15% of the mean, sign frequency 100%). The Scheffé surfaces are an honest, interpretable summary of the performance surface where they pass the gate (log-loss on Santander, Porto and UCI; everything on UCI) and an explicitly detected failure where they do not (AUC on Santander, everything on BNP). The interpretable content that survives is negative as often as positive: the largest interactions mark the weakest classifier, and the practically relevant structure (fronts concentrated on 2–4-model faces, a degenerate AUC-vs-log-loss axis, cost as the only real trade-off) is visible from the surfaces and the fronts together.
- **Surrogate approximation — the weak link.** Whenever the selected surface fails the reliability gate, surrogate-anchored NBI degrades (HV ratio 0.67–0.85, joint-ND ≈ 0); when it passes, surrogate NBI with real anchors is within IGD+ 0.01 of the reference. The surrogate is therefore usable only behind the gate and with real anchors.
- **NBI — works as a front constructor when its geometry is fed real information.** With real anchors and either reliable surfaces (Santander, BNP) or the real objectives themselves (all datasets), NBI produces validated fronts with HV ratio 0.97–0.99 and, in the metamodel-free form, the best fronts in the benchmark at 66 subproblems. Its failure modes are geometric, not numerical: vertex β subproblems at degenerate corners (handled by returning the anchor) and 50% subproblem failure on UCI even with reliable surfaces, plus no uniform-spacing advantage after revalidation.
- **Sensitivity to anchor construction — decisive where the surrogate is poor** (Santander, BNP: 10/10 replications improve), negligible where it is good.
- **Sensitivity to the cost definition — decisive for deployment conclusions.** The linear cost is what makes the optimization smooth; the support cost is what deployment pays, and it removes 70–85% of the weighted-cost front and reorders the methods on BNP.

Practical reading: the framework is reproducible and its DoE/RSM layer is interpretable; as an *optimizer* its defensible configuration is NBI with real anchors on the cached OOF (variant C) or, when the reliability gate passes, on the Scheffé surfaces (variant B), followed by real revalidation and a support-cost Pareto filter; random scalarization on reliable surfaces with real anchors is a strong, cheap baseline that surrogate NBI must beat. The single-objective SLSQP log-loss oracle remains the reference for the log-loss axis on every dataset, and the AUC axis adds almost nothing beyond it.

### Caveats that bound every statement above (from the adversarial verification of 12 pre-registered claims: all numbers reproduced; five inferences narrowed as written in §§ 4–9)

1. The empirical reference is the non-dominated union of the sample and every candidate set; NBI-B and NBI-C contribute 30–40% of its points, so convergence/coverage indicators and the size-matched spacing null are partly self-graded.
2. "Success" differs between variants (A/B: SLSQP-certified with residual < 10⁻³; C: feasibility under a lenient rule) and C ran at a reduced optimizer budget; success rates, compute ratios and validity-filtered indicators are not like-for-like.
3. All NBI variants optimize the weighted cost; support-cost results measure transfer, and the support-cost hypervolume is fragile to the 10⁻³ activity threshold.
4. HV = 0.00 means "outside the 1.1 reference box", and the random-Dirichlet baseline's magnitude is a property of the α = 1 sampler with a 150–250 ms/1k kNN in the pool.
5. Per-dataset statistics rest on n = 10 replications; pair-level correlations on n = 10 pairs; dataset-level "mechanism" statements are between-dataset associations over four datasets.
6. The reliability gate is a composite (R² and Spearman) and its borderline cells (Porto, BNP) are validation-draw sensitive; parsimony order selection uses the same 100 validation points; quadratic β's are reported irrespective of the selected order.
7. Several headline means summarize bimodal distributions (BNP anchor gains, UCI NBI-A/B success, BNP NBI-C support HV); medians and per-replication counts are given alongside.
8. The batched design-point AUC evaluator differs from midrank AUC by ≤ 3.4 × 10⁻⁴ at gnb/knn-dominated points (probability ties); no reliability count changes.
9. Holdout feature rows (unlabelled) were used for inference-cost timing; holdout labels only for confirmation metrics; holdout agreement is level agreement under weighted cost.
10. `README.md` front sizes are sample-only fronts and its BNP scalarization median front size (1.5) is printed as 2.

---

## 11. Output paths

- `reports/pco213_postwork_benchmark/README.md`, `summary.json`, `tables/*.csv` (12 tables), `manifests/*.json` (master + 4 dataset manifests), this `FINAL_REPORT.md`
- `figures/pco213_postwork_benchmark/fig01_methodology.png` … `fig17_nbi_variants_paired.png` (25 files: methodology, fronts and cost definitions per dataset, GD/IGD/IGD+/HV/spacing distributions, β_ij heatmaps, coefficient stability, active support, weight composition, N_eff, diversity vs β_ij, surrogate validation, NBI variant pairing)
- `experiments/pco213_postwork_benchmark/` (unversioned): `benchmark_manifest.json`, `benchmark.log`, `<dataset>/rep_XX/*`
- Scripts: `scripts/pco213_run_postwork_benchmark.py`, `scripts/pco213_benchmark_launch.sh`, `scripts/pco213_postwork_benchmark_report.py`; modules `src/mixens/{datasets,bench_models,fastmetrics,pareto_tools}.py` and the extended `nbi.py`, `scheffe.py`, `mixture_design.py`, `optimize.py`
