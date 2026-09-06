# Statistical analysis plan and conventions — R = 30 robustness extension

This document states how the replication-level statistics in `statistics/` are computed and how they may be read. It applies to the R = 30 extension of the four-dataset post-work benchmark (replications 10–29 added to the frozen replications 0–9 of tag `pco213-postwork-r10`, identical protocol, seeds `20260904 + rep`). Code: `scripts/pco213_postwork_benchmark_stats.py`; configuration snapshot: `statistics/analysis_config.json`.

## 1. Statistical unit and dependence

The unit of replication is one outer stratified 80/20 train/holdout partition of one dataset. All methods (three NBI variants, comparators, references, surfaces) are computed on the same partition, so every comparison is **paired by replication**. Within a dataset the 30 partitions overlap heavily (each holds 80% of the same rows), so the replication distribution measures **sensitivity to data partitioning**, not sampling from a population of datasets. Consequently:

- the **dataset** is the unit for cross-dataset generalization: results are summarized per dataset and the four dataset-level results are then compared; nothing is pooled over the 4 × 30 = 120 replications as if they were independent;
- no statement of the form "N = 120 independent experiments" is made;
- bootstrap intervals over the 30 replications describe Monte Carlo stability under the observed resampling process, not population uncertainty.

## 2. Pre-specified primary comparisons and endpoints

Primary comparisons (new method vs reference method):

1. NBI-B vs NBI-A (real anchors vs surrogate anchors, same Scheffé surfaces);
2. NBI-C vs NBI-B (metamodel-free on cached OOF vs surrogate with real anchors);
3. NBI-C vs random weighted scalarization on the surfaces with real anchors;
4. NBI-C vs budget-matched (66-point) random Dirichlet(1) search.

Primary endpoints, computed on real revalidated objectives against the empirical Pareto reference of the same replication: **IGD+** (lower is better) and **hypervolume ratio** (higher is better). The primary analysis uses the weighted (linear) cost, which is the objective every NBI variant optimized; the support-cost repetition is reported as a sensitivity analysis. All other indicators (GD, IGD, spacing, joint non-dominated fraction, runtime, front sizes, success rates) are descriptive.

## 3. Paired effects and sign convention

For each dataset × comparison × endpoint the paired difference per replication is

- ΔHV = HV_new − HV_ref,
- ΔIGD+ = IGD+_ref − IGD+_new,

so that **Δ > 0 always means the new (second-named) method is better**. Reported: n, mean, median, sd, quartiles, min, max, percentile-bootstrap 95% intervals for the mean and the median (10,000 resamples, seed 20260906), win/tie/loss counts (tie = |Δ| ≤ 10⁻⁴), win fraction with a Wilson 95% interval, and the matched-pairs rank-biserial correlation r = (T⁺ − T⁻)/(T⁺ + T⁻) as a nonparametric effect size (−1 … +1; ties dropped).

## 4. Formal tests and their correction

Because replications overlap, an ordinary paired t-test overstates precision. The formal test for each primary pair is the **Nadeau–Bengio corrected resampled t-test**: with J = 30 paired differences d_j, t = mean(d) / sqrt(var(d) · (1/J + ρ)), df = J − 1, ρ = n_test/n_train = 0.20/0.80 = 0.25, which is the setting the correction was derived for (the same procedure evaluated on random splits of one dataset). A Wilcoxon signed-rank test (distribution-free, uncorrected for overlap) is reported alongside as a check; it is not the primary test.

**Multiplicity.** Formal tests are restricted to the 4 primary comparisons × 2 primary endpoints. Within each dataset these 8 tests form one family and receive a Holm correction (`p_nb_holm_family_dataset`); no test is corrected across datasets because datasets are not pooled. Where the parametric assumptions look poor for a metric (bimodal or bounded distributions, ties), the paired-effect distribution, the bootstrap interval and the win fraction are the reported evidence and the p-value is not emphasized.

## 5. Proportions

Frequencies over replications (reliability-gate pass, win fractions, sign consistency of coefficients, NBI feasibility ≥ 0.9, holdout ranking agreement, best-set changes between cost definitions) are reported with **Wilson** and **Jeffreys** 95% intervals (`proportion_intervals.csv`).

## 6. Specific analyses

- **Reliability gate** (`reliability_gate_r30.csv`): P(pass) per dataset × response with intervals; the gate thresholds (external R² ≥ 0.5 and Spearman ≥ 0.9 on the 100 unseen Dirichlet points) are the R = 10 thresholds, unchanged. `reliability_gate_conditional_gain_r30.csv` gives the B − A gain split by gate outcome within each dataset (descriptive; the two groups have unequal n).
- **Coefficient stability** (`coefficient_stability_r30.csv`): per dataset × response × Scheffé term: mean, sd, CV, median, IQR, bootstrap CI, sign-positive frequency with Wilson interval, rank statistics and top-1/top-3 frequencies over the ten interactions; for interactions also the vertex-quality gap of the pair, the real 50/50 blend's performance relative to its better member, and the pair's participation on the empirical reference front. These support the R = 10 reading that a large β_ij marks a poor vertex rather than exploitable complementarity.
- **R = 10 vs R = 30 stability** (`r10_vs_r30_stability.csv`): every listed quantity estimated from replications 0–9 (identical to the frozen R = 10 analysis) and from 0–29, with absolute and relative change and the R = 30 bootstrap interval.
- **Holdout transfer** (`holdout_transfer_r30.csv`): for the knee pick of each set, OOF and holdout AUC/log-loss, paired holdout − OOF differences with intervals, and the frequency with which the OOF-best set is also the holdout-best set (level agreement vs ranking agreement are kept distinct). Holdout labels never enter selection; the OOF-chosen F1 threshold is applied unchanged.
- **AUC-vs-log-loss conflict** (`auc_logloss_conflict_r30.csv`): distributions of AUC and log-loss gaps between the direct-AUC optimum and the SLSQP log-loss optimum, with weight-vector L1 distance, support Jaccard, and weighted/support cost differences, to separate objective-space degeneracy from weight-space/deployment differences.
- **Cost-definition sensitivity** (`cost_definition_sensitivity_r30.csv`): per replication, the best set under weighted vs support cost and the frequency of disagreement (Wilson interval), plus the rank correlation of set rankings; the support threshold is the R = 10 value 10⁻³, unchanged.
- **Regimes** (`bimodality_regimes_r30.csv`): distribution summaries with Sarle's bimodality coefficient (> 0.555 suggests bimodality) and a one- vs two-component Gaussian-mixture BIC difference as exploratory diagnostics, and Spearman associations of NBI-A hypervolume with surrogate R², Spearman validation, anchor costs and the class prevalence of the outer split. These are exploratory; no formal multimodality test is claimed.

## 7. What may and may not be concluded

- Within a dataset: paired effects with intervals, win fractions and corrected tests support statements about that dataset's partition sensitivity.
- Across datasets: only the pattern over four dataset-level results is reported; any mechanism statement ("real anchors help when the surrogate is unreliable") is a between-dataset association over four datasets, not a tested hypothesis.
- Effect sizes and their consistency (win fractions, rank-biserial) take precedence over p-values.
