# Supplementary material

Every table in this document is generated from a committed artifact by
`scripts/build_supplement.py`. No figure is transcribed by hand.

**Manuscript source commit** `eeac042339b6` — the state of the
manuscript and analysis sources from which this supplement was generated.
The final package commit and tag are recorded in S17; they differ from the
source commit by the metadata-only step that records this provenance, and no
byte identity between the two is claimed.

Protocol tag `xgboost-hpo-protocol-v3`. Results tag
`xgboost-hpo-confirmatory-results-v1`.

## S1. Protocol lineage and tags

| tag | commit | what it records |
|---|---|---|
| `v0.1.0-dissertation` | `67d9fe5` | the historical implementation this study reconstructs |
| `xgboost-hpo-protocol-v1` | `355c290` | the specification frozen before the pilot |
| `xgboost-hpo-protocol-v2` | `c50c380` | the confirmatory protocol after pilot Stage A |
| `xgboost-hpo-protocol-v3` | `9b15ba7` | the final frozen protocol, before the confirmatory campaign |
| `xgboost-hpo-confirmatory-results-v1` | `a4f20ed` | the first adversarially verified analysis |

The protocol tag records what was specified before the confirmatory campaign began;
the results tag records the first verified analysis. Neither was moved.

## S2. Amendment chronology

The protocol carries 23 numbered amendments, each recording what caused it
and whether any arm result had been observed when it was made; the ledger records
that none had. It also records two process failures found by review rather than by the
author, and four engineering defects caught by pre-freeze smoke runs. The last of these
runs, on 2026-09-14, completed one full Spambase unit, every arm and the metrics stage,
before amendments 21 (primary reference) and 22 (Spambase's role) and before the v3 tag;
the ledger records that only its execution and accounting were inspected, and its
outputs are not archived. The ledger is reproduced in full in `PROTOCOL_AMENDMENTS.md`.
The corrections made after the campaign to two claims from the analysis are recorded in
`analysis/verified_claims.json`, not in the ledger.

## S3. Hyperparameter space

| hyperparameter | lower | upper | type |
|---|---:|---:|---|
| `subsample` | 0.05 | 1.0 | continuous |
| `colsample_bytree` | 0.05 | 1.0 | continuous |
| `colsample_bylevel` | 0.05 | 1.0 | continuous |
| `learning_rate` | 0.01 | 0.3 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.0 | continuous |
| `n_estimators` | 50 | 700 | integer |

All seven enter the design in coded units on $[-1, 1]$; integers are cast by
`int(round(·))` at evaluation time.

## S4. Responses and objective construction

| response | role | transform |
|---|---|---|
| `Accuracy_Mean` | quality | none |
| `Precision_Mean` | quality | none |
| `Recall_Mean` | quality | none |
| `Specificity_Mean` | quality | none |
| `RocAuc_Mean` | quality | none |
| `LogLoss_Mean` | quality | none |
| `Leaves_Mean` | cost | log1p |

### S4.1 Frozen factor models

| dataset | fitting sample | audit rows used | quality weights | Kaiser | $\lambda_3/\lambda_4$ |
|---|---:|---:|---|---:|---:|
| MAGIC | 88 pilot design rows | 0 | 0.5556, 0.4444 | 2 | 2.93 |
| Spambase | 88 pilot design rows | 0 | 0.7749, 0.2251 | 2 | 2.10 |
| Adult | 88 pilot design rows | 0 | 0.5470, 0.4530 | 1 | 1.34 |
| Bank Marketing | 88 pilot design rows | 0 | 0.5875, 0.4125 | 1 | 1.06 |

One model per dataset, fitted on the 88 design rows only and applied to all 30
replications. The fitting rows are the pre-campaign pilot (Stage A) evaluations of the
design: stratified, shuffled five-fold cross-validation on the 80% training part of one
stratified 80/20 partition per dataset (seed 20260913), the partition of the S5
screening, drawn separately from the 30 confirmatory partitions. Each replication
re-evaluates the same 88 configurations on its own partition and maps them through the
frozen model. Because the pilot partition is a separate draw, its training part
shares rows with every confirmatory holdout: a holdout is unseen within its replication,
not by the pilot that fixed the objective definition. The 78-point audit-only external
construction is excluded from the fit (amendment 23): 64 of its 78 coded points are
identical to the external validation set and the rest are the same axial runs.

Eigenvalues of the correlation matrix of the seven standardized responses $Z$ on the 88
fitting rows; the cumulative percentage of the total variance of $Z$ carried by the first
$k$ components, $100 \sum_{j \le k} \lambda_j / 7$; and the percentage of the variance of
the transformed leaf count that the first $k$ components reproduce. Varimax redistributes
the $k = 3$ total among the rotated factors without changing it. The frozen model is
applied unchanged to every replication, so these values are the same in every
replication. They are descriptive: the component count is fixed at three for every
dataset and was not chosen by them. From `analysis/factor_dimensionality.json`, written
by `scripts/factor_dimensionality.py` after the campaign from the committed pre-campaign
reference models.

| dataset | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | $\lambda_4$ | cumulative % of variance, $k = 1$ / 2 / 3 | leaf-count variance reproduced, %, $k = 2$ / 3 |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | 3.988 | 1.659 | 0.891 | 0.304 | 57.0 / 80.7 / 93.4 | 17.6 / 98.8 |
| Spambase | 4.455 | 1.092 | 0.865 | 0.412 | 63.6 / 79.2 / 91.6 | 68.2 / 84.0 |
| Adult | 4.775 | 0.851 | 0.701 | 0.522 | 68.2 / 80.4 / 90.4 | 78.3 / 98.1 |
| Bank Marketing | 4.664 | 0.853 | 0.625 | 0.588 | 66.6 / 78.8 / 87.7 | 75.3 / 89.1 |

### S4.2 Rotated loadings and objective coefficients

For each frozen model: the oriented rotated loadings $\Lambda_R$ of the seven
canonicalized (minimization-oriented) responses on the leading quality factor (the one with
the larger quality weight in S4.1), the second quality factor and the cost factor; the
correlation of each response with $f_{\text{quality}}$; and the coefficients of the two
objectives on the standardized responses $Z$ (`Leaves_Mean` after its $\log(1 + \cdot)$
transform). A loading is the Pearson correlation of a response with a factor score on the
88 fitting rows, so in the loading and correlation columns a positive entry means that the
response tends to worsen as the score rises on those rows. The frozen model is applied
unchanged to every other evaluation, where these correlations are not guaranteed to hold
(S4.3). In the coefficient columns a positive entry means that, with the other responses
held fixed, a worse value of that response raises the objective. The responses do not vary
one at a time, so a coefficient describes the fixed linear map, not an effect of that
response, and a coefficient and the loadings can differ in sign. Accuracy is, fold by fold,
the prevalence-weighted mean of recall and specificity, so the coefficients on those three
are not separately interpretable. Neither quality factor names the same quantity on two
datasets. The coefficients are read from `analysis/factor_dimensionality.json`.

| dataset | response | leading quality factor | second quality factor | cost factor | correlation with $f_{\text{quality}}$ | $f_{\text{quality}}$ coefficient | $f_{\text{cost}}$ coefficient |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGIC | `Accuracy_Mean` | +0.862 | -0.433 | -0.196 | +0.402 | +0.115 | -0.015 |
| MAGIC | `Precision_Mean` | -0.047 | +0.965 | +0.024 | +0.566 | +0.279 | -0.128 |
| MAGIC | `Recall_Mean` | +0.750 | -0.614 | -0.200 | +0.202 | +0.028 | -0.008 |
| MAGIC | `Specificity_Mean` | -0.167 | +0.955 | +0.145 | +0.466 | +0.243 | -0.015 |
| MAGIC | `RocAuc_Mean` | +0.933 | -0.018 | -0.161 | +0.717 | +0.262 | -0.030 |
| MAGIC | `LogLoss_Mean` | +0.898 | +0.049 | +0.061 | +0.732 | +0.275 | +0.192 |
| MAGIC | `Leaves_Mean` | -0.133 | +0.115 | +0.978 | -0.032 | +0.011 | +1.006 |
| Spambase | `Accuracy_Mean` | +0.894 | +0.179 | -0.348 | +0.909 | +0.195 | -0.005 |
| Spambase | `Precision_Mean` | +0.108 | +0.988 | -0.094 | +0.379 | +0.175 | +0.119 |
| Spambase | `Recall_Mean` | +0.917 | +0.166 | -0.311 | +0.927 | +0.209 | +0.031 |
| Spambase | `Specificity_Mean` | -0.922 | +0.004 | -0.145 | -0.884 | -0.302 | -0.393 |
| Spambase | `RocAuc_Mean` | +0.584 | +0.051 | -0.709 | +0.575 | -0.012 | -0.426 |
| Spambase | `LogLoss_Mean` | +0.885 | +0.055 | -0.392 | +0.865 | +0.159 | -0.066 |
| Spambase | `Leaves_Mean` | -0.129 | -0.134 | +0.898 | -0.161 | +0.178 | +0.706 |
| Adult | `Accuracy_Mean` | +0.776 | +0.585 | -0.180 | +0.971 | +0.189 | +0.132 |
| Adult | `Precision_Mean` | +0.582 | +0.473 | -0.134 | +0.750 | +0.146 | +0.104 |
| Adult | `Recall_Mean` | +0.861 | +0.444 | -0.216 | +0.947 | +0.182 | +0.099 |
| Adult | `Specificity_Mean` | -0.928 | -0.124 | +0.262 | -0.794 | -0.147 | -0.026 |
| Adult | `RocAuc_Mean` | +0.301 | +0.791 | -0.434 | +0.737 | +0.052 | -0.290 |
| Adult | `LogLoss_Mean` | +0.282 | +0.935 | -0.021 | +0.813 | +0.165 | +0.197 |
| Adult | `Leaves_Mean` | -0.242 | -0.130 | +0.951 | -0.269 | +0.165 | +0.992 |
| Bank Marketing | `Accuracy_Mean` | +0.731 | +0.607 | -0.235 | +0.947 | +0.209 | +0.138 |
| Bank Marketing | `Precision_Mean` | +0.636 | +0.426 | -0.015 | +0.766 | +0.232 | +0.328 |
| Bank Marketing | `Recall_Mean` | +0.866 | +0.343 | -0.316 | +0.906 | +0.196 | +0.080 |
| Bank Marketing | `Specificity_Mean` | -0.894 | -0.121 | +0.349 | -0.801 | -0.170 | -0.030 |
| Bank Marketing | `RocAuc_Mean` | +0.349 | +0.595 | -0.614 | +0.627 | -0.029 | -0.480 |
| Bank Marketing | `LogLoss_Mean` | +0.225 | +0.943 | -0.091 | +0.726 | +0.139 | +0.105 |
| Bank Marketing | `Leaves_Mean` | -0.298 | -0.078 | +0.893 | -0.288 | +0.189 | +0.883 |

| dataset | role-block mean, leading / second quality factor | `Leaves_Mean` on the cost factor | Spearman of $f_{\text{quality}}$ with equal-weight quality badness, fitting rows |
|---|---:|---:|---:|
| MAGIC | +0.538 / +0.150 | +0.978 | +0.929 |
| Spambase | +0.411 / +0.240 | +0.898 | +0.243 |
| Adult | +0.312 / +0.517 | +0.951 | +0.835 |
| Bank Marketing | +0.319 / +0.466 | +0.893 | +0.896 |

The sign rule (Section 5.5) makes the two role-block means and the `Leaves_Mean` loading
positive; it does not make every loading in a role block positive. The last column is the
frozen orientation check of amendment 19 on the fitting rows; S4.3 evaluates it beyond them.

At the 0.5 threshold, 0 (MAGIC), 1 (Spambase), 13 (Adult) and 16 (Bank Marketing) of the 88 fitting rows predict no positive case in any fold, so their recall and
precision are 0 and their specificity 1. All of them are among the 16 factorial
design rows with `learning_rate` 0.01 and `n_estimators` 50 (Section 5.4).

### S4.3 The quality composite beyond its fitting sample (post hoc)

Computed after the campaign by `scripts/composite_beyond_fitting_rows.py` from stored
evaluations, with the frozen models; it changes no frozen number and fed no arm. Alignment
is the frozen orientation check of amendment 19: the Spearman correlation of
$f_{\text{quality}}$ with the equal-weight mean of the six standardized canonicalized quality
responses, which must be positive. Before the campaign it was evaluated on the 88-row fitting
sample, and each replication recomputed it on its own design rows. Here it is also evaluated
on the 386 configurations that the random-search baseline drew uniformly from the
coded decision box in each replication. A reversal is a pair of configurations in which one
is at least as good on all six quality responses and strictly better on one, but receives
the larger (worse) $f_{\text{quality}}$; random-search pairs are formed within replications
and pooled.

| dataset | alignment, fitting sample | alignment, design rows (range over replications) | alignment, random-search rows: median (range) | replications with negative alignment | reversals, fitting sample (pairs) | reversal rate, random-search rows |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | +0.929 | +0.889 to +0.945 | +0.941 (+0.903 to +0.979) | 0/30 | 0/905 | <0.001 |
| Spambase | +0.243 | +0.188 to +0.310 | -0.387 (-0.540 to -0.182) | 30/30 | 1,045/1,311 | 0.751 |
| Adult | +0.835 | +0.809 to +0.850 | +0.562 (+0.461 to +0.639) | 0/30 | 150/615 | 0.262 |
| Bank Marketing | +0.896 | +0.873 to +0.916 | +0.694 (+0.602 to +0.761) | 0/30 | 149/421 | 0.146 |

### S4.4 Per-replication refit: Tucker congruence

Declared in the frozen protocol (section 7.2; amendment 14) and computed after the campaign
from stored outputs by `scripts/tucker_congruence_refit.py`. In each replication the factor
model refitted on that replication's own 88 design rows is compared with the frozen model by
the Tucker congruence of matched loading columns; the refit is never applied. The runner
recorded congruences under a greedy column matching that does not resolve factor sign, so a
negative value there can be an artifact of the matching. The optimal-assignment matching
beside it is post hoc: columns are matched by maximal total absolute congruence and each
matched column is oriented before its congruence is computed. A congruence of 0.95 or more
is conventionally read as equivalence, and 0.85 to 0.95 as fair similarity.

| dataset | greedy, as recorded: minimum absolute congruence | greedy: replications with a negative congruence | optimal (post hoc): minimum congruence | below 0.95 | below 0.85 | cost factor recovered |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | 0.895 | 0 of 30 | 0.895 | 6 of 30 | 0 of 30 | 30 of 30 |
| Spambase | 0.918 | 0 of 30 | 0.918 | 2 of 30 | 0 of 30 | 30 of 30 |
| Adult | 0.948 | 0 of 30 | 0.948 | 1 of 30 | 0 of 30 | 30 of 30 |
| Bank Marketing | 0.351 | 4 of 30 | 0.751 | 10 of 30 | 4 of 30 | 30 of 30 |

### S4.5 Front-level weighting sensitivity

Declared in the frozen protocol (section 11, item 1) and computed after the campaign from
stored outputs by `scripts/weighting_sensitivity_fronts.py`. The protocol asks how often
variance weighting and equal weighting of the two quality factors disagree about the
returned front. Two fronts differ if their non-dominated sets differ by at least one member
after deduplication at the evaluation tolerance, and if they differ in more than 20% of
replications on a dataset, the conclusions for that dataset are reported as conditional on
the weighting, in the abstract as well. The protocol did not fix how the equal-weight fronts
are obtained or which arms the rule counts. Post hoc, each arm's stored revalidated
candidates, and the CORE reference, are rescored with the equal-weight composite (no arm was
re-solved), configurations with identical stored evaluations count as one member, and the
rule is counted as met on a dataset if it is met for any of the four
arms. The design-row rank agreement of the two composites (Spearman 0.987 to 0.991,
Section 5.5) is a different estimand.

| dataset | HISTORICAL-WS: fronts differ | WS-S: fronts differ | NBI-S: fronts differ | NBI-R: fronts differ | more than 20% for any arm |
|---|---:|---:|---:|---:|---|
| MAGIC | 13 of 30 | 9 of 30 | 19 of 30 | 14 of 30 | yes |
| Spambase | 14 of 30 | 15 of 30 | 23 of 30 | 14 of 30 | yes |
| Adult | 13 of 30 | 17 of 30 | 14 of 30 | 18 of 30 | yes |
| Bank Marketing | 16 of 30 | 14 of 30 | 16 of 30 | 21 of 30 | yes |

The three identifying contrasts under the equal-weight composite, against the rescored
CORE reference: median paired difference in the hypervolume ratio (second minus first),
with W/T/L; descriptive and untested.

| dataset | contrast | median difference | W/T/L |
|---|---|---:|---:|
| MAGIC | HISTORICAL-WS $\rightarrow$ WS-S | +0.0000 | 14/3/13 |
| MAGIC | WS-S $\rightarrow$ NBI-S | +0.0395 | 27/0/3 |
| MAGIC | NBI-S $\rightarrow$ NBI-R | -0.0776 | 1/0/29 |
| Spambase | HISTORICAL-WS $\rightarrow$ WS-S | +0.0000 | 16/6/8 |
| Spambase | WS-S $\rightarrow$ NBI-S | +0.0223 | 16/2/12 |
| Spambase | NBI-S $\rightarrow$ NBI-R | -0.5934 | 5/1/24 |
| Adult | HISTORICAL-WS $\rightarrow$ WS-S | -0.0012 | 6/2/22 |
| Adult | WS-S $\rightarrow$ NBI-S | +0.2543 | 30/0/0 |
| Adult | NBI-S $\rightarrow$ NBI-R | -0.0266 | 12/0/18 |
| Bank Marketing | HISTORICAL-WS $\rightarrow$ WS-S | +0.0000 | 9/7/14 |
| Bank Marketing | WS-S $\rightarrow$ NBI-S | +0.3205 | 27/0/3 |
| Bank Marketing | NBI-S $\rightarrow$ NBI-R | -0.4389 | 7/0/23 |

### S4.6 Leaf count and measured time

Declared in the frozen protocol (`protocol/OBJECTIVE_DEFINITIONS.md`, section 9, and
`protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md`) and computed after the campaign from stored
outputs by `scripts/leaf_time_association.py`: the association between the leaf-count
objective and measured mean per-fold time over all 120 units. Before the campaign, raw total
leaf count tracked measured per-fold time at Spearman 0.86 on 22 of the 88
MAGIC design configurations, and two timings of one configuration differed by a median 1.3%
(`audits/cost_objective_selection.json`). Campaign timings were taken with several worker
processes running in parallel (S17), so they carry load-dependent noise. The association is
reported as measured; a weaker association is a finding about the proxy and changes no
objective. Each cell is the median over units of the per-unit Spearman correlation, with
its range. Post hoc choices: all evaluations are every cross-validated fit stored for the
unit, excluding holdout fits and the unmatched NSGA-II run, which lies outside the unit;
design rows are the 79 distinct design configurations, because the ten replicated centre
points were fitted once.

| dataset | units | all evaluations: raw leaf count | all evaluations: $f_{\text{cost}}$ | design rows: raw leaf count | design rows: $f_{\text{cost}}$ |
|---|---:|---:|---:|---:|---:|
| MAGIC | 30 | 0.933 (0.919 to 0.943) | 0.917 (0.899 to 0.930) | 0.907 (0.897 to 0.915) | 0.896 (0.886 to 0.904) |
| Spambase | 30 | 0.957 (0.953 to 0.961) | 0.919 (0.896 to 0.926) | 0.935 (0.920 to 0.941) | 0.820 (0.772 to 0.852) |
| Adult | 30 | 0.871 (0.844 to 0.885) | 0.852 (0.822 to 0.871) | 0.870 (0.864 to 0.876) | 0.795 (0.779 to 0.825) |
| Bank Marketing | 30 | 0.878 (0.847 to 0.895) | 0.851 (0.825 to 0.872) | 0.894 (0.887 to 0.903) | 0.729 (0.644 to 0.757) |

### S4.7 Cost-definition sensitivity

Declared in the frozen protocol (section 11, item 4): whether the primary cost objective and
measured time disagree about the winning arm, reported as a frequency over replications.
Computed after the campaign from stored outputs by `scripts/cost_definition_sensitivity.py`.
The protocol did not fix the operationalization. Post hoc, the winning arm is the one of
HISTORICAL-WS, WS-S, NBI-S and NBI-R with the largest CORE-relative hypervolume ratio; under
the time definition the cost axis of every candidate and of the CORE reference is the
measured mean per-fold time, with the quality objective unchanged. No arm was re-run, and the
timing noise of S4.6 applies. The time axis is in seconds, untransformed, and arms whose
ratios agree to within $10^{-12}$ share the win (listed as tied).

| dataset | replications in which the winning arm differs | winners, leaf-count cost | winners, time cost |
|---|---:|---|---|
| MAGIC | 9 of 30 | NBI-S 24, WS-S 4, HISTORICAL-WS 1, NBI-R 1 | NBI-S 17, NBI-R 6, WS-S 5, HISTORICAL-WS 2 |
| Spambase | 17 of 30 | NBI-S 14, WS-S 6, HISTORICAL-WS 3, NBI-R 3, HISTORICAL-WS/WS-S (tied) 2, HISTORICAL-WS/WS-S/NBI-S (tied) 2 | NBI-S 15, NBI-R 5, HISTORICAL-WS 4, WS-S 3, HISTORICAL-WS/WS-S/NBI-S/NBI-R (tied) 2, HISTORICAL-WS/WS-S (tied) 1 |
| Adult | 16 of 30 | NBI-S 19, NBI-R 11 | NBI-R 16, NBI-S 8, HISTORICAL-WS 4, WS-S 2 |
| Bank Marketing | 14 of 30 | NBI-S 17, NBI-R 7, HISTORICAL-WS 3, WS-S 2, HISTORICAL-WS/WS-S (tied) 1 | NBI-S 15, WS-S 7, HISTORICAL-WS 4, NBI-R 3, HISTORICAL-WS/WS-S/NBI-S/NBI-R (tied) 1 |

The three identifying contrasts under both cost definitions: median paired difference in
the CORE-relative hypervolume ratio (second minus first), with W/T/L, ties being differences
within $10^{-12}$; descriptive and untested. Under the leaf-count definition the values are
those of the primary analysis; with its exact-zero tie rule Spambase HISTORICAL-WS
$\rightarrow$ WS-S reads 16/7/7 rather than 16/8/6, one replication differing by
$9.7 \times 10^{-15}$. Under measured time the geometry contrast stays positive on every
dataset, smaller on Adult and Bank Marketing, and the anchor-provenance contrast changes
sign on Adult.

| dataset | contrast | leaf-count cost: median (W/T/L) | time cost: median (W/T/L) |
|---|---|---:|---:|
| MAGIC | HISTORICAL-WS $\rightarrow$ WS-S | +0.0000 (14/3/13) | -0.0000 (14/0/16) |
| MAGIC | WS-S $\rightarrow$ NBI-S | +0.0374 (26/0/4) | +0.0555 (23/0/7) |
| MAGIC | NBI-S $\rightarrow$ NBI-R | -0.0741 (2/0/28) | -0.0249 (7/0/23) |
| Spambase | HISTORICAL-WS $\rightarrow$ WS-S | +0.0018 (16/8/6) | +0.0060 (15/6/9) |
| Spambase | WS-S $\rightarrow$ NBI-S | +0.0500 (18/2/10) | +0.1875 (21/2/7) |
| Spambase | NBI-S $\rightarrow$ NBI-R | -0.6932 (5/0/25) | -0.9574 (7/3/20) |
| Adult | HISTORICAL-WS $\rightarrow$ WS-S | -0.0006 (9/4/17) | -0.0010 (13/2/15) |
| Adult | WS-S $\rightarrow$ NBI-S | +0.2549 (30/0/0) | +0.1195 (22/0/8) |
| Adult | NBI-S $\rightarrow$ NBI-R | -0.0192 (11/0/19) | +0.0696 (18/0/12) |
| Bank Marketing | HISTORICAL-WS $\rightarrow$ WS-S | +0.0000 (8/8/14) | +0.0030 (16/4/10) |
| Bank Marketing | WS-S $\rightarrow$ NBI-S | +0.2281 (24/0/6) | +0.0616 (19/1/10) |
| Bank Marketing | NBI-S $\rightarrow$ NBI-R | -0.4093 (7/0/23) | -0.4004 (5/1/24) |

## S5. Panel screening and dataset roles

| dataset | C1 raw conflict | C2 front / curvature | C3 min $R^2$ | C4 range | role |
|---|---:|---|---:|---:|---|
| MAGIC | -0.2509 | 7 / 0.1729 | 0.6913 | 2706.3× | `primary_geometry_confirmatory` |
| Spambase | -0.4621 | 2 / undefined | 0.8987 | 591.2× | `boundary_geometry_control` |
| Adult | -0.3977 | 6 / 0.1992 | 0.9078 | 3421.3× | `primary_geometry_confirmatory` |
| Bank Marketing | -0.4100 | 6 / 0.1983 | 0.8700 | 4355.5× | `primary_geometry_confirmatory` |

Roles were assigned from these pre-campaign measurements and never revised from
an optimizer outcome. They were measured on one partition per dataset, the pilot
Stage A partition (seed 20260913), whose design-row evaluations are also the fitting
sample of the frozen factor models (S4.1); it is not one of the 30 confirmatory
partitions. Spambase was retained as a boundary geometry control by a pre-campaign
amendment adopted after its criterion-2 failure was known (amendment 22): the replacement
rule frozen at protocol v1, before the pilot, required a dataset failing criterion 1 or 2
to be replaced from the registry, and retaining Spambase departs from that rule.

Every confirmatory replication evaluates the same 88 design rows on its own partition. The
number of those rows that are non-dominated under the frozen objectives is summarized below
as a post-campaign description; it changes no role.

| dataset | non-dominated design rows, screening partition | confirmatory partitions: median | minimum | partitions with exactly two |
|---|---:|---:|---:|---:|
| MAGIC | 7 | 9 | 6 | 0 of 30 |
| Spambase | 2 | 3 | 2 | 9 of 30 |
| Adult | 6 | 6 | 4 | 0 of 30 |
| Bank Marketing | 6 | 5 | 3 | 0 of 30 |

## S6. Method and control registry

| identifier | kind | runner stage | standalone budget | revalidated | isolates |
|---|---|---|---|---|---|
| `HISTORICAL-WS-asrun` | arm | `historical_ws_asrun` | 108 | yes | the frozen v0.1.0-dissertation solver, called unmodified, with the archived maximization orientation, observed-extrema normalization and asymmetric grid with no pure-quality vertex (20 points at the solver's default step), and no gate; its uncoded natural-unit surfaces follow the archived specification but are fitted by this study's code to this study's objectives. Reproduced, not repaired. |
| `HISTORICAL-WS` | arm | `historical_ws` | 186 | yes | the same weighted sum, solver, symmetric grid and surrogates as WS-S, differing ONLY in the normalization reference. WS-S to HISTORICAL-WS is therefore a single-factor contrast on normalization. |
| `WS-S` | arm | `ws_s` | 186 | yes | weighted-sum scalarization over the surrogate payoff reference. The geometry baseline for the primary contrast. |
| `NBI-S` | arm | `nbi_s` | 186 | yes | Normal Boundary Intersection on the same surrogates and the same surrogate reference as WS-S. WS-S to NBI-S isolates front-construction geometry. |
| `NBI-R` | arm | `nbi_r` | 386 | yes | NBI with anchors and payoff matrix from direct search on the REAL objectives. NBI-S to NBI-R isolates anchor and payoff provenance. |
| `ANCHOR-INJECTION-CONTROL` | control | `anchor_injection_control` | — | yes | NBI-S's own revalidated candidate set augmented with the same empirical anchors NBI-R receives, changing nothing else. It measures the indicator gain from adding those anchors alone to NBI-S's set; it was frozen to separate the set-composition part of the NBI-S to NBI-R gap from the relocated-geometry part, a reading that holds only if NBI-R returns its own anchors. Costs no new real evaluations. |
| `GRID` | baseline | `direct_baselines` | — | n/a | direct-search comparator at the frozen 386-evaluation comparator budget, which is matched to the most expensive arm (NBI-R) and not pairwise to each arm |
| `RANDOM` | baseline | `direct_baselines` | — | n/a | direct-search comparator at the frozen 386-evaluation comparator budget, matched to the most expensive arm and not pairwise to each arm |
| `BAYES-QUALITY` | baseline | `direct_baselines` | — | n/a | single-objective Bayesian optimization on quality, at the frozen comparator budget; excluded from front indicators and from the AUGMENTED reference |
| `BAYES-COST` | baseline | `direct_baselines` | — | n/a | single-objective Bayesian optimization on cost; same exclusion |
| `TPE-QUALITY` | baseline | `direct_baselines` | — | n/a | single-objective TPE on quality; same exclusion |
| `TPE-COST` | baseline | `direct_baselines` | — | n/a | single-objective TPE on cost; same exclusion |
| `NSGA2-MATCHED` | baseline | `direct_baselines` | — | n/a | NSGA-II at 32 x 12 = 384 evaluations, run at the frozen comparator budget to within a disclosed two-evaluation shortfall (384 against 386); this is a budget-level match to the most expensive arm, not a pairwise match to NBI-S |
| `NSGA2-UNMATCHED` | context | `nsga2_unmatched` | — | n/a | NSGA-II at 10x the matched budget, one replication per dataset. A CONTEXT baseline: no fairness claim attaches to it and it enters no budget-matched comparison. |
| `DESIGN` | shared stage | `design` | — | n/a | 88 face-centred central composite runs |
| `EXTERNAL-VALIDATION-AUDIT` | shared stage, AUDIT-ONLY | `external_validation` | — | n/a | 78 points: the design's complementary half fraction plus 14 axial runs at half radius. Reaches the reliability gate diagnostics and NOTHING else. A gate failure changes no execution. |
| `SURROGATE-ANCHORS` | reference construction | `surrogate_anchors` | — | n/a | per-objective minimization of the fitted surrogates over the coded box |
| `EMPIRICAL-ANCHORS` | reference construction | `empirical_anchors` | — | n/a | direct search on the REAL objectives, one budget per objective. Best found within the declared budget; never described as certified optima. |
| `REFERENCE-CORE` | reference construction | `reference_core` | — | n/a | the reference front used for indicators |
| `AUGMENTED-REFERENCE` | reference construction | `augmented_reference` | — | n/a | reports self_grading_share_of_front |
| `HOLDOUT-CONFIRMATION` | confirmation stage | `holdout_confirmation` | — | n/a | selected candidates re-measured on the held-out partition |

## S7. Budget

| method | stage | logical per unit |
|---|---|---:|
| `anchor_injection_control` | `candidate_validation` | 2 |
| `bayes_cost` | `direct_search` | 386 |
| `bayes_quality` | `direct_search` | 386 |
| `design` | `design` | 88 |
| `empirical_anchor_search` | `anchor` | 200 |
| `external_validation_audit` | `external_audit` | 78 |
| `grid` | `direct_search` | 386 |
| `historical_ws_asrun_revalidation` | `candidate_validation` | 20 |
| `historical_ws_revalidation` | `candidate_validation` | 20 |
| `holdout_confirmation` | `holdout_audit` | 5 |
| `nbi_r_revalidation` | `candidate_validation` | 20 |
| `nbi_s_revalidation` | `candidate_validation` | 20 |
| `nsga2` | `direct_search` | 384 |
| `random` | `direct_search` | 386 |
| `tpe_cost` | `direct_search` | 386 |
| `tpe_quality` | `direct_search` | 386 |
| `ws_s_revalidation` | `candidate_validation` | 20 |
| **total** | | **3,173** |

Of which 83 audit-only and 3,090 solution-producing.

| quantity | value |
|---|---:|
| logical evaluations per unit | 3,173 |
| confirmatory units | 120 |
| unmatched NSGA-II, outside the unit | 15,360 |
| **campaign total** | **396,120** |
| of which audit-only | 9,960 |
| unique physical evaluations | 377,316 |

**The campaign total already contains the unmatched NSGA-II charge**, inside the
four replication-0 ledgers. Adding it again gives 411,480 and is wrong.

### S7.1 Standalone cost per arm

| arm | standalone logical |
|---|---:|
| HISTORICAL-WS-asrun | 108 |
| HISTORICAL-WS | 186 |
| WS-S | 186 |
| NBI-S | 186 |
| NBI-R | 386 |
| each direct-search comparator except NSGA-II | 386 |
| NSGA-II, matched (population 32, 12 generations) | 384 |

The comparator budget is the maximum over arms (NBI-R). NSGA-II's frozen population and
generation count give 384 real evaluations, 2 fewer than that budget. A
direct-search comparator therefore receives about twice the real evaluations WS-S and
NBI-S require standalone, and any comparison against them must say so.

## S8. Randomness

Seeds derive from BLAKE2b over the namespace `xgboost-hpo-vrfnbi/v3` and the tuple
(dataset, replication, method, stage), fed to `numpy.random.SeedSequence`.

| dataset | unit seed, replication 0 | example method seed |
|---|---:|---:|
| MAGIC | 20261914 | 12501833259513610416 |
| Spambase | 20262914 | 1639200741726807977 |
| Adult | 20263914 | 3386437749439664214 |
| Bank Marketing | 20264914 | 7633668092872975921 |

All 3,840 campaign streams are distinct and stable across processes and
`PYTHONHASHSEED`.

## S9. Primary statistics, both references


### S9.1 CORE reference (primary)

| dataset | contrast | median Δ | 95% CI | W/T/L | rank-biserial | Holm $p$ | Nadeau–Bengio $p$ |
|---|---|---:|---|---:|---:|---:|---:|
| MAGIC | HISTORICAL-WS -> WS-S | +0.0000 | [-0.0022, +0.0056] | 14/3/13 | +0.14 | 0.5165 | 0.6697 |
| MAGIC | WS-S -> NBI-S | +0.0374 | [+0.0144, +0.0687] | 26/0/4 | +0.92 | 9.425e-07 | 0.0860 |
| MAGIC | NBI-S -> NBI-R | -0.0741 | [-0.0957, -0.0482] | 2/0/28 | -0.99 | 2.794e-08 | 0.0254 |
| Adult | HISTORICAL-WS -> WS-S | -0.0006 | [-0.0033, +0.0000] | 9/4/17 | -0.28 | 0.2087 | 0.7467 |
| Adult | WS-S -> NBI-S | +0.2549 | [+0.1740, +0.5614] | 30/0/0 | +1.00 | 5.588e-09 | 0.0148 |
| Adult | NBI-S -> NBI-R | -0.0192 | [-0.1610, +0.0299] | 11/0/19 | -0.37 | 0.1607 | 0.4161 |
| Bank Marketing | HISTORICAL-WS -> WS-S | +0.0000 | [-0.0088, +0.0000] | 8/8/14 | -0.21 | 0.3896 | 0.8893 |
| Bank Marketing | WS-S -> NBI-S | +0.2281 | [+0.1670, +0.4741] | 24/0/6 | +0.88 | 9.717e-06 | 0.0764 |
| Bank Marketing | NBI-S -> NBI-R | -0.4093 | [-0.5821, -0.2059] | 7/0/23 | -0.82 | 4.733e-05 | 0.0791 |
| Spambase | HISTORICAL-WS -> WS-S | +0.0018 | [+0.0000, +0.0398] | 16/7/7 | +0.41 | 0.1175 | 0.6846 |
| Spambase | WS-S -> NBI-S | +0.0500 | [-0.0043, +0.4369] | 18/2/10 | +0.41 | 0.1175 | 0.6550 |
| Spambase | NBI-S -> NBI-R | -0.6932 | [-0.9839, -0.5217] | 5/0/25 | -0.78 | 0.0005664 | 0.2763 |

The Holm $p$ column is the secondary test: a two-sided Wilcoxon signed-rank test,
Holm-adjusted within each dataset family (Spambase forms its own family). The
descriptive triple on the CORE reference (median $\Delta$ with its 95% percentile
bootstrap interval, W/T/L and rank-biserial) is the primary evidence; the AUGMENTED
reference is a pre-declared sensitivity. The Nadeau–Bengio column is the pre-declared
corrected resampled $t$, at an inflation of $\sqrt{8.5} = 2.9155$, not adjusted for
multiplicity. It is a sensitivity, not the study's significance test, and is reported in
full including where it is adverse to the finding.

### S9.2 AUGMENTED reference (mandatory sensitivity)

| dataset | contrast | median Δ | 95% CI | W/T/L | rank-biserial | Holm $p$ | Nadeau–Bengio $p$ |
|---|---|---:|---|---:|---:|---:|---:|
| MAGIC | HISTORICAL-WS -> WS-S | +0.0000 | [-0.0029, +0.0047] | 13/3/14 | +0.05 | 0.8288 | 0.7807 |
| MAGIC | WS-S -> NBI-S | +0.0382 | [+0.0179, +0.0668] | 27/0/3 | +0.95 | 2.049e-07 | 0.0927 |
| MAGIC | NBI-S -> NBI-R | -0.0794 | [-0.0939, -0.0451] | 1/0/29 | -0.99 | 2.794e-08 | 0.0286 |
| Adult | HISTORICAL-WS -> WS-S | -0.0006 | [-0.0018, +0.0000] | 8/4/18 | -0.23 | 0.3158 | 0.7431 |
| Adult | WS-S -> NBI-S | +0.1936 | [+0.1322, +0.3717] | 30/0/0 | +1.00 | 5.588e-09 | 0.0204 |
| Adult | NBI-S -> NBI-R | -0.0535 | [-0.1181, +0.0035] | 10/0/20 | -0.54 | 0.01741 | 0.3756 |
| Bank Marketing | HISTORICAL-WS -> WS-S | -0.0001 | [-0.0021, +0.0000] | 7/8/15 | -0.23 | 0.3382 | 0.7957 |
| Bank Marketing | WS-S -> NBI-S | +0.1792 | [+0.1074, +0.2045] | 25/0/5 | +0.94 | 7.655e-07 | 0.0643 |
| Bank Marketing | NBI-S -> NBI-R | -0.1699 | [-0.2369, -0.0772] | 7/0/23 | -0.82 | 4.733e-05 | 0.1109 |
| Spambase | HISTORICAL-WS -> WS-S | +0.0000 | [+0.0000, +0.0096] | 14/11/5 | +0.48 | 0.1283 | 0.6545 |
| Spambase | WS-S -> NBI-S | +0.0503 | [+0.0000, +0.2079] | 19/2/9 | +0.33 | 0.1283 | 0.6531 |
| Spambase | NBI-S -> NBI-R | -0.3464 | [-0.4880, -0.2553] | 4/1/25 | -0.84 | 0.0002382 | 0.0658 |

The Holm $p$ column is the secondary test: a two-sided Wilcoxon signed-rank test,
Holm-adjusted within each dataset family (Spambase forms its own family). The
descriptive triple on the CORE reference (median $\Delta$ with its 95% percentile
bootstrap interval, W/T/L and rank-biserial) is the primary evidence; the AUGMENTED
reference is a pre-declared sensitivity. The Nadeau–Bengio column is the pre-declared
corrected resampled $t$, at an inflation of $\sqrt{8.5} = 2.9155$, not adjusted for
multiplicity. It is a sensitivity, not the study's significance test, and is reported in
full including where it is adverse to the finding.

Self-grading share of the AUGMENTED front, which the frozen protocol requires beside
every AUGMENTED contrast; each contrast above compares two of the methods tabulated here.
The share is the fraction of the AUGMENTED reference's non-dominated points contributed
by the points a method is scored on (an arm's revalidated candidates, a comparator's
evaluations), median over the 30 replications, from
`analysis/indicators_long.csv`. A front point contributed by more than one method counts
once for each of them. The anchor-injection control contributes no candidates of its own
to the reference.

| dataset | HISTORICAL-WS-asrun | HISTORICAL-WS | WS-S | NBI-S | NBI-R | GRID | RANDOM | NSGA2-MATCHED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAGIC | 0.000 | 0.055 | 0.058 | 0.104 | 0.080 | 0.182 | 0.020 | 0.335 |
| Spambase | 0.000 | 0.036 | 0.080 | 0.200 | 0.000 | 0.121 | 0.000 | 0.000 |
| Adult | 0.000 | 0.070 | 0.069 | 0.135 | 0.132 | 0.212 | 0.000 | 0.180 |
| Bank Marketing | 0.000 | 0.093 | 0.059 | 0.121 | 0.018 | 0.204 | 0.000 | 0.214 |

## S10. Secondary indicators

| dataset | entity | HV ratio | IGD⁺ | GD | spacing | joint-ND | front size |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGIC | HISTORICAL-WS-asrun | 0.7627 | 0.1477 | 0.1174 | 0.0460 | 0.1458 | 8.0000 |
| MAGIC | HISTORICAL-WS | 0.8613 | 0.1028 | 0.1153 | 0.0890 | 0.5798 | 10.0000 |
| MAGIC | WS-S | 0.8614 | 0.0930 | 0.1013 | 0.0862 | 0.5857 | 9.0000 |
| MAGIC | NBI-S | 0.9430 | 0.0511 | 0.0781 | 0.0496 | 0.5917 | 11.0000 |
| MAGIC | NBI-R | 0.8588 | 0.0856 | 0.1015 | 0.0244 | 0.6667 | 9.0000 |
| MAGIC | ANCHOR-INJECTION-CONTROL | 0.9607 | 0.0444 | 0.0713 | 0.0979 | 0.6259 | 12.5000 |
| MAGIC | GRID | 1.0068 | 0.0036 | 0.0393 | 0.0706 | 0.9181 | 17.0000 |
| MAGIC | RANDOM | 0.8907 | 0.0629 | 0.0775 | 0.0382 | 0.4416 | 13.0000 |
| MAGIC | NSGA2-MATCHED | 0.9580 | 0.0373 | 0.0858 | 0.0188 | 0.6883 | 30.5000 |
| Spambase | HISTORICAL-WS-asrun | 0.0221 | 0.6561 | 0.7892 | 0.1642 | 0.0000 | 4.0000 |
| Spambase | HISTORICAL-WS | 0.8583 | 0.1312 | 0.1837 | 0.1445 | 0.8750 | 3.0000 |
| Spambase | WS-S | 0.8819 | 0.1054 | 0.2042 | 0.2049 | 1.0000 | 3.0000 |
| Spambase | NBI-S | 1.0812 | 0.0877 | 0.2343 | 0.2142 | 0.9444 | 4.0000 |
| Spambase | NBI-R | 0.0793 | 0.6200 | 0.5881 | 0.0000 | 1.0000 | 3.0000 |
| Spambase | ANCHOR-INJECTION-CONTROL | 1.0812 | 0.0877 | 0.2343 | 0.2142 | 0.9444 | 4.0000 |
| Spambase | GRID | 1.0192 | 0.0000 | 0.0394 | 0.2372 | 1.0000 | 5.0000 |
| Spambase | RANDOM | 0.0000 | 1.3453 | 1.3795 | 0.0256 | 0.0000 | 2.0000 |
| Spambase | NSGA2-MATCHED | 0.4164 | 0.4738 | 0.5878 | 0.1166 | 0.2500 | 5.0000 |
| Adult | HISTORICAL-WS-asrun | 0.5196 | 0.3404 | 0.2414 | 0.0176 | 0.0000 | 10.5000 |
| Adult | HISTORICAL-WS | 0.5422 | 0.1847 | 0.1483 | 0.0930 | 0.6970 | 7.5000 |
| Adult | WS-S | 0.6028 | 0.1688 | 0.1785 | 0.0890 | 0.7321 | 7.0000 |
| Adult | NBI-S | 0.9951 | 0.0579 | 0.1077 | 0.1329 | 0.8258 | 8.5000 |
| Adult | NBI-R | 0.9098 | 0.1201 | 0.1374 | 0.0503 | 0.7071 | 9.0000 |
| Adult | ANCHOR-INJECTION-CONTROL | 0.9951 | 0.0578 | 0.1397 | 0.1483 | 0.7386 | 10.0000 |
| Adult | GRID | 1.0476 | 0.0000 | 0.0465 | 0.0947 | 1.0000 | 8.5000 |
| Adult | RANDOM | 0.4369 | 0.3334 | 0.2441 | 0.0984 | 0.0000 | 5.0000 |
| Adult | NSGA2-MATCHED | 0.8650 | 0.1228 | 0.1821 | 0.0359 | 0.4715 | 28.0000 |
| Bank Marketing | HISTORICAL-WS-asrun | 0.2743 | 0.4275 | 0.2907 | 0.0784 | 0.0000 | 6.0000 |
| Bank Marketing | HISTORICAL-WS | 0.6082 | 0.1946 | 0.2087 | 0.1804 | 0.6333 | 7.0000 |
| Bank Marketing | WS-S | 0.6010 | 0.1948 | 0.2320 | 0.1862 | 0.5000 | 7.0000 |
| Bank Marketing | NBI-S | 0.9979 | 0.0915 | 0.2173 | 0.1219 | 0.6111 | 9.0000 |
| Bank Marketing | NBI-R | 0.5330 | 0.3582 | 0.2468 | 0.0505 | 0.2917 | 6.0000 |
| Bank Marketing | ANCHOR-INJECTION-CONTROL | 0.9979 | 0.0898 | 0.1754 | 0.1358 | 0.6667 | 8.5000 |
| Bank Marketing | GRID | 1.0625 | 0.0000 | 0.0481 | 0.1432 | 1.0000 | 8.0000 |
| Bank Marketing | RANDOM | 0.3190 | 0.3257 | 0.3119 | 0.1037 | 0.0000 | 7.0000 |
| Bank Marketing | NSGA2-MATCHED | 0.9674 | 0.0934 | 0.2182 | 0.0451 | 0.4929 | 20.5000 |

**Undefined spacing.** 140 cells are non-finite — spacing and
spacing_cv on the 70 method-by-reference blocks whose front has one
point. Schott spacing requires at least two gaps, so NaN is the correct value and
those units are excluded from spacing summaries rather than propagated.

## S11. Surrogate-gate regimes

| dataset | quality gate | cost gate | both | role |
|---|---:|---:|---:|---|
| MAGIC | 8/30 | 100% | 27% | `primary_geometry_confirmatory` |
| Spambase | 9/30 | 100% | 30% | `boundary_geometry_control` |
| Adult | 0/30 | 73% | 0% | `primary_geometry_confirmatory` |
| Bank Marketing | 0/30 | 90% | 0% | `primary_geometry_confirmatory` |

The gate is diagnostic, never adaptive: every arm ran at every replication
whatever the gate said.

Every primary contrast by composite-quality gate status, from
`analysis/indicators_long.csv`: median paired difference (second minus first) in
the CORE-relative hypervolume ratio, with W/T/L, descriptive and untested. The cost
gate passed in every MAGIC and Spambase replication, so these are also the strata of
the joint gate. No Adult or Bank Marketing replication passed, so for those datasets
the failing stratum is the whole sample of S9.1; n/a marks an empty stratum.

| dataset | contrast | gate passed: n, median difference, W/T/L | gate failed: n, median difference, W/T/L |
|---|---|---|---|
| MAGIC | HISTORICAL-WS -> WS-S | 8, -0.0031, 2/1/5 | 22, +0.0018, 12/2/8 |
| MAGIC | WS-S -> NBI-S | 8, +0.1173, 8/0/0 | 22, +0.0165, 18/0/4 |
| MAGIC | NBI-S -> NBI-R | 8, -0.0577, 1/0/7 | 22, -0.0901, 1/0/21 |
| Spambase (boundary) | HISTORICAL-WS -> WS-S | 9, +0.0000, 3/4/2 | 21, +0.0173, 13/3/5 |
| Spambase (boundary) | WS-S -> NBI-S | 9, +0.0000, 4/1/4 | 21, +0.0641, 14/1/6 |
| Spambase (boundary) | NBI-S -> NBI-R | 9, -0.5238, 3/0/6 | 21, -0.8770, 2/0/19 |
| Adult | HISTORICAL-WS -> WS-S | 0, n/a, n/a | 30, -0.0006, 9/4/17 |
| Adult | WS-S -> NBI-S | 0, n/a, n/a | 30, +0.2549, 30/0/0 |
| Adult | NBI-S -> NBI-R | 0, n/a, n/a | 30, -0.0192, 11/0/19 |
| Bank Marketing | HISTORICAL-WS -> WS-S | 0, n/a, n/a | 30, +0.0000, 8/8/14 |
| Bank Marketing | WS-S -> NBI-S | 0, n/a, n/a | 30, +0.2281, 24/0/6 |
| Bank Marketing | NBI-S -> NBI-R | 0, n/a, n/a | 30, -0.4093, 7/0/23 |

## S12. Controls

### S12.1 Anchor-injection control

| dataset | injection effect | full NBI-S→NBI-R gap | CHIM extent ratio (post hoc) |
|---|---:|---:|---:|
| MAGIC | +0.0133 | -0.0741 | 0.811 |
| Spambase | +0.0000 | -0.6932 | 0.370 |
| Adult | +0.0000 | -0.0192 | 0.204 |
| Bank Marketing | +0.0000 | -0.4093 | 0.272 |

The control set is a superset of NBI-S's, so its effect is non-negative by
construction. The CHIM extent ratio is post hoc: it was first computed after the
confirmatory results, from the recorded payoff matrices, and is not part of the frozen
protocol. Within-dataset Spearman between the per-replication CHIM extent ratio and the
hypervolume gap: MAGIC -0.11, Spambase +0.32, Adult +0.68, Bank Marketing +0.68. A post hoc association, not a cause: clear on Adult and Bank Marketing, weak on
Spambase and absent on MAGIC.

### S12.2 Solver health, NBI-S against NBI-R

| dataset | certified (S/R) | max equality residual (S/R) | max rounding displacement (S/R) | distinct configs (S/R) |
|---|---|---|---|---|
| MAGIC | 1.000 / 1.000 | 8.6e-10 / 9.3e-10 | 2.8e-02 / 5.2e-03 | 20 / 20 |
| Spambase | 1.000 / 1.000 | 5.9e-10 / 5.8e-10 | 1.7e-03 / 1.7e-12 | 20 / 20 |
| Adult | 1.000 / 1.000 | 4.4e-10 / 5.6e-10 | 2.7e-02 / 8.3e-04 | 20 / 20 |
| Bank Marketing | 1.000 / 1.000 | 6.1e-10 / 6.8e-10 | 8.2e-03 / 9.5e-04 | 20 / 20 |

Solver behaviour was comparable between `NBI-S` and `NBI-R` and does not explain
the deficit. Across the 240 arm-units the certified fraction had median 1.000,
with two exceptions at 0.900 and 0.950; the per-unit maximum equality residual had
median 6.6e-10, against a campaign maximum of 6.9e-1 on a single MAGIC `NBI-S`
unit; and both arms returned 20 distinct realized configurations in 239 of the
240 arm-units, with 19 in that same MAGIC `NBI-S` unit. The table entries are
per-dataset medians of per-unit values; the rounding-displacement column is, per unit,
the largest objective displacement induced by rounding on the fitted surfaces.
Solver behaviour is therefore comparable, but not identical, and the deficit is
not solver failure or candidate collapse. The protocol's fixed-integer re-solve was not
implemented (Section 8.7), so the rounded points actually evaluated carry no equality
certificate.

### S12.3 Historical reconstruction

| dataset | as-run | shared specification | WS-S | Δ(shared − as-run) |
|---|---:|---:|---:|---:|
| MAGIC | 0.7627 | 0.8613 | 0.8614 | +0.0994 |
| Spambase | 0.0221 | 0.8583 | 0.8819 | +0.5404 |
| Adult | 0.5196 | 0.5422 | 0.6028 | +0.1480 |
| Bank Marketing | 0.2743 | 0.6082 | 0.6010 | +0.3126 |

## S13. Baselines

| dataset | GRID | RANDOM | NSGA-II matched | WS-S | NBI-S | NBI-S wins vs GRID |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | 1.0068 | 0.8907 | 0.9580 | 0.8614 | 0.9430 | 0/30 |
| Spambase | 1.0192 | 0.0000 | 0.4164 | 0.8819 | 1.0812 | 12/30 |
| Adult | 1.0476 | 0.4369 | 0.8650 | 0.6028 | 0.9951 | 5/30 |
| Bank Marketing | 1.0625 | 0.3190 | 0.9674 | 0.6010 | 0.9979 | 10/30 |

The unmatched NSGA-II run receives ten times the matched budget on one
replication per dataset and is a context baseline: no fairness claim attaches
to it and it enters no budget-matched comparison and neither reference.

### S13.1 Single-objective Bayesian-optimization and TPE runs

Declared in the frozen protocol (section 3): the four single-objective runs produce
single-objective optima, not fronts, so they are reported on their own endpoints only,
labelled as single-objective runs, and enter neither the front-indicator tables nor either
reference. Computed after the campaign from stored outputs by
`scripts/single_objective_endpoints.py`. Each run used the comparator budget of 386 real
evaluations per replication. The endpoint is the best value of the run's own objective
($f_{\text{quality}}$ or $f_{\text{cost}}$, both minimized) among its real evaluations.
For context only, the last column gives the median best value of the same objective in
each replication's CORE reference (88 design rows and 200 anchor-search rows), to which no
compared method contributes; that comparison was not
pre-specified, is descriptive and carries no test.

| dataset | run | objective | best value: median (interquartile range) | CORE best value: median |
|---|---|---|---:|---:|
| MAGIC | BAYES-QUALITY | $f_{\text{quality}}$ | -0.810 (-0.826 to -0.798) | -0.735 |
| MAGIC | BAYES-COST | $f_{\text{cost}}$ | -1.800 (-1.847 to -1.785) | -2.052 |
| MAGIC | TPE-QUALITY | $f_{\text{quality}}$ | -0.805 (-0.822 to -0.791) | -0.735 |
| MAGIC | TPE-COST | $f_{\text{cost}}$ | -1.794 (-1.850 to -1.751) | -2.052 |
| Spambase | BAYES-QUALITY | $f_{\text{quality}}$ | -0.855 (-0.872 to -0.819) | -0.845 |
| Spambase | BAYES-COST | $f_{\text{cost}}$ | -2.586 (-2.710 to -2.363) | -3.028 |
| Spambase | TPE-QUALITY | $f_{\text{quality}}$ | -0.816 (-0.848 to -0.768) | -0.845 |
| Spambase | TPE-COST | $f_{\text{cost}}$ | -2.310 (-2.515 to -2.170) | -3.028 |
| Adult | BAYES-QUALITY | $f_{\text{quality}}$ | -0.707 (-0.712 to -0.700) | -0.683 |
| Adult | BAYES-COST | $f_{\text{cost}}$ | -2.237 (-2.310 to -2.188) | -2.387 |
| Adult | TPE-QUALITY | $f_{\text{quality}}$ | -0.706 (-0.713 to -0.700) | -0.683 |
| Adult | TPE-COST | $f_{\text{cost}}$ | -2.163 (-2.238 to -2.010) | -2.387 |
| Bank Marketing | BAYES-QUALITY | $f_{\text{quality}}$ | -0.930 (-0.946 to -0.910) | -0.887 |
| Bank Marketing | BAYES-COST | $f_{\text{cost}}$ | -2.583 (-2.639 to -2.457) | -2.516 |
| Bank Marketing | TPE-QUALITY | $f_{\text{quality}}$ | -0.919 (-0.934 to -0.909) | -0.887 |
| Bank Marketing | TPE-COST | $f_{\text{cost}}$ | -2.402 (-2.530 to -2.207) | -2.516 |

### S13.2 Unmatched NSGA-II run

Declared in `protocol/NSGA2_DECISION.md` (section 5): the unmatched run, ten times the
matched budget on one replication per dataset, asks only whether a weak matched result is
explained by budget starvation. It is not an efficiency comparison, no claim about any arm
is made relative to it, and it is reported in its own table with the generation count
stated. Computed after the campaign from stored outputs by `scripts/nsga2_unmatched_table.py`,
which scores the stored rows of the run against that replication's CORE reference with the
frozen indicator code.

| dataset | replication | real evaluations | population | generations | CORE-relative HV ratio, unmatched | matched NSGA-II, same replication | non-dominated rows (distinct points) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAGIC | 0 | 3,840 | 32 | 120 | 1.0269 | 0.9357 | 186 (105) |
| Spambase | 0 | 3,840 | 32 | 120 | 6.4459 | 0.0937 | 1091 (403) |
| Adult | 0 | 3,840 | 32 | 120 | 1.4053 | 0.8060 | 176 (108) |
| Bank Marketing | 0 | 3,840 | 32 | 120 | 2.0001 | 0.7862 | 177 (72) |

The unmatched budget is ten times the realized matched budget, 3,840 real evaluations
(ten times 384), as fixed in `protocol/NSGA2_DECISION.md` (section 5) and executed by the
runner; the protocol's general section states 3,860 (ten times 386). Non-dominated rows
count every non-dominated stored row, so repeated configurations of the converged
population are counted separately; the distinct points are in parentheses. Spambase's CORE
front has only three points in this replication, so its normalization box is narrow and
the hypervolume ratio is unbounded above (S15); its magnitude is not comparable with the
other datasets'.

## S14. Holdout confirmation

| dataset | arm | internal | holdout | median paired drop |
|---|---|---:|---:|---:|
| MAGIC | historical_ws | 0.8580 | 0.8604 | -0.0017 |
| MAGIC | historical_ws_asrun | 0.8477 | 0.8541 | -0.0034 |
| MAGIC | nbi_r | 0.8587 | 0.8603 | -0.0013 |
| MAGIC | nbi_s | 0.8593 | 0.8594 | -0.0015 |
| MAGIC | ws_s | 0.8598 | 0.8628 | -0.0016 |
| Spambase | historical_ws | 0.8871 | 0.8947 | -0.0059 |
| Spambase | historical_ws_asrun | 0.9046 | 0.9088 | -0.0044 |
| Spambase | nbi_r | 0.9185 | 0.9235 | -0.0117 |
| Spambase | nbi_s | 0.8898 | 0.9072 | -0.0110 |
| Spambase | ws_s | 0.8890 | 0.8990 | -0.0076 |
| Adult | historical_ws | 0.8408 | 0.8392 | +0.0016 |
| Adult | historical_ws_asrun | 0.8412 | 0.8406 | +0.0004 |
| Adult | nbi_r | 0.8433 | 0.8430 | -0.0018 |
| Adult | nbi_s | 0.8437 | 0.8445 | -0.0012 |
| Adult | ws_s | 0.8409 | 0.8415 | +0.0020 |
| Bank Marketing | historical_ws | 0.8985 | 0.8970 | +0.0009 |
| Bank Marketing | historical_ws_asrun | 0.8975 | 0.8974 | +0.0011 |
| Bank Marketing | nbi_r | 0.8984 | 0.8990 | -0.0002 |
| Bank Marketing | nbi_s | 0.8986 | 0.8973 | +0.0012 |
| Bank Marketing | ws_s | 0.8988 | 0.8973 | +0.0014 |

All magnitudes are below 0.012 and several are negative, meaning the held-out
partition scored better than the internal resampling. This is read as the absence
of gross selection optimism, **not** as a ranking of arms.

## S15. Finite-reference diagnostics

| dataset | rows with HV ratio > 1 | share | maximum |
|---|---:|---:|---:|
| MAGIC | 36/270 | 13% | 1.057 |
| Spambase | 95/270 | 35% | 9.912 |
| Adult | 76/270 | 28% | 1.176 |
| Bank Marketing | 93/270 | 34% | 2.349 |

Overall 27.8% of rows exceed 1, maximum
9.912. The CORE reference is a finite method-independent set of
288 points, not the true Pareto front; a ratio above 1 means the candidate set
improved on that finite reference.

## S16. Claims and evidence

The manuscript's headline numerical claims, each with its estimand, source
artifact, verification status, allowed wording and prohibited stronger wording,
are in `CONFIRMATORY_CLAIMS_AND_EVIDENCE.md`. Two are recorded in corrected form
because a post-campaign recomputation from the raw unit artifacts refuted the
author's version.

## S17. Reproducibility manifest

| item | value |
|---|---|
| manuscript source commit | `eeac042339b671b882a4c47d0fac2665f9e169e3` |
| final package tag | `paper2-manuscript-v6` |
| relationship | the package commit adds only this provenance metadata and the compiled PDFs; no manuscript text, analysis artifact or number differs |
| protocol tag | `xgboost-hpo-protocol-v3` |
| results tag | `xgboost-hpo-confirmatory-results-v1` |
| datasets | MAGIC, Spambase, Adult, Bank Marketing |
| replications | 30 per dataset, 120 units |
| primary endpoint | hv_ratio against the CORE reference |
| campaign runtime | 9 h 46 min, 14 workers × 1 thread |

Raw datasets and evaluation caches are deliberately unversioned; the design, the
frozen factor models, every analysis artifact and every script are committed.

