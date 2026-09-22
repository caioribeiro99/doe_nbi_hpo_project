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
| `xgboost-hpo-protocol-v3` | `9b15ba7` | the final frozen protocol, before any comparative result |
| `xgboost-hpo-confirmatory-results-v1` | `a4f20ed` | the first adversarially verified analysis |

The protocol tag records what was specified **before** any comparative result
existed; the results tag records the first verified analysis. Neither was moved.

## S2. Amendment chronology

The protocol carries **23 numbered amendments**, each recording what caused it
and whether any arm result had been observed when it was made. None had. The ledger
also records two process failures found by review rather than by the author, four
engineering defects caught by a pre-freeze smoke run, and the corrections made to
two claims after independent verification refuted the author's version. It is
reproduced in full in `PROTOCOL_AMENDMENTS.md`.

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
| MAGIC | 88 design rows | 0 | 0.5556, 0.4444 | 2 | 2.93 |
| Spambase | 88 design rows | 0 | 0.7749, 0.2251 | 2 | 2.10 |
| Adult | 88 design rows | 0 | 0.5470, 0.4530 | 1 | 1.34 |
| Bank Marketing | 88 design rows | 0 | 0.5875, 0.4125 | 1 | 1.06 |

One model per dataset, fitted on the 88 design rows only and applied to all 30
replications. The 78-point audit-only external construction is excluded from the
fit (amendment 23): 64 of its 78 coded points are identical to the external
validation set and the rest are the same axial runs.

## S5. Panel screening and dataset roles

| dataset | C1 raw conflict | C2 front / curvature | C3 min $R^2$ | C4 range | role |
|---|---:|---|---:|---:|---|
| MAGIC | -0.2509 | 7 / 0.1729 | 0.6913 | 2706.3× | `primary_geometry_confirmatory` |
| Spambase | -0.4621 | 2 / undefined | 0.8987 | 591.2× | `boundary_geometry_control` |
| Adult | -0.3977 | 6 / 0.1992 | 0.9078 | 3421.3× | `primary_geometry_confirmatory` |
| Bank Marketing | -0.4100 | 6 / 0.1983 | 0.8700 | 4355.5× | `primary_geometry_confirmatory` |

Roles were assigned from these pre-campaign measurements and never revised from
an optimizer outcome.

## S6. Method and control registry

| identifier | kind | runner stage | standalone budget | revalidated | isolates |
|---|---|---|---|---|---|
| `HISTORICAL-WS-asrun` | arm | `historical_ws_asrun` | 108 | yes | the dissertation's pipeline reproduced bit-faithfully by calling the frozen v0.1.0-dissertation solver: maximization orientation, uncoded natural-unit surfaces, observed-extrema normalization, asymmetric 20-point grid with no pure-quality vertex, and no gate. Reproduced, not repaired. |
| `HISTORICAL-WS` | arm | `historical_ws` | 186 | yes | the same weighted sum, solver, symmetric grid and surrogates as WS-S, differing ONLY in the normalization reference. WS-S to HISTORICAL-WS is therefore a single-factor contrast on normalization. |
| `WS-S` | arm | `ws_s` | 186 | yes | weighted-sum scalarization over the surrogate payoff reference. The geometry baseline for the primary contrast. |
| `NBI-S` | arm | `nbi_s` | 186 | yes | Normal Boundary Intersection on the same surrogates and the same surrogate reference as WS-S. WS-S to NBI-S isolates front-construction geometry. |
| `NBI-R` | arm | `nbi_r` | 386 | yes | NBI with anchors and payoff matrix from direct search on the REAL objectives. NBI-S to NBI-R isolates anchor and payoff provenance. |
| `ANCHOR-INJECTION-CONTROL` | control | `anchor_injection_control` | — | yes | NBI-S's own revalidated candidate set augmented with the same empirical anchors NBI-R receives, changing nothing else. Separates the part of any NBI-S to NBI-R gap that is set composition from the part that is relocated geometry. Costs no new real evaluations. |
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
| unique physical fits | 377,316 |

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
| each direct-search comparator | 386 |

The comparator budget is the **maximum over arms** (NBI-R). A direct-search
comparator therefore receives about twice the real evaluations WS-S and NBI-S
require standalone, and any comparison against them must say so.

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

The Nadeau–Bengio column is the pre-declared corrected resampled $t$, at an
inflation of $\sqrt{8.5} = 2.9155$. It is a sensitivity, not the primary test,
and is reported in full including where it is adverse to the finding.

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

The Nadeau–Bengio column is the pre-declared corrected resampled $t$, at an
inflation of $\sqrt{8.5} = 2.9155$. It is a sensitivity, not the primary test,
and is reported in full including where it is adverse to the finding.

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

## S12. Controls

### S12.1 Anchor-injection control

| dataset | injection effect | full NBI-S→NBI-R gap | CHIM extent ratio |
|---|---:|---:|---:|
| MAGIC | +0.0133 | -0.0741 | 0.811 |
| Spambase | +0.0000 | -0.6932 | 0.370 |
| Adult | +0.0000 | -0.0192 | 0.204 |
| Bank Marketing | +0.0000 | -0.4093 | 0.272 |

The control set is a superset of NBI-S's, so its effect is non-negative by
construction. Within-dataset Spearman between the per-replication CHIM extent
ratio and the hypervolume gap: MAGIC -0.11, Spambase +0.32, Adult +0.68, Bank Marketing +0.68. Association, not cause, and absent on MAGIC.

### S12.2 Solver health, NBI-S against NBI-R

| dataset | certified (S/R) | max equality residual (S/R) | distinct configs (S/R) |
|---|---|---|---|
| MAGIC | 1.000 / 1.000 | 8.6e-10 / 9.3e-10 | 20 / 20 |
| Spambase | 1.000 / 1.000 | 5.9e-10 / 5.8e-10 | 20 / 20 |
| Adult | 1.000 / 1.000 | 4.4e-10 / 5.6e-10 | 20 / 20 |
| Bank Marketing | 1.000 / 1.000 | 6.1e-10 / 6.8e-10 | 20 / 20 |

Solver behaviour was comparable between `NBI-S` and `NBI-R` and does not explain
the deficit. Across the 240 arm-units the certified fraction had median 1.000,
with two exceptions at 0.900 and 0.950; the per-unit maximum equality residual had
median 6.6e-10, against a campaign maximum of 6.9e-1 on a single MAGIC `NBI-S`
unit; and both arms returned 20 distinct realized configurations in every unit.
Solver behaviour is therefore comparable, but not identical, and the deficit is
not solver failure, rounding or candidate collapse.

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
replication per dataset and is a **context baseline**: no fairness claim attaches
to it and it enters no budget-matched comparison and neither reference.

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
because independent verification refuted the author's version.

## S17. Reproducibility manifest

| item | value |
|---|---|
| manuscript source commit | `eeac042339b671b882a4c47d0fac2665f9e169e3` |
| final package tag | `paper2-manuscript-v5` |
| relationship | the package commit adds only this provenance metadata and the compiled PDFs; no manuscript text, analysis artifact or number differs |
| protocol tag | `xgboost-hpo-protocol-v3` |
| results tag | `xgboost-hpo-confirmatory-results-v1` |
| datasets | MAGIC, Spambase, Adult, Bank Marketing |
| replications | 30 per dataset, 120 units |
| primary endpoint | hv_ratio against the CORE reference |
| campaign runtime | 9 h 46 min, 14 workers × 1 thread |

Raw datasets and evaluation caches are deliberately unversioned; the design, the
frozen factor models, every analysis artifact and every script are committed.

