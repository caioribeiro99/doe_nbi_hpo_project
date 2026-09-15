# Protocol v3 — freeze report

**Status.** This protocol is **prospectively frozen, after documented
measurement-validation pilots and before the confirmatory campaign.** It is *not* described as
pre-registered: tag `xgboost-hpo-protocol-v1` is the evidence that a specification existed before
Stage A ran, and everything since has been amended in the open, in `PROTOCOL_AMENDMENTS.md`, with the
cause of each change recorded beside it.

> **No confirmatory comparative arm-level result was inspected before this protocol freeze.**

**Reviewed commit.** `b9a97c47121c2b6ecd06b79a15b5d1659435fde9` (`b9a97c4`), reviewed statically and
read-only by V14. Tree hash `7ff811c37d3a33c540b07cbc42a5bb8e6985848d`, byte-identical at the start
and end of the review; `git status` clean throughout.

**V14 verdict: GO TO FREEZE.** Zero blockers against the nine declared blocking conditions.

---

## 1. How this protocol arrived here

| Point | Tag / commit | What it is |
|---|---|---|
| Dissertation freeze | `v0.1.0-dissertation` | the historical implementation this paper reconstructs |
| Protocol v1 | `xgboost-hpo-protocol-v1` | frozen before the pilot |
| Stage A | — | measurement-validation pilot: design, external set, screening |
| Protocol v2 | `xgboost-hpo-protocol-v2` | the confirmatory protocol, amended by Stage A |
| Stage B | — | throughput and statistical-sensitivity calibration |
| Adversarial review | V10 → V14 | seven independent reviews; 23 numbered amendments |
| **Protocol v3** | `xgboost-hpo-protocol-v3` | **this document** |

Seven reviews ran. The first found ten blocking defects; the last found none. What each one cost is
recorded in `PROTOCOL_AMENDMENTS.md` rather than summarized away, including the six regressions the
author introduced while fixing earlier findings, and the two occasions on which a commit message
asserted a green test suite that was in fact red.

---

## 2. The two objectives

Seven responses are measured per evaluation. Six carry the `quality` role — `Accuracy_Mean`,
`Precision_Mean`, `Recall_Mean`, `Specificity_Mean`, `RocAuc_Mean`, `LogLoss_Mean` — and one carries
the `cost` role, `Leaves_Mean`.

**Objective 1, composite quality.** Responses are canonicalized to badness, standardized, reduced by
PCA to `k = 3` components and Varimax-rotated. Scores are formed as

```
S = Z · V · diag(1/√λ) · R
```

— standardized **before** rotation, so the factors are Pearson-orthogonal and the rotated loadings
describe the scores actually in use. The quality composite is the weighted mean of the quality-role
factors, weighted by their **normalized rotated sums of squared loadings**. Both corrections were
made after the original construction was found to produce factors correlated 0.379 to 0.698 and
weights read from unrotated eigenvalues indexed by rotated component.

**Objective 2, model complexity.** The cost-role factor, dominated by log leaf count. It is a
**deterministic model-complexity proxy** and is never described as training time; measured wall-clock
time is a secondary audit variable only.

Both objectives are minimized.

---

## 3. The factor model — fitted once per dataset, on the design rows only

`EXPERIMENT_PROTOCOL.md` §7.2, as corrected by **amendment 23**: one factor model per dataset, fitted
on **the Stage A 88 design rows and nothing else**, and **applied** to every replication. The
per-replication refit is computed and reported as a Tucker-congruence sensitivity, never applied.

The reference set originally included the 78-point complement, 166 points. That was withdrawn because
**the complement is the audit-only external validation construction** — 64 of its 78 coded points are
identical to `design.external_validation_set()` and the remaining 14 are the same axial runs,
differing only by integer rounding of `max_depth`. Fitting the objective definition on them made the
surrogate gate validate a fitted surface against data that had helped define that surface's target.

| dataset | fitting sample | SHA256 (16) | audit-only rows used |
|---|---:|---|---:|
| MAGIC | 88 design rows | `b7df350a109bc61c` | 0 |
| Spambase | 88 design rows | `2896e4e8182428f6` | 0 |
| Adult | 88 design rows | `922b43a05c5b3b88` | 0 |
| Bank Marketing | 88 design rows | `e516540cab3ebcc0` | 0 |

Why one model per dataset: a per-replication refit makes the objective a different variable in every
pair, so 30 paired indicator values would not live in one objective space and no normalized indicator
is invariant to that. Fitting per dataset on the design rows alone satisfies that requirement and the
leakage rule simultaneously.

---

## 4. Screening, and the dataset roles

### 4.1 Criterion 1 — the latent estimator is withdrawn, the construct is kept

**Amendment 20.** Criterion 1's scientific purpose is retained: establish before confirmatory
execution that a genuine predictive-quality-versus-model-complexity trade-off exists. Its
**operationalization as the Spearman between the latent quality composite and the latent cost factor
is permanently withdrawn.**

The reason is structural. Both are rotated factors of one orthogonal basis, so their linear
association is zero by construction on the fitting sample, and the rank statistic that remains has no
cross-dataset coherent direction — negative on MAGIC, positive on Spambase and Adult, indeterminate
on Bank Marketing. It does not operationalize one common directional construct across datasets.

Criterion 1 now uses the **canonical raw-response measurement**:
`doe_xgb.campaign.factor_model.raw_conflict`, the equally weighted standardized quality badness
against the **raw `Leaves_Mean` complexity response**, with no factor stage on either side. It is
imported, never reimplemented, and a test enforces that.

### 4.2 Criterion 2 — unchanged

Its threshold was not moved, its definition not rewritten, and no alternative formulation was sought.

### 4.3 The panel

Authoritative artifact: `audits/final_panel_screening.json`.

| dataset | C1 raw conflict | C2 front / curvature | C3 | C4 | role | units |
|---|---:|---|:--:|:--:|---|---:|
| MAGIC | −0.251 | 7 points / 0.173 | met | met | `primary_geometry_confirmatory` | 30 |
| Adult | −0.398 | 6 points / 0.199 | met | met | `primary_geometry_confirmatory` | 30 |
| Bank Marketing | −0.410 | 6 points / 0.198 | met | met | `primary_geometry_confirmatory` | 30 |
| **Spambase** | −0.462 | **2 points / undefined** | met | met | **`boundary_geometry_control`** | 30 |

**On what amendment 23 changed.** The screening measurements were **not** numerically unchanged by
removing the audit-set leakage. MAGIC's non-dominated set went from 8 points to 7 and its curvature
from 0.166 to 0.173. What survived is the **prospectively assigned dataset roles**, and that is the
stronger statement: a classification robust to refitting the objective on a different sample is
better evidence than one that never faced the test.

### 4.4 Spambase — a retained boundary control, not a replacement case

**Amendment 22.** A two-point non-dominated set has no interior: every scalarization returns the same
two extreme points, so there is no interior front geometry for the weighted-sum-versus-NBI contrast
to separate. The replacement rule of `dataset_selection.md` was **considered and deliberately not
exercised**; **no replacement dataset was selected.**

Spambase is **executed in full** — same arms, same budgets, same seeds, same evaluation machinery, 30
replications — and excluded **only** from the primary inferential family. The interpretation is fixed
in advance:

- a **null** WS-S → NBI-S difference on Spambase is **not** evidence against the geometry mechanism,
  because screening established before execution that the required interior geometry is absent;
- a **non-null** difference is reported and investigated only **after** campaign completion, and the
  protocol is not changed in response to it.

What the manuscript must say: *three datasets passed the geometry screening and constitute the
primary mechanistic panel; a fourth pre-screened dataset, Spambase, failed the nonlinear-front
criterion and was retained prospectively as a boundary control.*

---

## 5. Endpoints, references and the statistical family

**Amendment 21.** The **method-independent CORE reference is primary**: the 88 design rows plus the
200 anchor-search rows, 288 points before Pareto filtering, which no compared method contributes to.
The **AUGMENTED reference is a mandatory sensitivity** — the core together with every compared
method's real-revalidated candidates — reported for every contrast beside its
`self_grading_share_of_front`, and **never an alternate primary test**.

A core-relative hypervolume ratio above 1 is **not an error**: it means the method found points that
dominate part of the finite independent reference. It is a score relative to a finite
method-independent empirical reference and is never described as a fraction of the true Pareto
hypervolume.

**Primary endpoint:** the hypervolume ratio against the core reference.
**Secondary and descriptive:** IGD⁺, generational distance, Schott spacing, joint non-dominated
fraction. **IGD⁺ was never frozen as a co-primary endpoint** — at tag `xgboost-hpo-protocol-v2` no
indicator was designated primary at all; the hypervolume ratio became primary later as the resolution
of review finding MF17. That earlier declared status is preserved; amendment 21 resolves reference
provenance, not which endpoint is primary.

**The primary family.** Within each of MAGIC, Adult and Bank Marketing: the three identifying
contrasts on the core-reference hypervolume ratio, Holm-corrected **within that dataset**. **Dataset
is the generalization unit; there is no pooling** and no 90-replication analysis. Spambase's
identical comparisons form the boundary-control analysis.

If the direction of a headline comparison differs between the core and augmented references, **that
disagreement is itself a result and is reported prominently.** Neither may be chosen after the fact.

### Contrasts

| contrast | isolates |
|---|---|
| **WS-S → NBI-S** | front-construction geometry, and nothing else |
| **NBI-S → NBI-R** | anchor / payoff provenance, and nothing else |
| WS-S → HISTORICAL-WS | normalization (secondary) |

Isolation is enforced at run time: `_contrast_fingerprint` captures the solver configuration,
realizer, weight grid, surrogate identities and reference, and `_require_same` raises a
`MethodologicalFailure` at each arm's call site if any of them differs beyond the one declared
mechanism. `quasi_normal` is **derived from each arm's own payoff matrix** as n̂ = −Φ·1/‖Φ·1‖ and
must therefore differ between NBI-S and NBI-R; an identical one would mean the contrast varies
nothing, and that also fails.

---

## 6. The method and control registry

Every executed scientific entity, by exact identifier. Generated by
`scripts/method_registry.py`.

| identifier | kind | runner stage | standalone budget | revalidated | isolates |
|---|---|---|---|---|---|
| `HISTORICAL-WS-asrun` | arm | `historical_ws_asrun` | 108 | yes | the dissertation's pipeline reproduced bit-faithfully by calling the frozen v0.1.0-dissertation solver: maximization orientation, uncoded natural-unit surfaces, observed-extrema normalization, asymmetric 20-point grid with no pure-quality vertex, and no gate. Reproduced, not repaired. |
| `HISTORICAL-WS` | arm | `historical_ws` | 186 | yes | the same weighted sum, solver, symmetric grid and surrogates as WS-S, differing ONLY in the normalization reference. WS-S to HISTORICAL-WS is therefore a single-factor contrast on normalization. |
| `WS-S` | arm | `ws_s` | 186 | yes | weighted-sum scalarization over the surrogate payoff reference. The geometry baseline for the primary contrast. |
| `NBI-S` | arm | `nbi_s` | 186 | yes | Normal Boundary Intersection on the same surrogates and the same surrogate reference as WS-S. WS-S to NBI-S isolates front-construction geometry. |
| `NBI-R` | arm | `nbi_r` | 386 | yes | NBI with anchors and payoff matrix from direct search on the REAL objectives. NBI-S to NBI-R isolates anchor and payoff provenance. |
| `ANCHOR-INJECTION-CONTROL` | control | `anchor_injection_control` | — | yes | NBI-S's own revalidated candidate set augmented with the same empirical anchors NBI-R receives, changing nothing else. Separates the part of any NBI-S to NBI-R gap that is set composition from the part that is relocated geometry. Costs no new real evaluations. |
| `GRID` | baseline | `direct_baselines` | — | n/a | evaluation-matched direct search |
| `RANDOM` | baseline | `direct_baselines` | — | n/a | evaluation-matched direct search |
| `BAYES-QUALITY` | baseline | `direct_baselines` | — | n/a | single-objective Bayesian optimization on quality; excluded from front indicators and from the augmented reference |
| `BAYES-COST` | baseline | `direct_baselines` | — | n/a | single-objective Bayesian optimization on cost; same exclusion |
| `TPE-QUALITY` | baseline | `direct_baselines` | — | n/a | single-objective TPE on quality; same exclusion |
| `TPE-COST` | baseline | `direct_baselines` | — | n/a | single-objective TPE on cost; same exclusion |
| `NSGA2-MATCHED` | baseline | `direct_baselines` | — | n/a | NSGA-II at 32 x 12 = 384 evaluations, evaluation-matched against the comparator budget to within its stated shortfall |
| `NSGA2-UNMATCHED` | context | `nsga2_unmatched` | — | n/a | NSGA-II at 10x the matched budget, one replication per dataset. A CONTEXT baseline: no fairness claim attaches to it and it enters no budget-matched comparison. |
| `DESIGN` | shared stage | `design` | — | n/a | 88 face-centred central composite runs |
| `EXTERNAL-VALIDATION-AUDIT` | shared stage, AUDIT-ONLY | `external_validation` | — | n/a | 78 points: the design's complementary half fraction plus 14 axial runs at half radius. Reaches the reliability gate diagnostics and NOTHING else. A gate failure changes no execution. |
| `SURROGATE-ANCHORS` | reference construction | `surrogate_anchors` | — | n/a | per-objective minimization of the fitted surrogates over the coded box |
| `EMPIRICAL-ANCHORS` | reference construction | `empirical_anchors` | — | n/a | direct search on the REAL objectives, one budget per objective. Best found within the declared budget; never described as certified optima. |
| `REFERENCE-CORE` | reference construction | `reference_core` | — | n/a | the reference front used for indicators |
| `AUGMENTED-REFERENCE` | reference construction | `augmented_reference` | — | n/a | reports self_grading_share_of_front |
| `HOLDOUT-CONFIRMATION` | confirmation stage | `holdout_confirmation` | — | n/a | selected candidates re-measured on the held-out partition |

The two historical entities are **separate identifiers and cost different amounts**.
`HISTORICAL-WS-asrun` calls the frozen `v0.1.0-dissertation` solver unmodified, with its
maximization orientation, uncoded natural-unit surfaces, observed-extrema normalization and
asymmetric 20-point grid whose pure-quality vertex is absent; it has no gate and never pays
`B_surrogate_validation`. `HISTORICAL-WS` uses WS-S's own gated surrogates and differs from WS-S only
in the normalization reference. They were once both labelled `HISTORICAL-WS`, which would have merged
them in any table grouping by arm.

`bayes_quality`, `bayes_cost`, `tpe_quality` and `tpe_cost` are single-objective and are reported on
their own endpoint only; they enter neither the front-indicator table nor the augmented reference.

---

## 7. Randomness

Seeds are derived per `(dataset, replication, method, stage)` by BLAKE2b over the namespace
`xgboost-hpo-vrfnbi/v3`, fed to `numpy.random.SeedSequence`. The replication seed remains a
component, so a paired design still pairs: every method sees the same data split for a given
replication.

This replaced a single shared stream under which the grid's padding points were **bit-identical** to
random search's first 258 draws and the Bayesian and Parzen initial designs to its first 77 — five
baselines that were substantially one baseline. All 3,840 campaign streams are distinct, stable
across processes and `PYTHONHASHSEED`, and independent of execution order and worker count.
`seeding.as_uint32` narrows a seed only where a third-party API requires 32 bits; numpy generators
keep the full width.

Unit seeds: `SEED_BASE = 20260914`, e.g. MAGIC rep 0 → 20261914, Spambase rep 0 → 20262914, Adult
20263914, Bank Marketing 20264914.

---

## 8. Budget

The logical budget is **request-based**: every request is charged whether it hits the cache or not,
so rounding, duplicate candidates and physical cache hits cannot reduce a method's charge. Physical
fits are an engineering quantity and enter no fairness comparison.

### Per unit

| method | stage | logical |
|---|---|---:|
| design | design | 88 |
| external_validation_audit | external_audit | 78 |
| empirical_anchor_search | anchor | 200 |
| historical_ws_asrun_revalidation | candidate_validation | 20 |
| historical_ws_revalidation | candidate_validation | 20 |
| ws_s_revalidation | candidate_validation | 20 |
| nbi_s_revalidation | candidate_validation | 20 |
| nbi_r_revalidation | candidate_validation | 20 |
| anchor_injection_control | candidate_validation | 2 |
| holdout_confirmation | holdout_audit | 5 |
| grid / random / bayes_quality / bayes_cost / tpe_quality / tpe_cost | direct_search | 386 each |
| nsga2 | direct_search | 384 |
| **total** | | **3,173** |

Of which **83 audit-only** (78 external validation + 5 holdout confirmation) and **3,090
solution-producing**.

### Campaign, reconstructed independently

```
3,173 per unit × 120 units                = 380,760
unmatched NSGA-II, 32 × 12 × 10 × 4       =  15,360
                                            -------
                                             396,120
```

**396,120 logical evaluations** = 386,160 solution-producing + 9,960 audit-only, reconstructed from
the method-stage registry with **zero unexplained remainder** and confirmed against a completed
engineering unit's own request log, which charged 7,013 against 7,013 declared.

Two audit-only stages are **declared, not omitted**. Both were previously performed outside every
ledger — 240 anchor-injection and 600 holdout evaluations per campaign — while a reconciliation gate
that compared a sum of the registry against the same sum reported everything as balanced.

### Standalone per-arm cost

| arm | `B_total_solution` |
|---|---:|
| HISTORICAL-WS-asrun | 108 |
| HISTORICAL-WS | 186 |
| WS-S | 186 |
| NBI-S | 186 |
| NBI-R | **386** |
| each comparator | 386 |

NBI-R is the only arm that buys anchors with real evaluations, and the comparator budget is set to
the maximum over arms so the comparison is evaluation-matched.

**NSGA-II status.** The matched run is 32 × 12 = 384, two short of the comparator budget, and the
shortfall is published rather than hidden. The **unmatched** run receives ten times that budget on
**one replication per dataset** and is a **context baseline**: no fairness claim attaches to it, and
it enters no budget-matched comparison and neither reference.

---

## 9. Statistical sensitivity

Nadeau and Bengio (2003), derived rather than cited:

```
SE_corr² = (1/R + n_test/n_train) · s²
```

With `R = 30` and an 80/20 outer split, `n_test/n_train = 0.25`, so

```
SE_corr / SE_naive = √((1/30 + 0.25)/(1/30)) = √8.5 = 2.9155
```

This is **not** ρ = 0.25. An equicorrelation reading of ρ = 0.25 gives 3.3166 and corresponds to a
75/25 split, not this protocol's. The two coincide only when ρ = n_test/(n_test + n_train).

| dataset | detectable HV difference (80% power) | under the corrected test |
|---|---:|---:|
| MAGIC | 0.0079 | 0.0232 |
| Adult | 0.0152 | 0.0442 |
| Bank Marketing | 0.0226 | 0.0660 |
| Spambase (boundary control) | 0.0275 | 0.0802 |

The design resolves a **medium** effect, 0.532 paired standard deviations. It cannot resolve small
ones, and that is stated before any result exists. The corrected test is a **sensitivity and never
the primary test**: it was derived for the generalization error of a learner under repeated
resampling, and a Pareto quality indicator computed on a returned set is not that quantity.

---

## 10. Isolation guarantees

**External validation is audit-only.** The 78-point complementary half fraction plus 14 axial runs at
half radius reaches the surrogate reliability gate and **nothing else**. It is structurally
unreachable by any fitting stage: every fitting stage precedes it in `STAGES`, its rows exist only
inside one block, and nothing loads that checkpoint downstream. As of amendment 23 it is also absent
from the factor model's fitting sample — verified as a **zero** intersection in coded space on all
four datasets.

**The gate is diagnostic, not adaptive.** Every dataset runs every arm at every replication whatever
the gate says. Its pass rate is a result, not a filter, and every primary comparison is additionally
reported conditioned on gate status.

**Holdout labels are unavailable until confirmation.** One holdout flag, one holdout cache view,
created inside the holdout stage guard, with `metrics` the only stage after it. Proved dynamically
from a real unit's request log: every holdout evaluation is keyed under `fold_id="holdout"`, belongs
to the `holdout_audit` stage and to no other, and none is requested before that stage begins.

**Resume is equivalent to uninterrupted execution.** Request rows carry an attempt epoch and only the
latest attempt per `(method, stage)` is counted, so a mid-stage interrupt no longer double-charges the
scientific budget. Before this, killing a process part-way through one stage took that stage's charge
from 386 to 772 while the physical fit count stayed correct — invisible in the cache statistics, and
asymmetric across the comparison because it landed only on whichever method was interrupted.

---

## 11. Verification at the freeze

| | |
|---|---|
| **Reviewed commit** | `b9a97c47121c2b6ecd06b79a15b5d1659435fde9` |
| **V14 verdict** | **GO TO FREEZE**, zero blockers against nine declared conditions |
| Tree during review | byte-identical at start and end; `git status` clean |
| Scientific-executable change since the previously validated commit | **none** — the diff is one documentation file and one test file, zero lines under `src/` or `scripts/` |
| **Test suite** | **588 passed, 0 failed** |
| Dry run | clean, every frozen pre-launch proof green |
| Audit artifacts | all three reproduce byte-identically under `--check` |
| Claim blacklist | passes, 27 patterns over 18 documents |
| Campaign budget | 396,120, reconstructed independently with zero remainder |
| Engineering smoke | one full unit, 20 of 20 stages, real learner, ledger reconciling exactly |
| **Confirmatory arm-level result inspected** | **NO** |

Confirmatory blindness was checked four ways: no `experiments/xgboost_hpo_vrfnbi_confirmatory/`
output exists at any commit; the working tree holds only a plan file there; no evaluation database is
tracked in git; and no arm output artifact appears in `git log --diff-filter=A`.

---

## 12. What this freeze does not claim

- It does not claim the protocol was pre-registered. It was prospectively frozen, amended in the
  open after documented pilots, and every amendment records whether an arm result had been observed
  when it was made. None had.
- It does not claim the screening measurements were unchanged by amendment 23. They moved; the
  **roles** were robust to the movement.
- It does not claim the surrogate is reliable. In the corrected pre-campaign screening the frozen
  reliability criterion was not met by the composite-quality surrogate on three of four datasets.
  That is **screening evidence**, measured on one pilot partition per dataset. The confirmatory
  campaign recomputes the gate 30 times per dataset, and only those figures may be reported as
  findings.
- It does not claim criterion 1's replacement is as strong a screen as what it replaced. It is
  weaker: it cannot fail a dataset whose raw metrics trade off while its latent composite does not.
  That limitation is carried into the manuscript.
- It does not claim the review process was clean. Seven reviews were needed; the author introduced
  six regressions while fixing earlier findings and twice asserted a green test suite that was red.
  Those are recorded in `PROTOCOL_AMENDMENTS.md` in the same form as everything else.
