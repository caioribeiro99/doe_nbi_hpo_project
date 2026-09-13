# Budget accounting

**Unit.** One *real evaluation* is one stratified 5-fold cross-validation of one hyperparameter
vector on one dataset partition, that is five XGBoost fits. Every term below counts real evaluations
unless it says otherwise. Surrogate evaluations are counted separately and never mixed in, because
they cost microseconds against seconds.

**Rule.** Every arm is charged for everything it needs in order to run on its own. An arm is never
given, free, a stage that another arm paid for. Paper 1 established that quoting solver-stage cost
alone understates a real-anchor arm by an order of magnitude, and the same trap is present here.

---

## The ledger

| Term | Definition | Who pays it |
|---|---|---|
| `B_design` | evaluations of the design matrix | every arm, identically |
| `B_surrogate_validation` | evaluations at held-out compositions used to score the surrogate externally | every arm that uses a surrogate |
| `B_anchor` | evaluations spent finding anchors on the **real** objectives | NBI-R only |
| `B_candidate_validation` | re-evaluation of every returned candidate on the real objectives | every arm |
| `B_direct_search` | evaluations spent by a comparator that does not use the design | comparators only |
| `B_total_solution` | what it costs to produce one arm's returned set | per arm, the sum of the rows it pays |
| `B_total_audit` | evaluations spent only to grade the arms, charged to no arm | the study |
| `B_total_experiment` | every real evaluation the campaign performs | the study |

`B_total_audit` covers the empirical Pareto reference, the held-out scoring and the anchor-injection
control. It must be reported and must not be charged to any arm, because no arm needs it in order to
run. Paper 1's Table 5 is the model for how this is presented.

## Per-arm composition

| Arm | `B_design` | `B_surrogate_validation` | `B_anchor` | `B_candidate_validation` |
|---|---|---|---|---|
| HISTORICAL-WS | yes | no (the dissertation has no gate) | no | yes |
| WS-S | yes | yes | no | yes |
| NBI-S | yes | yes | no | yes |
| NBI-R | yes | yes | **yes** | yes |
| Comparators | no | no | no | `B_direct_search` only |

The asymmetry to state plainly: **NBI-R is the only arm that buys anchors with real evaluations, and
it is not free.** Whatever NBI-R gains, the honest comparison is against a comparator given the same
`B_total_solution`, not against NBI-S at a smaller budget.

## Known values at the historical protocol

From the frozen code and the reproduction in `audits/provenance/`:

| Term | Value | Source |
|---|---|---|
| `B_design` | 88 | the version-controlled design, one evaluation per row |
| `B_candidate_validation` | 20 | 20 weight pairs at step 0.05, one candidate each |
| `B_surrogate_validation` | 0 | no external validation exists in the frozen pipeline |
| `B_anchor` | 0 | the observed-extremes box costs nothing beyond the design |
| `B_total_solution`, HISTORICAL-WS | **108** | 88 + 20 |
| comparator budget as the dissertation set it | 108 each | `benchmark_budget = len(doe_df) + len(cand_params)` in `scripts/run_replica.py` |

The dissertation's fairness rule is therefore already evaluation-matched, and correctly so. Paper 2
keeps the rule and extends it to the terms the dissertation had no need for.

## Measured, by pilot Stage A

| Term | Value | Source |
|---|---|---|
| `B_surrogate_validation` | **78** | the design's complementary half fraction (64 corners) plus 14 axial runs at half the axial distance |
| `B_total_solution` | 108 / 186 / 186 / 386 for HISTORICAL-WS / WS-S / NBI-S / NBI-R at q = 2 | the ledger above |
| comparator budget | 386 | the most expensive arm |
| seconds per real evaluation, 8 threads | 1.02 to 1.69, mean 1.40 | measured on all four panel datasets |
| campaign total at q = 2 | 285,120 evaluations, 111 hours, 4.6 days serial | 2,376 per replication per dataset x 30 x 4 |
| campaign total at q = 3 | 357,120 evaluations, 5.8 days serial | over the ceiling; the campaign runs at q = 2 |

The original ledger assumed 100 validation evaluations, copied from Paper 1. Stage A found that a
random held-out set cannot validate a surface fitted to a factorial design, and replaced it with the
design's complementary half fraction, which is both correct and 22 evaluations cheaper. See
`../audits/PILOT_STAGE_A_FINDINGS.md` Finding 4.

## What must be decided before the freeze

1. **`B_surrogate_validation`.** Paper 1 used 100 held-out compositions per replication and a
   pass/fail gate on external R² and Spearman. An equivalent here costs 100 real evaluations per
   replication per dataset, which nearly doubles `B_total_solution`. The alternative, cross-validating
   the surrogate on the design rows themselves, is cheaper and weaker. **Recommendation: pay for the
   external set.** Paper 1's central finding was that an unvalidated surrogate fails in ways nothing
   else detects, and repeating the omission here would ignore the study's own strongest result.

2. **`B_anchor` for NBI-R.** Direct search for each objective's real optimum, at a pre-declared
   budget per objective. The budget must be fixed in advance and identical across datasets, so it is
   a constant of the design and not a tuning knob.

3. **The comparator budget.** Set to `B_total_solution` of the most expensive arm, so that no
   comparator is handicapped. Report the realized ratio, as Paper 1 did.

4. **Whether the design cost is shared or charged per arm.** All arms use the same design on the same
   partition, so the campaign evaluates it once. For the **accounting** each arm is still charged
   the full 88, because each would have to pay it alone. The distinction is stated explicitly and the
   wall-clock total reports the shared figure.

## Wall-clock anchor

Superseded by the measured figures above. The original projection, retained so the estimate can be
compared against what happened: the dissertation reproduction measured 4.6 minutes for 88 evaluations
on MAGIC, about 3.1 seconds per evaluation. A first-order projection for a campaign of `D` datasets and
`R = 30` replications:

```
evaluations = D x 30 x (88 design + 100 validation + 20 candidates x 4 arms + B_anchor + audit)
```

At 3.1 s per evaluation and D = 4 this is roughly 4 x 30 x 300 x 3.1 s ≈ 31 hours of single-machine
compute before the anchor and audit terms, which are the uncertain ones. MAGIC is a small dataset,
so per-evaluation cost on a larger panel member will be higher. **The pilot exists to replace this
projection with a measurement**, and the five-day ceiling is checked against the measurement, not
against this estimate.

## Reporting

The manuscript reports, per arm and per dataset: each `B_*` term, `B_total_solution`, the ratio to
the cheapest arm, the realized comparator budget ratio, and `B_total_audit` separately. Surrogate
evaluation counts are reported in the same table in their own column, labelled as such.
