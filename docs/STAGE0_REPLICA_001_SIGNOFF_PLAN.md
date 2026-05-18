# Stage 0 replica 1 — aggregate signoff plan

This document is the operator-facing review surface for stage 0
replica 1. It explains *why* sign-off is separated from execution,
*what* needs to be reviewed before the sign-off file is created,
and *what* the later sign-off commit must record.

This is a **planning-only** commit. It does NOT create
`jobs/doctoral/openml_cc18/stage3_signoff.json`.

## Current lane state

Stage 0 replica 1 is now **lane-complete**:

| lane | commit | source-of-truth summary | n_jobs_executed |
|---|---|---|---:|
| standard | Commit 40 (`daae8ab`) | `experiments/_stage_runs/stage0_standard_lane_latest_summary.json` | 684 / 684 |
| heavy | Commit 41 (`ddb657d`) | `experiments/_stage_runs/stage0_heavy_lane_latest_summary.json` | 156 / 156 |
| extreme | Commit 43 (`28961fe`) | `experiments/_stage_runs/stage0_extreme_lane_latest_summary.json` | 24 / 24 |
| **aggregate** | **Commit 44 (this commit)** | `experiments/_stage_runs/stage0_replica_001_signoff_plan_latest_summary.json` | **864 / 864 canary success** |

All four artifacts share the same `policy_version`:

```
47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36
```

`jobs/doctoral/openml_cc18/stage3_signoff.json` is **intentionally
absent**. The aggregate summary published by this commit records
`signoff_status = "planned_not_signed"`.

## Why sign-off is separate from execution

Creating `stage3_signoff.json` is not an accounting step. It is a
**capacity decision**:

- `scripts/cc18_runner.py` is wired to refuse stage-3 top-up rows
  (those whose `notes` carry
  `requires_manual_signoff_before_stage3`) until that file exists.
  Once it lands, the runner becomes capable of dispatching the
  stage-3 jobs — 54,720 cells across the heavy/extreme/standard
  panels combined.
- The four published stage-0 summaries describe a single replica
  (~88,500 s of runner CPU across the three lanes); stage 3
  would scale that by ~20× (5 → 30 replicas via top-ups), most
  of it on Devnagari-Script and the other heavy tasks.

A typo or hurried commit that materializes `stage3_signoff.json`
without operator review would let the next `cc18_runner` invocation
start claiming stage-3 rows. The sign-off file is therefore kept
strictly behind an explicit, reviewed commit *after* this aggregate
plan is read.

## What the aggregate review must verify

The script `scripts/build_stage0_replica_signoff.py` (this commit)
produces the JSON/MD pair that captures these invariants. The
operator reviewing the artifact should walk through:

1. **Lane counts.** 684 + 156 + 24 = 864 canary successes; 1,440
   non-canary refusals across the three lanes; 2,304 - 864 -
   1,440 = 0 stragglers.
2. **`policy_version` consistency.** All three lane summaries +
   the extreme-lane plan summary + the live `heavy_task_policy.csv`
   share the same SHA-256.
3. **`source_shards_unchanged`** in every lane summary. The
   committed shards under `jobs/doctoral/openml_cc18/shards/` are
   byte-identical to the Commit 28 baseline (`shard_00.sqlite`
   still hashes to `91e7a861ea73daf82694029d6c590e54`).
4. **No pending / running / failed / failed_timeout cells** across
   the three lanes. The aggregator's
   `no_pending_running_failed_all_lanes` flag must be `true`.
5. **Metric completeness.** Binary tasks must carry the
   `accuracy / precision / recall / specificity` quadruple;
   multiclass tasks must carry
   `accuracy / balanced_accuracy / f1_macro / mcc` (plus
   `roc_auc_ovr_macro / pr_auc_ovr_macro / brier_multiclass /
   ece_multiclass` when probabilities are available). The
   aggregator's `metric_aggregates_by_task_type` block lists any
   missing keys.
6. **Runtime outliers.** The aggregator's `slowest_cells_overall`
   and `slowest_tasks_overall` lists make Devnagari-Script's
   ~10 ks `doe_rsm` cells visible. Confirm none came near the
   14,400 s extreme timeout (they did not — peak was 10,663 s).
7. **Comparability caveats.** See below.

## Important caveats

These are the *known* asymmetries that any downstream analysis of
this replica must respect.

### Caveat 1 — `isolet` (task 3481) ran in the standard lane but was slow

Commit 40 observed `isolet / doe_rsm_vrf_true_nbi / catboost` at
1,078.6 s. That is over the policy's 900 s
observed-runtime threshold for promotion to **heavy**, but the
metadata-based classification placed `isolet` in **standard**
(7,800 × 617 × 26 classes — under all three metadata rules for
heavy).

Two paths:

- **Do not retroactively re-lane `isolet` for replica 1.** The
  three lane summaries already pin a single `policy_version`;
  changing the assignment now would invalidate that pin and
  force a full re-run.
- **Recalibrate the policy between replicas.** Re-running
  `scripts/build_cc18_heavy_task_policy.py` against the standard-
  lane summary will pick up the observed-runtime rule and
  promote `isolet` to `heavy`. That produces a new
  `policy_version` and starts the next replica. This is the
  intended path.

The aggregator's caveats list flags `isolet` as a future-
recalibration candidate; the operator reviewing the sign-off plan
should acknowledge this before authorising stage-3 top-ups.

### Caveat 2 — `Devnagari-Script` ran at `max_evaluations=1`

`runtime_guardrails.yaml` sets `extreme.stage0_max_evaluations =
1`. Commit 43 honored that default. The standard and heavy lanes
both ran at `max_evaluations=5`. As a result:

- `random_search`, `tpe_optuna`, and `default_gbdt` cells on
  Devnagari-Script exercised fewer configurations than the rest
  of the panel — the per-task headline accuracy / balanced
  accuracy / f1_macro for Devnagari is lower-bound by HPO budget
  rather than method capability.
- `doe_rsm_vrf_true_nbi` is mostly unaffected because it
  selects `n_doe = max(2·d, max_evaluations) = 8` for the 4-
  dimensional GBDT search spaces — same as batch_03 ran. The
  Devnagari `doe_rsm` headline is therefore directly comparable
  to batch_03; the other three methods are not directly
  comparable to standard/heavy.

Any **panel-average** metric headline that aggregates over all 72
tasks should footnote this. The `metric_aggregates_per_method`
and `metric_aggregates_per_lane` tables in the aggregator's MD
make the asymmetry visible.

### Caveat 3 — lane assignments are a reproducibility feature, not a result filter

Heavy- and extreme-lane handling exists to bound per-cell wall-
clock and protect the worker. It is NOT a quality filter.
Downstream analyses (Friedman / Nemenyi, win-rate, etc.) should
treat the 864 canary successes as the full replica and account
for the budget asymmetry rather than subsetting "only the
standard lane".

## What a later signoff commit should do

A separate, operator-reviewed commit (Commit 45+) may create
`jobs/doctoral/openml_cc18/stage3_signoff.json`. That file should
record, at minimum:

- the **final `policy_version`** (currently
  `47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`);
- the **SHA-256 of all three lane summary JSONs** (these are
  recorded in this commit's aggregate plan);
- explicit **approval metadata** (operator name / handle,
  ISO-8601 timestamp, free-form justification — at least one
  sentence acknowledging both caveats above);
- a **declared scope** for what the sign-off unlocks
  (specifically: which top-up stages, which methods, which
  workers);
- a **policy_version transition note** if the operator intends
  to recalibrate (e.g. promote `isolet` to heavy) before stage
  3 — in which case the sign-off file applies only to the
  current replica and a new replica starts under a fresh
  policy.

Sign-off is not a "yes, run more" button. It is a structured
artifact that downstream readers (replica 2 operators, the
article reviewers, an external auditor) can trace from raw shards
through lane summaries to the operator's recorded approval.

Until that commit ships, the runner refuses stage-3 rows and the
pipeline stays in pre-stage-3 territory.
