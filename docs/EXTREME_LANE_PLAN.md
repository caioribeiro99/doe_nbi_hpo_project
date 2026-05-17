# Stage 0 extreme-lane plan

This document is the planning frontier for the third (and last)
stage-0 lane. It exists so that no operator runs the extreme
lane by accident: a single Devnagari-Script cell can occupy the
worker for several hours, and stage 0 replica 1 must not be
blocked on a long run we never explicitly approved.

## Why this needs explicit planning

Commit 37 (batch_03) consumed ~56,710 s of runner CPU. The
**eight** slowest cells were all `167121 / Devnagari-Script`
and together accounted for ~92 % of that runtime:

| method | algorithm | runtime (s) |
|---|---|---:|
| doe_rsm_vrf_true_nbi | xgboost | 11,090.6 |
| doe_rsm_vrf_true_nbi | catboost | 10,575.0 |
| tpe_optuna | catboost | 7,944.0 |
| random_search | catboost | 7,646.9 |
| tpe_optuna | xgboost | 6,524.3 |
| random_search | xgboost | 5,435.7 |
| default_gbdt | catboost | 1,594.8 |
| doe_rsm_vrf_true_nbi | lightgbm | 1,505.0 |

The heavy-task runtime guardrails introduced in Commit 38 mark
`167121` (Devnagari-Script) and `6` (letter) as **extreme**,
and `runtime_guardrails.yaml` sets `extreme.include_by_default
= false`. Stage-0 standard and heavy lanes both refuse to
dispatch extreme rows. Commit 42 is the dedicated planning
step that asks: *when, how, and at what evaluation budget do we
run the extreme lane?*

## Current stage 0 status (Commit 42)

- **standard lane** (Commit 40, `daae8ab`): **green**
  - 684 / 684 success across 57 standard tasks × 4 canary × 3
    algorithms;
  - `source_shards_unchanged: true`,
    `stage3_signoff_present: false`;
  - `policy_version:
    47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`.
- **heavy lane** (Commit 41, `ddb657d`): **green**
  - 156 / 156 success across 13 heavy tasks × 4 canary × 3
    algorithms;
  - same invariants and same `policy_version`.
- **extreme lane** (this plan): **pending — deferred**.

Stage 0 replica 1 is **not** considered complete until the
extreme lane lands a green stage-run summary that pins to the
same `policy_version`. Promotion criteria:

1. Three stage-run summaries published under
   `experiments/_stage_runs/stage0_*_lane_latest_summary.json`,
   one per lane (standard / heavy / extreme).
2. All three carry the same `policy_version` SHA-256.
3. All three carry `source_shards_unchanged: true`.
4. All three carry `stage3_signoff_present: false`.
5. The extreme summary's `execution_status` is `"executed"`
   (Commit 42 publishes `execution_status:
   "planned_not_executed"` — that does **not** count toward
   promotion).

Only after promotion may policy recalibration (e.g. promoting
`isolet` to heavy via the observed-runtime rule) and a new
replica start.

## Extreme tasks

| task_id | dataset | rows | features | classes | metadata trigger | observed worst cell (batch_03) |
|---:|---|---:|---:|---:|---|---:|
| 6 | letter | 20,000 | 16 | 26 | `n_classes ≥ 25 AND n_rows ≥ 20000` | 26.3 s (default_gbdt lightgbm) |
| 167121 | Devnagari-Script | 92,000 | 1,024 | 46 | all 3 extreme rules fire (observed 11,091 s; rows×features extreme; classes×rows extreme) | 11,090.6 s (doe_rsm xgboost) |

The letter task is "extreme by metadata only" — batch_03
observed it running fast (26 s max). It earned the lane
strictly because `n_classes >= 25 AND n_rows >= 20000` fires
on its 26-class, 20 k-row shape. The expectation is that
letter remains cheap in practice; the lane is precautionary.

## Expected stage-0 extreme-lane row counts (across all 10 shards)

| bucket | count |
|---|---:|
| `runnable_extreme_canary` (2 tasks × 4 canary × 3 algorithms) | **24** |
| `refused_not_in_canary_set` (2 tasks × 7 non-canary × 3 algorithms — modulo parego subset) | 42 |
| `skipped_standard_lane_already_completed` | 1,815 |
| `skipped_heavy_lane_already_completed` | 423 |
| total | 2,304 |

The 24 / 42 / 1,815 / 423 split was verified by the dry-run
runner committed alongside this doc (see
`scripts/run_stage0_extreme_lane.py --dry-run`).

## Whether to run letter and Devnagari-Script together

The cheap call would be to run both in one extreme-lane pass:
twelve `letter` cells (fast) + twelve `Devnagari-Script` cells
(slow). The runtime profile would be ~92 % Devnagari-Script
anyway, so coupling them wastes nothing.

The more careful call is to publish them as **two separate
extreme-lane summaries**, because:

1. If something goes wrong with Devnagari-Script (transient
   OpenML download, OOM, kernel weirdness), we don't lose the
   `letter` provenance.
2. The promotion-criteria machinery (above) is easier to reason
   about per-task: `stage0_extreme_lane_latest_summary.json`
   can be the joint summary, but the operator may also
   publish per-task summaries
   `stage0_extreme_lane_letter_summary.json` and
   `stage0_extreme_lane_devnagari_summary.json` for diagnosis.

**Recommendation:** Commit 43 should publish **one combined
extreme-lane summary** (`stage0_extreme_lane_latest_summary.json`)
that includes per-task status. If Devnagari-Script fails or
times out, the runner's failure-handling marks the
Devnagari cells as `failed`/`failed_timeout` while preserving
the letter cells' success — same model the heavy lane already
uses. A separate per-task split is only worth the extra
plumbing if the operator wants to publish `letter` early.

## Runtime expectations (anchored on batch_03)

For each `167121` cell, batch_03 observed the worst-case at
the same `max_evaluations=5` budget batch_03 used. Letter
cells were uniformly cheap.

Pessimistic per-cell extrapolations:

| task | method × algorithm (per cell) | expected (s) | n_cells | total (s) |
|---|---|---:|---:|---:|
| 6 (letter) | any canary × any algo | ≤ 60 | 12 | ≤ 720 |
| 167121 | doe_rsm × xgboost | 11,091 | 1 | 11,091 |
| 167121 | doe_rsm × catboost | 10,575 | 1 | 10,575 |
| 167121 | doe_rsm × lightgbm | 1,505 | 1 | 1,505 |
| 167121 | tpe_optuna × catboost | 7,944 | 1 | 7,944 |
| 167121 | tpe_optuna × xgboost | 6,524 | 1 | 6,524 |
| 167121 | tpe_optuna × lightgbm | ~1,500 | 1 | ~1,500 |
| 167121 | random_search × catboost | 7,647 | 1 | 7,647 |
| 167121 | random_search × xgboost | 5,436 | 1 | 5,436 |
| 167121 | random_search × lightgbm | ~1,500 | 1 | ~1,500 |
| 167121 | default_gbdt × catboost | 1,595 | 1 | 1,595 |
| 167121 | default_gbdt × xgboost | ~200 | 1 | ~200 |
| 167121 | default_gbdt × lightgbm | ~200 | 1 | ~200 |
| **total** | — | — | **24** | **≈ 56,400 s (~15.7 h)** |

That is essentially the same shape we saw in batch_03 (15.7 h),
because batch_03 already ran every Devnagari-Script canary
cell at max_evaluations=5 plus six other tasks; the difference
here is we'd be paying that same Devnagari-Script tax again on
top of the standard + heavy lanes already done. **Operationally
this is a one-day run.**

## max_evaluations: 1 (YAML default) vs 5 (override)

`runtime_guardrails.yaml` sets `extreme.stage0_max_evaluations
= 1` and `extreme.gate_max_evaluations = 1`. The runtime
helper's `get_effective_max_evaluations(167121, ...,
context="stage0")` therefore caps at **1**. Running at the
default would:

- cut Devnagari-Script's HPO-style methods (`random_search`,
  `tpe_optuna`, `doe_rsm`) to a single configuration each, ~5×
  faster than the batch_03 numbers above;
- yield aggregate metrics on Devnagari that are NOT directly
  comparable to the standard / heavy lanes (which ran at 5
  evals);
- break the "same budget across the panel" property that the
  doctoral comparative protocol expects.

The doctoral campaign's headline claim is "DoE + RSM + true
N-objective NBI is competitive with the dominant HPO families
at the same budget." Running Devnagari-Script at 1 eval and
the rest at 5 would muddle that claim for the headline figure
that aggregates over all 72 tasks. The clean alternative is to
override to `max_evaluations=5` for the extreme lane and pay
the ~15.7 h tax, OR to run Devnagari-Script at 5 evals only
for the comparative table and footnote that letter ran at the
same budget as the rest of the panel.

**Recommendation:** override to `--max-evaluations 5` for
Commit 43 (extreme execution), keeping the lane budget parity
with standard / heavy. Use the YAML's `stage0_max_evaluations
= 1` as the fallback for emergency-only runs where wall-clock
is the binding constraint.

The runner's `--max-evaluations` flag already caps at the
lane's effective budget, so explicitly passing
`--max-evaluations 5` AND keeping the YAML at 1 produces
`min(5, 1) = 1` — wrong. The override path is either:
- temporarily edit `runtime_guardrails.yaml` (recorded in the
  policy_version drift), OR
- add a `--override-stage0-max-evaluations N` CLI knob to the
  extreme runner that bypasses
  `get_effective_max_evaluations` for the extreme lane only.
The second option is preferred because it leaves the policy
file untouched; the override is recorded in the stage-run
summary's `runner_invocations` block so the choice is
auditable.

## Operational notes: local Mac vs dedicated Mac

- **Dedicated Mac (Factored-LWCTW4633L, the one running every
  batch so far)** — currently the only worker. The 15.7 h run
  would block the worker for ~one calendar day; plan around
  thermal headroom and other commitments. Run in a `tmux`
  session.
- **Local Mac (laptop)** — do NOT run. The doe_rsm catboost
  cell on Devnagari-Script touches ~10 GB of RAM during the
  46-class softmax; on a laptop with cooling constraints, that
  realistically produces a `failed_timeout` even at the
  14,400 s extreme timeout.

## Timeout behaviour

Per `runtime_guardrails.yaml`:

- `extreme.timeout_seconds_per_cell = 14,400` (4 h);
- `disposition_on_timeout = failed_timeout`.

The extreme runner must honor this. If a single cell exceeds
4 h, it is recorded as `last_error = "failed_timeout"`, the
overall summary counts it under `n_jobs_failed_timeout`, and
the verdict drops to `GATE FAIL`. There is no silent skip; a
timeout is the same as a method failure for downstream gating.

For Devnagari-Script: only `doe_rsm × xgboost` (11,091 s in
batch_03) is over the 4-h ceiling. The operator must either:
- raise the extreme-lane per-cell timeout to 18,000 s (5 h)
  before running, OR
- accept that one cell will surface as `failed_timeout` and
  the extreme lane gates as `GATE FAIL` until investigated.

**Recommendation:** raise the extreme cell timeout to 18,000 s
in `runtime_guardrails.yaml` ONCE between Commit 42 and
Commit 43, regenerate the policy file, and accept the
resulting `policy_version` bump as the start of a new replica
slice (so standard + heavy are not retroactively under the
new policy version). Alternative: pass an explicit
`--override-extreme-timeout 18000` CLI knob to the extreme
runner so the YAML and `policy_version` stay pinned.

## Commit 42 vs Commit 43 boundary

| | Commit 42 (this commit) | Commit 43 (later) |
|---|---|---|
| **doc** | publishes this plan | references this plan |
| **runner** | refuses to execute unless `--execute-extreme-lane` is passed (a flag this commit does NOT use) | explicitly passes `--execute-extreme-lane`, the chosen `--max-evaluations`, and any timeout override |
| **summary** | publishes `stage0_extreme_lane_plan_latest_summary.{json,md}` with `execution_status = "planned_not_executed"` | publishes `stage0_extreme_lane_latest_summary.{json,md}` with `execution_status = "executed"` and the usual per-cell breakdown |
| **OpenML downloads** | none (dry-run); letter and Devnagari-Script payloads are not even loaded | downloads / hits the warm cache for both task payloads (`data/source/openml_cc18/{6,167121}/`) |
| **expected wall-clock** | < 1 minute | ~15.7 h on the dedicated Mac |
| **promotion of stage 0 replica 1** | not changed; replica still incomplete | candidate-complete pending operator review |

## What the dry-run runner does in Commit 42

`scripts/run_stage0_extreme_lane.py --dry-run` performs the
read-only inventory and refuses every other action:

1. Verifies the standard-lane and heavy-lane summaries are
   green and share the pinned `policy_version`.
2. Loads the policy via `RuntimeGuardrails.load()`.
3. Inventories all 10 committed shards under
   `jobs/doctoral/openml_cc18/shards/stage0_replica_001/`
   (read-only SQLite URI; never mutates).
4. Classifies every row into the four buckets above.
5. Asserts the runnable count matches the expected 24.
6. Writes the dry-run summary to
   `experiments/_stage_runs/stage0_extreme_lane_plan_latest_summary.{json,md}`
   with `execution_status = "planned_not_executed"`.

Without `--execute-extreme-lane` the runner exits there. It
does NOT create execution copies, does NOT contact OpenML, does
NOT touch `runs/cc18/`, and does NOT change any committed
artifact under `jobs/` or `benchmarks/`.

## Promotion criteria recap (for the future Commit 43 reviewer)

Stage 0 replica 1 is **complete** when the following all hold
simultaneously:

- `experiments/_stage_runs/stage0_standard_lane_latest_summary.json`
  is green (`n_jobs_failed == 0`, `n_jobs_pending_after == 0`,
  `source_shards_unchanged == true`).
- `experiments/_stage_runs/stage0_heavy_lane_latest_summary.json`
  is green by the same criteria.
- `experiments/_stage_runs/stage0_extreme_lane_latest_summary.json`
  is green by the same criteria AND its `execution_status` is
  `"executed"` (not `"planned_not_executed"`).
- All three summaries share the same `policy_version`.
- `jobs/doctoral/openml_cc18/stage3_signoff.json` does NOT
  exist (no operator has signed off on the long top-up tier
  yet, and Commit 42 does not change that).

Only after that point is policy recalibration (re-running
`scripts/build_cc18_heavy_task_policy.py` to promote `isolet`
via the observed-runtime rule, for instance) the right next
step — and the new policy_version starts replica 2.
