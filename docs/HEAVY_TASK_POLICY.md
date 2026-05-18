# OpenML-CC18 heavy-task policy

This document is the contract that lets the doctoral pipeline
schedule large CC18 tasks without letting any single dataset
dominate the wall-clock budget. It is the policy frontier for
every CC18 batch / stage runner from Commit 38 onward.

## tl;dr

- The CC18 panel ships 72 tasks. They are **not** equally cheap.
- A single task — **167121 Devnagari-Script** (92 000 × 1 024 ×
  46 classes) — burned ~92 % of batch_03's 56 710 s runner CPU
  on its 8 cells alone (Commit 37).
- Letting that pattern repeat in full stage 0 would block the
  worker for days.
- Commit 38 splits the panel into three **lanes** (`standard`,
  `heavy`, `extreme`), with per-lane budgets in
  `benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml` and
  per-task assignments in
  `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv`.
- The `src/doe_xgb/runtime_guardrails.py` helper is the API every
  runner consults before dispatching a cell.

## Why

batch_03 (`experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json`)
reported 216 / 216 cells green in 15 h 45 min of runner CPU.
The 8 slowest cells were all 167121 Devnagari-Script:

| method | algorithm | runtime (s) |
|---|---|---:|
| doe_rsm_vrf_true_nbi | xgboost | 11 090.6 |
| doe_rsm_vrf_true_nbi | catboost | 10 575.0 |
| tpe_optuna | catboost | 7 944.0 |
| random_search | catboost | 7 646.9 |
| tpe_optuna | xgboost | 6 524.3 |
| random_search | xgboost | 5 435.7 |
| default_gbdt | catboost | 1 594.8 |
| doe_rsm_vrf_true_nbi | lightgbm | 1 505.0 |

The next-slowest task (3573 mnist_784) maxed out at 1 507 s in
batch_02 — three orders of magnitude lighter per cell than the
worst Devnagari cell.

GBDT HPO cost scales (roughly) with `n_rows × n_features × tree
count × CV folds × HPO evaluations × (number of classes for
multiclass)`. Devnagari-Script has all five factors above the
panel average; multi:softprob with 46 classes turns each gradient
update into a 46-vector. The combination makes the task
*structurally* extreme rather than transiently slow.

## Three lanes

| lane | what it is | who lives here |
|---|---|---|
| **standard** | small / well-behaved CC18 tasks; no individual cell expected to exceed 30 min at gate budget. | 57 of the 72 tasks (Commit 38). |
| **heavy** | large rows, high feature dimensions, or many classes; cells may run for tens of minutes; still scheduled inside the regular run window. | 13 tasks: `pendigits, electricity, mnist_784, adult, Bioresponse, cnae-9, bank-marketing, connect-4, Fashion-MNIST, jungle_chess_..., numerai28.6, CIFAR_10, Internet-Advertisements`. |
| **extreme** | a small set of tasks whose cells exceed an hour and risk blocking the worker for the rest of the day; do NOT run unless the caller passes `--include-extreme-tasks` AND the operator has reviewed the per-task plan. | 2 tasks: `letter, Devnagari-Script`. |

Lane assignment is reproducible from
`tasks.csv` + the latest `batch_02` / `batch_03` summaries via:

```bash
python scripts/build_cc18_heavy_task_policy.py
```

which writes `heavy_task_policy.csv` and the human-readable
`heavy_task_policy_report.md` under the same directory.

## Lane defaults

The numbers below come straight from
`benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml` and are
the only place those constants live (the helper reads them at
load time; the policy CSV may override per-task).

| field | standard | heavy | extreme |
|---|---:|---:|---:|
| `timeout_seconds_per_cell` | 1 800 | 7 200 | 14 400 |
| `default_max_evaluations` | 5 | 5 | 5 |
| `gate_max_evaluations` | 5 | 3 | 1 |
| `stage0_max_evaluations` | 5 | 5 | 1 |
| `include_by_default` | ✅ | ✅ | ❌ |
| `requires_manual_review_before_full_stage0` | no | no | yes |

`disposition_on_timeout` is `failed_timeout` — a cell that exceeds
its timeout is converted into a failed cell with a tagged
`last_error` so the stage-run summary can distinguish "method
failure" from "ran out of time".

## How the runner uses the policy

`src/doe_xgb/runtime_guardrails.py` exposes four methods:

```python
from doe_xgb.runtime_guardrails import RuntimeGuardrails
g = RuntimeGuardrails.load()         # uses default paths
g.get_task_lane(167121)              # -> "extreme"
g.get_timeout_seconds(167121)        # -> 14_400.0
g.get_effective_max_evaluations(
    167121, requested_max_evaluations=5, context="gate",
)                                    # -> 1
g.should_defer_task(167121, include_extreme=False)  # -> True
```

Existing batches 00 → 03 do **not** retroactively rerun through
this layer — their published gate artifacts and summaries already
stand. Every CC18 runner introduced from Commit 38 onward must:

1. Load the policy at start-up.
2. Before each cell, check `should_defer_task(task_id, include_extreme=...)`.
   - Deferred tasks are skipped, surface in the summary as
     `n_skipped` (and as a `deferred_extreme_tasks` list), and do
     NOT count as failures.
3. Cap `max_evaluations` via `get_effective_max_evaluations`.
4. Apply `get_timeout_seconds` as the per-cell subprocess
   timeout. A timeout shows up in the stage-run summary's
   `failures_grouped` block with `last_error_sample:
   "failed_timeout"`.

## Stage 0 progress (Commits 40 → 45)

The stage 0 split materialized as expected. As of Commit 45:

- **standard lane** ran in Commit 40 (`daae8ab`): 684 / 684 cells
  green;
- **heavy lane** ran in Commit 41 (`ddb657d`): 156 / 156 cells
  green;
- **extreme lane** ran in Commit 43 (`28961fe`): 24 / 24 cells
  green at the policy's `extreme.stage0_max_evaluations = 1`,
  per-cell timeout 14,400 s. Devnagari-Script `doe_rsm` topped
  out at 10,663 s — under the timeout. See
  `docs/EXTREME_LANE_PLAN.md` for the budget-parity caveat.
- **aggregate signoff plan** shipped in Commit 44 via
  `scripts/build_stage0_replica_signoff.py`, originally publishing
  `experiments/_stage_runs/stage0_replica_001_signoff_plan_latest_summary.{json,md}`
  with `signoff_status = "planned_not_signed"`.
- **operator signoff** shipped in Commit 45 via
  `scripts/sign_stage0_replica_001.py`, which writes
  `jobs/doctoral/openml_cc18/stage3_signoff.json` (operator
  metadata + both required caveat acknowledgements +
  `downstream_execution_authorized_in_this_commit = false`) and
  re-runs the aggregator so the published plan summary now reads
  `signoff_status = "signed"`. See
  `docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md`.

All three lanes pin the same `policy_version` SHA-256 so the
three stage-run summaries belong to the same replica.

## Stage 0 split

Full stage 0 will no longer run as a single 2 304-job pass.
Instead:

- **standard lane stage 0** runs the 57 standard tasks × 3 GBDT
  algorithms × N methods × 1 replica. Same budget as today;
  expected wall-clock is bounded by the slowest standard cell.
- **heavy lane stage 0** runs the 13 heavy tasks on the same
  shard layout but with the heavy-lane timeout. Operators can
  promote individual cells to a longer budget by editing
  `heavy_task_policy.csv` (per-task `timeout_seconds_per_cell`
  override) without changing the YAML.
- **extreme lane stage 0** runs only after explicit operator
  approval and only on a worker reserved for the long pass. The
  pass uses `--include-extreme-tasks` and the
  `extreme.stage0_max_evaluations = 1` budget.

The summary published by each lane is a normal stage-run summary
(`experiments/_stage_runs/<lane>_summary.{json,md}`). Downstream
analysis joins them at the per-task level.

## How summaries communicate heavy-task status

Stage-run summaries gain two optional blocks when the policy is
in play:

- `policy_version`: hash of the
  `heavy_task_policy.csv` used at runtime;
- `deferred_extreme_tasks`: list of task ids that were skipped
  because `include_extreme=False`. Without this, a reader cannot
  tell whether the run actually visited the extreme lane or
  silently defer-skipped it.

A reader on the personal Mac who pulls the summary should be
able to answer "did batch_04 cover Devnagari-Script?" simply by
checking whether `167121` appears in `deferred_extreme_tasks`.

## Re-classification rules

The builder is reproducible: given the same `tasks.csv` and the
same batch summaries, it always produces the same lane
assignments. Two paths cause re-classification:

1. **Observed-runtime escalation.** When a new batch is run and
   its summary lands under `experiments/_stage_runs/`, regenerate
   the policy with `scripts/build_cc18_heavy_task_policy.py`. Any
   task whose cell runtime crossed 900 s in the new summary is
   promoted to `heavy`; any whose cell crossed 3 600 s is
   promoted to `extreme`.
2. **Metadata escalation.** A `tasks.csv` update (e.g. a new CC18
   suite import) re-applies the metadata rules:
   - `extreme`: rows ≥ 75 000 AND features ≥ 500, OR classes ≥ 25
     AND rows ≥ 20 000.
   - `heavy`: rows ≥ 40 000, OR features ≥ 750, OR
     categorical ≥ 500, OR (classes ≥ 10 AND rows ≥ 10 000).

The rules never downgrade a task — a task that was once heavy
stays heavy until a human edits the CSV. This is intentional:
once we've observed a heavy cell we should not silently shorten
its budget.

## Note: letter (task 6)

`letter` (20 000 × 16 × 26 classes) matches the extreme rule
(`n_classes >= 25 AND n_rows >= 20 000`) even though its observed
batch_03 max cell was 26 s. The strict-rule classification is
intentionally conservative: at full stage budgets (more replicas,
heavier DoE+NBI configurations, longer trees) a 26-class softmax
can grow much faster than the 26-class linear extrapolation. If
the operator later observes consistently fast `letter` cells, the
CSV row may be hand-edited to `heavy` (with a note explaining the
downgrade).
