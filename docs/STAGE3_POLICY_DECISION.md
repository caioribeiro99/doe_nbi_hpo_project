# Stage-3 / top-up policy decision note

This is the **decision note** that accompanies the top-up planning
work in Commit 46. It does not change `policy_version`. It does
not promote any task between lanes. It documents the options the
operator must choose between *before* Commit 47 (the tiny pilot)
can dispatch any cell of any top-up tier.

## Context

Commit 45 signed `stage0_replica_001` under

```
policy_version =
  47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36
```

(SHA-256 of `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv`).
The signed signoff acknowledges two caveats that have direct
implications for the top-up plan:

### Caveat 1 — `isolet_future_recalibration_candidate`

`isolet` (OpenML task 3481) is currently assigned to the
**standard** lane by the Commit 38 policy. The standard lane's
`timeout_seconds_per_cell = 1800`. In Commit 40, the
`doe_rsm_vrf_true_nbi` × catboost cell on isolet ran for
**~1078.6 s** — comfortably under the 1800-s standard timeout but
**above the 900-s promotion threshold** used by
`scripts/build_cc18_heavy_task_policy.py` to mark a task as a
heavy-lane candidate. The Commit 45 signoff explicitly captures
this as a "future recalibration candidate": isolet succeeded in
the standard lane at R=1, but a future re-run of the policy
generator under R=5 / R=10 / R=30 data may promote it to heavy.

### Caveat 2 — `devnagari_extreme_budget_non_equivalence`

`Devnagari-Script` (OpenML task 167121) ran in the **extreme**
lane with `stage0_max_evaluations = 1` per
`runtime_guardrails.yaml`. Standard- and heavy-lane cells ran at
`max_evaluations = 5`. Three of the four canary methods
(`random_search`, `tpe_optuna`, `default_gbdt`) therefore exercised
fewer configurations on Devnagari than on the rest of the panel.
`doe_rsm_vrf_true_nbi` floors at `n_doe = max(2 * d, max_evaluations)
= 8` for `d = 4` and is unchanged. The signoff notes that headline
panel-average metrics aggregating Devnagari with the other 70
tasks should footnote this asymmetry.

These two caveats define the operational space for the top-up
policy decision.

## Decision options

### Option A — Freeze the current `policy_version` for all top-up replicas

**What it means.** Replicas 002 … 030 all run under
`policy_version = 47b6b50…`. isolet stays in standard. Devnagari
stays in extreme with `stage0_max_evaluations = 1`.

**Pros.**
- Strongest reproducibility guarantee: every cell of every
  replica was budgeted by the same policy.
- Simplest aggregation: panel-average metrics across R=1 … R=30
  are directly comparable for every task except the two flagged
  caveats (which remain footnotes).
- No mid-campaign policy change to defend in the dissertation.
- Stage-3 signoff and the per-tier signoffs all reference the
  same `policy_version` SHA-256.

**Cons.**
- isolet stays in the standard lane even if its R=5 cells reveal
  a stable runtime >900 s. The standard-lane timeout still
  protects against true regressions, but the heavy-promotion
  signal is shelved.
- Devnagari stays at extreme `stage0_max_evaluations = 1` across
  all 30 replicas — extreme HPO budget asymmetry is permanent.

### Option B — New post-stage0 `policy_version` for replicas 002+

**What it means.** A new policy generation pass produces
`heavy_task_policy.csv` with isolet promoted to heavy (using the
R=1 observed runtime) and/or Devnagari's `stage0_max_evaluations`
raised to match the rest of the panel. `stage0_replica_001` stays
signed under the old `policy_version`. Replicas 002 … 030 run
under the new `policy_version`.

**Pros.**
- Addresses both caveats at the point where their cost is
  smallest (before the top-up burns runtime on the wrong lane /
  wrong budget).
- isolet's heavy-lane timeout (7200 s) gives more headroom; the
  Devnagari budget bump produces directly comparable HPO budgets
  across the panel for replicas 002+.

**Cons.**
- Two `policy_version` SHAs in the same campaign. Panel-average
  metrics across R=1 vs R≥2 are not policy-equivalent for the
  two affected tasks (isolet on a different lane;
  Devnagari on a different budget).
- The dissertation must defend two policies, the rationale for
  the switch, and the boundary at R=1 → R=2.
- Per-tier aggregators need to be policy-aware (they currently
  refuse on policy drift).
- The signed `stage3_signoff.json` becomes a "frozen R=1 only"
  artifact rather than a "this is the policy for the whole
  campaign" artifact.

### Option C — Produce both manifests and require operator choice (recommended for Commit 46)

**What it means.** Commit 46 produces, alongside the topup plan,
a **candidate policy-drift report** that quantifies what would
change if Option B were applied (which task moves to which lane,
what budget Devnagari would get, what the new policy_version SHA
would be), but **does not** create that policy on disk. The
operator reads the report and explicitly chooses A or B in a later
commit. Commit 46 is policy-neutral.

**Pros.**
- Commit 46 stays scoped to planning; no surprise policy change.
- The candidate drift report gives the operator concrete numbers
  (how much extra runtime Option B costs, what proportion of
  cells move) to base the decision on.
- The choice between A and B can be made *after* the Commit 47
  pilot, which gives one more replica's worth of data on the
  isolet runtime.
- Aligns with the signoff's
  `downstream_execution_authorized_in_this_commit = false`:
  Commit 46 prepares, does not commit.

**Cons.**
- Defers a decision the operator must eventually make.
- Adds a small artifact (the drift report) that has to be
  reviewed before R=2 dispatch.

## Recommendation

**Commit 46 implements Option C.** It produces:

1. `experiments/_stage_runs/stage3_topup_plan_latest_summary.json`
   (and `.md`) — the top-up plan under the **current** policy.
2. (When `--allow-policy-drift-report-only` is passed) a candidate
   policy-drift report describing what Option B would change. The
   report is not a new policy file; it is a planning artifact.
3. `benchmarks/doctoral/openml_cc18/stage3_topup_manifest.csv` and
   `.md` — the per-tier execution units under the *current* policy.

Commit 46 does **not** change `policy_version`. It does **not**
write a new `heavy_task_policy.csv`. It does **not** promote or
demote any task. Doing any of those is a separate, operator-
reviewed commit that follows the operator's decision between A and
B.

## Operational guardrails

These remain true regardless of A vs B:

- **Lane assignment changes scheduling and timeout, not necessarily
  HPO budget.** `runtime_guardrails.yaml` defines per-lane
  `timeout_seconds_per_cell` and `stage0_max_evaluations`. A
  promotion from standard to heavy reduces the chance of timeout
  on long cells but does *not* automatically change the number of
  HPO configurations evaluated unless the lane's
  `max_evaluations` differs. The heavy lane currently uses
  `default_max_evaluations = 5` (same as standard); the extreme
  lane uses `stage0_max_evaluations = 1`. A lane move therefore
  has operational consequences (timeout protection, gate cadence,
  worker assignment) that must be tracked even if the HPO budget
  number is identical.
- **No silent policy change.** Any new `policy_version` requires
  a dedicated commit that rebuilds `heavy_task_policy.csv` via
  `scripts/build_cc18_heavy_task_policy.py`, regenerates the
  policy report, and updates the relevant docs. The per-tier
  aggregator refuses to mix `policy_version`s across the lane
  summaries within a single tier.
- **The signed `stage3_signoff.json` stays frozen.** It records
  the policy under which R=1 was signed. A future Option-B
  commit must not overwrite it; instead it should add a sibling
  signoff (e.g. `stage1_topup_signoff.json`) that records the
  new `policy_version` and the deliberate scope change.

## What the planner script (`scripts/plan_stage3_topup.py`) enforces

The Commit 46 planner refuses if:

- `jobs/doctoral/openml_cc18/stage3_signoff.json` is missing;
- the signoff exists but `signoff_status != "signed"`;
- the signoff's recorded `policy_version` does not match the
  live SHA-256 of `heavy_task_policy.csv`, **unless** the caller
  passes `--allow-policy-drift-report-only`, in which case it
  emits the candidate drift report and refuses to produce the
  ordinary plan.

In all cases the planner emits
`execution_status = "planned_not_executed"` and produces no
execution artifact. The decision between A and B is escalated to
the operator, not silently resolved by the script.

## Pointer to the next operator action

After Commit 46 lands:

1. Review
   `experiments/_stage_runs/stage3_topup_plan_latest_summary.{json,md}`.
2. Decide between Option A and Option B. If A, no further
   policy work is required and Commit 47 can be scheduled.
   If B, schedule a separate operator commit that runs
   `scripts/build_cc18_heavy_task_policy.py` to produce the new
   `heavy_task_policy.csv` and a fresh signoff sibling.
3. Run Commit 47 (the tiny pilot: replica 002, shard 00, standard
   lane, canary only, no heavy, no extreme).
