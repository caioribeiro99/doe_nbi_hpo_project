# Stage-3 / replica_002 extreme-lane plan (Commit 50)

This document captures the **planning-only** Commit 50 step: it
explains *why* the replica_002 extreme lane is still pending, what
the future Commit 51 execution should look like, and what an
operator must validate before scaling beyond replica_002.

Commit 50 produces:

- `scripts/plan_stage3_replica002_extreme_lane.py` — the read-only
  planner;
- `experiments/_stage_runs/`
  `stage3_replica_002_extreme_lane_plan_latest_summary.{json,md}` —
  the machine-readable plan artifact;
- `tests/unit/test_stage3_replica002_extreme_lane_plan.py` —
  contract tests.

No training, no execution SQLite, no `runs/` artifacts, no new
signoff file, no policy edits, no `topup_to_5` scale-out, and no
shard mutation happen in Commit 50.

## Where we stand on replica_002

| Lane     | Commit | Cells | Result                              |
|----------|-------:|------:|-------------------------------------|
| standard | 48     |   684 | 684 / 684 success, 0 failures       |
| heavy    | 49     |   156 | 156 / 156 success, 0 failures       |
| extreme  |  —     |   (24) | **pending — planned in Commit 50** |

The 24 extreme cells = 2 extreme tasks × 4 canary methods × 3
algorithms.

## Extreme tasks under the pinned policy

`policy_version = 47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`

| openml_task_id | dataset            |
|---------------:|--------------------|
| 6              | letter             |
| 167121         | Devnagari-Script   |

These are the only two tasks classified `extreme` in
`benchmarks/doctoral/openml_cc18/heavy_task_policy.csv` under the
signoff-pinned policy. The planner refuses to proceed if the live
universe ever drifts from `(6, 167121)`.

## Why Commit 50 is planning-only

1. **Devnagari-Script previously dominated runtime.** In batch_03,
   task 167121 ran 11,090 s on `xgboost / doe_rsm_vrf_true_nbi`,
   10,575 s on `catboost / doe_rsm_vrf_true_nbi`, 7,944 s on
   `catboost / tpe_optuna`, and 7,647 s on `catboost /
   random_search`. Executing the full extreme lane without an
   operator review risks tying up the worker for many hours.
2. **The extreme lane requires explicit operator opt-in.** Under
   `runtime_guardrails.yaml`, `extreme.include_by_default = false`
   and `extreme.requires_manual_review_before_full_stage0 = true`.
   Replica_002's extreme lane inherits the same opt-in contract:
   the future Commit 51 runner must require both an
   `--include-extreme-tasks` and an `--execute-extreme-lane`
   flag.
3. **`policy_version` must stay pinned during replica_002.** All
   three lanes of replica_002 (standard / heavy / extreme) must
   run under the same signed-off policy so the cells stack with
   replica_001 under the doctoral comparative protocol. Any
   change to `heavy_task_policy.csv` or
   `runtime_guardrails.yaml` between lanes invalidates the
   replica.

## Expected execution size

- 2 extreme tasks × 4 canary methods × 3 algorithms = **24 cells**.
- 42 non-canary extreme rows (2 tasks × 7 non-canary methods ×
  partial algorithm coverage) will be **refused** in the future
  Commit 51 run.
- 1,815 standard-lane rows + 423 heavy-lane rows from the source
  template stay marked as "already completed" (Commits 48 / 49
  authoritative summary).

## Likely runtime

- `letter` is comparatively cheap (small row count, modest
  feature dim) — its 12 cells should land in well under an
  hour.
- `Devnagari-Script` is the wall-clock dominator. Under the
  policy-defined `extreme.stage0_max_evaluations = 1` budget,
  random-search / tpe_optuna / default_gbdt cells exercise a
  single configuration each; only `doe_rsm_vrf_true_nbi` floors
  at `n_doe = max(2 * d, max_evaluations) = 8` for `d = 4`.
  Total replica_002 extreme runtime is expected to track
  Commit 43's stage-0 extreme lane (~30,844 s ≈ 8h34m, signoff
  caveat 2).

## Execution policy for a future Commit 51

| Knob                                       | Value                          | Source                                     |
|--------------------------------------------|--------------------------------|--------------------------------------------|
| `--include-extreme-tasks` flag             | **required**                   | matches `runtime_guardrails.yaml` opt-in  |
| `--execute-extreme-lane` flag              | **required**                   | matches Commit 50 plan recommendation     |
| `max_evaluations`                          | **1** (`extreme.stage0_max_evaluations`) | policy `extreme.stage0_max_evaluations`   |
| timeout                                    | **14,400 s / cell** (4 h)      | policy `extreme.timeout_seconds_per_cell` |
| canary methods                             | `default_gbdt`, `random_search`, `tpe_optuna`, `doe_rsm_vrf_true_nbi` | comparative protocol |
| algorithms                                 | `xgboost`, `lightgbm`, `catboost` | comparative protocol                   |
| stage label in execution SQLite            | `stage1_topup_to_005`          | Commit 47 / 48 / 49 convention             |
| replica                                    | `2`                            | Commit 47 / 48 / 49 convention             |
| `policy_version`                           | pinned (must equal signoff)    | refuse on drift                            |

## Why replicas 003 – 005 must wait

The doctoral comparative protocol requires every replica to
clear *all three lanes* under the same `policy_version` before
the next replica can begin. Concretely, before any
replica_003 execution:

1. **Commit 50** publishes this plan (this commit).
2. **Commit 51** executes the replica_002 extreme lane under the
   pinned policy and the explicit `--include-extreme-tasks` /
   `--execute-extreme-lane` flags.
3. A future **replica_002 aggregate review** validates the
   four-summary chain (signoff + Commit 48 standard + Commit 49
   heavy + Commit 51 extreme) end-to-end.
4. Only after step 3 does the operator decide whether to
   recalibrate the policy (e.g. promote isolet to heavy — see
   signoff caveat 1) or proceed with replica_003 under the same
   policy.

## What a replica_002 aggregate review must validate

- All three lane summaries report `execution_status = executed`,
  zero failures, zero pending, zero running, and
  `source_shards_unchanged = true`.
- All four artifacts (signoff + 3 lane summaries) pin the same
  `policy_version`.
- No `runs/` artifact, execution SQLite, OpenML payload, fitted
  model, fold CSV, notebook, or fairness artifact was committed
  by Commits 47 – 51.
- No `topup_to_5` scale-out (no replica_003 / 004 / 005) was
  triggered by any of those commits.
- The signoff's `n_canary_success_total` (currently 864 from
  replica_001) plus the three lane successes for replica_002
  (684 + 156 + 24 = 864) sum to **1,728** canary successes across
  replicas 1–2, which is the value an aggregate signoff for
  replica_002 should record.

## Expected next step after Commit 50

After Commit 50 lands and an operator reviews this plan, Commit
51 should execute the replica_002 extreme lane. Commit 51 must:

- chain the same four gates Commit 50 verified (signoff →
  top-up plan → Commit 48 standard → Commit 49 heavy);
- additionally read this Commit 50 plan summary by SHA-256 and
  refuse on drift;
- enforce both `--include-extreme-tasks` and
  `--execute-extreme-lane` flags;
- use `max_evaluations = 1` and `timeout = 14,400 s / cell`;
- never run the standard or heavy lane, never scale to
  replica_003+.
