# Stage-3 / top-up worker plan

This document is the **machine-assignment plan** that pairs each
top-up tier × lane in `stage3_topup_manifest.csv` with a recommended
worker from `stage3_worker_plan.csv`. The CSV sibling is the
machine-readable form.

The plan is **planning-only** (Commit 46). Actual dispatch is a
separate operator-reviewed commit.

## Workers

### `personal_mac_m4_pro` — MacBook M4 Pro

- **Role.** Primary planning surface. Runs
  `scripts/plan_stage3_topup.py`,
  `scripts/build_stage0_replica_signoff.py`, the per-tier
  aggregators (to be added), and the docs review loop.
- **Dedicated.** Partial. This machine is also the developer's
  laptop; long unattended runs are discouraged.
- **Lanes.** `standard` (light) — small / well-behaved tasks where
  a partial-day run is fine.
- **Tiers.** `topup_to_5` (sampled, e.g. the Commit 47 pilot).
- **Not suitable for.** `heavy` and `extreme` lanes — wall-clock
  cost is too high to leave on a laptop overnight without a UPS
  and `caffeinate`.

### `dedicated_mac` — dedicated MacBook / Mac Pro-like worker

- **Role.** Primary execution host for `topup_to_5` and
  `topup_to_10`.
- **Dedicated.** Full. 24/7 availability, `caffeinate` always on.
- **Lanes.** `standard` (all) + `heavy` (all).
- **Tiers.** `topup_to_5` standard + heavy, `topup_to_10`
  standard + heavy. Extreme is fine here if the university Mac is
  not yet available.
- **Prerequisites.** `bash scripts/setup_dedicated_mac.sh`,
  `python scripts/audit_method_capabilities.py`, refresh the
  capability audit before any new tier.
- **Not suitable for.** `topup_to_30` heavy + extreme if the
  university Mac is available — that workload is too long to keep
  the dedicated Mac monopolized.

### `university_mac_max` — future university Mac Pro Max (contingent)

- **Role.** Long-tail execution for `topup_to_30` heavy + extreme
  if/when access is provisioned.
- **Dedicated.** Full (assumed).
- **Lanes.** `heavy` + `extreme`.
- **Tiers.** `topup_to_10` heavy/extreme (off-load from dedicated
  Mac), `topup_to_30` heavy + extreme (primary host).
- **Not suitable for.** Light validation work — keep the workload
  long-running so the access window is amortized.
- **Note.** Do **not** assume this machine is available. The plan
  must remain executable on `dedicated_mac` alone, with a
  longer wall-clock budget.

### `optional_cloud` — cloud VM (optional future)

- **Role.** Optional burst capacity for `topup_to_30` heavy +
  extreme cells if neither Mac worker can finish in time.
- **Dedicated.** Funded only. Explicitly **not** the default.
- **Lanes.** `heavy` + `extreme` (burst only).
- **Note.** The Commit-45 signoff scope explicitly says
  "no cloud execution is authorized by this commit". A future
  operator commit may authorize cloud burst, but Commit 46 only
  documents it as a contingency.

## Assignment overview

| tier | lane | primary worker | fallback worker |
|---|---|---|---|
| topup_to_5 | standard | `dedicated_mac` | `personal_mac_m4_pro` (sampled) |
| topup_to_5 | heavy | `dedicated_mac` | `university_mac_max` |
| topup_to_5 | extreme | `dedicated_mac` | `university_mac_max` |
| topup_to_10 | standard | `dedicated_mac` | `personal_mac_m4_pro` (sampled) |
| topup_to_10 | heavy | `dedicated_mac` / `university_mac_max` | — |
| topup_to_10 | extreme | `dedicated_mac` / `university_mac_max` | — |
| topup_to_30 | standard | `dedicated_mac` / `university_mac_max` | — |
| topup_to_30 | heavy | `university_mac_max` (preferred) | `dedicated_mac` |
| topup_to_30 | extreme | `university_mac_max` (preferred) | `dedicated_mac` (long) |

## Operating principles

- **No cloud by default.** Cloud execution is explicitly out of
  scope for Commits 46 and 47. The plan must stay reproducible on
  the documented physical workers.
- **One worker, one shard at a time.** Concurrent runs against the
  same shard race on the SQLite claim lease. Run one shard per
  worker; parallelize across workers, not within.
- **Per-worker provenance.** Every summary records its worker's
  `hostname`, `uname`, `python_version`, and package versions
  (already wired through `collect_package_versions`). A later
  audit script should compare these across workers.
- **`caffeinate` mandatory on Mac workers.** A sleeping worker
  loses its claim lease and the runner may end up with stale
  rows.

## Coordination flow

1. Personal Mac runs the planner and publishes the plan summary
   and manifests (Commit 46).
2. Personal Mac runs the Commit-47 pilot (a single standard shard
   of replica 002) and verifies the result locally.
3. Dedicated Mac pulls the branch, runs its assigned shards,
   exports summaries, pushes a small commit with only the JSON/MD
   summary.
4. Personal Mac aggregates per-tier summaries via the (planned)
   per-tier aggregator script.
5. Operator review → per-tier signoff commit → next tier.
6. When the university Mac becomes available, repeat steps 3–5
   with shifted assignments per the table above.
