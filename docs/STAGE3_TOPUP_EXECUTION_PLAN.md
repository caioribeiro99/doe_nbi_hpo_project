# Stage-3 / top-up execution plan

This document is the operator-facing **execution plan** for scaling
the OpenML-CC18 doctoral campaign from the signed
`stage0_replica_001` baseline (Commit 45) to the three top-up tiers
defined by the frozen comparative protocol (Commit 27):

- `topup_to_5`  — replicas 002 … 005   (4 additional replicas)
- `topup_to_10` — replicas 006 … 010   (5 additional replicas)
- `topup_to_30` — replicas 011 … 030   (20 additional replicas)

It is a **planning** document. Nothing in this file authorizes
execution. The planner script
(`scripts/plan_stage3_topup.py`, Commit 46) emits the
machine-readable equivalent under `experiments/_stage_runs/`. A
later, operator-reviewed commit (Commit 47 onward) will run a
controlled pilot before the full `topup_to_5` is dispatched.

## What `stage0_replica_001` already completed

Commit 45 signed off `jobs/doctoral/openml_cc18/stage3_signoff.json`
with the following invariants frozen:

- `policy_version =
  47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`
  (SHA-256 of `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv`).
- `lane_success_counts = {standard: 684, heavy: 156, extreme: 24}`.
- `n_canary_success_total = 864`.
- `source_shards_unchanged_all_lanes = true` (committed SQLite
  shards under `jobs/doctoral/openml_cc18/shards/stage0_replica_001/`
  are byte-identical to the Commit 28 baseline).
- Both required caveat acknowledgements present:
  - `isolet_future_recalibration_candidate`
  - `devnagari_extreme_budget_non_equivalence`
- `downstream_execution_authorized_in_this_commit = false`.

The signed signoff **unlocks planning** of stage-3 / top-up commits.
It does not unlock execution. Each top-up dispatch commit must
record its own operator review.

## What "top-up" means in this repo

A *top-up* commit extends the existing replica panel by drawing N
additional replicas of the *same* (task × algorithm × method) matrix
that was executed in `stage0_replica_001`. Top-ups never re-run
finished replicas; the SQLite `replica` field is the deduplication
key (`shard_summary.json` `stage3_signoff_rows` confirms the panel
that stage-3 unlocks).

The shard layout under `jobs/doctoral/openml_cc18/shards/` already
mirrors the four tiers (Commit 28):

| stage subdir          | shards | rows  | scope                                |
|-----------------------|-------:|------:|--------------------------------------|
| `stage0_replica_001`  | 10     |  2,304 | replica 001 (signed)                 |
| `stage1_topup_to_005` | 10     |  9,216 | replicas 002–005 (4 replicas)        |
| `stage2_topup_to_010` | 10     | 13,680 | replicas 006–010 (5 replicas)        |
| `stage3_topup_to_030` | 10     | 54,720 | replicas 011–030 (20 replicas)       |
| **total**             | **40** | **79,920** |                                |

Stage-3 rows carry the
`requires_manual_signoff_before_stage3` note. The runner refuses to
claim them in the absence of `stage3_signoff.json`; that file now
exists (Commit 45), so the *runner* no longer refuses on absence —
but actual dispatch still needs the corresponding operator commit.

## Why we scale in tiers (R=5 → R=10 → R=30)

The protocol target is 30 replicas per (task, algorithm) pair for
statistical power. Jumping straight from R=1 to R=30 is **not** the
right operational move:

1. **Runtime risk concentrates in the heavy/extreme lanes.** Stage 0
   replica 1 spent ~88,500 s of runner CPU across the three lanes.
   `topup_to_30` projects ~20× of *that*, dominated by
   Devnagari-Script (extreme), isolet, CIFAR_10, Fashion-MNIST,
   mnist_784, and Internet-Advertisements. A single misconfigured
   guardrail at R=30 wastes ~weeks of dedicated-Mac time.
2. **Calibration before commitment.** R=5 produces enough data to
   re-evaluate the lane policy (isolet runtime, Devnagari budget,
   heavy timeouts) without committing to 30 replicas of the same
   plan. R=10 produces stable means for headline metrics; R=30
   produces tight CIs.
3. **Independent gating.** Each tier publishes its own
   `stage{1,2,3}_topup_to_{5,10,30}_signoff.json` (planned in the
   per-tier signoff scripts; not built in Commit 46). A failure in
   tier N does not invalidate tier N-1's signoff, and a tier-N
   commit cannot start until tier N-1 is signed.
4. **Recovery.** Resuming from a partially-failed `topup_to_30`
   would require shard-level forensic. Resuming from a
   partially-failed `topup_to_5` recovers in hours.

## Expected cell counts

One replica of the protocol is **72 tasks × 4 canary methods × 3
algorithms = 864 executable canary cells**. That matches the
Commit-45 signed `n_canary_success_total = 864`.

| tier         | replicas added | additional canary cells | cumulative (canary) |
|--------------|---------------:|------------------------:|--------------------:|
| stage0 base  |              1 |                     864 |                 864 |
| topup_to_5   |              4 |                   3,456 |               4,320 |
| topup_to_10  |              5 |                   4,320 |               8,640 |
| topup_to_30  |             20 |                  17,280 |              25,920 |
| **total**    |         **30** |             **25,056**  |          **25,920** |

The protocol also defines non-canary panel methods (`random_search`,
`tpe_optuna`, etc. cover canary; `asha`, `bohb`, `dehb`, `nsga2`,
`motpe`, `parego`, plus the two stub_only ablations). Non-canary
rows are currently `stub_only` and dispatched as `skipped` /
`refused_not_in_canary_set` — the **canary** cell counts above are
what each tier must execute end-to-end. The full SQLite shard
populations (2,304 + 9,216 + 13,680 + 54,720 = 79,920 jobs) include
every method row; only the canary subset trains. The planner script
emits both the canary executable count and the full per-shard row
count for traceability.

## Lane split

Each replica retains the Commit 38 heavy-task policy split:

| lane     | tasks | canary cells / replica | notes                              |
|----------|------:|-----------------------:|------------------------------------|
| standard |    57 |                    684 | `timeout_seconds_per_cell = 1800` |
| heavy    |    13 |                    156 | `timeout_seconds_per_cell = 7200` |
| extreme  |     2 |                     24 | `timeout_seconds_per_cell = 14400`; `stage0_max_evaluations = 1` |
| **total**| **72**|                **864** |                                    |

Per-tier additional canary cells, per lane:

| tier        | standard | heavy   | extreme | total   |
|-------------|---------:|--------:|--------:|--------:|
| stage0 base |      684 |     156 |      24 |     864 |
| topup_to_5  |    2,736 |     624 |      96 |   3,456 |
| topup_to_10 |    3,420 |     780 |     120 |   4,320 |
| topup_to_30 |   13,680 |   3,120 |     480 |  17,280 |

These are budget-relevant numbers. Wall-clock will be dominated by
heavy + extreme even though the count is concentrated in standard.

## Why lane summaries remain the unit of reproducibility

Stage 0 published one summary per lane
(`experiments/_stage_runs/stage0_{standard,heavy,extreme}_lane_latest_summary.json`)
plus an aggregator. Top-up tiers continue the same pattern:

- one `stageN_topup_to_{5,10,30}_{lane}_lane_summary.json` per
  lane (planned schema, not yet finalized);
- a per-tier aggregator that mirrors
  `scripts/build_stage0_replica_signoff.py`;
- a per-tier signoff file that mirrors `stage3_signoff.json`.

Each lane summary is the smallest committable replication unit. We
do **not** commit fold CSVs, fitted models, raw OpenML payloads,
execution SQLite, or `catboost_info` — those are reconstructed from
the shard + policy. The summary is the gate.

## Per-machine execution split

`benchmarks/doctoral/openml_cc18/stage3_worker_plan.csv` (Commit
46) is the source of truth. The high-level mapping is:

| worker                  | lane(s)                | scope                                       |
|-------------------------|------------------------|---------------------------------------------|
| personal MacBook M4 Pro | standard (light)       | planning, summary aggregation, validation, occasional standard-lane shard |
| dedicated Mac           | standard, heavy        | `topup_to_5` / `topup_to_10` standard + heavy |
| future university Mac   | heavy, extreme         | `topup_to_30` heavy + extreme               |

The personal Mac is **not** 100 % dedicated; it runs the planner,
aggregator, and review surface. The dedicated Mac and the future
university worker run the actual training shards.

## How summaries should be committed

For each top-up tier and lane:

1. Worker runs the lane against its assigned shards (e.g.
   `--shards stage1_topup_to_005/shard_03`).
2. Worker exports a JSON / MD summary via
   `scripts/export_cc18_run_summary.py` (same path the stage-0
   lanes used).
3. Worker pushes a small commit that contains **only** the JSON /
   MD summary plus any updated docs.
4. The aggregator runs on the personal Mac and produces the per-
   tier signoff plan.
5. Operator review → tier signoff commit → next tier authorized.

## How large local artifacts stay outside Git

`.gitignore` already excludes:

- `runs/` (per-run materialized execution SQLite + per-cell fold
  outputs);
- `catboost_info/`;
- `data/source/openml_cc18/` raw payloads;
- fitted models, nested run outputs, notebooks, fairness artifacts.

The commit guardrails in this repo refuse to stage these paths.
**Do not** add any of them to a top-up commit. The summary JSON /
MD pair is sufficient to reproduce the result with the same shard +
policy + git_sha.

## Retry / resume

Per-cell retry semantics are inherited from the cc18_runner
(Commit 28+):

- `claim_lease_seconds` is honored at the shard level. A killed
  worker releases the lease after `claim_lease_seconds`; the next
  worker picks the cell up automatically.
- `status = failed_timeout` is the lane-policy outcome when the
  cell exceeds `timeout_seconds_per_cell`. Re-running the same
  cell will produce the same outcome unless the policy is changed
  — top-up commits **must not** silently rebudget cells across
  tiers (see `docs/STAGE3_POLICY_DECISION.md`).
- `status = failed_other` is a real failure and must be triaged
  before the tier signoff.

After interruption:

1. `git pull --ff-only` on the worker.
2. `python scripts/cc18_runner.py --resume --shards <shard>` (the
   runner skips `success` / `skipped` rows, retries `pending`).
3. Re-export the summary and re-push.

## Stale / drifted / wrong-policy detection

Each summary embeds:

- `policy_version` (SHA-256 of `heavy_task_policy.csv`);
- `source_git_sha` (the git_sha the worker ran at);
- `package_versions` (xgboost, lightgbm, catboost, optuna, openml,
  scikit-learn, numpy, pandas, etc.);
- `source_shards_unchanged` (per-cell MD5 cross-check against the
  committed shard);
- `stage3_signoff_present` (and the signoff hash on disk).

The per-tier aggregator (planned mirror of
`scripts/build_stage0_replica_signoff.py`) refuses if any of these
disagree across the lanes within the tier. The planner script
(Commit 46) refuses if the signed `policy_version` no longer
matches the live `heavy_task_policy.csv` — unless explicitly
passed `--allow-policy-drift-report-only`, in which case the
planner produces a candidate drift report (a separate artifact, not
a new signoff).

A worker is **stale** if its `source_git_sha` does not match the
HEAD of `origin/repo-publication-readiness` at the time the summary
was exported. A worker is **drifted** if its `policy_version`
differs from the signed `policy_version`. A worker is **using the
wrong policy_version** if it reads `heavy_task_policy.csv` that
hashes to anything other than
`47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`.

All three conditions are surfaced by the planner JSON output as
top-level booleans and by the per-tier aggregator as a refusal.

## What Commit 46 does and does not do

Commit 46 (this commit) is a **planning / scaffolding** commit
only. It:

- documents the top-up strategy (this file and
  `docs/STAGE3_POLICY_DECISION.md`);
- adds `scripts/plan_stage3_topup.py` (read-only planner);
- emits one new JSON / MD pair under
  `experiments/_stage_runs/stage3_topup_plan_latest_summary.{json,md}`;
- emits two new manifests under
  `benchmarks/doctoral/openml_cc18/stage3_topup_manifest.csv`
  and `stage3_worker_plan.csv` (+ their `.md` siblings);
- adds tests for the planner;
- updates docs to reflect the new tier roadmap.

Commit 46 explicitly does **not**:

- run any OpenML training;
- mutate committed SQLite shards;
- create execution SQLite files;
- stage raw OpenML payloads, fold CSVs, fitted models, runs/,
  catboost_info, nested run outputs, notebooks, fairness
  artifacts;
- regenerate `heavy_task_policy.csv`, its report, or
  `runtime_guardrails.yaml`;
- change `policy_version`;
- promote any task between lanes;
- consume the signoff (the signoff is still
  `downstream_execution_authorized_in_this_commit = false`);
- dispatch on cloud.

## Expected next step after Commit 46

Commit 47 is the **tiny Stage-3 / top-up pilot**:

- `replica_002` only;
- `shard_00` only;
- `standard` lane only;
- canary methods only (`default_gbdt`, `random_search`,
  `tpe_optuna`, `doe_rsm_vrf_true_nbi`);
- no heavy / extreme.

After Commit 47 publishes a clean pilot summary and an operator
review, the next commit dispatches the rest of `topup_to_5` per
the worker plan.
