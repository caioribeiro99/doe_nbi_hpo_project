# CC18 result handoff protocol

This document is the contract that lets the personal Mac, the
dedicated Mac, and any future worker share doctoral-CC18 results
*without ever shipping mutable execution state through Git*. It
formalizes the operational pattern that batches 00 and 01 already
follow, so batch 02 onward can reuse it without surprises.

## tl;dr

1. **Git is the control plane.** Every artifact that crosses
   machines goes through `git pull` / `git push` on the
   `repo-publication-readiness` branch.
2. **Committed SQLite shards under `jobs/doctoral/openml_cc18/shards/`
   are immutable job-queue templates.** Workers never `--train`
   directly against them; they always copy first.
3. **Execution SQLite files live under `runs/cc18/<run_id>/` and
   are gitignored.** Status transitions, fitted models, fold CSVs,
   `catboost_info`, OpenML payloads and other large outputs all
   stay there.
4. **Small JSON / Markdown summaries land under
   `experiments/_stage_runs/` and are committed.** They are how
   the dedicated Mac tells the personal Mac (and the article)
   what happened.
5. **Large artifacts (full per-cell manifests, fitted models,
   logs, OpenML caches) ship out-of-band** to an external archive,
   referenced by SHA-256 in the committed summary.

## Why

Two reasons drove the split:

- **SQLite mutability.** A worker has to flip `status` on a job
  row to `claimed` / `running` / `success` / `failed`. If the
  committed shard was the live job-queue, every dedicated-Mac run
  would dirty Git history with non-reproducible bytes, and a
  personal-Mac `git pull` could clobber an in-flight run.
- **Repo size.** A full stage-0 run over 2,304 jobs already
  produces tens of MB of per-cell manifests, fold CSVs, and
  CatBoost training caches. Multiplying by 30 replicas would
  push the repo into hundreds of MB. Small JSON / MD summaries
  keep the repo navigable.

## Layout

### Committed (the control plane)

```
jobs/doctoral/openml_cc18/
├── schema.sql                       # cc18_jobs schema
├── README.md
└── shards/                          # 40 immutable .sqlite files
    ├── shard_summary.json
    ├── shard_summary.md
    ├── stage0_replica_001/shard_00.sqlite ... shard_09.sqlite
    ├── stage1_topup_to_005/...
    ├── stage2_topup_to_010/...
    └── stage3_topup_to_030/...

experiments/
├── _capability_audit/               # capability report (json + md)
├── _batch_runs/                     # batch_00 / batch_01 / ... gates (json + md)
└── _stage_runs/                     # NEW: per-run summaries (json + md)
    ├── <run_id>_summary.json
    ├── <run_id>_summary.md
    └── ...

docs/RESULT_HANDOFF_PROTOCOL.md      # this file
```

### Gitignored (local execution state, per-machine)

```
runs/                                # whole tree gitignored
└── cc18/
    └── <run_id>/                    # one per (machine, batch / stage, attempt)
        ├── run_manifest.json        # source SHA, source-shard MD5s, config
        ├── shards/
        │   └── <stage>/
        │       ├── shard_00.execution.sqlite
        │       └── ...
        ├── outputs/                 # per-cell fold metrics, fitted models, logs
        │   └── <job_id>/
        │       ├── manifest.json
        │       ├── fold_metrics.json
        │       └── catboost_info/
        └── archives/                # tarballs prepared for external upload

data/source/openml_cc18/             # already gitignored (Commit 34)
└── _openml_cache/                   # OpenML library cache
└── <openml_task_id>/                # per-task pickle payload + manifest
```

### External (out of Git, optionally hashed in summaries)

Large artifacts that need to round-trip between machines (per-cell
manifests for a full stage, fitted CatBoost forests, fold CSVs)
ship through an out-of-band archive:

- **Default:** a `tar.zst` under `runs/cc18/<run_id>/archives/`
  uploaded to the operator's chosen bucket / storage. The
  `<run_id>_summary.json` records:
  - `archive_path` (URI or local path on the publishing machine);
  - `archive_sha256`;
  - `archive_size_bytes`.
- A future puller is expected to use those fields to download +
  verify before post-processing.

## Lifecycle

### 1. Plan (any machine)

The control-plane operator picks a stage / batch and assigns a
`run_id`. The convention is:

```
<batch_id_or_stage>__<host>__<utc_yyyymmddThhmmssZ>
```

For example:
```
batch_02_cc18_small_12_tasks__Factored-LWCTW4633L__2026-05-04T18-30-00Z
```

A `run_id` is opaque outside this protocol; it just has to be
unique on the publishing machine.

### 2. Materialize (worker machine)

```bash
python scripts/create_cc18_run_dir.py \
    --run-id "<run_id>" \
    --shards-dir jobs/doctoral/openml_cc18/shards/stage0_replica_001 \
    --shard shard_00.sqlite --shard shard_01.sqlite ...
```

`create_cc18_run_dir.py`:

- creates `runs/cc18/<run_id>/shards/<stage>/`;
- copies each requested committed shard into that path with a
  `.execution.sqlite` suffix;
- never opens the source for write — sources are only read via
  ``mode=ro`` URIs and are MD5-hashed before the copy and after
  the copy, with both hashes recorded in `run_manifest.json`;
- writes `runs/cc18/<run_id>/run_manifest.json` with: `run_id`,
  `created_at`, `source_git_sha`, `host`, `python`, the per-shard
  source MD5s, and the per-shard execution-copy MD5s.

### 3. Execute (worker machine)

The runner (`scripts/cc18_runner.py`, or batch-specific wrappers
like `scripts/run_batch_01_cc18_tiny_3_tasks.py`) consumes only
the `.execution.sqlite` files under `runs/cc18/<run_id>/shards/`.

Status transitions and per-cell outputs land entirely under
`runs/cc18/<run_id>/`. The committed shards are never touched —
the runner refuses to claim a row written through a write
handle to the canonical path.

### 4. Summarize and publish (worker machine)

```bash
python scripts/export_cc18_run_summary.py \
    --run-dir runs/cc18/<run_id> \
    --out-json experiments/_stage_runs/<run_id>_summary.json \
    --out-md  experiments/_stage_runs/<run_id>_summary.md \
    --include-shard-hashes
```

The exporter inspects every `.execution.sqlite` under the run
dir and records:

- `run_id`, `stage`, `batch_id` (when the run dir was created
  for a named batch);
- source git SHA, hostname, started_at / finished_at when
  available, total runtime;
- counts of `pending` / `claimed` / `running` / `success` /
  `failed` / `skipped` rows;
- failure rollups by `(method, algorithm, openml_task_id)` plus
  short error tails;
- per-shard row counts, SHA-256 of each execution SQLite file
  (when `--include-shard-hashes` is set);
- the recorded source-shard MD5s from `run_manifest.json`,
  re-checked against the live committed shard so the summary
  proves the source is untouched;
- optional `archive_path` / `archive_sha256` fields when an
  archive is being published alongside.

The two output files are committed and pushed:

```bash
git add experiments/_stage_runs/<run_id>_summary.json
git add experiments/_stage_runs/<run_id>_summary.md
git commit -m "Run <run_id>: summary"
git push
```

### 5. Sync (other machines)

The personal Mac (or any reader) only needs:

```bash
git fetch origin
git switch repo-publication-readiness
git pull --ff-only
```

The summaries under `experiments/_stage_runs/` show what ran,
where, and with what status. The execution SQLite files and
fitted models stay on the publishing machine; pull them through
the archive URI in the summary if the reader needs the raw
artifacts.

## Heavy-task policy interaction (Commit 38)

From Commit 38 onward, every CC18 runner also consults the
heavy-task policy at `benchmarks/doctoral/openml_cc18/{runtime_guardrails.yaml,
heavy_task_policy.csv}` via `src/doe_xgb/runtime_guardrails.py`.
The interaction with the handoff protocol is additive:

- the policy determines whether a task is deferred (extreme lane)
  or runs with a lane-specific timeout / max_evaluations cap;
- deferred tasks appear in the stage-run summary as
  `n_skipped` cells with `last_error = "deferred_extreme_lane"`
  (or surface in a top-level `deferred_extreme_tasks` field);
- the rest of the protocol — `runs/cc18/<run_id>/` for execution
  state, `experiments/_stage_runs/<run_id>_summary.{json,md}`
  for the cross-machine handoff, run_manifest.json for source
  MD5s — is unchanged.

See `docs/HEAVY_TASK_POLICY.md` for the lane definitions and the
classification rules.

## Stage 0 lane summaries (Commits 40 → 44)

Stage 0 publishes one summary per lane under
`experiments/_stage_runs/`:

- `stage0_standard_lane_latest_summary.{json,md}` — Commit 40;
- `stage0_heavy_lane_latest_summary.{json,md}` — Commit 41;
- `stage0_extreme_lane_plan_latest_summary.{json,md}` —
  Commit 42 (planning-only; `execution_status =
  "planned_not_executed"`);
- `stage0_extreme_lane_latest_summary.{json,md}` — Commit 43
  (`28961fe`); ships with `execution_status = "executed"`.
- `stage0_replica_001_signoff_plan_latest_summary.{json,md}` —
  Commit 44 published this with `signoff_status =
  "planned_not_signed"`. **Commit 45 re-runs the aggregator** so
  it now reads `signoff_status = "signed"`,
  `final_recommendation = "signed_ready_for_next_stage_planning"`,
  carries `stage3_signoff_sha256` (the on-disk SHA-256 of the
  Commit 45 signoff JSON), and includes a `signoff_record`
  block with operator metadata. The aggregator cross-checks the
  signoff's recorded lane SHA-256s against the live ones on
  every run; mismatch raises `SignoffRefusalError`.
- `jobs/doctoral/openml_cc18/stage3_signoff.json` — created by
  Commit 45 via `scripts/sign_stage0_replica_001.py`. Carries
  `downstream_execution_authorized_in_this_commit = false`; see
  `docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md`.

The dedicated planning summary keeps the same JSON layout used
by the other lanes, with an extra top-level field
`execution_status` ∈ {`planned_not_executed`, `executed`} so a
reader can tell at a glance whether the extreme lane was
actually run. Stage 0 replica 1 is *complete* only when all
three lanes have green summaries pinned to the same
`policy_version`, all carry `source_shards_unchanged: true`,
all carry `stage3_signoff_present: false`, and the extreme
summary's `execution_status` is `executed`.

## Refusal rules

- A worker MUST NOT pass a path under `jobs/doctoral/openml_cc18/shards/`
  to `--train`. The dedicated-Mac workflow always copies first.
- An execution SQLite file MUST live under `runs/`. The exporter
  refuses to summarize a file that lives under `jobs/`.
- The exporter MUST verify, against `run_manifest.json`, that the
  committed source shards still hash to their pre-run MD5. If they
  drifted, the summary records `source_shards_unchanged: false`,
  and downstream gates treat the run as poisoned.
- `stage3_signoff.json` is never created by the export protocol
  itself. It was created exactly once by Commit 45's
  `scripts/sign_stage0_replica_001.py`, which checks every gate
  the Commit 44 aggregator advertised, writes the file with
  operator metadata + both required caveat acknowledgements +
  `downstream_execution_authorized_in_this_commit = false`, and
  re-publishes the aggregate plan. The runner refuses stage-3
  rows tagged `requires_manual_signoff_before_stage3` while the
  file is absent; once present, it still does not auto-dispatch
  — actual stage-3 execution is a separate, operator-reviewed
  commit.
- Stage-3 / top-up *planning* (Commit 46 onward) is also read-
  only against the handoff protocol. `scripts/plan_stage3_topup.py`
  reads the signoff + lane summaries + `heavy_task_policy.csv` +
  shard MD5s and publishes
  `experiments/_stage_runs/stage3_topup_plan_latest_summary.{json,md}`.
  The planner refuses if the live `policy_version` differs from
  the signed one (unless `--allow-policy-drift-report-only` is
  passed) and if any lane summary's SHA-256 drifted since
  signoff. The planner does not create execution SQLite files
  and does not mutate committed shards.

## What gets committed vs. what does not

| Artifact | Committed? |
|---|---|
| Committed shards (`jobs/doctoral/openml_cc18/shards/`) | Yes — immutable |
| Capability audit (`experiments/_capability_audit/`) | Yes (json + md only) |
| Batch gate artifacts (`experiments/_batch_runs/`) | Yes (json + md only) |
| **Stage-run summaries (`experiments/_stage_runs/`)** | **Yes (json + md only)** |
| Execution SQLite (`runs/cc18/<run>/shards/...execution.sqlite`) | No — gitignored |
| Per-cell fold metrics / manifests under `runs/cc18/<run>/outputs/` | No — gitignored |
| Fitted models (`*.cbm`, `*.txt`, `*.json` under runs/) | No — gitignored |
| `catboost_info/` | No — gitignored |
| Raw OpenML payloads (`data/source/openml_cc18/<task>/payload.pkl`) | No — gitignored |
| Run-level archives (`runs/cc18/<run>/archives/*.tar.zst`) | No — external |

## Multi-machine sanity invariants

A read of `git log` plus the committed summaries should answer:

- "Has stage 0 / batch X been completed on the dedicated Mac?"
  — yes if a green summary under `experiments/_stage_runs/`
  references that stage / batch.
- "Were the committed shards mutated during run Y?"
  — `source_shards_unchanged` field in the summary, plus the
  per-shard `md5_before` / `md5_after` block.
- "Where are the fitted models for run Y?"
  — `archive_path` field (if published) or "publishing machine
  only" when no archive was created.
- "Which packages were in play for run Y?"
  — `package_versions` block in the summary (sourced from
  `doe_xgb._versions.collect_package_versions`).

If the answer to any of those questions is ambiguous, the
protocol has been violated and the run should be re-published.
