# OpenML-CC18 doctoral job matrix

SQLite-backed job queues for the doctoral campaign. Schema in
`schema.sql`. Shards under `shards/<stage>/shard_NN.sqlite` are
generated deterministically by `scripts/generate_cc18_job_shards.py`
(landed in Commit 28).

## Layout

```
jobs/doctoral/openml_cc18/
├── README.md
├── schema.sql
└── shards/
    ├── shard_summary.json
    ├── shard_summary.md
    ├── stage0_replica_001/
    │   ├── shard_00.sqlite
    │   ├── shard_01.sqlite
    │   ├── ...
    │   └── shard_09.sqlite
    ├── stage1_topup_to_005/
    │   ├── shard_00.sqlite
    │   └── ...
    ├── stage2_topup_to_010/
    │   └── ...
    └── stage3_topup_to_030/
        └── ...
```

40 shard files at 10 shards per stage × 4 stages. Shard files are
**deterministic job queues**, not benchmark results, and are
committed to the repository so that the dedicated MacBook Pro can
fetch them by `git pull` rather than regenerate them.

## Granularity

One job = one OpenML task × algorithm × method × replica.

For the frozen comparative protocol (Commit 27):

| Stage | Replicas added | Jobs added | Cumulative |
|---|---:|---:|---:|
| stage0_replica_001  |  1 |  2,304 |  2,304 |
| stage1_topup_to_005 |  4 |  9,216 | 11,520 |
| stage2_topup_to_010 |  5 | 13,680 | 25,200 |
| stage3_topup_to_030 | 20 | 54,720 | **79,920** |

The exact composition is in `shards/shard_summary.json` /
`shards/shard_summary.md`.

## Why stage-separated

Each stage lives under its own subdirectory so the dedicated Mac can
copy / claim / drain stages independently:

- Stage 0 is the cheap sizing pass; it is always run first.
- Top-up stages do not re-run earlier replicas; the replica field is
  the deduplication key.
- Stage 3 jobs of every tier-1+ method carry the
  `requires_manual_signoff_before_stage3` note; a future runner
  will refuse to claim them until the stage-3 sign-off file exists
  (planned at `jobs/doctoral/openml_cc18/stage3_signoff.json`).

## Regenerating shards

```bash
# Dry run (compute counts and write summary; no SQLite written).
python scripts/generate_cc18_job_shards.py --dry-run

# Materialize all four stages × 10 shards.
python scripts/generate_cc18_job_shards.py --shards 10 --force

# Restrict to stage 0 only (e.g., for a sizing run).
python scripts/generate_cc18_job_shards.py \
    --shards 10 --force --stage stage0_replica_001
```

The generator reads:

- `benchmarks/doctoral/openml_cc18/tasks.csv`
- `benchmarks/doctoral/openml_cc18/method_matrix.csv`
- `benchmarks/doctoral/openml_cc18/execution_policy.csv`
- `benchmarks/doctoral/openml_cc18/parego_subset.csv`
- `jobs/doctoral/openml_cc18/schema.sql`

It is the single point of truth for shard contents; **no method
names, scope rules, or stage-gating logic are hardcoded**. Edit the
CSVs and re-run the generator to alter the job matrix.

## Inspecting a shard

```bash
sqlite3 jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \
    "SELECT method, count(*) FROM cc18_jobs GROUP BY method ORDER BY method"

sqlite3 jobs/doctoral/openml_cc18/shards/stage3_topup_to_030/shard_00.sqlite \
    "SELECT count(*) FROM cc18_jobs
     WHERE notes='requires_manual_signoff_before_stage3'"
```

`shards/shard_summary.json` and `shards/shard_summary.md` carry the
full breakdown by stage / shard / method / algorithm.

## Copying to the dedicated Mac

The repository at `https://github.com/caioribeiro99/doe_nbi_hpo_project`
is the transport. On the dedicated Mac:

```bash
git fetch origin
git switch repo-publication-readiness
git pull --ff-only

# Inspect the shards that will be claimed.
sqlite3 jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \
    "SELECT count(*), stage FROM cc18_jobs GROUP BY stage"
```

The runner at `scripts/cc18_runner.py` opens a shard, selects
pending jobs, resolves the method adapter, and logs a dispatch
decision. The default `--no-train` mode briefly claims + releases
each job without training. With **both** `--canary-only` and
`--train`, the four canary adapters
(`default_gbdt`, `random_search`, `tpe_optuna`,
`doe_rsm_vrf_true_nbi`) execute end-to-end on a `--synthetic-task`;
non-canary methods are still refused
(`refused_not_in_canary_set`).

Before running on the dedicated Mac, refresh the capability audit:

```bash
pip install -e ".[gbdt,hpo_baselines,doctoral,dev]"
python scripts/audit_method_capabilities.py
```

Stage 3 is locked: the runner refuses to claim any job carrying the
`requires_manual_signoff_before_stage3` note unless
`jobs/doctoral/openml_cc18/stage3_signoff.json` exists. The signoff
file was created in Commit 45 (operator-reviewed; see
`docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md`), but it carries
`downstream_execution_authorized_in_this_commit = false` — so the
*runner* will no longer refuse on absence, but actual stage-3
dispatch still needs a separate, operator-reviewed commit.

## Heavy-task policy (Commit 38)

CC18 tasks are now split into three lanes (`standard`, `heavy`,
`extreme`) via
`benchmarks/doctoral/openml_cc18/{runtime_guardrails.yaml,
heavy_task_policy.csv}`. Per-lane budgets enforce timeouts and
max_evaluations caps; extreme tasks (`Devnagari-Script`,
`letter`) are deferred unless the runner is invoked with
`--include-extreme-tasks`. batch_04 onward MUST consult the
policy via `src/doe_xgb/runtime_guardrails.py`. Full stage 0
splits into a standard / heavy / extreme pass, each with its own
stage-run summary under `experiments/_stage_runs/`. See
`docs/HEAVY_TASK_POLICY.md`.

## Stage 0 lane progress (Commits 40 → 45)

Stage 0 splits into three independent lanes per the heavy-task
policy (Commit 38), followed by aggregate planning and operator
signoff:

- standard (Commit 40 `daae8ab`): 684 / 684 green;
- heavy (Commit 41 `ddb657d`): 156 / 156 green;
- extreme (Commit 43 `28961fe`): 24 / 24 green at the policy's
  `extreme.stage0_max_evaluations = 1`;
- aggregate signoff plan (Commit 44): 864 / 864 canary cells
  across 72 tasks, all four artifacts pinned to the same
  `policy_version`; initially `signoff_status =
  "planned_not_signed"`;
- operator signoff (Commit 45): `stage3_signoff.json` written
  via `scripts/sign_stage0_replica_001.py` with operator
  metadata, both required caveat acknowledgements, and
  `downstream_execution_authorized_in_this_commit = false`. The
  aggregator is re-run; the published plan summary now reads
  `signoff_status = "signed"`.

All lanes pin the same `policy_version`. The committed shards
under `shards/stage0_replica_001/` are byte-identical to the
Commit 28 baseline (each lane uses copies under
`runs/cc18/<run_id>/`). `stage3_signoff.json` is now present
(see Commit 45); it unlocks the *planning* of stage-3 top-up
commits but does not itself dispatch any cells. See
`docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md` for the review
surface and the fields the signoff records.

## Result handoff protocol (Commit 35)

The committed shards under `shards/` are *immutable job-queue
templates*. From batch_02 onward, every worker materializes a
gitignored execution copy under `runs/cc18/<run_id>/shards/<stage>/`
via `scripts/create_cc18_run_dir.py`, runs against those copies,
and publishes a small JSON/MD summary under
`experiments/_stage_runs/<run_id>_summary.{json,md}` via
`scripts/export_cc18_run_summary.py`. The summary verifies that
each source shard's MD5 still matches the value recorded in the
run-dir's `run_manifest.json`; mismatches surface as
`source_shards_unchanged: false` and stop downstream promotion.
See `docs/RESULT_HANDOFF_PROTOCOL.md` for the full contract.

batch_00 and batch_01 used a slimmer temp-shard pattern; their
artifacts at `experiments/_batch_runs/batch_0X_..._latest.{json,md}`
are conceptually a specialization of the same protocol.

## Reduced-execution batches (Commit 31)

Before stage 0 runs, the dedicated Mac walks through five
pre-stage-0 batches manifested under
`benchmarks/doctoral/openml_cc18/batches/`. Use
`scripts/filter_cc18_shard_for_batch.py` to derive a batch shard
from a committed shard (the source is opened read-only via SQLite
URI mode, so the committed shards under `shards/` are never
mutated). Output goes under `jobs/doctoral/openml_cc18/batch_shards/`
or a tmp directory.

**Stage 0 must not start** until batches A→E land green sign-off
artifacts. The first gate (`batch_00_synthetic_canary`) is
scripted in Commit 32:

```bash
bash scripts/setup_dedicated_mac.sh
python scripts/audit_method_capabilities.py
python scripts/run_batch_00_synthetic_canary.py
```

Artifacts land at
`experiments/_batch_runs/batch_00_synthetic_canary_latest.{json,md}`
and are committed only when produced on the dedicated Mac.

## Task-based vs dataset-based identity

A few CC18 tasks share the same underlying OpenML dataset (different
target column or train/test split). The job matrix therefore keys
both `openml_task_id` and `openml_dataset_id`, so the campaign can
report results "per task" (the OpenML convention) without losing the
dataset deduplication needed for storage / loader caching.

## Versus `jobs/doctoral_82/`

`jobs/doctoral_82/schema.sql` is the pre-pivot scaffolding.
**Deprecated** in Commit 25; do not generate new shards there. The
table name there (`doctoral_jobs`) and the table here
(`cc18_jobs`) intentionally differ so a stray import cannot mix
the two.
