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

The runner skeleton at `scripts/cc18_runner.py` (Commit 29) opens a
shard, selects pending jobs, resolves the method adapter, logs a
dispatch decision, and (in non-`--dry-run` mode) briefly claims +
releases each job without training. It does **not** execute any
HPO yet — every adapter's `run()` raises `NotImplementedError`.

Before running on the dedicated Mac, refresh the capability audit:

```bash
pip install -e ".[gbdt,hpo_baselines,doctoral,dev]"
python scripts/audit_method_capabilities.py
```

Stage 3 is locked: the runner refuses to claim any job carrying the
`requires_manual_signoff_before_stage3` note unless
`jobs/doctoral/openml_cc18/stage3_signoff.json` exists. Commit 29
deliberately does not create that file.

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
