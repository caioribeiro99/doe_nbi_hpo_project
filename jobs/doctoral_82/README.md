# Doctoral benchmark — SQLite job matrix

This directory holds the per-shard SQLite job-queue files that drive
the doctoral campaign. Schema in `schema.sql`. Concrete shard files
(`shards/shard_*.sqlite`) are generated deterministically by
`scripts/generate_doctoral_job_shards.py` (planned for Commit 25).

## Layout

```
jobs/doctoral_82/
├── README.md                 (this file)
├── schema.sql                (single source of truth)
└── shards/
    ├── shard_001.sqlite      (gitignored unless tiny)
    ├── shard_002.sqlite
    └── ...
```

## Why SQLite shards

- One file per worker keeps lock contention out of the way; the
  campaign runs locally on a 10-worker dedicated Mac.
- Shards are deterministic: `scripts/generate_doctoral_job_shards.py
  --panel-csv benchmarks/doctoral_82/datasets.csv --shards 10` always
  produces the same `job_id` set for the same panel snapshot.
- The granularity is one job = `(dataset, algorithm, method, replica)`
  so the staged top-up (1 → 5 → 10 → 30) is a simple `WHERE stage=
  'stage<N>...' AND status='pending'` query.

## Job lifecycle

```
pending  ──claim──▶  claimed  ──start──▶  running
                                  │
                                  ├──success──▶  success
                                  │
                                  └──fail────▶  failed (retry_count++)
```

Re-runnable jobs after a failure: bump `retry_count`; abandon after a
configurable threshold (default 3). The `assigned_worker` column is
the local hostname or PID-prefix that grabbed the job.

## Generation policy

The generator must:

1. Validate the panel CSV (`registry.validate_registry`).
2. Refuse to write shards if any `include=True` row has
   `loader_status != 'registered'`.
3. Emit deterministic `job_id` via
   `doctoral_benchmark.jobs.job_id(...)`.
4. Honour the four stage names defined in
   `STAGE_NAMES`. Stage `stage0_replica_001` always lands first; the
   top-up stages append rows for replicas 2–5, 6–10, 11–30.
5. Round-robin shard assignment by `(dataset_id, algorithm)` so a
   single worker's shard never hammers the same dataset back-to-back
   on every replica.

## Total job count for the headline doctoral target

Once the panel reaches 82 datasets:

```
82 datasets × 3 algorithms × 1 method × 30 replicas = 7,380 jobs
```

For the staged subsets:

| Stage | Replicas in stage | Total jobs added | Cumulative |
|---|---:|---:|---:|
| stage0_replica_001 | 1 | 246 | 246 |
| stage1_topup_to_005 | 4 | 984 | 1,230 |
| stage2_topup_to_010 | 5 | 1,230 | 2,460 |
| stage3_topup_to_030 | 20 | 4,920 | 7,380 |

## What this directory does NOT contain

- Real raw / processed dataset payloads (gitignored).
- Per-replica result CSVs (those land under
  `experiments/doctoral_82/<dataset>/<algorithm>/<method>/replica_NNN/`,
  gitignored).
- Cloud / GitHub Actions worker provisioning. The doctoral campaign
  runs locally on the dedicated Mac by default.
