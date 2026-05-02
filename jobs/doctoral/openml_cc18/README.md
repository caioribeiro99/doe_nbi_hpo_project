# OpenML-CC18 doctoral job matrix

SQLite-backed job queues for the doctoral campaign. Schema in
`schema.sql`. Concrete shard files (`shards/shard_*.sqlite`) are
generated deterministically by `scripts/generate_cc18_job_shards.py`
(planned for Commit 26).

## Layout

```
jobs/doctoral/openml_cc18/
├── README.md
├── schema.sql
└── shards/
    ├── shard_001.sqlite      (gitignored unless tiny)
    └── ...
```

## Granularity

One job = one OpenML task × algorithm × method × replica.

For the headline doctoral target:

```
72 tasks × 3 algorithms × 1 method × 30 replicas = 6,480 jobs
```

Stage breakdown:

| Stage | Replicas added | Jobs added | Cumulative |
|---|---:|---:|---:|
| stage0_replica_001 | 1 | 216 | 216 |
| stage1_topup_to_005 | 4 | 864 | 1,080 |
| stage2_topup_to_010 | 5 | 1,080 | 2,160 |
| stage3_topup_to_030 | 20 | 4,320 | 6,480 |

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
