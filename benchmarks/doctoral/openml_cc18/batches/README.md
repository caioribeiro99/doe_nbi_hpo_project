# OpenML-CC18 reduced execution batches

- generated_at: `2026-05-02T22:55:33Z`
- batch_seed: `20260502`
- source: `benchmarks/doctoral/openml_cc18/tasks.csv`

These batches let us run small, representative subsets of OpenML-CC18 before kicking off any full stage. They are deterministic: regenerating with the same `tasks.csv` produces byte-identical batch CSVs.

## Execution order

| step | batch | scope |
|---|---|---|
| A | `batch_00_synthetic_canary` | 4 canary methods on synthetic binary; no OpenML data |
| B | `batch_01_cc18_tiny_3_tasks` | 3 real CC18 tasks: small binary numeric / small categorical / small multiclass |
| C | `batch_02_cc18_small_12_tasks` | 12 real CC18 tasks; stratified by task_type / categorical / size / balance |
| D | `batch_03_cc18_representative_18_tasks` | 18 real CC18 tasks; broader coverage |
| E | `batch_04_stage0_shard00_only` | one existing stage-0 shard from Commit 28 |
| F | full stage 0 | 2,304 jobs across the 10 stage-0 shards |
| G | top-up to stages 1 / 2 / 3 | gated by manual sign-off as documented in `execution_tiers.md` |

Steps A-D are pre-stage-0 pilots: they validate the adapters, the OpenML loader, and the runner. Step E is an operational dry run on the smallest real worker shard. Steps F and G follow only after each prior step lands a green sign-off artifact under `experiments/_canary_runs/` or `experiments/_batch_runs/`.

## Files in this directory

| file | rows | purpose |
|---|---:|---|
| `batch_00_synthetic_canary.json` | — | synthetic binary; canary methods only |
| `batch_01_cc18_tiny_3_tasks.csv` | 3 | 3 real CC18 tasks (binary numeric / categorical / multiclass) |
| `batch_02_cc18_small_12_tasks.csv` | 12 | 12 real CC18 tasks; stratified pilot |
| `batch_03_cc18_representative_18_tasks.csv` | 18 | 18 real CC18 tasks; pre-stage0 research pilot |
| `batch_04_stage0_shard00_only.json` | — | pointer to one existing stage-0 SQLite shard |

Each `.csv` batch ships a `.meta.json` sidecar with the selection rule, the deterministic seed, and the explicit task-id list.

## Regenerating the batches

```bash
python scripts/create_cc18_batches.py --force
```

## Filtering a SQLite shard for a batch

```bash
python scripts/filter_cc18_shard_for_batch.py \
    --source jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \
    --batch-file benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv \
    --out jobs/doctoral/openml_cc18/batch_shards/batch_01_shard_00.sqlite
```
The filter NEVER mutates the source shard.
