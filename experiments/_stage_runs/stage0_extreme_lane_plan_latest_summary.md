# stage0 extreme-lane plan — stage0_extreme_lane_plan_latest

- run_id: `stage0_extreme_lane_plan_latest`
- batch_id: `stage0_extreme_lane_plan`
- lane: `extreme`
- stage: `stage0_replica_001`
- exported_at: `2026-05-17T13:48:17Z`
- **execution_status: `planned_not_executed`**
- plan_doc: `docs/EXTREME_LANE_PLAN.md`
- policy_version: `47b6b50c6d1e1d09` (pinned: `47b6b50c6d1e1d09`)

## Universe of stage-0 rows

- total: 2304
- expected extreme-canary cells: 24
- runnable extreme canary (this lane would execute): **24**
- skipped standard (already completed in Commit 40): 1815
- skipped heavy (already completed in Commit 41): 423
- refused (extreme × non-canary methods): 42

- extreme tasks to execute: [6, 167121]
- non_canary_methods_refused_on_extreme: ['asha', 'bohb', 'dehb', 'motpe', 'nsga2', 'parego', 'smac3']

## stage0 standard-lane pre-flight

- exported_at: `2026-05-16T04:44:08Z`
- age_days: 1.38
- n_executed=684, failed=0, pending=0
- policy_version: `47b6b50c6d1e1d09`

## stage0 heavy-lane pre-flight

- exported_at: `2026-05-17T04:31:54Z`
- age_days: 0.39
- n_executed=156, failed=0, pending=0
- policy_version: `47b6b50c6d1e1d09`

## Per-shard plan

| shard | total | runnable_extreme | skip_std | skip_heavy | refused |
|---|---:|---:|---:|---:|---:|
| `shard_00.sqlite` | 219 | 4 | 177 | 31 | 7 |
| `shard_01.sqlite` | 230 | 0 | 199 | 31 | 0 |
| `shard_02.sqlite` | 223 | 0 | 192 | 31 | 0 |
| `shard_03.sqlite` | 235 | 4 | 202 | 22 | 7 |
| `shard_04.sqlite` | 226 | 4 | 193 | 22 | 7 |
| `shard_05.sqlite` | 226 | 4 | 160 | 55 | 7 |
| `shard_06.sqlite` | 228 | 0 | 162 | 66 | 0 |
| `shard_07.sqlite` | 270 | 0 | 182 | 88 | 0 |
| `shard_08.sqlite` | 237 | 4 | 182 | 44 | 7 |
| `shard_09.sqlite` | 210 | 4 | 166 | 33 | 7 |

## Source-shard MD5 (read-only)

| shard | md5 |
|---|---|
| `shard_00.sqlite` | `91e7a861ea73daf82694029d6c590e54` |
| `shard_01.sqlite` | `b94e71ccb24d5d184c3346d336c2691d` |
| `shard_02.sqlite` | `38e0208538432577d82840d356ca039d` |
| `shard_03.sqlite` | `198c30f36e040c18af674eb6510ccd1d` |
| `shard_04.sqlite` | `c5eb54e008f90abf7a3e47e7f4a22584` |
| `shard_05.sqlite` | `ff9d67f50910ba1753602a5eac16905c` |
| `shard_06.sqlite` | `4f6d062e42e4df8b72c82803fec1b814` |
| `shard_07.sqlite` | `83fb2d1e840aff2376ee70959d1961dd` |
| `shard_08.sqlite` | `711d28b2ce61381a4b72e24a90b107af` |
| `shard_09.sqlite` | `f2c5f528ad680b0c4c670b8bdc11bde7` |

## Runtime ETA (anchored on batch_03)

- expected total runner CPU: ~56032 s
- dedicated Mac wall-clock estimate: ~15.7 h
- local laptop: DO NOT RUN — Devnagari catboost OOM risk under thermal limits

## Promotion criteria for stage 0 replica 1

- standard / heavy / extreme stage-run summaries all green
- all three summaries share the same policy_version
- all three carry source_shards_unchanged=True
- all three carry stage3_signoff_present=False
- extreme summary execution_status == 'executed' (planned_not_executed does not count)

## What happens next

After human review of docs/EXTREME_LANE_PLAN.md, Commit 43 may invoke this same script with --execute-extreme-lane. Commit 42 must not.
