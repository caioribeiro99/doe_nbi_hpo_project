# Stage 3 / replica_002 extreme-lane plan (Commit 50)

- run_id: `stage3_replica_002_extreme_lane_plan_latest`
- batch_id: `stage3_replica_002_extreme_lane_plan`
- stage: `stage1_topup_to_005`
- topup_tier: `topup_to_5_partial`
- **execution_status: `planned_not_executed`**
- replica: **2** (source template replica = 1)
- lane: `extreme`
- git_sha: `7c1777d200f0`
- host: `Factored-LWCTW4633L`
- policy_version: `47b6b50c6d1e1d09`
- policy_version_pinned: `47b6b50c6d1e1d09`

## Gate chain

- signoff: `jobs/doctoral/openml_cc18/stage3_signoff.json` sha256=`3f3f1b1fd6819344` (signed / stage0_replica_001 / operator caioribeiro99)
- stage3 top-up plan: `experiments/_stage_runs/stage3_topup_plan_latest_summary.json` sha256=`bb79eaee6b5a4da2` (planned_not_executed)
- Commit 48 standard summary: `experiments/_stage_runs/stage3_replica_002_standard_lane_latest_summary.json` sha256=`6d028cef7c3715cf` (success=684, runtime≈7288s)
- Commit 49 heavy summary: `experiments/_stage_runs/stage3_replica_002_heavy_lane_latest_summary.json` sha256=`95fba21f38db3453` (success=156, runtime≈37301s)

## Scope of this plan

- expected runnable extreme canary cells: **24**
- planned runnable extreme canary cells: **24**
- refused extreme non-canary rows: 42
- standard rows already completed (Commit 48): 1815
- heavy rows already completed (Commit 49): 423
- n_jobs_total across 10 source shards: 2304

## Extreme task universe (pinned policy)

- task **6** / `letter` — 12 planned canary cells
- task **167121** / `Devnagari-Script` — 12 planned canary cells

## Execution policy for the future Commit 51

- require explicit `--include-extreme-tasks` flag: True
- require explicit `--execute-extreme-lane` flag: True
- max_evaluations (recommended): **1** (policy `extreme.stage0_max_evaluations`)
- timeout per cell (recommended): **14400 s** (policy `extreme.timeout_seconds_per_cell`)

Stage-3 top-up cells stack with the signed-off stage-0 cells under the same policy_version. The pinned policy_version sets extreme.stage0_max_evaluations=1 (see signoff caveat 2 / devnagari_extreme_budget_non_equivalence). Commit 51 should reuse this budget so the extreme cells in replica_002 are directly comparable to the extreme cells signed off in replica_001.

## Per-shard planned counts

| shard | n_jobs | runnable_ext | refused_ext | std_done | hvy_done |
|---|---:|---:|---:|---:|---:|
| `shard_00.sqlite` | 219 | 4 | 7 | 177 | 31 |
| `shard_01.sqlite` | 230 | 0 | 0 | 199 | 31 |
| `shard_02.sqlite` | 223 | 0 | 0 | 192 | 31 |
| `shard_03.sqlite` | 235 | 4 | 7 | 202 | 22 |
| `shard_04.sqlite` | 226 | 4 | 7 | 193 | 22 |
| `shard_05.sqlite` | 226 | 4 | 7 | 160 | 55 |
| `shard_06.sqlite` | 228 | 0 | 0 | 162 | 66 |
| `shard_07.sqlite` | 270 | 0 | 0 | 182 | 88 |
| `shard_08.sqlite` | 237 | 4 | 7 | 182 | 44 |
| `shard_09.sqlite` | 210 | 4 | 7 | 166 | 33 |

## Devnagari-Script runtime caveat

task 167121 / Devnagari-Script previously dominated runtime in batch_03 (xgboost / doe_rsm_vrf_true_nbi ~ 11,090 s; catboost / doe_rsm_vrf_true_nbi ~ 10,575 s; catboost / tpe_optuna ~ 7,944 s; catboost / random_search ~ 7,647 s). Under the pinned policy_version's extreme.stage0_max_evaluations=1, those cells run with a tighter configuration budget; doe_rsm_vrf_true_nbi is unchanged because it floors at n_doe=max(2*d, max_evaluations)=8 for d=4. Headline panel-average metrics must footnote this asymmetry (signoff caveat 2).

## What this commit does NOT do

- ✓ no training was run (`no_training_run_by_this_script` = `True`)
- ✓ no execution SQLite files were created (`no_execution_sqlite_created_by_this_script` = `True`)
- ✓ no `runs/` directory artifacts were created (`no_runs_directory_artifacts_created_by_this_script` = `True`)
- ✓ no raw OpenML payloads were staged (`no_raw_openml_payloads_staged_by_this_script` = `True`)
- ✓ standard lane was not rerun (Commit 48 stands) (`no_standard_lane_rerun_by_this_script` = `True`)
- ✓ heavy lane was not rerun (Commit 49 stands) (`no_heavy_lane_rerun_by_this_script` = `True`)
- ✓ extreme lane was not executed (`no_extreme_lane_executed_by_this_script` = `True`)
- ✓ full topup_to_5 was not executed (`no_full_topup_to_5_executed_by_this_script` = `True`)
- ✓ no replica_003 / 004 / 005 was executed (`no_other_replica_executed_by_this_script` = `True`)
- ✓ no new signoff file was created (`no_new_signoff_file_created_by_this_script` = `True`)
- ✓ heavy_task_policy.csv was not regenerated (`no_policy_csv_regenerated_by_this_script` = `True`)
- ✓ runtime_guardrails.yaml was not regenerated (`no_guardrails_yaml_regenerated_by_this_script` = `True`)
- ✓ no committed SQLite shard was modified (`no_committed_shard_modified_by_this_script` = `True`)

## Verdict

**PLAN PASS — operator review required before Commit 51.** All gates verified, the planned extreme-lane scope matches the policy-defined count, and no committed shard was modified. Commit 51 may execute the replica_002 extreme lane under the pinned policy_version, but only with an explicit operator-confirmed scope and the `--include-extreme-tasks` + `--execute-extreme-lane` guards.

## Next recommended step

After Commit 50 is operator-reviewed, Commit 51 should execute the replica_002 extreme lane under the pinned policy_version (extreme.stage0_max_evaluations=1, per-cell timeout=14,400 s). Commit 51 MUST require an explicit --include-extreme-tasks / --execute-extreme-lane flag pair and MUST NOT scale to replica_003-005 until replica_002 standard + heavy + extreme have all been operator-reviewed end-to-end.
