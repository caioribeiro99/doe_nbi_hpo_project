# OpenML-CC18 SQLite shard summary

- generated_at: `2026-05-02T22:07:11Z`
- panel_version: `cc18_v1`
- shards per stage: 10
- dry_run: False
- total rows: **79920**

## Rows by stage

| stage | rows |
|---|---:|
| `stage0_replica_001` | 2304 (cum 2304) |
| `stage1_topup_to_005` | 9216 (cum 11520) |
| `stage2_topup_to_010` | 13680 (cum 25200) |
| `stage3_topup_to_030` | 54720 (cum 79920) |

## Rows by method

| method | rows |
|---|---:|
| `asha` | 6480 |
| `bohb` | 6480 |
| `default_gbdt` | 6480 |
| `dehb` | 6480 |
| `doe_rsm_vrf_true_nbi` | 6480 |
| `doe_rsm_vrf_true_nbi_no_mbpa` | 5400 |
| `legacy_weighted_sum_scalarization` | 5400 |
| `motpe` | 6480 |
| `nsga2` | 6480 |
| `parego` | 4320 |
| `random_search` | 6480 |
| `smac3` | 6480 |
| `tpe_optuna` | 6480 |

## Rows by algorithm

| algorithm | rows |
|---|---:|
| `xgboost` | 26640 |
| `lightgbm` | 26640 |
| `catboost` | 26640 |

## Tier counters

- ParEGO subset rows: **4320**
- Ablation rows: **10800**
- Literature-only rows: **0** (must be 0)
- Stage-3 manual-signoff rows: **46080**

## Rows by (stage, shard)

| stage | shard | rows |
|---|---:|---:|
| `stage0_replica_001` | 00 | 219 |
| `stage0_replica_001` | 01 | 230 |
| `stage0_replica_001` | 02 | 223 |
| `stage0_replica_001` | 03 | 235 |
| `stage0_replica_001` | 04 | 226 |
| `stage0_replica_001` | 05 | 226 |
| `stage0_replica_001` | 06 | 228 |
| `stage0_replica_001` | 07 | 270 |
| `stage0_replica_001` | 08 | 237 |
| `stage0_replica_001` | 09 | 210 |
| `stage1_topup_to_005` | 00 | 876 |
| `stage1_topup_to_005` | 01 | 920 |
| `stage1_topup_to_005` | 02 | 892 |
| `stage1_topup_to_005` | 03 | 940 |
| `stage1_topup_to_005` | 04 | 904 |
| `stage1_topup_to_005` | 05 | 904 |
| `stage1_topup_to_005` | 06 | 912 |
| `stage1_topup_to_005` | 07 | 1080 |
| `stage1_topup_to_005` | 08 | 948 |
| `stage1_topup_to_005` | 09 | 840 |
| `stage2_topup_to_010` | 00 | 1305 |
| `stage2_topup_to_010` | 01 | 1370 |
| `stage2_topup_to_010` | 02 | 1325 |
| `stage2_topup_to_010` | 03 | 1395 |
| `stage2_topup_to_010` | 04 | 1340 |
| `stage2_topup_to_010` | 05 | 1340 |
| `stage2_topup_to_010` | 06 | 1350 |
| `stage2_topup_to_010` | 07 | 1600 |
| `stage2_topup_to_010` | 08 | 1405 |
| `stage2_topup_to_010` | 09 | 1250 |
| `stage3_topup_to_030` | 00 | 5220 |
| `stage3_topup_to_030` | 01 | 5480 |
| `stage3_topup_to_030` | 02 | 5300 |
| `stage3_topup_to_030` | 03 | 5580 |
| `stage3_topup_to_030` | 04 | 5360 |
| `stage3_topup_to_030` | 05 | 5360 |
| `stage3_topup_to_030` | 06 | 5400 |
| `stage3_topup_to_030` | 07 | 6400 |
| `stage3_topup_to_030` | 08 | 5620 |
| `stage3_topup_to_030` | 09 | 5000 |
