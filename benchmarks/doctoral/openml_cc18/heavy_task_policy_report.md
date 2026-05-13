# OpenML-CC18 heavy-task policy report

- generated_at: `2026-05-13T21:25:31Z`
- tasks_csv: `benchmarks/doctoral/openml_cc18/tasks.csv`
- input summaries:
  - `experiments/_stage_runs/batch_02_cc18_small_12_tasks_latest_summary.json` (read)
  - `experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json` (read)

## Lane counts (72 CC18 tasks)

| lane | count |
|---|---:|
| `standard` | 57 |
| `heavy` | 13 |
| `extreme` | 2 |

## Lane: extreme (2)

| task_id | dataset | rows | features | classes | categorical | reason |
|---:|---|---:|---:|---:|---:|---|
| 6 | `letter` | 20000 | 16 | 26 | 0 | n_classes>=25 AND n_rows>=20000 |
| 167121 | `Devnagari-Script` | 92000 | 1024 | 46 | 0 | observed_max_runtime_s=11091>=3600; n_rows>=75000 AND n_features>=500; n_classes>=25 AND n_rows>=20000 |

## Lane: heavy (13)

| task_id | dataset | rows | features | classes | categorical | reason |
|---:|---|---:|---:|---:|---:|---|
| 32 | `pendigits` | 10992 | 16 | 10 | 0 | n_classes>=10 AND n_rows>=10000 |
| 219 | `electricity` | 45312 | 8 | 2 | 1 | n_rows>=40000 |
| 3573 | `mnist_784` | 70000 | 784 | 10 | 0 | observed_max_runtime_s=1507>=900; n_rows>=40000; n_features>=750; n_classes>=10 AND n_rows>=10000 |
| 7592 | `adult` | 48842 | 14 | 2 | 8 | n_rows>=40000 |
| 9910 | `Bioresponse` | 3751 | 1776 | 2 | 0 | n_features>=750 |
| 9981 | `cnae-9` | 1080 | 856 | 9 | 0 | n_features>=750 |
| 14965 | `bank-marketing` | 45211 | 16 | 2 | 9 | n_rows>=40000 |
| 146195 | `connect-4` | 67557 | 42 | 3 | 42 | n_rows>=40000 |
| 146825 | `Fashion-MNIST` | 70000 | 784 | 10 | 0 | n_rows>=40000; n_features>=750; n_classes>=10 AND n_rows>=10000 |
| 167119 | `jungle_chess_2pcs_raw_endgame_complete` | 44819 | 6 | 3 | 0 | n_rows>=40000 |
| 167120 | `numerai28.6` | 96320 | 21 | 2 | 0 | n_rows>=40000 |
| 167124 | `CIFAR_10` | 60000 | 3072 | 10 | 0 | n_rows>=40000; n_features>=750; n_classes>=10 AND n_rows>=10000 |
| 167125 | `Internet-Advertisements` | 3279 | 1558 | 2 | 1555 | n_features>=750; categorical_feature_count>=500 |

## Lane: standard (57)

| task_id | dataset | rows | features | classes | categorical | reason |
|---:|---|---:|---:|---:|---:|---|
| 3 | `kr-vs-kp` | 3196 | 36 | 2 | 36 | metadata within standard envelope |
| 11 | `balance-scale` | 625 | 4 | 3 | 0 | metadata within standard envelope |
| 12 | `mfeat-factors` | 2000 | 216 | 10 | 0 | metadata within standard envelope |
| 14 | `mfeat-fourier` | 2000 | 76 | 10 | 0 | metadata within standard envelope |
| 15 | `breast-w` | 699 | 9 | 2 | 0 | metadata within standard envelope |
| 16 | `mfeat-karhunen` | 2000 | 64 | 10 | 0 | metadata within standard envelope |
| 18 | `mfeat-morphological` | 2000 | 6 | 10 | 0 | metadata within standard envelope |
| 22 | `mfeat-zernike` | 2000 | 47 | 10 | 0 | metadata within standard envelope |
| 23 | `cmc` | 1473 | 9 | 3 | 7 | metadata within standard envelope |
| 28 | `optdigits` | 5620 | 64 | 10 | 0 | metadata within standard envelope |
| 29 | `credit-approval` | 690 | 15 | 2 | 9 | metadata within standard envelope |
| 31 | `credit-g` | 1000 | 20 | 2 | 13 | metadata within standard envelope |
| 37 | `diabetes` | 768 | 8 | 2 | 0 | metadata within standard envelope |
| 43 | `spambase` | 4601 | 57 | 2 | 0 | metadata within standard envelope |
| 45 | `splice` | 3190 | 60 | 3 | 61 | metadata within standard envelope |
| 49 | `tic-tac-toe` | 958 | 9 | 2 | 9 | metadata within standard envelope |
| 53 | `vehicle` | 846 | 18 | 4 | 0 | metadata within standard envelope |
| 2074 | `satimage` | 6430 | 36 | 6 | 0 | metadata within standard envelope |
| 2079 | `eucalyptus` | 736 | 19 | 5 | 5 | metadata within standard envelope |
| 3021 | `sick` | 3772 | 29 | 2 | 22 | metadata within standard envelope |
| 3022 | `vowel` | 990 | 12 | 11 | 2 | metadata within standard envelope |
| 3481 | `isolet` | 7797 | 617 | 26 | 0 | metadata within standard envelope |
| 3549 | `analcatdata_authorship` | 841 | 70 | 4 | 0 | metadata within standard envelope |
| 3560 | `analcatdata_dmft` | 797 | 4 | 6 | 4 | metadata within standard envelope |
| 3902 | `pc4` | 1458 | 37 | 2 | 0 | metadata within standard envelope |
| 3903 | `pc3` | 1563 | 37 | 2 | 0 | metadata within standard envelope |
| 3904 | `jm1` | 10885 | 21 | 2 | 0 | metadata within standard envelope |
| 3913 | `kc2` | 522 | 21 | 2 | 0 | metadata within standard envelope |
| 3917 | `kc1` | 2109 | 21 | 2 | 0 | metadata within standard envelope |
| 3918 | `pc1` | 1109 | 21 | 2 | 0 | metadata within standard envelope |
| 9946 | `wdbc` | 569 | 30 | 2 | 0 | metadata within standard envelope |
| 9952 | `phoneme` | 5404 | 5 | 2 | 0 | metadata within standard envelope |
| 9957 | `qsar-biodeg` | 1055 | 41 | 2 | 0 | metadata within standard envelope |
| 9960 | `wall-robot-navigation` | 5456 | 24 | 4 | 0 | metadata within standard envelope |
| 9964 | `semeion` | 1593 | 256 | 10 | 0 | metadata within standard envelope |
| 9971 | `ilpd` | 583 | 10 | 2 | 1 | metadata within standard envelope |
| 9976 | `madelon` | 2600 | 500 | 2 | 0 | metadata within standard envelope |
| 9977 | `nomao` | 34465 | 118 | 2 | 29 | metadata within standard envelope |
| 9978 | `ozone-level-8hr` | 2534 | 72 | 2 | 0 | metadata within standard envelope |
| 9985 | `first-order-theorem-proving` | 6118 | 51 | 6 | 0 | metadata within standard envelope |
| 10093 | `banknote-authentication` | 1372 | 4 | 2 | 0 | metadata within standard envelope |
| 10101 | `blood-transfusion-service-center` | 748 | 4 | 2 | 0 | metadata within standard envelope |
| 14952 | `PhishingWebsites` | 11055 | 30 | 2 | 30 | metadata within standard envelope |
| 14954 | `cylinder-bands` | 540 | 39 | 2 | 21 | metadata within standard envelope |
| 14969 | `GesturePhaseSegmentationProcessed` | 9873 | 32 | 5 | 0 | metadata within standard envelope |
| 14970 | `har` | 10299 | 561 | 6 | 0 | metadata within standard envelope |
| 125920 | `dresses-sales` | 500 | 12 | 2 | 11 | metadata within standard envelope |
| 125922 | `texture` | 5500 | 40 | 11 | 0 | metadata within standard envelope |
| 146800 | `MiceProtein` | 1080 | 81 | 8 | 4 | metadata within standard envelope |
| 146817 | `steel-plates-fault` | 1941 | 27 | 7 | 0 | metadata within standard envelope |
| 146819 | `climate-model-simulation-crashes` | 540 | 20 | 2 | 0 | metadata within standard envelope |
| 146820 | `wilt` | 4839 | 5 | 2 | 0 | metadata within standard envelope |
| 146821 | `car` | 1728 | 6 | 4 | 6 | metadata within standard envelope |
| 146822 | `segment` | 2310 | 19 | 7 | 0 | metadata within standard envelope |
| 146824 | `mfeat-pixel` | 2000 | 240 | 10 | 0 | metadata within standard envelope |
| 167140 | `dna` | 3186 | 180 | 3 | 180 | metadata within standard envelope |
| 167141 | `churn` | 5000 | 20 | 2 | 4 | metadata within standard envelope |

## Classification rules

- `extreme`:
  - observed any cell runtime >= 3600 s, OR
  - n_rows >= 75000 AND n_features >= 500, OR
  - n_classes >= 25 AND n_rows >= 20000
- `heavy` (and not extreme):
  - observed any cell runtime >= 900 s, OR
  - n_rows >= 40000, OR
  - n_features >= 750, OR
  - categorical_feature_count >= 500, OR
  - n_classes >= 10 AND n_rows >= 10000
- `standard` otherwise.

Lane defaults (timeouts, max_evaluations, include-by-default) live in `runtime_guardrails.yaml`. The `src/doe_xgb/runtime_guardrails.py` helper exposes `get_task_lane`, `get_timeout_seconds`, `get_effective_max_evaluations`, and `should_defer_task`.
