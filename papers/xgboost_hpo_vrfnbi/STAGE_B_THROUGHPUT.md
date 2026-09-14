# Stage B: parallel-throughput study

**Engineering only.** No arm was run, no method was compared, and no configuration was selected
because an optimizer liked it. The benchmark set is pre-specified from the design's own structure.

Machine: Apple M4 Max, **14 physical cores**, 36 GB, macOS 26.6.2. Reproduce with
`scripts/stage_b_throughput.py` and `scripts/stage_b_calibration.py`; machine-readable records are
`audits/stage_b_throughput.json` and `audits/stage_b_calibration.json`.

---

## 1. The benchmark workload

Seventeen pre-specified configurations: the box centre; a cheap corner and an expensive corner
constructed from the coded extremes; six further corners taken from the design's own half fraction
**by index**; and eight axial runs. Nothing was chosen by outcome.

## 2. Layout sweep

Layouts are (process workers) × (XGBoost threads per fit). Any combination whose product exceeds 14
was skipped, because nested oversubscription measures the scheduler rather than the workload: `4×4`,
`6×4`, `8×2`, `8×4`, `10×2`, `10×4`, `12×2`, `12×4`, `14×2`, `14×4`.

Dataset MAGIC, 68 evaluations per layout:

| layout | evals/s | s/eval | CPU % | peak RSS GB | swap MB |
|---|---:|---:|---:|---:|---:|
| 1w × 1t | 1.959 | 0.510 | 9.3 | 1.00 | 0 |
| 1w × 2t | 1.989 | 0.503 | 10.0 | 1.01 | 0 |
| 1w × 4t | 1.987 | 0.503 | 10.9 | 0.52 | 0 |
| 2w × 1t | 3.699 | 0.270 | 16.4 | 1.61 | 0 |
| 2w × 2t | 2.839 | 0.352 | 22.6 | 1.58 | 0 |
| 2w × 4t | 2.161 | 0.463 | 29.5 | 1.40 | 0 |
| 4w × 1t | 6.785 | 0.147 | 28.7 | 2.03 | 0 |
| 4w × 2t | 3.998 | 0.250 | 39.8 | 2.03 | 0 |
| 6w × 1t | 9.493 | 0.105 | 41.5 | 2.46 | 0 |
| 6w × 2t | 4.753 | 0.210 | 59.5 | 2.40 | 0 |
| 8w × 1t | 11.656 | 0.086 | 59.6 | 2.83 | 0 |
| 10w × 1t | 12.196 | 0.082 | 79.0 | 3.29 | 0 |
| 12w × 1t | 12.865 | 0.078 | 73.9 | 3.69 | 0 |
| **14w × 1t** | **13.615** | **0.073** | 80.3 | 4.13 | 0 |

Two results, both clear:

**Process parallelism wins and thread parallelism does not.** One worker gains nothing from extra
XGBoost threads on this workload: 1×1, 1×2 and 1×4 are within 1.5% of each other. Worse, once there
is more than one worker, adding threads actively *hurts* — 2×2 is 23% slower than 2×1 and 2×4 is 42%
slower, at higher CPU usage. The threads contend without doing useful work.

**Throughput is still rising at the core count.** 14w × 1t is the best measured layout and nothing
else is within 5% of it, so the selection rule (best throughput, then simplest layout within 5%)
returns it uniquely. Speedup over single-worker: **6.95×**.

## 3. Determinism across layouts

Every layout produced the identical fingerprint `3fc2870df94924eb`, a hash over leaf count, accuracy,
area under the curve and log loss for all 17 configurations. **How the work is distributed does not
change any result.** Measured time is excluded from the fingerprint, since it must vary.

## 4. Sustained interval

Four segments at 14w × 1t over 3.5 minutes:

| segment | evals/s |
|---|---:|
| 0 | 19.256 |
| 1 | 19.866 |
| 2 | 19.842 |
| 3 | 19.937 |

Degradation **−3.5%**, that is throughput *improved* by 3.5% as the pool warmed. No thermal
throttling. CPU 92% mean, peak resident memory 14.36 GB against 36 GB available, **swap delta 0 MB**.

The segments are faster than the sweep because the sweep charges pool startup to each layout while
the segments amortize it. The sweep number is the conservative one and is what the comparison used;
the calibration below is what the projection uses.

## 5. Calibration on the campaign's real workload

Comparing layouts needs a fixed workload; projecting the campaign needs a **representative** one. The
17-configuration benchmark is neither guaranteed to be representative nor intended to be. Each
replication's real workload is known exactly: the 88-run design plus the 78-run external validation
set. Both were run, per dataset, at the chosen layout.

| dataset | evaluations | wall s | s/eval | evals/s | peak RSS GB | swap MB |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | 166 | 24.8 | 0.1497 | 6.68 | 5.80 | 0 |
| Spambase | 166 | 11.7 | 0.0702 | 14.25 | 5.01 | 0 |
| Adult | 166 | 41.5 | 0.2503 | 4.00 | 4.55 | 0 |
| Bank Marketing | 166 | 27.2 | 0.1636 | 6.11 | 4.11 | 0 |

**Panel mean 0.1585 s per real evaluation, 6.31 evaluations per second.** Against Stage A's serial
measurement of 1.40 s, that is an **8.8× speedup** on the workload that actually matters.

The per-dataset spread is a factor of 3.6, so a projection using a single mean is an approximation.
It is the right one here because the campaign runs all four datasets in fixed proportion.

## 6. Campaign projection

| | two objectives | three objectives |
|---|---:|---:|
| `B_total_solution`: HISTORICAL-WS / WS-S / NBI-S / NBI-R | 108 / 186 / 186 / 386 | 108 / 186 / 186 / 486 |
| comparator budget | 386 | 486 |
| evaluations per replication per dataset | 2,376 | 2,976 |
| campaign evaluations (R = 30, 4 datasets) | 285,120 | 357,120 |
| **projected wall clock** | **12.6 h, 0.52 days** | **15.7 h, 0.66 days** |
| with the unmatched NSGA-II run | 0.55 days | 0.69 days |

Against a 5-day ceiling, **both formulations fit with an order of magnitude to spare**. The serial
projection that motivated choosing two objectives — 4.6 days against 5.8 — was an artefact of
measuring single-process throughput.

## 7. The frozen execution layout

| Setting | Value |
|---|---|
| Process workers | **14** |
| XGBoost threads per fit | **1** |
| Thread environment | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to 1 in each worker, before the numerical libraries are first used |
| Start method | `spawn`, so workers do not inherit parent state |
| Chunk size | 1, so a slow configuration does not stall a worker's queue |
| Dataset loading | once per worker process, not once per evaluation |

Frozen before any confirmatory execution. Selection rule applied: highest measured throughput, then
the simplest layout within 5%; only one layout was in the band.

## 8. Objective-count decision

`protocol/OBJECTIVE_COUNT_DECISION.md` established **before** any throughput number was read that the
third objective is **complementary and arguably central**, not redundant: the two quality factors
rank configurations in opposing directions on two of four datasets (Spearman −0.308 and −0.533), the
non-dominated set grows by 1.3× to 10.8× when they are kept separate, and the aggregation weighting
that the PCA/Varimax audit showed to matter disappears entirely.

The same document fixed "comfortably at or under five days" as **at or under four days** before the
measurement. The measurement is **0.69 days**.

Branch 2 of the decision rule therefore applies: the third objective is **not silently added**. A
short adversarial protocol review runs, the amendment is documented, and `xgboost-hpo-protocol-v3`
is cut before any confirmatory method comparison. Outcome recorded in
`protocol/Q3_AMENDMENT_REVIEW.md`.
