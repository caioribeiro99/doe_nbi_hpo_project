# Doctoral benchmark — pivot summary

The article-track repository now targets a **doctoral-scale benchmark
campaign**, not the intermediate 12-dataset article. The pivot landed
in Commit 24.

## Headline scope

- **82 public benchmark datasets** (binary + multiclass tabular).
- **3 GBDT algorithms**: XGBoost, LightGBM, CatBoost.
- **30 replicas** per (dataset, algorithm) pair as the target for
  robust statistical validity.
- **Method**: DOE + RSM + VRF/FMSE + true N-objective NBI + conditional
  MBPA, exactly as validated by the article-track end-to-end smoke
  (Commit 23).

## Staged execution

| Stage | Replicas | Purpose |
|---|---|---|
| `stage0_replica_001` | 1 | sizing / compatibility pass per dataset; surface loader bugs, hyperparameter range issues, NBI residual issues. |
| `stage1_topup_to_005` | up to 5 | first statistically meaningful estimate per pair. |
| `stage2_topup_to_010` | up to 10 | inter-replica intervals for rank-based comparisons. |
| `stage3_topup_to_030` | up to 30 | final headline tables for the doctoral / journal artifact. |

Each stage tops up the previous one rather than re-running. The
`replica` field in the SQLite job matrix is the deduplication key.

## Status of the 12-dataset panel

The 12 datasets used in earlier commits remain the **smoke / profiling
subset** of the doctoral panel:

| ID | Source | Status |
|---|---|---|
| `magic`, `breast_cancer`, `pima_diabetes`, `spambase`, `adult`, `bank_marketing`, `credit_card_default`, `german_credit`, `wine_quality`, `dry_bean`, `mushroom`, `phishing` | UCI / OpenML / sklearn | seeded into `benchmarks/doctoral_82/datasets.csv` with `include=true` |

The remaining 70 entries are placeholders (`loader_status=pending`,
`include=false`) until the selection policy in
`benchmarks/doctoral_82/selection_policy.md` is executed.

## Where things live

- **Registry**: `benchmarks/doctoral_82/datasets.csv` (committed CSV);
  `benchmarks/doctoral_82/dataset_schema.json` (column contract).
- **Selection policy**: `benchmarks/doctoral_82/selection_policy.md`.
- **Importer**: `scripts/import_doctoral_benchmark_datasets.py`.
- **Job matrix schema**: `jobs/doctoral_82/schema.sql`.
- **Generator (Commit 25)**: `scripts/generate_doctoral_job_shards.py`
  (planned).

## Capacity assumptions

The doctoral campaign runs locally on the dedicated MacBook Pro by
default, with the Caio personal Mac as an opportunistic supplement.
The cost estimator in `src/doe_xgb/cost_estimator.py` carries:

- `LocalProfile` (single machine).
- `MultiMachineProfile` (Commit 24): aggregates a dedicated-Mac and an
  optional Caio-Mac profile into a single daily-CPU-hours figure.
- Dedicated-Mac efficiency scenarios: **0.75** (conservative),
  **0.85** (realistic with cooling), **0.90** (optimistic). The
  previous 0.70 is reserved for the Caio Mac, *not* the dedicated one.

## What this commit does NOT do

- Does **not** finalize the 82-dataset list.
- Does **not** generate any job SQLite shards (Commit 25).
- Does **not** run any benchmark.
- Does **not** kill the article-track smoke / profiler scripts; they
  remain useful for the v1 subset and as integration tests.

## Provisional cost estimate (anchored on the 12-dataset profile)

The 82-dataset projection is **provisional** until the actual
82-dataset profiler runs. Using the mean per-pair time from
`experiments/_runtime_profile/v1_full_dataset_5fold_profile.json` and
the new dedicated-Mac efficiency assumptions:

| Scope | Total CPU-h (4× infl.) | Dedicated Mac eff 0.75 | eff 0.85 | eff 0.90 |
|---|---:|---:|---:|---:|
| 82 × 3 × 1 | ~142 | 0.79 d | 0.69 d | 0.66 d |
| 82 × 3 × 5 | ~708 | 3.93 d | 3.47 d | 3.28 d |
| 82 × 3 × 10 | ~1,416 | 7.87 d | 6.94 d | 6.56 d |
| 82 × 3 × 30 | ~4,248 | 23.6 d | 20.8 d | 19.7 d |

Sources of error: (i) the doctoral 70 datasets may be heavier or
lighter than the v1 panel mean; (ii) the inflation multiplier scales
with the actual DoE/NBI configurations chosen; (iii) thermal /
sustained-load efficiency on a dedicated Mac is empirical. Re-anchor
these numbers after Commit 25 / 26 once the actual 82-dataset list
exists and a real 82-dataset profiler runs.
