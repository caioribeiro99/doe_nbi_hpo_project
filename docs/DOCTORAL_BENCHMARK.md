# Doctoral benchmark — pivot summary

The article-track repository now targets a **doctoral-scale benchmark
campaign**. As of **Commit 25**, the canonical primary benchmark is the
**OpenML-CC18 curated classification suite** (`suite_id = 99`, 72
standardized tasks). The earlier "82 datasets" framing (Commit 24)
was an arbitrary list and has been retired; see
`benchmarks/doctoral_82/` for the deprecated scaffolding kept only as
historical record.

## Headline scope

- **OpenML-CC18 suite** — 72 curated classification tasks (binary +
  multiclass, tabular only). Identity is by **OpenML task id**, not
  dataset id (a few CC18 tasks share the same underlying OpenML
  dataset with different target columns or train/test splits).
- **3 GBDT algorithms**: XGBoost, LightGBM, CatBoost.
- **30 replicas** per (task, algorithm) pair as the target for
  robust statistical validity.
- **Method**: DOE + RSM + VRF/FMSE + true N-objective NBI + conditional
  MBPA, exactly as validated by the article-track end-to-end smoke
  (Commit 23).

Total job count: `72 × 3 × 1 × 30 = 6,480` (one method = `doe_nbi`).

## Staged execution

| Stage | Replicas in stage | Jobs added | Cumulative |
|---|---:|---:|---:|
| `stage0_replica_001` | 1 | 216 | 216 |
| `stage1_topup_to_005` | 4 | 864 | 1,080 |
| `stage2_topup_to_010` | 5 | 1,080 | 2,160 |
| `stage3_topup_to_030` | 20 | 4,320 | 6,480 |

Each stage tops up the previous one rather than re-running. The
`replica` field in the SQLite job matrix is the deduplication key.

## Status of the 12-dataset internal panel

The 12 datasets shipped in earlier commits (`magic`, `breast_cancer`,
`pima_diabetes`, `spambase`, `adult`, `bank_marketing`,
`credit_card_default`, `german_credit`, `wine_quality`, `dry_bean`,
`mushroom`, `phishing`) are **not** part of the CC18 doctoral
benchmark. They are demoted to a **smoke / profiling / development
fixture** and live at `benchmarks/doctoral/internal_smoke_panel/`.
They keep the existing v1 smoke and profiler scripts working but are
no longer in the canonical benchmark count.

## Where things live

- **Primary benchmark registry**:
  - `benchmarks/doctoral/openml_cc18/tasks.csv` (one row per task).
  - `benchmarks/doctoral/openml_cc18/datasets.csv` (one row per
    unique OpenML dataset; tracks how many CC18 tasks each
    dataset backs).
  - `benchmarks/doctoral/openml_cc18/openml_cc18_metadata.json` (suite
    snapshot: suite id, name, importer timestamp, openml package
    version, full task-id list).
  - `benchmarks/doctoral/openml_cc18/coverage_report.md` (binary /
    multiclass / categorical / imbalance / size-bucket summary).
- **Importer**: `scripts/import_openml_cc18.py` (network-bound;
  `--dry-run` and `--validate-only` exercise the code paths offline).
- **Job-matrix schema**: `jobs/doctoral/openml_cc18/schema.sql`
  (`cc18_jobs` table; `(openml_task_id, algorithm, method, replica)`
  is the unique key).
- **Internal smoke panel**:
  `benchmarks/doctoral/internal_smoke_panel/datasets.csv` (the 12 v1
  datasets, used for development / smoke / runtime profiling).

## Deprecated locations (kept for historical context)

- `benchmarks/doctoral_82/` — the pre-pivot 82-arbitrary-list
  scaffolding (registry CSV, schema, selection policy). The README
  there carries the deprecation banner.
- `jobs/doctoral_82/schema.sql` — the pre-pivot SQLite schema with the
  `doctoral_jobs` table. Renamed intentionally to `cc18_jobs` in the
  new schema so a stray import cannot cross-pollinate the two.
- `doctoral_82_datasets_3_algorithms_*` cost-estimator presets — kept
  as deprecated aliases that emit a `DeprecationWarning` when
  resolved. Use the `openml_cc18_72_tasks_3_algorithms_*` presets
  instead.

## Capacity assumptions

The doctoral campaign runs locally on the dedicated MacBook Pro by
default, with the Caio personal Mac as an opportunistic supplement.
The cost estimator in `src/doe_xgb/cost_estimator.py` carries:

- `LocalProfile` (single machine).
- `MultiMachineProfile`: aggregates a dedicated-Mac and an optional
  Caio-Mac profile into a single daily-CPU-hours figure.
- Dedicated-Mac efficiency scenarios: **0.75** (conservative),
  **0.85** (realistic with cooling, default), **0.90** (optimistic).
  The 0.70 figure is reserved for the Caio Mac, *not* the dedicated
  one.

## Provisional cost estimate (anchored on the v1 mean per-pair time)

The CC18 projection uses the same v1-mean per-pair 5-fold runtime
and 4× inflation multiplier as the deprecated 82-anchored table; the
arithmetic just scales by 72/82. These remain **provisional** until
a real CC18 profiler runs.

| Scope | Total CPU-h (4× infl.) | Dedicated 0.75 | 0.85 | 0.90 |
|---|---:|---:|---:|---:|
| 72 × 3 × 1 | ~124 | 0.69 d | 0.61 d | 0.58 d |
| 72 × 3 × 5 | ~622 | 3.46 d | 3.05 d | 2.88 d |
| 72 × 3 × 10 | ~1,243 | 6.91 d | 6.09 d | 5.76 d |
| 72 × 3 × 30 | ~3,730 | 20.7 d | 18.3 d | 17.3 d |

Sources of error: (i) CC18 task heaviness vs the v1 panel mean is
unknown — large CC18 tasks (`adult`, `bank-marketing`, etc. recur
inside CC18) skew the mean upward; (ii) the inflation multiplier
scales with the actual DoE / NBI configuration chosen; (iii) thermal
/ sustained-load efficiency on a dedicated Mac is empirical.
Re-anchor these numbers after the first real CC18 profiler pass.

## Comparative protocol (Commit 26)

The doctoral campaign is not a single-method run; it is a comparative
study against the dominant HPO families. The authoritative narrative
specification is `docs/COMPARATIVE_PROTOCOL.md`; the canonical
machine-readable list is
`benchmarks/doctoral/openml_cc18/method_matrix.csv`. Every method
participating in the SQLite job matrix MUST come from a row of that
CSV — the shard generator (planned next commit) reads the method
list from the CSV and never hardcodes method names. Method families
covered: classical (random search, Optuna TPE), SMAC family (SMAC3),
multi-fidelity (ASHA / BOHB / DEHB), evolutionary multi-objective
(NSGA-II), other multi-objective (MOTPE, ParEGO on subset),
proposed (DOE+RSM+VRF+true NBI+conditional MBPA), and ablations
(no-MBPA, legacy weighted-sum). AutoML systems (Auto-sklearn, FLAML,
AutoGluon) are cited as context, not benchmarked, with FLAML
optionally promoted to a baseline once the open items at the bottom
of `docs/COMPARATIVE_PROTOCOL.md` are resolved.

## Next operational step

The next operational step is **not** "run the 12-dataset campaign";
it is:

1. Resolve the open items in `docs/COMPARATIVE_PROTOCOL.md` (FLAML
   inclusion, ASHA vs Hyperband, ParEGO subset, TODO references in
   `article/references.bib`).
2. Freeze `benchmarks/doctoral/openml_cc18/method_matrix.csv`.
3. Implement `scripts/generate_cc18_job_shards.py` (Commit 27 plan)
   to materialize SQLite shards under
   `jobs/doctoral/openml_cc18/shards/`, driven by the frozen CSV.
4. Profile a single CC18 task at `R=1` per (task, algorithm, method)
   to re-anchor the wall-clock estimates above.

The method list must be frozen before any SQLite shards are
committed.

## What this commit does NOT do

- Does **not** download CC18 dataset payloads (only OpenML metadata).
- Does **not** generate any job SQLite shards.
- Does **not** run any benchmark.
- Does **not** kill the article-track smoke / profiler scripts; they
  remain useful for the 12-dataset smoke panel and as integration
  tests.
