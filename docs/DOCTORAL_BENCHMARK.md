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

## SQLite shards landed in Commit 28

The deterministic SQLite job queues are materialized under
`jobs/doctoral/openml_cc18/shards/<stage>/shard_NN.sqlite`:
40 files (4 stages × 10 shards), 79,920 rows total
(2,304 / 9,216 / 13,680 / 54,720 by stage). The
`scripts/generate_cc18_job_shards.py` generator is driven by:

- `benchmarks/doctoral/openml_cc18/method_matrix.csv`
  (frozen 16-row method matrix);
- `benchmarks/doctoral/openml_cc18/execution_policy.csv`
  (per-method per-stage gating, manual sign-off flag);
- `benchmarks/doctoral/openml_cc18/parego_subset.csv`
  (48 ParEGO subset task IDs);
- `benchmarks/doctoral/openml_cc18/tasks.csv` (72 CC18 tasks);
- `jobs/doctoral/openml_cc18/schema.sql` (`cc18_jobs` schema).

**No method names, scope rules, or stage-gating logic are hardcoded
in the shard generator.** The generated shard counts match the
projection in `benchmarks/doctoral/openml_cc18/job_count_projection.md`:
2,304 at stage 0; 11,520 through stage 1; 25,200 through stage 2;
79,920 through stage 3. Stage 3 jobs of every tier-1+ method carry
the `requires_manual_signoff_before_stage3` note in
`cc18_jobs.notes`, so the runner can ship a stage-2 (10-replica)
snapshot if the stage-3 cost is judged unacceptable.

## Method capability audit + runner skeleton (Commit 29)

Adapters for every non-literature method live under
`src/doe_xgb/methods/`. The capability audit
(`scripts/audit_method_capabilities.py`) imports each adapter,
records its declared `run_status` (`stub_only` / `dispatch_only` /
`smoke_ready` / `full_ready`), and inspects whether the required
package is importable. The audit report lives at
`experiments/_capability_audit/cc18_capability_report.{json,md}`
and is the gating artifact between protocol freeze and the actual
benchmark run.

The local runner (`scripts/cc18_runner.py`) is a skeleton: it
opens a shard, selects pending jobs, resolves the adapter, and
logs a dispatch decision **without training**. `--dry-run` opens
the database read-only; the default `--no-train` mode briefly
claims a job and immediately releases it, so the row's logical
state is unchanged. Tests always copy a shard to a tmp directory
before exercising the runner; the committed shard files are not
mutated by tests.

Stage-3 jobs of every tier-1+ method are still locked behind a
manual sign-off file at
`jobs/doctoral/openml_cc18/stage3_signoff.json`; this commit does
**not** create that file. Without it, the runner refuses to claim
stage-3 jobs and reports them as `refused_stage3_signoff_missing`.

## Executable canary adapters (Commit 30)

The four canary adapters are now `smoke_ready` and execute
end-to-end on a synthetic binary task:

- `default_gbdt` — one fit per CV split with library defaults;
- `random_search` — uniform draws over the canary search space at
  `--max-evaluations` budget;
- `tpe_optuna` — Optuna `TPESampler` at the same budget; raises a
  clear `ImportError` when `optuna` is not installed;
- `doe_rsm_vrf_true_nbi` — Latin-hypercube DOE → quadratic RSM →
  true 2-objective NBI (mean accuracy max, mean fold runtime min) →
  distance-to-utopia selection → conditional MBPA diagnostics →
  retrain at the chosen hyperparameters.

The runner default is still `--no-train`; training is **only**
allowed when `--canary-only` and `--train` are set together. Even
under `--canary-only --train`, non-canary methods are refused
(`refused_not_in_canary_set`), and stage-3 jobs carrying the
`requires_manual_signoff_before_stage3` note are still refused
without the sign-off file.

Running the synthetic canary on a temp shard:

```bash
# On the dedicated Mac, after `pip install -e .[gbdt,hpo_baselines,doctoral,dev]`:
cp jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite /tmp/canary.sqlite
sqlite3 /tmp/canary.sqlite "DELETE FROM cc18_jobs WHERE method NOT IN \
  ('default_gbdt','random_search','tpe_optuna','doe_rsm_vrf_true_nbi')"
python scripts/cc18_runner.py \
    --shard /tmp/canary.sqlite --max-jobs 12 \
    --canary-only --train --synthetic-task \
    --max-evaluations 5 --n-folds 2 \
    --output-root experiments/_canary_runs

# Refresh the capability audit so the report reflects the dedicated Mac.
python scripts/audit_method_capabilities.py
```

**Stage 0 must not start until the canary above passes on the
dedicated Mac**, with all four adapters marked `success` and the
audit reporting zero missing packages for the canary set.

The other 9 adapters (`smac3`, `asha`, `bohb`, `dehb`, `nsga2`,
`motpe`, `parego`, `doe_rsm_vrf_true_nbi_no_mbpa`,
`legacy_weighted_sum_scalarization`) remain `stub_only` /
`dispatch_only` and are wired in later commits.

## Reduced-execution batches (Commit 31)

Before any full-stage run, the dedicated Mac walks through a
deterministic ladder of pre-stage-0 batches. The batch manifests
live under `benchmarks/doctoral/openml_cc18/batches/` and are
generated reproducibly from `tasks.csv` by
`scripts/create_cc18_batches.py`:

| step | batch | scope |
|---|---|---|
| A | `batch_00_synthetic_canary` | 4 canary methods on synthetic binary; no OpenML data |
| B | `batch_01_cc18_tiny_3_tasks` | 3 real CC18 tasks (small binary numeric, small categorical, small multiclass) |
| C | `batch_02_cc18_small_12_tasks` | 12 real CC18 tasks; stratified pilot |
| D | `batch_03_cc18_representative_18_tasks` | 18 real CC18 tasks; broader coverage |
| E | `batch_04_stage0_shard00_only` | one existing stage-0 shard from Commit 28 |
| F | full stage 0 | 2,304 jobs across the 10 stage-0 shards |
| G | top-up to stages 1 / 2 / 3 | gated by manual sign-off (`execution_tiers.md`) |

A→D are pre-stage-0 pilots that validate the adapters, the OpenML
loader, and the runner. E is an operational dry run on the smallest
real worker shard. F and G follow only after each prior step lands
a green sign-off artifact. Each batch CSV ships a `.meta.json`
sidecar with the selection rule and the deterministic seed so the
chosen task IDs are reproducible from `tasks.csv`. To filter an
existing SQLite shard down to a batch's task IDs without ever
mutating the source:

```bash
python scripts/filter_cc18_shard_for_batch.py \
    --source jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \
    --batch-file benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv \
    --out jobs/doctoral/openml_cc18/batch_shards/batch_01_shard_00.sqlite
```

## Dedicated Mac batch_00 gate (Commit 32)

The first required gate is `batch_00_synthetic_canary`, which the
operator must run on the dedicated MacBook Pro before any real CC18
batch is allowed. The procedure is fully scripted:

```bash
# On the dedicated Mac (the script requires Python >= 3.10):
bash scripts/setup_dedicated_mac.sh
python scripts/audit_method_capabilities.py
python scripts/run_batch_00_synthetic_canary.py
```

The setup script installs the minimum environment for the gate
(`pip install -e ".[gbdt,doctoral,dev]"` plus `optuna>=3.5`) and
attempts the broader `[hpo_baselines]` extras (failures tolerated).
The batch_00 runner copies the committed shard
`jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite`
to a private temp path, prunes it to the 12-cell canary slice
(4 canary methods × 3 algorithms × 1 replica), and dispatches via
`scripts/cc18_runner.py --canary-only --train --synthetic-task`.
**No CC18 dataset is downloaded; no real OpenML task is touched.**

The gate artifact lands at
`experiments/_batch_runs/batch_00_synthetic_canary_latest.{json,md}`
with: git SHA, hostname, Python version, package versions,
runner command, per-cell status, capability audit summary, source
shard MD5 before/after (must match), and confirmation that the
stage-3 sign-off file was NOT created. The artifact is committed
**only when produced on the dedicated Mac** — never fabricated.

**Batch 01 (`batch_01_cc18_tiny_3_tasks`) is blocked** until the
gate artifact above shows `n_cells_failed == 0` and
`source_shard_unchanged == true`.

## Result handoff protocol (Commit 35)

`docs/RESULT_HANDOFF_PROTOCOL.md` formalizes how the dedicated Mac
publishes results back to the personal Mac (and to any future
worker):

- committed shards under `jobs/doctoral/openml_cc18/shards/` are
  immutable job-queue templates;
- execution SQLite files live under `runs/cc18/<run_id>/` and are
  gitignored, alongside per-cell fold metrics, fitted models,
  `catboost_info`, and OpenML payload caches;
- small JSON/MD summaries land under
  `experiments/_stage_runs/<run_id>_summary.{json,md}` and are
  committed — that is the only handoff that crosses Git;
- large bundles ship out-of-band, referenced in the summary by
  SHA-256 / size / archive path.

Two scripts implement the protocol:

- `scripts/create_cc18_run_dir.py` — copies committed shards into
  `runs/cc18/<run_id>/shards/<stage>/` with an `.execution.sqlite`
  suffix and writes `run_manifest.json` (source MD5s, host, git SHA);
- `scripts/export_cc18_run_summary.py` — reads the execution SQLites
  and emits the committed JSON/MD summary, re-checking each
  committed source shard against its recorded MD5 to surface drift.

batch_02 onward must use this protocol; batch_00 / batch_01 can
keep their existing temp-shard pattern, but their gate artifacts
(`experiments/_batch_runs/`) are conceptually a special case of
this protocol where the "run dir" was a `tempfile.mkdtemp()`.

## What this commit does NOT do

- Does **not** download CC18 dataset payloads (only OpenML metadata).
- Does **not** generate any job SQLite shards.
- Does **not** run any benchmark.
- Does **not** kill the article-track smoke / profiler scripts; they
  remain useful for the 12-dataset smoke panel and as integration
  tests.
