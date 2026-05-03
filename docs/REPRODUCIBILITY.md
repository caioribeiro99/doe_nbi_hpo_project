# Reproducibility guide

This branch is engineered to be reproducible end-to-end at three
fidelity levels.

## Level 1 — Synthetic smoke

Goal: ≤ 1 minute, no external data. Used by CI and as a sanity check.

```bash
make install-dev
make test-unit
make test-methodology
make smoke
```

## Level 2 — Reduced reproduction

Goal: a few minutes. Three replicas, deterministic XGBoost, small CV
budget. Demonstrates the pipeline qualitatively on the canonical
dataset (or a synthetic stand-in if MAGIC is not available).

```bash
make data            # downloads MAGIC, verifies SHA-256
make repro-mini      # = python -m doe_xgb.cli run --config configs/reduced_repro.yaml
```

## Level 3 — Full reproduction

Goal: matches the dissertation tables (within determinism caveats).
Long: 30 replicas, full benchmark suite, `tree_method="exact"` for the
canonical headline run.

```bash
make repro-full      # = python -m doe_xgb.cli run --config configs/dissertation_baseline_xgb_magic.yaml
make tables          # regenerates aggregated tables under experiments/
```

## Doctoral benchmark scope (Commit 24, retargeted in Commit 25,
## comparative protocol frozen in Commit 26)

The repository targets a doctoral-scale campaign on the
**OpenML-CC18 curated classification suite** (`suite_id = 99`,
**72 standardized tasks**) × **3 GBDT algorithms** (XGBoost, LightGBM,
CatBoost) × **N methods** × **30 replicas**, staged through
1 → 5 → 10 → 30 replicas. The headline job count for the methods that
run on all 72 tasks is `72 × 3 × N_full × 30`, plus the per-replica
jobs contributed by subset-only methods on their defined CC18 subset;
the exact total is computed at shard-generation time from
`benchmarks/doctoral/openml_cc18/method_matrix.csv` and is not
hardcoded. The methods themselves are listed in
`docs/COMPARATIVE_PROTOCOL.md`.

**Protocol freeze gate (cleared in Commit 27).** The four open
items from Commit 26 are resolved: FLAML stays `literature_only`,
ASHA was chosen over Hyperband (and the `method_id` was renamed),
the ParEGO subset is frozen at 48 of 72 tasks
(`benchmarks/doctoral/openml_cc18/parego_subset.csv`), and the
two remaining TODO references do not block shard generation.

**SQLite shards (Commit 28).** `scripts/generate_cc18_job_shards.py`
materializes the deterministic job queues at
`jobs/doctoral/openml_cc18/shards/<stage>/shard_NN.sqlite` (40 files
total: 4 stages × 10 shards) by reading
`method_matrix.csv` + `execution_policy.csv` + `parego_subset.csv`
+ `tasks.csv` against `schema.sql`. No method list is hardcoded.
Total rows: **79,920** (2,304 / 9,216 / 13,680 / 54,720 by stage).
The shard files are committed to the repository so the dedicated
Mac can fetch them by `git pull` rather than regenerate. Re-run
the generator after any edit to the four CSVs:

```bash
python scripts/generate_cc18_job_shards.py --shards 10 --force
```

**Capability audit + runner skeleton (Commit 29).** Adapters for
every non-literature method live under `src/doe_xgb/methods/`. Run

```bash
pip install -e ".[gbdt,hpo_baselines,doctoral,dev]"
python scripts/audit_method_capabilities.py
```

to refresh `experiments/_capability_audit/cc18_capability_report.{json,md}`
on the dedicated Mac; the report lists which adapters are
`stub_only` / `dispatch_only` / `smoke_ready` and which optional
packages are missing. Stage 3 is gated by a sign-off file at
`jobs/doctoral/openml_cc18/stage3_signoff.json`; the runner refuses
to claim stage-3 jobs until that file exists.

**Executable canary adapters (Commit 30).** Four adapters are now
`smoke_ready`: `default_gbdt`, `random_search`, `tpe_optuna`,
`doe_rsm_vrf_true_nbi`. They run end-to-end on a synthetic binary
task. The runner default is still `--no-train`; training requires
both `--canary-only` and `--train`, and even then only the four
canary methods are dispatched. Sample command (on a temp-copied
shard):

```bash
cp jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite /tmp/canary.sqlite
python scripts/cc18_runner.py \
    --shard /tmp/canary.sqlite --max-jobs 12 \
    --canary-only --train --synthetic-task \
    --max-evaluations 5 --n-folds 2 \
    --output-root experiments/_canary_runs
```

The committed shards under `jobs/doctoral/openml_cc18/shards/`
must not be passed to `--train` directly — always copy first.

**Stage 0 must not start** until the canary above passes on the
dedicated Mac with all four adapters marked `success` and the
audit reporting zero missing canary packages. The other 9
adapters remain stub/dispatch-only and are wired in later commits.

**Reduced-execution batches (Commit 31).** The dedicated Mac walks
through five pre-stage-0 batches before touching the full
benchmark. Manifests at
`benchmarks/doctoral/openml_cc18/batches/`:
`batch_00_synthetic_canary` (no OpenML data) →
`batch_01_cc18_tiny_3_tasks` (3 real CC18 tasks) →
`batch_02_cc18_small_12_tasks` (12 stratified) →
`batch_03_cc18_representative_18_tasks` (18; broader coverage) →
`batch_04_stage0_shard00_only` (one existing shard). Generated by
`scripts/create_cc18_batches.py` deterministically. A SQLite shard
can be filtered down to a batch's task IDs without mutating the
source via:

```bash
python scripts/filter_cc18_shard_for_batch.py \
    --source jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \
    --batch-file benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv \
    --out jobs/doctoral/openml_cc18/batch_shards/batch_01.sqlite
```

Full stage 0 runs only after batches A→E land green sign-off
artifacts.

**Dedicated Mac batch_00 gate (Commit 32).** Procedure on the
dedicated MacBook Pro (Python >= 3.10 required):

```bash
bash scripts/setup_dedicated_mac.sh
python scripts/audit_method_capabilities.py
python scripts/run_batch_00_synthetic_canary.py
```

The runner copies the committed shard to a private temp path,
prunes it to a 12-cell canary slice (4 canary methods × 3
algorithms × 1 replica), and runs `cc18_runner.py --canary-only
--train --synthetic-task`. The committed shards are never
mutated. The gate artifact at
`experiments/_batch_runs/batch_00_synthetic_canary_latest.{json,md}`
is committed only when produced on the dedicated Mac. Batch 01
is blocked until that artifact shows zero failures.

**Result handoff protocol (Commit 35).** From batch_02 onward,
results cross machines through `docs/RESULT_HANDOFF_PROTOCOL.md`.
The committed SQLite shards under `jobs/doctoral/openml_cc18/shards/`
are immutable templates. Execution copies live under
`runs/cc18/<run_id>/shards/<stage>/shard_NN.execution.sqlite`
(gitignored, like every fitted model, fold CSV, `catboost_info/`
and OpenML payload). Small JSON/MD summaries land under
`experiments/_stage_runs/<run_id>_summary.{json,md}` and are the
only artifact that crosses Git. Large bundles ship out-of-band
with SHA-256 captured in the summary. Two helper scripts implement
the protocol:

```bash
python scripts/create_cc18_run_dir.py \
    --run-id "<batch_or_stage>__<host>__<utc>" \
    --stage stage0_replica_001 \
    --shard shard_00.sqlite

# ... runner consumes the .execution.sqlite copies ...

python scripts/export_cc18_run_summary.py \
    --run-dir runs/cc18/<run_id> \
    --out-json experiments/_stage_runs/<run_id>_summary.json \
    --out-md  experiments/_stage_runs/<run_id>_summary.md \
    --include-shard-hashes
```

The CC18 task / dataset registry lives at
`benchmarks/doctoral/openml_cc18/{tasks.csv, datasets.csv,
openml_cc18_metadata.json, coverage_report.md}` and is generated by
`scripts/import_openml_cc18.py` (network-bound). The SQLite job-matrix
schema lives at `jobs/doctoral/openml_cc18/schema.sql` (table:
`cc18_jobs`). See `docs/DOCTORAL_BENCHMARK.md` for the full pivot
summary.

The 12-dataset article v1 panel is **demoted** to a smoke / profiling
/ development fixture and lives at
`benchmarks/doctoral/internal_smoke_panel/datasets.csv`. The existing
scripts under `scripts/run_v1_*` and `scripts/profile_v1_*` keep
working and remain useful for integration testing and capacity
calibration; they do not contribute to the CC18 benchmark.

The earlier "82 datasets" framing (Commit 24) and its scaffolding
under `benchmarks/doctoral_82/` and `jobs/doctoral_82/` are
**deprecated** but kept as historical context.

## End-to-end article-track smoke (Commit 23)

A reduced article-track smoke is committed for sanity checks:

```bash
python scripts/run_article_true_nbi_magic_smoke.py
```

Single replica on MAGIC + XGBoost, q=2 NBI, simplex_lattice {2, 10}.
Drives the full pipeline (DOE → FA → RSM → true N-objective NBI →
confirmation → conditional MBPA), uses
`evaluation.assert_metric_set_compatible_with_task` for the binary
guardrail, and writes
`experiments/_v1_smoke/article_true_nbi_magic_smoke.{json,md}` with
per-stage timings, NBI residual statistics, and the MBPA decision.
Total wall-clock ~86 s on Apple Silicon. The legacy weighted-sum
scalarization is never invoked.

## What every replica writes

```
experiments/<dataset>/<design>/replica_XX/
├── manifest.json                       # seeds, sha256, config, system fingerprint
├── doe_results.csv
├── doe_results_with_scores.csv
├── factor_loadings.csv
├── factor_scores.csv
├── factor_diagnostics.json
├── rsm_coefficients_<obj>.csv
├── nbi_anchors.csv
├── nbi_chim.json
├── nbi_candidates.csv
├── nbi_subproblem_diagnostics.csv
├── nbi_candidate_evaluations.csv
├── confirmation_vrf.csv
├── frontier_quality.json
├── post_optimization_diagnostics.json   # always written; describes whether MBPA fired
├── post_optimization_mixture_fit.csv    # only if MBPA fired
├── post_optimization_refined_candidate.csv
├── confirmation_summary.csv
├── fold_metrics.csv
└── run_replica.log
```

## System fingerprint

`manifest.json` includes platform, CPU model, RAM, OS, Python version,
and a `pip freeze` snapshot. This makes "I cannot reproduce" tickets
diagnosable.

## Deterministic mode

For bit-stable headline tables, set `experiment.deterministic: true` in
the YAML config or pass `--deterministic` on the CLI. This forces
`tree_method="exact"` and `n_jobs=1`. It roughly doubles wall-time on
laptop-class hardware and is the recommended setting for the article
tables.
