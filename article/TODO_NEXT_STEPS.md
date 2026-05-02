# Article v1 — next steps (now subordinate to the doctoral benchmark)

> **Pivot (Commit 24, retargeted in Commit 25, comparative protocol
> frozen in Commit 26).** The repository now targets the
> **doctoral-scale benchmark on OpenML-CC18** (`suite_id = 99`,
> 72 standardized classification tasks × 3 GBDT algorithms × N
> methods × 30 replicas) as its scientific objective; the
> 12-dataset article v1 campaign is no longer the main target. The
> publication will be extracted from the doctoral-scale CC18
> results, not from a separate intermediate campaign. The 12-dataset
> panel is **demoted** to a smoke / profiling / development fixture
> under `benchmarks/doctoral/internal_smoke_panel/datasets.csv` and
> is **not** part of the CC18 benchmark count. The previous
> "82 datasets" framing (`benchmarks/doctoral_82/`) is deprecated.
> The comparison methods (random search, Optuna TPE, SMAC3,
> ASHA / BOHB / DEHB, NSGA-II, MOTPE, ParEGO on a subset, plus the
> proposed method and its two ablations) are frozen by
> `benchmarks/doctoral/openml_cc18/method_matrix.csv`; see
> `docs/COMPARATIVE_PROTOCOL.md` and `docs/DOCTORAL_BENCHMARK.md`.
>
> **Next operational step is not "run the 12-dataset campaign".** It
> is: (i) resolve the open items at the bottom of
> `docs/COMPARATIVE_PROTOCOL.md`, (ii) freeze
> `method_matrix.csv`, (iii) generate SQLite shards driven by the
> CSV via `scripts/generate_cc18_job_shards.py` (planned for the
> next commit). The method list must be frozen before any SQLite
> shards are committed.

These are the actions needed to take the manuscript from scaffold to
submitted draft. They are deliberately ordered: each step gates the
next.

## Phase A — confirm and prepare data (no full runs) [DONE in Commit 15]

1. **Dataset availability check.** Done. All 12 entries probed; report
   at `data/source/AVAILABILITY_CHECK.md` and JSON registry at
   `data/source/dataset_registry.json`. CLI:
   `doe-xgb datasets check-availability`.
2. **Loaders implemented.** `src/doe_xgb/datasets/loaders.py` exposes
   one loader per dataset returning `(X, y, metadata)`. Loaders read
   from `data/source/<id>/` if cached and otherwise raise
   `DatasetUnavailableError` with the canonical URL.
3. **Download scripts.** Done in Commit 16. Per-dataset
   `scripts/fetch_<dataset>_dataset.py` for the eleven non-sklearn
   datasets, plus a shared helper at `scripts/_dataset_fetch_base.py`
   and a downloader utility module at
   `src/doe_xgb/datasets/download.py`. Each script writes raw +
   processed + manifest, and the aggregated checksums land in
   `data/source/CHECKSUMS.txt`. The CLI helpers
   `doe-xgb datasets fetch [--all] [--force]` and
   `doe-xgb datasets verify-checksums` wrap the per-dataset scripts.
   Real dataset payloads are gitignored; only manifests / checksums
   are versioned (after the first authoritative run on the user's
   machine).

## Phase B — sizing [DONE in Commit 17]

4. **Cost calibration.** Done. Calibration JSONs at
   `experiments/_cost_calibration/{xgboost,lightgbm,catboost}.json`
   on Apple Silicon Mac. Best-of-N seconds per fit on a 1500-row ×
   12-feature synthetic problem at default hyperparameters:
   XGBoost ~0.054 s, LightGBM ~0.041 s, CatBoost ~0.135 s.
   Worst-case used for planning: **0.135 s/fit** (CatBoost).
5. **Projected v1 wall-clock and cost.** Sizing report at
   `experiments/_cost_calibration/article_v1_cost_estimates.{json,md}`.
   At a realistic 4× inflation factor for full DoE / heavy CV
   configurations (~0.54 s/fit), the headline 12 × 3 × 10 scenario
   finishes in **1.2 days on a dedicated Mac**, **0.8 days on
   combined two Macs (16 workers @ 24 h, eff 0.70)**, or **7.5 h /
   $24 on a 32-worker $0.10/h cloud**. The pessimistic 8× scenario
   finishes in 2.4 days dedicated or $48 cloud.
6. **Sign-off.** Recommended scope: **run the full 12 × 3 × 10
   headline panel**. Drop to 5 replicas only selectively on the
   heaviest datasets if real-world inflation exceeds 8×.

### Dry Bean (multiclass) -- config + guardrail landed in Commit 19

The configuration gap reported in Commit 18 is now closed:

- `configs/article_3vrf_dry_bean.yaml` ships the multiclass
  appendix config: 8 response keys (`F1Macro_Mean`,
  `BalancedAccuracy_Mean`, `MCC_Mean`, `ROCAUC_OVR_Mean`,
  `PRAUC_OVR_Mean`, `BrierMC_Mean`, `ECE_Mean`, `Time_MeanFold`),
  explicit objective directions (maximize quality / probability;
  minimize Brier / ECE / Time), and XGBoost `multi:softprob` /
  `num_class=7`.
- `src/doe_xgb/datasets/guardrails.py::validate_task_metric_compatibility(config, ...)`
  resolves the dataset id from a YAML config and applies the
  Commit 18 assertion. It raises `MultiClassNotConfiguredError`
  for Dry Bean + binary defaults; passes for binary configs and for
  Dry Bean + the multiclass response set.
- The future orchestrator should call this helper after dataset
  resolution and before stage 1 of FA / NBI. Until that orchestrator
  lands, the helper is exercised by unit tests and can be invoked
  manually inside `scripts/run_replica.py`.

Remaining items before a real Dry Bean smoke:

1. Run the Dry Bean downloader once on the contributor's machine:
   `python scripts/fetch_dry_bean_dataset.py`. The downloader already
   exists; the actual fetch is local-only and not committed.
2. Run a tiny **binary** smoke first (2 binary datasets × 3 GBDT × 1
   replica) to validate the orchestrator on the headline path.
3. Then run a Dry Bean appendix smoke (1 replica, XGBoost only) using
   `configs/article_3vrf_dry_bean.yaml`.

Decision unchanged: **Option B -- Dry Bean is the secondary multiclass
stress test reported in the appendix/supplementary**, not a headline
v1 dataset. The 11 binary datasets remain the headline panel.

## Phase C — small smoke run [v1 binary smoke landed in Commit 20]

6. **Tiny binary smoke (DONE).** `scripts/run_v1_binary_smoke.py`
   loads `german_credit`, `pima_diabetes`, and `spambase` (and
   optionally Breast Cancer); runs `evaluate_xgb_cv` at a single
   safe hyperparameter point with 2-fold CV; asserts the
   dissertation-era binary keys are populated; writes
   `experiments/_v1_smoke/binary_smoke_output.json`. Total runtime
   ~1.3 s. Does **not** run DOE / RSM / NBI / MBPA.
7. **Three-algorithm smoke (DONE).**
   `scripts/run_v1_binary_3alg_smoke.py` evaluates each of the
   three fetched datasets with XGBoost / LightGBM / CatBoost at one
   safe hyperparameter point under 2-fold CV; writes
   `experiments/_v1_smoke/binary_3alg_smoke_output.json`. CatBoost
   uses native categorical handling on `german_credit` and falls
   back to encoded ints on the all-numeric datasets. Total runtime
   ~1.7 s. All 9 (dataset, algorithm) pairs passed the binary-key
   and accuracy-floor assertions.
8. **Single-dataset DOE+NBI smoke (later).** Once the orchestrator
   wrapper lands, run `doe-xgb run --config configs/article_3vrf_xgb_magic.yaml`
   on MAGIC at `n_replicas=2` and confirm NBI residuals < 1e-3,
   MBPA decision recorded, per-fold metrics CSV has the expected
   columns.
9. **Dry Bean appendix smoke (later).** After 7 and 8 pass, run
   `configs/article_3vrf_dry_bean.yaml` at `n_replicas=1` with
   XGBoost only.

## Phase D — full v1 campaign

> **Sizing update (Commit 22).** The full-dataset 5-fold runtime
> profile at
> `experiments/_runtime_profile/v1_full_dataset_5fold_profile.{json,md}`
> projects the headline 12 × 3 × 10 panel at **~1.2 days dedicated
> Mac (eff 0.70)** under the 4× realistic inflation multiplier, or
> **~7.5 h / $24 on the 32-worker $0.10/h cloud**. 30 replicas at
> 4× project to ~3.7 days dedicated. Recommendation: run the full
> 12 × 3 × 10 locally on the dedicated Mac. Reserve 30 replicas for
> selected datasets only.
>
> **Article-track end-to-end smoke (Commit 23).** Real-data
> validation of the full pipeline: DOE → FA → RSM → **true
> N-objective NBI** → confirmation → conditional MBPA, on MAGIC +
> XGBoost, n_replicas=1, q=2, simplex_lattice {2, 10}. All 7
> stages succeeded; NBI residuals `max=1.6e-10, median=4.9e-12`;
> MBPA fired on a `high_weight_concentration` trigger. Total wall
> time ~86 s. See
> `experiments/_v1_smoke/article_true_nbi_magic_smoke.{json,md}`.
> The legacy weighted-sum scalarization was **not** invoked at any
> stage of the smoke.

8. **Run the v1 campaign.** Schedule
   `12 datasets x 3 algorithms x 10 replicas` on the chosen profile.
   Use the cost-estimator batching plan to checkpoint every
   `checkpoint_frequency_replicas` replicas.
9. **Aggregate.** Run `make tables` to produce
   `article/tables/*.csv`. Inspect outliers; if any (dataset, algorithm)
   cell has an obvious outlier replica, investigate before accepting
   the table.
10. **Statistical analysis.** Friedman + Nemenyi notebook produces
    `article/tables/friedman_nemenyi.csv` and the critical-difference
    diagram.

## Phase E — paper drafting

11. **Fill `05_results.tex`.** Replace every TODO with a single line
    of LaTeX referring to the aggregated CSV. Numbers are inserted
    via `\input{}` from a small `tables/*.tex` companion if you want
    the article to never touch hand-typed numbers.
12. **Fill `06_discussion.tex`.** Use the discussion checklist; do not
    over-claim.
13. **Refresh `00_abstract.tex`.** Insert quantitative headline
    numbers verbatim from the result CSVs.
14. **Choose a venue and switch the LaTeX template.** The current
    preamble is venue-agnostic; switch to elsarticle / IEEE / Springer
    once the target is decided.
15. **Final compile + ruff/test green check** on the artifact branch
    before tagging `v0.3.0-articlev1`.

## Out of scope for v1

- OpenML-CC18 doctoral benchmark (72 tasks × 3 algorithms × 30
  replicas; runs as a separate campaign on the dedicated Mac).
- Deep-tabular baselines.
- Calibration as a primary objective (only reported in v1).
- Fairness extension (lives on a separate branch).
