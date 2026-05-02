# Article v1 — next steps

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

## Phase C — small smoke run

6. **Single-dataset smoke.** Re-run the existing tests; then run
   `doe-xgb run --config configs/article_3vrf_xgb_magic.yaml` (after
   the orchestrator follow-up branch lands) for `n_replicas=2` on
   MAGIC only. Confirm:
   - manifest written;
   - NBI residuals < 1e-3 across all sub-problems;
   - MBPA decision recorded;
   - per-fold metrics CSV has the expected columns.
7. **Three-algorithm smoke.** Repeat on a tiny synthetic dataset
   spanning XGBoost / LightGBM / CatBoost to confirm the loader
   abstractions work uniformly.

## Phase D — full v1 campaign

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

- 82-dataset doctoral benchmark.
- Deep-tabular baselines.
- Calibration as a primary objective (only reported in v1).
- Fairness extension (lives on a separate branch).
