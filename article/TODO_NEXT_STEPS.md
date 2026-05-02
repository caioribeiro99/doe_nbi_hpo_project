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

## Phase B — sizing

4. **Cost calibration.** Run
   `doe-xgb estimate-cost --calibrate
   --calibration-output cost_estimate_calibration.json --algorithm
   xgboost` on the target machine; repeat with `--algorithm
   lightgbm` and `--algorithm catboost`. Save the JSON files under
   `experiments/_cost_calibration/`.
5. **Project the v1 wall-clock and dollar cost.** Run
   `doe-xgb estimate-cost --preset
   article_v1_12_datasets_3_algorithms_10_replicas` for both the
   local profile (Mac defaults) and a candidate cloud profile.
   Sign-off if the projected wall-clock is acceptable; else reduce
   replicas or evaluation budget.

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
