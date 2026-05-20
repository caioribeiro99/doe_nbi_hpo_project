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
> **Protocol freeze gate cleared in Commit 27.** FLAML stays
> `literature_only`; ASHA was chosen over Hyperband; the ParEGO
> subset is frozen at 48 of 72 tasks
> (`benchmarks/doctoral/openml_cc18/parego_subset.csv`); the
> per-method execution-tier policy lives at
> `benchmarks/doctoral/openml_cc18/execution_policy.csv` and
> `execution_tiers.md`.
>
> **SQLite shards landed in Commit 28.**
> `scripts/generate_cc18_job_shards.py` materializes 40 deterministic
> shard files at
> `jobs/doctoral/openml_cc18/shards/<stage>/shard_NN.sqlite`,
> totalling 79,920 jobs (2,304 / 9,216 / 13,680 / 54,720 by stage).
> The shards are committed to the repository so the dedicated Mac
> fetches them by `git pull`.
>
> **Capability audit + runner skeleton landed in Commit 29.** All 13
> non-literature methods have an adapter under `src/doe_xgb/methods/`;
> the audit script
> (`scripts/audit_method_capabilities.py`) writes
> `experiments/_capability_audit/cc18_capability_report.{json,md}`.
>
> **Executable canary adapters landed in Commit 30.** Four adapters
> are `smoke_ready` (`default_gbdt`, `random_search`, `tpe_optuna`,
> `doe_rsm_vrf_true_nbi`); the rest stay `stub_only` /
> `dispatch_only`. The runner now supports `--canary-only --train
> --synthetic-task` to exercise the four adapters on a synthetic
> binary task on a temp-copied shard.
>
> **Reduced-execution batches landed in Commit 31.** The dedicated
> Mac walks through five pre-stage-0 batches before any full stage
> runs: `batch_00_synthetic_canary` →
> `batch_01_cc18_tiny_3_tasks` (3 tasks) →
> `batch_02_cc18_small_12_tasks` (12 tasks) →
> `batch_03_cc18_representative_18_tasks` (18 tasks) →
> `batch_04_stage0_shard00_only`.
>
> **Dedicated Mac batch_00 gate scripted in Commit 32.**
> `scripts/setup_dedicated_mac.sh` installs the minimum
> environment; `scripts/run_batch_00_synthetic_canary.py` copies a
> stage-0 shard to a temp path, prunes to the 12-cell canary
> slice, and runs the canary via `cc18_runner.py`. Artifacts at
> `experiments/_batch_runs/batch_00_synthetic_canary_latest.{json,md}`
> are committed only when produced on the dedicated Mac.
>
> **Dedicated Mac batch_01 ran green in Commit 34** (36/36 cells,
> source shards unchanged, OpenML payloads cached under the
> gitignored `data/source/openml_cc18/`).
>
> **Heavy-task policy + runtime guardrails in Commit 38**
> (`docs/HEAVY_TASK_POLICY.md`). batch_03 ran 216 / 216 cells
> green but spent ~92 % of 56 710 s of runner CPU on 8
> Devnagari-Script cells (task 167121, 92 000 × 1 024 × 46
> classes). To stop that pattern from blocking full stage 0,
> Commit 38 splits CC18 into three lanes — 57 standard, 13
> heavy, 2 extreme (`letter`, `Devnagari-Script`). Per-lane
> defaults live in
> `benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml`;
> per-task assignments in
> `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv` (built
> reproducibly by `scripts/build_cc18_heavy_task_policy.py`).
> `src/doe_xgb/runtime_guardrails.py` is the runtime API every
> CC18 runner from Commit 38 onward consults. Extreme tasks are
> deferred unless the runner is invoked with
> `--include-extreme-tasks`. Full stage 0 splits into separate
> standard / heavy / extreme passes, each publishing its own
> stage-run summary.
>
> **Result handoff protocol formalized in Commit 35**
> (`docs/RESULT_HANDOFF_PROTOCOL.md`). Committed SQLite shards stay
> immutable; execution copies live under `runs/cc18/<run_id>/` and
> are gitignored; small JSON/MD summaries under
> `experiments/_stage_runs/` are the only artifact that crosses
> Git. Two helper scripts implement the protocol:
> `scripts/create_cc18_run_dir.py` and
> `scripts/export_cc18_run_summary.py`.
>
> **Stage 0 progress (Commits 39 → 42).** Commit 39 ran
> `batch_04_stage0_shard00_only` (operational dry run on one
> committed shard; 80/80 canary cells green). Commit 40 ran
> the standard-lane pass of stage 0 (684 / 684 green). Commit 41
> ran the heavy-lane pass (156 / 156 green). Commit 42 is the
> dedicated planning step for the extreme lane: it ships
> `scripts/run_stage0_extreme_lane.py` in PLANNING-ONLY mode
> (real execution is locked behind `--execute-extreme-lane`,
> which Commit 42 does NOT pass) and publishes a dry-run plan at
> `experiments/_stage_runs/stage0_extreme_lane_plan_latest_summary.{json,md}`
> with `execution_status = "planned_not_executed"`. See
> `docs/EXTREME_LANE_PLAN.md` for the runtime forecast (~15.7 h
> on the dedicated Mac, dominated by Devnagari-Script), the
> max_evaluations 1-vs-5 tradeoff, and the promotion criteria
> for stage 0 replica 1 complete.
>
> **Commit 43 (`28961fe`) ran the extreme lane.** 24 / 24 cells
> green in 30,844 s of runner CPU at the policy default
> `extreme.stage0_max_evaluations = 1`. Devnagari-Script
> doe_rsm peaked at 10,663 s (under the 14,400 s timeout).
> All four stage-0 artifacts (standard / heavy / extreme-plan /
> extreme) pin the same `policy_version`.
>
> **Commit 44 planned the aggregate signoff.**
> `scripts/build_stage0_replica_signoff.py` reads the three
> lane summaries, records cross-lane invariants (same
> policy_version, all green, source_shards_unchanged,
> stage3_signoff_present=false), aggregates metrics by
> lane / method / algorithm / task_type, and publishes
> `experiments/_stage_runs/stage0_replica_001_signoff_plan_latest_summary.{json,md}`,
> originally with `signoff_status = "planned_not_signed"`. See
> `docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md` for the operator
> review surface and the `isolet` / `Devnagari-Script`
> caveats.
>
> **Commit 45 (this commit) signs off stage 0 replica 1.**
> `scripts/sign_stage0_replica_001.py` re-verifies every gate
> the aggregate plan advertised, writes
> `jobs/doctoral/openml_cc18/stage3_signoff.json` with operator
> metadata (`Caio Tertuliano Ribeiro` / `caioribeiro99`), both
> required caveat acknowledgements
> (`isolet_future_recalibration_candidate` +
> `devnagari_extreme_budget_non_equivalence`), and
> `downstream_execution_authorized_in_this_commit = false`,
> then re-runs the aggregator so the published plan summary now
> reads `signoff_status = "signed"`,
> `final_recommendation = "signed_ready_for_next_stage_planning"`.
> The signoff freezes the lane summary SHA-256s; the aggregator
> refuses on any post-signoff lane-summary tampering. No OpenML
> training, shard mutation, or downstream execution is performed
> by this commit.
>
> **Commit 46 plans stage-3 / top-up dispatch.**
> `scripts/plan_stage3_topup.py` reads the Commit-45 signoff and
> the three stage-0 lane summaries, computes the three top-up
> tiers (`topup_to_5` / `topup_to_10` / `topup_to_30`, totalling
> 25,056 additional canary cells across 30 replicas of the
> 864-cell panel), records SHA-256-cross-checked invariants
> against the live `heavy_task_policy.csv`, and publishes
> `experiments/_stage_runs/stage3_topup_plan_latest_summary.{json,md}`
> plus the per-tier / per-lane manifest
> (`benchmarks/doctoral/openml_cc18/stage3_topup_manifest.{csv,md}`)
> and the worker plan
> (`benchmarks/doctoral/openml_cc18/stage3_worker_plan.{csv,md}`).
> Three new docs frame the work:
> `docs/STAGE3_TOPUP_EXECUTION_PLAN.md` (strategic context),
> `docs/STAGE3_POLICY_DECISION.md` (Option A vs B vs C),
> `docs/STAGE3_DISTRIBUTED_RUNBOOK.md` (per-worker operator
> runbook). No Stage-3 / top-up execution happens in Commit 46.
> No policy change happens either: Commit 46 implements
> Option C and stays policy-neutral.
>
> **Next operational step:** Commit 47 should run a tiny
> Stage-3 / top-up pilot — `replica_002`, `shard_00`, standard
> lane, canary methods only, no heavy / extreme — and an
> operator should review the pilot summary before scaling to
> the full `topup_to_5`.
>
> **Commit 47 (this commit) ran the tiny Stage-3 / top-up
> pilot.**
> `scripts/run_stage3_pilot_replica002_shard00_standard_lane.py`
> verifies the Commit-45 signoff and the Commit-46 plan
> summary (refusing on missing files, drifted
> `policy_version`, or a plan that excludes `replica = 2`
> from `topup_to_5`), copies `shard_00.sqlite` from
> `shards/stage0_replica_001/` into `runs/cc18/<run_id>/`,
> rewrites the copy so every row carries `replica = 2` and
> `stage = 'stage1_topup_to_005'`, defers heavy and extreme
> rows, refuses non-canary rows, and executes the 68
> standard-lane canary cells on shard_00 (4 canary methods ×
> 3 algorithms × 17 standard tasks ≈ 68 with the catboost-
> excluded shape baked into the schedule). The committed
> source shard is byte-identical pre/post pilot; no
> execution SQLite, fitted model, raw OpenML payload,
> notebook, or fairness artifact is staged. The committed
> summary lives at `experiments/_stage_runs/`
> `stage3_pilot_replica_002_shard00_standard_lane_latest_summary.{json,md}`.
> Commit 47 does **not** run the full `topup_to_5` tier, the
> heavy lane, or the extreme lane, and does **not**
> regenerate `heavy_task_policy.csv`,
> `heavy_task_policy_report.md`, or
> `runtime_guardrails.yaml`. Operator review of the pilot
> summary is required before any further Stage-3 / top-up
> dispatch.
>
> **Next operational step (after Commit 47):** only after
> the pilot summary has been operator-reviewed, Commit 48
> should plan or run a slightly larger Stage-3 pilot — e.g.
> `replica_002` across all 10 standard-lane shards, or
> `shard_00` standard + a selected heavy probe. Do not
> scale directly to the full `topup_to_5` tier without
> reviewing the Commit-47 summary.
>
> **Commit 48 (this commit) expanded the pilot to replica_002
> standard lane across all 10 shards.**
> `scripts/run_stage3_replica002_standard_lane.py` chains three
> gates — Commit 45 signoff, Commit 46 plan summary, Commit 47
> pilot summary — refusing on missing files, drifted
> `policy_version`, a failed pilot, or a pilot that exercised a
> different scope. It copies all ten `shard_NN.sqlite` files
> from `shards/stage0_replica_001/` into `runs/cc18/<run_id>/`,
> rewrites every copy so every row carries `replica = 2` and
> `stage = 'stage1_topup_to_005'`, defers heavy / extreme rows,
> refuses non-canary rows, and executes the 684 standard-lane
> canary cells (57 standard tasks × 4 canary methods × 3
> algorithms). The 10 committed source shards are
> byte-identical pre/post run; no execution SQLite, fitted
> model, raw OpenML payload, notebook, or fairness artifact is
> staged. The committed summary lives at
> `experiments/_stage_runs/`
> `stage3_replica_002_standard_lane_latest_summary.{json,md}`.
> Commit 48 does **not** run the full `topup_to_5` tier, the
> heavy lane, the extreme lane, or any other replica.
>
> **Next operational step (after Commit 48):** only after the
> Commit-48 summary has been operator-reviewed, Commit 49
> should decide whether to: (a) run replica_002 heavy lane;
> (b) run a selected heavy-lane pilot first; or (c) create an
> aggregate review for replica_002 standard before heavy
> execution. Do not scale directly to the full `topup_to_5`
> tier without reviewing the Commit-48 summary.

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
