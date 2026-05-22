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

Stage-3 jobs of every tier-1+ method are gated by the manual
sign-off file at
`jobs/doctoral/openml_cc18/stage3_signoff.json`. The runner
refuses to claim stage-3 jobs without it and reports them as
`refused_stage3_signoff_missing`. Commit 45 created the signoff
file via `scripts/sign_stage0_replica_001.py` (operator metadata
+ both required caveat acknowledgements, with
`downstream_execution_authorized_in_this_commit = false`).
Stage-3 dispatch itself is still a separate, operator-reviewed
commit; Commit 45 only unlocks the planning of that commit.

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

## Heavy-task policy (Commit 38)

batch_03 (Commit 37) ran 216 / 216 cells green but spent ~92 % of
its 56 710 s runner CPU on 8 cells of `Devnagari-Script` (task
167121, 92 000 × 1 024 × 46 classes). To prevent that pattern
from blocking full stage 0, Commit 38 introduces a three-lane
heavy-task policy:

- `benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml` —
  per-lane defaults (timeouts, max_evaluations, include-by-
  default flag);
- `benchmarks/doctoral/openml_cc18/heavy_task_policy.csv` —
  per-task lane assignment (57 standard / 13 heavy / 2 extreme);
- `scripts/build_cc18_heavy_task_policy.py` regenerates both from
  `tasks.csv` + the latest batch summaries;
- `src/doe_xgb/runtime_guardrails.py` exposes the runtime API
  (`get_task_lane`, `get_timeout_seconds`,
  `get_effective_max_evaluations`, `should_defer_task`).

batch_04 onward MUST consult the policy. Extreme tasks (currently
`Devnagari-Script`, `letter`) are deferred unless the caller
passes `--include-extreme-tasks`. Full stage 0 splits into a
standard / heavy / extreme pass, each with its own published
stage-run summary. See `docs/HEAVY_TASK_POLICY.md` for the
contract.

## Stage 0 lane progress (Commits 40 → 45)

Per Commit 38's heavy-task policy, full stage 0 runs as three
independent lanes, then a planning commit, then operator signoff:

| lane | commit | runner | status |
|---|---|---|---|
| standard (57 tasks, 684 cells) | Commit 40 (`daae8ab`) | `scripts/run_stage0_standard_lane.py` | green |
| heavy (13 tasks, 156 cells) | Commit 41 (`ddb657d`) | `scripts/run_stage0_heavy_lane.py` | green |
| extreme (2 tasks, 24 cells) | Commit 43 (`28961fe`) | `scripts/run_stage0_extreme_lane.py --execute-extreme-lane` | green |
| aggregate signoff plan | Commit 44 — **planning only** | `scripts/build_stage0_replica_signoff.py` | planned_not_signed |
| operator signoff | Commit 45 | `scripts/sign_stage0_replica_001.py` | signed |

All three lanes share the same `policy_version` SHA-256
(`47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`).
Stage 0 replica 1 is lane-complete (864 / 864 canary cells
green: 684 standard + 156 heavy + 24 extreme). Commit 44
published the aggregate signoff plan at
`experiments/_stage_runs/stage0_replica_001_signoff_plan_latest_summary.{json,md}`
with `signoff_status = "planned_not_signed"`. Commit 45 then
created `jobs/doctoral/openml_cc18/stage3_signoff.json` — the
operator-reviewed file that unlocks stage-3 top-up *planning*
(actual dispatch is a separate, future commit) — and re-ran the
aggregator so the published summary now reads
`signoff_status = "signed"`. See
`docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md` for the review
surface, the `isolet` / `Devnagari-Script` caveats, and the
exact fields the signoff records.

## Stage-3 / top-up planning (Commit 46)

Commit 46 is the planning commit that scaffolds the scaling
from `stage0_replica_001` to the three top-up tiers
(`topup_to_5`, `topup_to_10`, `topup_to_30` — 4 / 5 / 20
additional replicas, +3,456 / +4,320 / +17,280 canary cells).
It adds `scripts/plan_stage3_topup.py` (read-only planner),
emits a JSON / MD plan summary under
`experiments/_stage_runs/stage3_topup_plan_latest_summary.{json,md}`,
and publishes machine-readable manifests under
`benchmarks/doctoral/openml_cc18/stage3_topup_manifest.{csv,md}`
and `stage3_worker_plan.{csv,md}`. Three new docs frame the
work: `docs/STAGE3_TOPUP_EXECUTION_PLAN.md`,
`docs/STAGE3_POLICY_DECISION.md`, and
`docs/STAGE3_DISTRIBUTED_RUNBOOK.md`. No Stage-3 / top-up
execution happens in Commit 46. The next execution commit
(Commit 47) is expected to be a tiny Stage-3 pilot:
`replica_002`, `shard_00`, standard lane, canary methods only.

## Stage-3 / top-up tiny pilot (Commit 47)

Commit 47 is the **first real Stage-3 / top-up execution** on
this repository. It is intentionally tiny:

- one shard (`shard_00`);
- one replica (`replica = 2`, the first replica of the
  `topup_to_5` tier);
- standard lane only;
- canary methods only.

The pilot script is
`scripts/run_stage3_pilot_replica002_shard00_standard_lane.py`.
It copies `shard_00.sqlite` under `runs/cc18/<run_id>/`,
rewrites the **copy** to carry `replica = 2` and
`stage = 'stage1_topup_to_005'` (the closest existing label in
the SQLite CHECK constraint for the `topup_to_5` tier), defers
heavy / extreme rows, refuses non-canary rows, and executes
the 68 standard-lane canary cells. The committed source shard
stays byte-identical. The committed summary lands at
`experiments/_stage_runs/`
`stage3_pilot_replica_002_shard00_standard_lane_latest_summary.{json,md}`.

Commit 47 does **not** run the full `topup_to_5` tier, the
heavy lane, or the extreme lane. Operator review of the
pilot summary is required before any further Stage-3 / top-up
dispatch.

## Stage-3 / top-up replica_002 standard lane (Commit 48)

Commit 48 expands the Commit 47 single-shard pilot to the full
standard lane for one replica: **all 10** source template
shards, ``replica = 2`` only, standard lane only, canary methods
only. The runner
`scripts/run_stage3_replica002_standard_lane.py` chains three
gates (Commit 45 signoff → Commit 46 plan → Commit 47 pilot
summary) before copying any shard. It copies all ten
`shard_NN.sqlite` files into `runs/cc18/<run_id>/`, rewrites
every copy so all rows carry ``replica = 2`` and
``stage = 'stage1_topup_to_005'``, defers heavy / extreme rows,
refuses non-canary rows, and executes the 684 standard-lane
canary cells. The 10 committed source shards remain
byte-identical. The committed summary lives at
`experiments/_stage_runs/`
`stage3_replica_002_standard_lane_latest_summary.{json,md}`.

Commit 48 still does **not** run the full `topup_to_5` tier,
heavy lane, extreme lane, or any other replica. Operator review
is required before any heavy-lane or broader top-up execution.

## Stage-3 / top-up replica_002 heavy lane (Commit 49)

Commit 49 runs the heavy-lane companion on replica 2: all 10
source template shards, heavy lane only, four canary methods
only — 156 executable heavy-lane canary cells total. The runner
`scripts/run_stage3_replica002_heavy_lane.py` chains four gates
(Commit 45 signoff → Commit 46 plan → Commit 48 standard-lane
summary → defensive isolet-lane guard) before copying any shard.
It rewrites the 10 execution copies to `replica = 2` and
`stage = 'stage1_topup_to_005'`, defers standard / extreme rows,
refuses non-canary rows, and executes the 156 heavy-lane canary
cells at the policy's `stage0_max_evaluations = 5` budget with
the 7,200 s per-cell timeout. The 10 committed source shards
remain byte-identical. The committed summary lives at
`experiments/_stage_runs/`
`stage3_replica_002_heavy_lane_latest_summary.{json,md}`.

Commit 49 does **not** run the full `topup_to_5` tier, the
extreme lane, or any other replica, and does **not** rerun the
standard lane. It also does **not** promote `isolet` (task 3481)
into the heavy lane — isolet remains a future policy
recalibration candidate under signoff caveat 1, but the pinned
`policy_version` keeps it standard for this commit. Operator
review is required before the extreme lane or broader top-up
execution.

## Stage-3 / top-up replica_002 extreme-lane plan (Commit 50)

Commit 50 is the **planning** companion to Commit 49: it chains
four gates (signoff → Commit 46 plan → Commit 48 standard
summary → Commit 49 heavy summary), inspects the 10 committed
source template shards, and projects the future Commit 51
extreme execution. The planner
`scripts/plan_stage3_replica002_extreme_lane.py` enumerates the
24 runnable extreme canary cells (2 extreme tasks — `letter` and
`Devnagari-Script` — × 4 canary methods × 3 algorithms), the 42
non-canary extreme rows it would refuse, and the 1,815 + 423
standard / heavy rows already completed in Commits 48 / 49. The
plan summary lives at
`experiments/_stage_runs/`
`stage3_replica_002_extreme_lane_plan_latest_summary.{json,md}`,
with a longer narrative at
`docs/STAGE3_REPLICA002_EXTREME_PLAN.md`.

Commit 50 explicitly does **not** run training, create execution
SQLite under `runs/`, mutate any committed shard, regenerate
`heavy_task_policy.csv` or `runtime_guardrails.yaml`, change
`policy_version`, create a new signoff file, rerun the standard
or heavy lanes, execute the extreme lane, or scale to replicas
003 – 005. Operator review of the plan summary is required
before Commit 51 may execute the extreme lane under the
policy-defined `extreme.stage0_max_evaluations = 1` budget and
`extreme.timeout_seconds_per_cell = 14,400 s`.

## Stage-3 / top-up replica_002 extreme-lane execution (Commit 51)

Commit 51 runs the 24 extreme-lane canary cells planned in
Commit 50: all 10 source template shards, `replica = 2`,
extreme lane only, four canary methods × three algorithms, two
extreme tasks (`6 / letter` and `167121 / Devnagari-Script`).
The runner `scripts/run_stage3_replica002_extreme_lane.py`
chains **five** gates (signoff → Commit 46 plan → Commit 48
standard summary → Commit 49 heavy summary → Commit 50 extreme
plan summary), refuses real execution unless **both**
`--include-extreme-tasks` and `--execute-extreme-lane` flags
are passed, and uses the policy-defined extreme budget
(`stage0_max_evaluations = 1`, per-cell timeout 14,400 s). The
committed source shards remain byte-identical. The committed
summary lives at
`experiments/_stage_runs/`
`stage3_replica_002_extreme_lane_latest_summary.{json,md}`.

Commit 51 does **not** rerun the standard / heavy lanes, run
the full `topup_to_5` tier, touch replicas 003–005, change
`policy_version`, create a new signoff file, or commit any
execution SQLite / OpenML payload. Operator review of the
Commit 51 summary is required before any aggregate replica_002
review or signoff.

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
