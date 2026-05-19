# Stage-3 / top-up manifest

This manifest is the **per-tier per-lane** rollup of the OpenML-CC18
top-up plan. One row per (tier × lane) execution unit; the CSV
sibling (`stage3_topup_manifest.csv`) is the machine-readable form.

The manifest is **planning-only** (Commit 46). It does not authorize
dispatch. Each row describes the work that would be required for the
corresponding (tier, lane) pair under the **current**
`policy_version = 47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36`
and the Commit-45 signed
`stage0_replica_001` baseline.

## Schema

| column | description |
|---|---|
| `topup_tier` | one of `topup_to_5`, `topup_to_10`, `topup_to_30` |
| `replica_start` | first replica id in this tier (inclusive, 1-indexed) |
| `replica_end` | last replica id in this tier (inclusive) |
| `replica_count` | replicas added by this tier (= `replica_end - replica_start + 1`) |
| `lane` | `standard`, `heavy`, or `extreme` |
| `source_stage` | shard subdir under `jobs/doctoral/openml_cc18/shards/` |
| `task_count` | number of CC18 tasks in this lane (57 / 13 / 2) |
| `executable_canary_cell_count` | `task_count × 4 canary methods × 3 algorithms × replica_count` |
| `deferred_or_refused_count_estimate` | non-canary rows the runner will refuse (`stub_only` methods + ParEGO subset, etc.) |
| `recommended_worker_type` | `dedicated_mac`, `dedicated_mac_or_university`, `university_mac_preferred` |
| `estimated_runtime_seconds_p50` | per-replica observed total × replica_count |
| `estimated_runtime_seconds_p90` | per-cell observed p90 × cells (loose upper bound) |
| `estimated_runtime_seconds_max` | per-cell observed max × cells (worst-case bound) |
| `can_run_on_personal_mac` | `yes`, `no`, `partial` (partial = some shards but not the long-pole ones) |
| `can_run_on_dedicated_mac` | `yes`, `no`, `partial` |
| `requires_manual_review` | `yes` if a per-tier operator review is required before dispatch |
| `notes` | free text |

## Rationale per (tier, lane)

- **standard tiers.** Standard lanes scale roughly linearly with
  replica count. The R=1 observation of ~1.89 h of runner CPU per
  replica suggests `topup_to_5` standard finishes in ~8 h on a
  dedicated Mac; `topup_to_30` standard runs out to ~38 h. These
  cells fit on either the dedicated Mac or the personal Mac
  (partial). `mnist_784` and `Bioresponse` rows belong to the
  heavy lane and are not part of these standard numbers.
- **heavy tiers.** Heavy lanes are dominated by `CIFAR_10`,
  `Fashion-MNIST`, `mnist_784`, and `Internet-Advertisements`.
  The R=1 observed total was ~9.69 h per replica. Heavy `topup_to_30`
  exceeds the dedicated Mac's reasonable wall-clock and should run
  on the future university Mac Pro Max if available.
- **extreme tiers.** `Devnagari-Script` (task 167121) and `letter`
  (task 6) sit here. Devnagari is the dominant cost (~10 ks per
  `doe_rsm_vrf_true_nbi` cell at R=1). The Commit-45 caveat
  `devnagari_extreme_budget_non_equivalence` notes that
  `stage0_max_evaluations = 1` was used for Devnagari at R=1.
  Each new replica adds the same asymmetry under Option A
  (`docs/STAGE3_POLICY_DECISION.md`).

## Worst-case bounds

The `estimated_runtime_seconds_max` column uses the worst per-cell
runtime observed in `stage0_replica_001` times the cell count in
that lane × tier. This is a **very** conservative upper bound: in
practice only a handful of cells per task hit the per-cell max
(typically the `doe_rsm_vrf_true_nbi` × catboost cell on the
largest dataset in the lane). Use the `_p50` column as the
operational planning number.

## How to consume this manifest

1. Read `stage3_topup_manifest.csv` programmatically.
2. Cross-check against
   `experiments/_stage_runs/stage3_topup_plan_latest_summary.json`
   (which records the same numbers under `tier_plans[].lanes`).
3. Pair each (tier, lane) row with the matching
   `stage3_worker_plan.csv` row to identify the worker.
4. Run the per-tier pilot (Commit 47 onward) before scheduling the
   full tier dispatch.
