# Projected job counts and wall-clock — frozen comparative protocol

Computed at protocol-freeze time (Commit 27) from:

- `benchmarks/doctoral/openml_cc18/method_matrix.csv` (16 methods)
- `benchmarks/doctoral/openml_cc18/execution_policy.csv` (per-method
  per-stage gating)
- `benchmarks/doctoral/openml_cc18/parego_subset.csv` (48 ParEGO
  tasks)
- `benchmarks/doctoral/openml_cc18/tasks.csv` (72 CC18 tasks)

The shard generator (next commit) will write these counts as
`shard_meta.notes` so the SQLite shards self-describe their scope.

## Method counts by category

| category | count | method_ids |
|---|---:|---|
| full_cc18 (run on all 72 tasks) | 12 | `default_gbdt`, `random_search`, `tpe_optuna`, `smac3`, `asha`, `bohb`, `dehb`, `nsga2`, `motpe`, `doe_rsm_vrf_true_nbi`, `doe_rsm_vrf_true_nbi_no_mbpa`, `legacy_weighted_sum_scalarization` |
| subset_only (ParEGO subset = 48 tasks) | 1 | `parego` |
| literature_only (excluded from jobs) | 3 | `flaml_optional`, `auto_sklearn_context`, `autogluon_context` |
| ablation (joins at stage 2) | 2 | `doe_rsm_vrf_true_nbi_no_mbpa`, `legacy_weighted_sum_scalarization` |

## Job-count projection (jobs = task × algorithm × method × replica)

Computed from the execution-policy gating in
`execution_policy.csv`. Tier 4 ablations join at stage 2; tier ∞
literature-only methods never enter the job matrix.

| Stage | Jobs added | Cumulative jobs |
|---|---:|---:|
| `stage0_replica_001`     |  2,304 |  2,304 |
| `stage1_topup_to_005`    |  9,216 | 11,520 |
| `stage2_topup_to_010`    | 13,680 | 25,200 |
| `stage3_topup_to_030`    | 54,720 | **79,920** |

Stage 0 includes 10 methods × 3 algorithms × 1 replica × 72 tasks
plus 1 ParEGO method × 3 algorithms × 1 replica × 48 tasks. Stage 2
adds 2 ablation methods × 3 × 5 × 72 on top of the staged top-ups
for the rest. Stage 3 multiplies all active methods by 20 additional
replicas; this is the dominant cost.

## Wall-clock projection (dedicated MacBook Pro)

Anchored on the same per-cell rate used in `docs/COST_ESTIMATOR.md`
(0.75 s mean per-pair × 690 evaluations × 4× inflation ≈ 0.575
CPU-h per (task, algorithm, method, replica) for HPO methods).
`default_gbdt` is weighted at 1/690 of that rate (it does no search:
one fit per CV split per replica). Multi-fidelity methods retain the
nominal 1.0× weight because their `B × max_iter` total boosting-
iteration budget is matched to the fixed-evaluation budget, not added
on top.

| Stage cumulative | CPU-h | Eff 0.75 | Eff 0.85 | Eff 0.90 |
|---|---:|---:|---:|---:|
| through stage 0      |  1,201 |   6.7 d |   5.9 d |   5.6 d |
| through stage 1      |  6,004 |  33.4 d |  29.4 d |  27.8 d |
| through stage 2      | 13,250 |  73.6 d |  65.0 d |  61.3 d |
| through stage 3      | **42,233** | **234.6 d** | **207.0 d** | **195.5 d** |

The dedicated-Mac daily-CPU-hours assumption is 10 workers × 24 h ×
efficiency, matching `dedicated_mac_profile()` in the cost
estimator.

## Warning — replica multiplication

A campaign with **12 full-CC18 methods at 30 replicas** is
**~6.8 months of dedicated-Mac wall-clock at efficiency 0.85**.
This is well outside the original "single-method 18.3 days"
projection in `docs/COST_ESTIMATOR.md`, which referred to the
proposed method only. The execution-tier policy
(`execution_tiers.md`) addresses this by:

- gating stage 3 (top-up to 30 replicas) behind a manual sign-off
  for every tier 1+ method, so the campaign can ship a stage-2
  (10-replica) snapshot if the stage-3 cost is judged unacceptable;
- splitting expensive primary methods into tier 2 explicitly, so
  the runner can serialize them rather than running them in parallel
  across the dedicated Mac.

A stage-2-only campaign is **65.0 days at efficiency 0.85**, which
is an acceptable doctoral-budget envelope. The decision of whether
to top up to stage 3 is intentionally deferred to after stage 2
finishes and a real wall-clock measurement replaces the projection
above.

## Sanity-check rules (enforced by tests)

- `full_cc18 + subset_only + literature_only = total methods`
  (mutually exclusive partition).
- `parego_subset.csv` has 48 rows.
- Stage 0 cumulative = 2,304 jobs.
- Stage 3 cumulative = 79,920 jobs.
- Every method with `primary_or_ablation != literature_only` has at
  least one `stage*=true` in `execution_policy.csv`.
- No method has both `full_cc18=true` and `subset_only=true`.

These invariants are enforced by `tests/unit/test_openml_cc18_benchmark.py`
so a future edit to either CSV cannot silently break the protocol.
