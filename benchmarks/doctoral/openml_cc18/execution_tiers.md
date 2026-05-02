# Execution-tier policy — OpenML-CC18 doctoral campaign

This document is the narrative version of
`benchmarks/doctoral/openml_cc18/execution_policy.csv`. Both must
agree; if they disagree, the CSV wins.

The shard generator (planned next commit) reads `method_matrix.csv`
*and* `execution_policy.csv` to decide:

- which methods enter the SQLite job matrix at all;
- which methods run on the full 72 CC18 tasks vs the ParEGO subset
  (`parego_subset.csv`);
- which methods materialize jobs at every stage vs only stage 0;
- whether stage 3 (top-up to 30 replicas) requires a manual
  sign-off before being unlocked.

No method list is hardcoded in the shard generator.

## Why tiers

The doctoral campaign multiplies cost across (task × algorithm ×
method × replica). With 72 tasks × 3 algorithms × 30 replicas, every
new full-CC18 method adds ~6,480 jobs. The single-method
provisional estimate (CC18 72 × 3 × 30 ≈ 18.3 dedicated-Mac days at
efficiency 0.85) becomes prohibitive once multiplied by the headline
method count. The execution-tier policy decouples *what is in the
benchmark* from *how aggressively each method is run*, so we can:

- run every primary baseline at stage 0 to validate plumbing;
- gate the stage-1/2/3 top-ups behind a cost sign-off that uses
  real stage-0 wall-clock data;
- treat ablations and subset-only methods as second-priority work
  that fills the dedicated Mac when the headline methods are idle.

## Tier definitions

### Tier 0 — controls / cheap baselines

`default_gbdt`, `random_search`.

These cost less than a full HPO run per (task, algorithm) cell;
`default_gbdt` is one fit per CV split with no search. They run on
the full 72 tasks at every stage with no manual sign-off; they are
the cheapest way to anchor the comparative table.

### Tier 1 — full-CC18 primary baselines (single-objective)

`tpe_optuna`, `smac3`.

Standard fixed-evaluation HPO with `B = 138` configurations per
replica. These are the headline single-objective baselines. Run
full 72 tasks; stage 0 unconditional; stage 1/2 unconditional; stage
3 (top-up to 30 replicas) **requires a manual sign-off** that uses
the real stage-2 wall-clock data to project the stage-3 cost.

### Tier 2 — expensive primary methods (multi-fidelity + multi-objective)

`asha`, `bohb`, `dehb`, `nsga2`, `motpe`, `doe_rsm_vrf_true_nbi`.

Multi-fidelity methods spend `B × max_iter` total boosting
iterations per replica per cell. Multi-objective methods spend the
same `B = 138` evaluations as the single-objective tier-1 methods
but materialize a Pareto front rather than a single incumbent. The
proposed method `doe_rsm_vrf_true_nbi` is in this tier because it
is the headline, not because it is more expensive than tier-1
baselines.

Run full 72 tasks; stage 0 unconditional; stage 1/2 unconditional;
stage 3 **requires a manual sign-off**. The sign-off is one decision
that covers all tier-1 + tier-2 methods — no per-method gating, just
a single "scale to 30 replicas yes/no" call after stage 2.

### Tier 3 — subset-only methods

`parego`.

Runs on the ParEGO subset (`parego_subset.csv`, 48 of 72 tasks)
only. Stage 0 runs on the full ParEGO subset; stage 1/2/3 follow
the same staged top-up. Stage 3 requires the same manual sign-off as
tier 2.

### Tier 4 — ablations

`doe_rsm_vrf_true_nbi_no_mbpa`,
`legacy_weighted_sum_scalarization`.

Ablations are second-priority work. They join the campaign at
**stage 2** (top-up to 10 replicas) — not earlier. Stage 3 requires
manual sign-off. If the dedicated Mac is saturated by tier 0–2,
ablations are queued; if a ablation cannot complete its stage-3
target before the publication deadline, it is reported at the
maximum stage that completed (stage 2 = 10 replicas) rather than
blocking the headline.

### Tier ∞ — literature-only (excluded from jobs)

`flaml_optional`, `auto_sklearn_context`, `autogluon_context`.

Cited in the manuscript as context. Never enter the job matrix.
The shard generator must **not** materialize jobs for any row of
`method_matrix.csv` whose `primary_or_ablation` is `literature_only`.

## Per-stage participation summary

| Tier | Methods | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|---|
| 0 | `default_gbdt`, `random_search` | yes | yes | yes | yes |
| 1 | `tpe_optuna`, `smac3` | yes | yes | yes | sign-off |
| 2 | `asha`, `bohb`, `dehb`, `nsga2`, `motpe`, `doe_rsm_vrf_true_nbi` | yes | yes | yes | sign-off |
| 3 | `parego` (ParEGO subset only) | yes | yes | yes | sign-off |
| 4 | `doe_rsm_vrf_true_nbi_no_mbpa`, `legacy_weighted_sum_scalarization` | no | no | yes | sign-off |
| ∞ | `flaml_optional`, `auto_sklearn_context`, `autogluon_context` | no | no | no | no |

## Manual sign-off rule

Stage 3 is the only stage that requires a sign-off. The shard
generator marks stage-3 jobs `pending` but with a flag
`requires_manual_signoff_before_stage3=true` in `cc18_jobs.notes`,
which the orchestrator must clear before claiming. Tier 0 jobs
unlock stage 3 unconditionally because they are cheap.

The sign-off is recorded as a small JSON file
(`jobs/doctoral/openml_cc18/stage3_signoff.json`, planned) that
records the decision, the user, the date, the projected wall-clock
under each efficiency scenario, and the actually observed
stage-2 wall-clock that informed the call.

## Why this is not in the manuscript

The execution-tier policy is operational, not methodological. The
manuscript reports results for whatever stage actually completed;
the staged top-up is mentioned as the cadence, but the per-tier
gating is engineering hygiene, not a result. The article-track
manuscript will cite this document at most as a one-line aside in
the experimental-design section.

## Frozen versus open

**Frozen (Commit 27):**

- The five tiers above and the literature-only exclusion.
- The stage-2 entry of ablations.
- The per-tier stage-3 sign-off requirement.

**Open (resolved at orchestrator time, not now):**

- The exact wording / format of the stage-3 sign-off JSON.
- Whether the sign-off can be partial (e.g., approve stage 3 for
  tier 0–2 but not tier 4).
- Whether the sign-off can be retracted mid-campaign.

These are operational details that will be settled by the runner
script in a later commit; they do not affect job-matrix generation.
