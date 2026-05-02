# Comparative protocol — doctoral benchmark

This document is the authoritative specification of *which methods*
the proposed DOE + RSM + VRF + true N-objective NBI + conditional MBPA
pipeline is compared against, on which subset of the OpenML-CC18
suite, and under what budget-equivalence rule. The machine-readable
source of truth is

```
benchmarks/doctoral/openml_cc18/method_matrix.csv
```

The shard generator that builds the SQLite job matrix (planned for
the next commit) MUST read its method list from that CSV and MUST NOT
hardcode method names.

This commit (Commit 26) produces only the protocol; it does not
generate any shards, run any benchmark, or download any dataset
payloads.

## Method families

| family | included? | source of methods |
|---|---|---|
| classical HPO | full | random search, Optuna TPE |
| SMAC family | full | SMAC3 (RF surrogate, intensification) |
| multi-fidelity | full | ASHA-or-Hyperband, BOHB, DEHB |
| evolutionary / DFO | full (via NSGA-II) | NSGA-II as multi-objective representative |
| multi-objective | full + subset | NSGA-II, MOTPE; ParEGO on subset only |
| AutoML systems | literature-only | Auto-sklearn, FLAML, AutoGluon (cited, not benchmarked) |
| proposed + ablations | full | DOE+RSM+VRF+true NBI; same with MBPA off; legacy weighted-sum |

## Method status (frozen by `method_matrix.csv`)

Every row of `method_matrix.csv` carries:

- `method_id` — string used as the SQLite job matrix `method` value.
- `method_family` — coarse family label.
- `primary_or_ablation` — one of `primary`, `subset`, `ablation`,
  `literature_only`.
- `objective_mode` — `single_objective` or `multi_objective`.
- `implementation` — short tag of the concrete implementation choice.
- `package` — Python package(s) the method depends on.
- `full_cc18` — `true` iff the method runs on all 72 CC18 tasks.
- `subset_only` — `true` iff the method runs only on a defined subset
  of CC18 tasks (the subset definition lives in
  `benchmarks/doctoral/openml_cc18/comparative_protocol.md`).
- `budget_unit` — `evaluations`, `fidelity_units`, or `seconds`.
- `budget_equivalence_rule` — the formal mapping from the headline
  budget `B` to the method's native budget unit (see below).
- `supports_multiclass` — `true` iff the method supports multiclass
  classification natively without re-encoding the target.
- `supports_categorical_native` — `true` if the underlying GBDT
  library handles categorical features natively, otherwise the
  evaluator passes label-encoded ints.
- `notes` — free text, including TODO markers.

## Budget equivalence rule

Single-objective baselines (`random_search`, `tpe_optuna`, `smac3`,
`nsga2`, `motpe`, `parego`, the proposed method, both ablations) all
get the same headline budget per replica:

```
B = DOE_RUNS + NBI_EVAL_K = 88 + 50 = 138 configurations
```

Each configuration is evaluated under a stratified 5-fold CV (or the
OpenML task-defined folds if available), so every method spends the
same number of fold-fits per replica. This matches the dissertation
default and is the same `B` used by the legacy weighted-sum ablation.

Multi-fidelity methods (`hyperband_or_asha`, `bohb`, `dehb`) cannot
be capped by configuration count without distorting their schedule.
Their fairness rule is

```
total_n_estimators_budget = B * max_iter
```

where `max_iter` is the per-algorithm maximum boosting-iteration count
fixed in `configs/`. This makes the *total cost of boosting work* the
shared resource between low-fidelity and full-fidelity methods.

For evolutionary methods (currently only `nsga2`), we choose the
population size and number of generations so `pop_size * gens = B`,
which keeps the total number of fitness evaluations equal to the
headline budget.

AutoML systems are listed for context only. If FLAML is added to the
benchmark, its budget unit is wall-clock seconds; the equivalence rule
is `time_budget = walltime(random_search_at_B)`. This is a TODO before
protocol freeze.

## Per-stage method participation

The staged execution rolls out methods in lock-step with replicas:

| Stage | Replicas added | Methods included |
|---|---|---|
| `stage0_replica_001` | 1 | every `primary` method (subset methods skipped) |
| `stage1_topup_to_005` | 4 | every `primary` method, plus `subset` methods on the defined CC18 subset |
| `stage2_topup_to_010` | 5 | every `primary` and `subset` method, plus `ablation` methods on the same coverage as the proposed method |
| `stage3_topup_to_030` | 20 | every method except `literature_only` |

This means stage 0 is a quick smoke / sizing pass for every primary
baseline; stage 1 brings in subset-only methods (currently ParEGO);
stage 2 unlocks the ablations; stage 3 fills out the full 30 replicas
for the headline tables.

## Multiclass and categorical handling

Every CC18 task whose `task_type` is `multiclass` flows through the
same evaluator that the article-track end-to-end smoke validated; the
binary guardrail in `validate_task_metric_compatibility` is enforced
per task before stage 1 of FA / NBI.

Categorical features are passed natively to CatBoost. For XGBoost and
LightGBM, categoricals are label-encoded by the evaluator
(both libraries have native support, but to keep the comparison
honest across HPO baselines that do not understand categorical
columns, encoded ints are the default; native categorical support is
toggleable per algorithm and is logged in the per-replica manifest).

## What this protocol freezes

- The list of methods participating in the CC18 doctoral benchmark.
- The objective mode of each method (single- vs multi-objective).
- The budget equivalence rule.
- Which methods run on the full 72 tasks vs the defined subset.
- The stage at which each method enters the campaign.

## What this protocol leaves open (TODO before shard freeze)

1. Decide whether FLAML is included as a single-algorithm GBDT
   baseline or kept as literature-only.
2. Pick exactly one of `hyperband_or_asha` for the multi-fidelity
   slot — both are listed but only one will run, to control the
   benchmark blast radius. ASHA is recommended for parallelism on the
   dedicated Mac.
3. Choose the ParEGO subset definition: imbalanced + multiclass +
   categorical-heavy tasks (recommended) or a fully random sample of
   ~12–18 tasks.
4. Settle the verified DOI / pages / venue for any reference still
   carrying a `TODO` marker in `article/references.bib`.

## What this commit deliberately does NOT do

- Does NOT generate any SQLite shard files.
- Does NOT run any HPO baseline.
- Does NOT download any CC18 dataset payload.
- Does NOT change core implementation modules.
- Does NOT alter `main`.

The next operational commit (Commit 27) is `scripts/generate_cc18_job_shards.py`,
which reads `method_matrix.csv` to build the per-shard job rows. Until
that commit lands, the SQLite job matrix is intentionally empty.
