# Baseline gap assessment

**Question.** Which comparators must the campaign run for the result to be believed, and does that
include an evolutionary multiobjective optimizer?

**Why it is asked now.** Paper 1's most consequential reviewer objection was the absence of an
evolutionary baseline, and adding one changed what the paper could claim: NSGA-II at a matched
real-evaluation budget beat the metamodel-free arm on both endpoints, on all four datasets, under
both reference definitions. Discovering that after the campaign cost a second experiment. The
decision is therefore taken before the freeze, not after.

---

## What the dissertation compared against

Coarse grid search, random search, Bayesian optimization via `scikit-optimize`, and the
tree-structured Parzen estimator via `hyperopt`, each at `88 + n_candidates = 108` evaluations.

Two structural problems, both from `protocol/original_thesis_protocol.md`:

1. **Every method, including the proposed multiobjective one, was scored by `max(Accuracy_Mean)`.**
   A single-objective rule applied to a Pareto front discards the trade-off the front exists to
   show, and it flatters whichever method happens to contain the single most accurate point.
2. **The comparators are single-objective.** Grid, random, Bayesian optimization and the Parzen
   estimator as configured all optimize one scalar. Comparing a Pareto front against a single-objective
   optimizer's best point is not a like-for-like comparison in either direction.

Both must be fixed regardless of what else is added.

## The gap against the current literature

`literature_review.md` Cluster D establishes that multiobjective hyperparameter optimization is an
active field with its own baseline conventions (Guerrero-Viu et al. 2021) and benchmark suites
(HPOBench, YAHPO Gym). Against those conventions the dissertation's comparator set is missing:

| Missing | Consequence if still missing at submission |
|---|---|
| any **multiobjective** comparator | the central claim has no like-for-like competitor; this is fatal |
| any **multi-fidelity** method (Hyperband, BOHB) | a reviewer asks why a method that spends 108 full evaluations is compared only against methods that also do, when the field's answer to expensive evaluation is to not pay full price |
| **random search reported as a Pareto front** | random search is the field's floor; if it is not beaten as a front, nothing else matters |

## Recommendation on the evolutionary baseline

**LIKELY REQUIRED.** Not merely useful.

The reasoning is specific to this paper rather than borrowed from Paper 1:

1. The proposed method's output **is** a Pareto front. NSGA-II's output is a Pareto front. It is the
   only comparator in common use whose output type matches, so it is the only one that can be scored
   by the same indicators without an argument about what is being compared.
2. Paper 1 already reports NSGA-II beating the metamodel-free NBI arm at a matched budget, on a
   different problem, and that result is in the authors' own frozen manuscript. A reviewer who finds
   it will ask why the follow-up did not run the comparison the authors themselves showed to matter.
   Omitting it would be a knowing omission.
3. The evaluation-matched machinery already exists and is tested: `src/mixens/nsga2_baseline.py` on
   the Paper-1 branch, with a deterministic simplex repair, a budget-matching helper whose
   off-by-one was found and fixed, and 15 unit tests. The decision space differs (a hyperparameter
   box rather than a simplex), so the repair is replaced by box clipping with integer rounding, but
   the budget arithmetic and the harness carry over.

**Against running it**, honestly stated: NSGA-II at 108 evaluations on a seven-dimensional problem
is a small budget for a population method, and it may perform poorly for that reason rather than
for any interesting one. The protocol must therefore report the population size and generation count
explicitly and must not present a budget-starved NSGA-II as evidence that evolutionary methods do
not work here. If the matched budget gives fewer than roughly four generations, that fact is
reported next to the result.

## The comparator set to freeze

| Comparator | Output | Budget | Role |
|---|---|---|---|
| Random search | front over all sampled points | `B_total_solution` of the most expensive arm | the floor; if the proposed method does not beat it, nothing else is worth reporting |
| NSGA-II | Pareto front | same, evaluation-matched, realized ratio reported | the like-for-like multiobjective competitor |
| Bayesian optimization, single objective per objective | two fronts' worth of endpoints | same, split across objectives and reported as split | continuity with the dissertation's comparator set |
| Tree-structured Parzen estimator | as above | same | continuity |
| Coarse grid | front over grid points | same | continuity; expected to be weak, reported anyway |

Multi-fidelity methods (Hyperband, BOHB) are **not** in the frozen set. They change the cost model
rather than the search, and folding them into an evaluation-matched comparison requires deciding
what a partial-fidelity evaluation costs, which is a separate design question. The manuscript states
this as a scoping decision and names it as the obvious next comparison, rather than leaving a
reviewer to notice the absence.

## Scoring, fixed here

- The full front is persisted for every method and every replication.
- Primary endpoints are computed against an empirical reference built independently of any single
  method, and repeated against a sampled core that the compared methods do not contribute to. Paper 1
  established that a self-graded reference is a real objection and that the fix is cheap.
- `max(Accuracy_Mean)` is reported **only** as a legacy column for continuity with the dissertation,
  labelled as such, and is never a primary endpoint.
