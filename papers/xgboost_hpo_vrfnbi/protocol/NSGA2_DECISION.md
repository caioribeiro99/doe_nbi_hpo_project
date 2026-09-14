# NSGA-II: decision for the confirmatory campaign

**Decision: USEFUL, and included — as a clearly labelled secondary external-context baseline at a
matched logical solution-producing budget. It is not a primary causal contrast and its result can
never replace one.**

Settled before the protocol freeze, with no arm-level result in existence.

---

## 1. The question

The mechanistic study already carries eight methods: four arms (HISTORICAL-WS, WS-S, NBI-S, NBI-R)
and four direct baselines (coarse grid, random search, Bayesian optimization, tree-structured Parzen
estimator). The two primary contrasts, NBI-S against WS-S and NBI-R against NBI-S, are internal to
the arm set and do not need NSGA-II to be well posed.

So the question is narrow: does the **intended venue** require an evolutionary multiobjective
comparator for the result to be believed?

## 2. Assessment

**Against including it.** The paper is a mechanistic decomposition, not a horse race. Its claims are
about which of three bundled choices the outcome depends on, and NSGA-II answers none of them. Every
comparator added enlarges the results surface and invites a reading the paper is not making.

**For including it, and this is decisive.** Three reasons, in order:

1. **Output type.** The arms return Pareto fronts. Of the eight methods already present, none of the
   four baselines does: grid and random search return sets scored post hoc, and Bayesian optimization
   and the Parzen estimator as the dissertation configured them return single-objective optima.
   NSGA-II is the only widely used comparator whose **output type matches the arms'**, so it is the
   only one that can be scored by the same indicators without an argument about what is being
   compared.
2. **The authors' own prior result.** Paper 1 reports NSGA-II beating the metamodel-free NBI arm at a
   matched real-evaluation budget, on all four of its datasets, under both reference definitions.
   That manuscript is frozen at `paper-submission-v2`. A reviewer who finds it will ask why the
   follow-up omitted the comparison its own authors showed to matter. Omitting it would be a knowing
   omission.
3. **Venue.** `literature_review.md` Cluster D establishes that multiobjective hyperparameter
   optimization has an agreed baseline convention (Guerrero-Viu et al. 2021) and dedicated benchmark
   suites. Engineering Applications of Artificial Intelligence and Applied Soft Computing both draw
   reviewers from that community.

**Not LIKELY REQUIRED**, which is the classification Paper 1 gave it, because there the
metamodel-free arm's standing rested on being the best available approximation and NSGA-II directly
contested that. Here the arms are compared against **each other**, so no arm's standing depends on
NSGA-II's result.

## 3. Frozen configuration

One canonical configuration, committed before execution, **not tuned**:

| Setting | Value |
|---|---|
| Implementation | `pymoo` 0.6.2, `NSGA2` |
| Population size | **32** |
| Generations | **12** |
| Real evaluations consumed | 32 × 12 = **384** |
| Crossover | simulated binary, `eta = 15`, probability 0.9 (pymoo default for real-valued problems) |
| Mutation | polynomial, `eta = 20`, probability 1/7 |
| Selection | binary tournament on rank and crowding distance, pymoo default |
| Duplicate elimination | on, pymoo default |
| Seed | the replication seed, so it is paired with every other method |
| Decision space | the same coded `[−1, 1]⁷` box as the arms |
| Realization | the same deterministic `Realizer`: clip, map to natural units, round integers half away from zero, clip again |
| Objectives | **the same two frozen objectives**, computed by the same frozen factor transformation (`OBJECTIVE_DEFINITIONS.md` §7) |
| Evaluator | **the same** stratified 5-fold internal resampling evaluator as every other method, through its own isolated `MethodView` |

No hyperparameter of NSGA-II is tuned, at any point, by anything.

## 4. Budget matching, stated exactly

The comparison is at **matched logical solution-producing real-evaluation budget**.

| Method | logical solution-producing budget |
|---|---:|
| HISTORICAL-WS | 108 |
| WS-S | 186 |
| NBI-S | 186 |
| NBI-R | **386** |
| grid, random, Bayesian optimization, Parzen estimator | 386 each |
| **NSGA-II** | **384** |

The comparator budget is the `B_total_solution` of the most expensive arm, 386. NSGA-II gets 384,
because 386 = 2 × 193 admits no factorization into a population and at least ten generations. **The
shortfall is 2 evaluations, 0.5%, it runs against NSGA-II, and it is reported.**

Three properties of this matching, all stated rather than assumed:

- **The 78-run external validation set is audit-only and is charged to no comparator.** It is not
  solution-producing information and giving a direct optimizer access to it would be a leak, not a
  budget adjustment.
- **Physical cache reuse does not reduce NSGA-II's logical budget.** Every evaluation NSGA-II
  requests is charged to NSGA-II whether or not another method already computed it, and NSGA-II sees
  only its own history.
- **The cheaper arms are compared against comparators given up to 3.6× their budget.** That is the
  conservative direction for the paper's own claims and is reported as such.

## 5. The unmatched run, and what it may not be called

NSGA-II is additionally run at **ten times** the matched budget, 3,840 evaluations, on **one
replication per dataset**. Measured cost: about 6 hours; on all 30 replications it would be about 180
hours.

**This is a clearly labelled secondary, unmatched-context run.** It exists to answer one question:
whether a weak matched result is explained by budget starvation. At 384 evaluations on a
seven-dimensional problem NSGA-II gets 12 generations, which is a real evolutionary run but a short
one, and a reviewer is entitled to ask.

It is **not** an efficiency comparison and is never described as one. No claim about any arm is made
relative to it. It is reported in its own table, on one replication per dataset, with the generation
count stated beside it.

## 6. What NSGA-II does not become

The two primary scientific contrasts are **NBI-S against WS-S** and **NBI-R against NBI-S**,
regardless of what NSGA-II does. No NSGA-II result may replace, reframe or substitute for either, and
no conclusion about the geometry or anchor-provenance factors may be drawn from it.

NSGA-II is one canonical configuration of one evolutionary algorithm. The campaign runs no others:
there is no evolutionary zoo, and a second multiobjective evolutionary method is not added after
seeing this one's result.
