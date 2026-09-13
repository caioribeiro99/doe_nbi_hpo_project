# NBI geometry audit: the article-track implementation on non-convex fronts

**Status:** complete, defect found and fixed. **Scope:** `src/doe_xgb/nbi_core.py` on the
article track. This is *not* a historical artifact; it is the current implementation that Paper 2's
canonical-NBI arms would use, and it was changed as a result of this audit.
**Tests:** `tests/methodology/test_nbi_geometry_validation.py` (12 tests).

---

## 1. Why this audit exists

`METHODOLOGICAL_IDENTITY_AUDIT.md` establishes that the dissertation optimizer is weighted-sum
scalarization. The single strongest reason to prefer Normal Boundary Intersection over weighted sum
is that weighted sum cannot reach a non-convex stretch of a Pareto front, whereas NBI can. Before
building a paper on that distinction, the implementation has to be shown to actually have the
property.

The existing suite (`tests/methodology/test_nbi_core.py`) exercises separable convex quadratics.
On a convex front weighted sum and NBI agree, so those tests cannot separate a correct NBI from a
weighted sum under another name. They also never run a weighted sum to compare against.

## 2. The test problem

On `x ∈ [0, 1]`, minimize

    f1(x) = x,    f2(x) = 1 − x²

Every `x` is Pareto optimal and the front is the curve `f2 = 1 − f1²`, whose second derivative is
`−2`. The attainable set is non-convex.

*Weighted sum.* `w·f1 + (1−w)·f2` has second derivative `−2(1−w) < 0`, so every interior stationary
point is a maximum and the minimum is at an endpoint. Weighted sum returns only the two anchors, for
any weight grid.

*NBI.* Anchors are `x = 0` and `x = 1`, so `utopia = (0, 0)` and `Φ = [[0, 1], [1, 0]]`. The
equality constraint fixes `f1 − f2 = β₂ − β₁`, giving a closed form:

    x*(β) = ( −1 + sqrt( 5 − 4(β₁ − β₂) ) ) / 2

which is interior for every interior `β`. A correct NBI must reproduce it.

## 3. The defect

`solve_nbi_subproblem` bounded the auxiliary variable:

```python
bounds = [(float(lo), float(hi)) for lo, hi in cfg.bounds] + [(0.0, None)]
```

Das and Dennis place **no sign restriction on `t`**. The restriction matters exactly on non-convex
fronts: a concave stretch lies on the *far* side of the CHIM from the utopia point, so reaching it
requires moving against the quasi-normal, that is `t < 0`. With `t ≥ 0` the equality constraint is
infeasible for every interior `β`.

Measured on the test problem, before the fix:

| β₁ | solver status | t | residual norm |
|---|---|---|---|
| 0.875 | failed | 0.0000 | 1.3e−01 |
| 0.750 | failed | 0.0000 | 1.7e−01 |
| 0.625 | failed | 0.0000 | 1.8e−01 |
| 0.500 | failed | 0.0000 | 1.7e−01 |
| 0.375 | failed | 0.0000 | 1.4e−01 |
| 0.250 | failed | 0.0000 | 1.0e−01 |
| 0.125 | failed | 0.0000 | 5.5e−02 |

Every interior subproblem failed, with `t` pinned at its lower bound and the equality constraint
violated. The anchors (β₁ ∈ {0, 1}) solved, because there `t = 0` is correct.

The failure is quiet in one important way: `run_nbi` still returns a candidate for each `β`, holding
SLSQP's least-infeasible iterate. On this one-dimensional problem those iterates happen to sit on
the front, so a caller reading only the returned decision vectors would see plausible output. The
`success` flag and the residual are the only signals, and a pipeline that does not check them would
report weighted-sum-like results as NBI results.

## 4. The diagnosis, confirmed

Re-solving the identical subproblems with `t` unbounded below:

| β₁ | x returned, t free | analytic x* | status |
|---|---|---|---|
| 0.875 | 0.20711 | 0.20711 | solved |
| 0.750 | 0.36603 | 0.36603 | solved |
| 0.625 | 0.50000 | 0.50000 | solved |
| 0.500 | 0.61803 | 0.61803 | solved |
| 0.375 | 0.72474 | 0.72474 | solved |
| 0.250 | 0.82288 | 0.82288 | solved |
| 0.125 | 0.91421 | 0.91421 | solved |

Every subproblem reaches the closed-form NBI solution to five decimal places.

## 5. The fix

`NBIConfig` gains `restrict_t_nonnegative: bool = False`, and the bound becomes

```python
t_lo = 0.0 if cfg.restrict_t_nonnegative else None
bounds = [(float(lo), float(hi)) for lo, hi in cfg.bounds] + [(t_lo, None)]
```

`t` is free by default, with the reason recorded at the declaration. The old behaviour is reachable
behind the flag, and one of the new tests asserts that it still fails on the non-convex case, so the
reason for the default cannot be lost.

All 22 pre-existing methodology tests continue to pass.

## 6. The tests added

`tests/methodology/test_nbi_geometry_validation.py`:

1. the test front is genuinely non-convex;
2. weighted sum returns only anchors, at five interior weights;
3. anchors, utopia, zero diagonal of `Φ` and the quasi-normal are as Das and Dennis define them;
4. NBI reproduces the closed-form solution to 1e−4 with residual below 1e−6;
5. at least seven of nine weights give strictly interior points;
6. `t` is negative at every interior point of a concave front;
7. the restricted solver provably fails the same case;
8. a three-objective non-convex problem on the decision simplex solves at every weight.

## 7. Consequence for Paper 2

- The canonical-NBI arms are only meaningful with the fix in place. Any result produced by the
  restricted solver on a non-convex front would be an artifact.
- Every NBI subproblem in the campaign must record `success` and `residual_norm`, and the protocol
  must report the certified fraction per arm. Silent least-infeasible iterates are the failure mode
  this audit found, and the only defence is to publish the certification rate.
- This is a defect in code written for the article track, found before any campaign was run. It is
  not a dissertation error and must not be reported as one.
