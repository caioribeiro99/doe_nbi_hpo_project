# Methodological identity audit: what the dissertation optimizer actually is

**Status:** complete. **Scope:** the optimizer named "NBI" in the master's dissertation and in the
code frozen at tag `v0.1.0-dissertation` (commit `67d9fe5`).
**Rule observed throughout:** no historical artifact was edited. The frozen code was extracted to a
scratch directory and read there.

---

## 1. The question

The dissertation presents a Normal Boundary Intersection method. Before any Paper-2 experiment is
designed, we need to know whether the code that produced the dissertation's numbers implements NBI.
The answer determines what a follow-up study can honestly compare against, and it determines what
the follow-up's control arms have to be.

## 2. Evidence

### 2.1 What the frozen optimizer computes

`src/doe_xgb/nbi.py` at `v0.1.0-dissertation` exposes one solver, `run_nbi_weighted_sum`. Its inner
objective, in full:

```python
nadir  = np.array(observed_nadir,  dtype=float)
utopia = np.array(observed_utopia, dtype=float)
_, norm = preds_and_norm(x_vec, nadir, utopia)
return -float(np.dot(betas_arr, norm))
```

with

```python
denom = np.where(np.abs(utopia - nadir) < 1e-12, 1.0, (utopia - nadir))
norm  = (preds - nadir) / denom
```

This is min–max normalized **weighted-sum scalarization**: for each weight pair it maximizes
`Σ_j β_j · f̄_j(x)` by SLSQP from ten multistarts, over a grid of twenty β pairs at step 0.05.

An optional inequality keeps both predictions inside the observed `[nadir, utopia]` box. That is a
feasibility restriction on the prediction range, not a scalarization structure.

### 2.2 What is absent

Every structural element of NBI is missing from the frozen solver:

| NBI element (Das and Dennis, 1998) | Present in `run_nbi_weighted_sum`? |
|---|---|
| Per-objective anchor minimizers `x_j*` | No |
| Payoff matrix `Φ` with columns `F_j* − utopia` | No |
| Convex hull of individual minima (CHIM) | No |
| Quasi-normal direction `n̂ = −Φ·1 / ‖Φ·1‖` | No |
| Subproblem `max t s.t. Φβ + t·n̂ = F(x)` | No |
| An auxiliary variable `t` at all | No |

Searching the frozen module for `Phi`, `chim`, `n_hat`, `anchor`, `payoff` and `max t` returns
nothing. `β` in this code indexes a weight grid, not a CHIM coordinate.

### 2.3 Where the reference point comes from

`scripts/run_nbi.py` at the same tag:

```python
utopia = (float(df["Score_Quality"].max()), float(df["Score_Cost"].max()))
nadir  = (float(df["Score_Quality"].min()), float(df["Score_Cost"].min()))
```

Both ends of the normalization box are **observed extremes of the design rows**, taken
component-wise. They are not the values of each objective at the other objective's minimizer, which
is what a payoff matrix supplies. This matters for Paper 2 because it means the historical method
differs from a canonical NBI in *two* independent respects, not one: the scalarization structure and
the reference construction.

### 2.4 The discrepancy was already documented by the author

`docs/METHODOLOGY_DECISIONS.md`, entry **D1**, on the `repo-publication-readiness` branch, records
it directly: the dissertation text describes `max t s.t. F(x) = F^c(β) + t·n` with anchors, a CHIM
matrix and a quasi-normal, while the dissertation code "implements a *normalized weighted-sum
scalarization* ... No anchors, no `Φ`, no `n`. This is *not* NBI."

The same entry states the remedy already taken: true NBI in `src/doe_xgb/nbi_core.py`, the legacy
routine preserved under `src/doe_xgb/scalarization.py` and "**never** referred to as NBI", and the
old import path kept as a deprecation shim.

This audit therefore **confirms** an existing internal record rather than discovering something new.
That is the honest framing, and Paper 2 must use it.

## 3. Verdict

The dissertation's optimizer is normalized weighted-sum scalarization over a twenty-point weight
grid, with an observed-extremes normalization box. It is not Normal Boundary Intersection.

## 4. What follows for Paper 2

### 4.1 What may and may not be written

Permitted, because it is supported by the code and by the author's own log:

- "The dissertation implementation optimizes a normalized weighted-sum scalarization."
- "Pereira et al. (2025) formulate the method using canonical NBI." *(a statement about the
  published formulation)*

Not permitted:

- "Their code implements canonical NBI." We have not audited the EAAI 2025 implementation and make
  no claim about it.
- Any retroactive relabelling of the weighted-sum results as NBI results.
- Any sensationalized framing. The discrepancy is a documented divergence between a text and an
  implementation, of a kind common in research code, and the author recorded it before we looked.

### 4.2 Consequence for the experimental design

A two-arm comparison of HISTORICAL-WS against a canonical NBI would confound two changes at once:

1. the scalarization geometry (weighted sum versus CHIM-and-quasi-normal), and
2. the normalization and reference construction (observed component-wise extremes versus a payoff
   matrix and its utopia).

Isolating the scalarization therefore requires an intermediate arm that changes only the reference
construction. This is the reason the arm set in `protocol/method_arms.md` has four members rather
than two.

## 5. Reproducing this audit

```
python papers/xgboost_hpo_vrfnbi/scripts/methodological_identity_audit.py
```

It extracts `v0.1.0-dissertation` to a scratch directory, asserts the absence of each NBI structural
element by parsing the frozen source, and writes `audits/methodological_identity.json`.
