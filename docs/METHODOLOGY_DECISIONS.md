# Methodology decisions log

This document records every place where the implementation on
`repo-publication-readiness` differs from the closed dissertation text, and
why. The dissertation itself is *not* edited; it remains the historical
academic baseline. The article-track repository may, however, adopt
methodologically stronger choices, especially when:

1. The dissertation **code** and the dissertation **text** already disagreed
   (this happened during late textual corrections that could not be
   re-implemented in time);
2. The peer-reviewed EAAI 2025 article
   (Costa Pereira, Tertuliano Ribeiro, et al., *Engineering Applications of
   Artificial Intelligence*, vol. 162, art. 112510, 2025;
   DOI [10.1016/j.engappai.2025.112510](https://doi.org/10.1016/j.engappai.2025.112510))
   provides a stronger, peer-validated formulation;
3. The dissertation simplification was acceptable as a closed academic
   artifact but is too narrow for a publishable framework.

Every entry below has the same shape: **what the dissertation says**, **what
the dissertation code does on `main`**, and **what the publication branch
chose, with rationale**.

---

## D1 — "NBI" vs weighted-sum scalarization

**Dissertation text (§4.4.3, Eq. 4.16).**
NBI is described as the constrained subproblem
`max t s.t. F(x) = F^c(β) + t·n`, with anchors, CHIM matrix `Φ`, and a
quasi-normal direction `n`. This is faithful to Das & Dennis (1998).

**Dissertation code on `main` (`src/doe_xgb/nbi.py`, function
`run_nbi_weighted_sum`).**
Implements a *normalized weighted-sum scalarization*:
`min −β · norm(F̂(x))` over a β-grid. No anchors, no `Φ`, no `n`. This is
*not* NBI; it can fail on non-convex Pareto fronts and gives non-uniform
spacing on the front.

**Publication-branch decision.**
1. Implement true N-objective NBI in `src/doe_xgb/nbi_core.py`. This is
   the article-track method.
2. Preserve the legacy implementation under
   `src/doe_xgb/scalarization.py` (renamed honestly). It remains available
   as an ablation baseline. It is **never** referred to as NBI.
3. The classic `nbi.py` import path is kept as a thin compatibility shim
   that re-exports the legacy symbols and emits a `DeprecationWarning`.
4. The article will report both methods side-by-side so the reviewer can
   see why the correction matters.

---

## D2 — N-objective generality

**Dissertation case study.** Practically uses `q = 2` (Score_Quality vs
Score_Cost) for the headline numerical comparisons, and discusses `q = 3`
(VRF1, VRF2, VRF3) descriptively in the corrected final text.

**Publication-branch decision.**
The NBI mathematical kernel must support arbitrary `n_objectives ≥ 2`. No
`if q == 2` or `if q == 3` is allowed in the math core; visualization
helpers may be 2D/3D-specific and explicitly raise for higher q.
The repository's automated tests cover `q ∈ {2, 3, 4, 5}` (`q = 7`
exercised as a slow-marked test).

---

## D3 — Objective direction handling

**Dissertation code.** Direction is *implicit*: time is "minimized" via the
sign of the time-factor loading; quality factors are oriented by the
average sign of their loadings on quality metrics. This is fragile:
loadings can flip; metric names can be misread.

**Publication-branch decision.**
Every objective entering the NBI core must declare its direction explicitly
through an `ObjectiveSpec` (`name`, `direction`, `transform`, `role`,
optional `target`, optional `weight`, optional `group`).
- `direction ∈ {minimize, maximize, target}` is required (no default).
- The core canonicalizes everything to **minimization** before NBI.
- A convenience layer (config loader) may emit *warnings* for common
  metric names but never silently infers direction.

---

## D4 — FMSE / target-seeking objective wrapping

**EAAI 2025 (Eqs. 17–22).** Each VRF objective is rewritten as
`VRFᵢ(x) = [F̃ᵢ(x) − Tᵢ]² + σ²_F̃ᵢ`, i.e. a target-seeking quadratic loss
plus a factor variance penalty. This converts mixed-direction factor
loadings into a uniform "minimize" convention and makes the math
direction-agnostic.

**Publication-branch decision.**
- `transform = fmse` is the default for **factor-score** objectives
  (article-track VRF mode).
- `transform = raw` (with explicit direction) is the default for
  **direct ML metric** objectives.
- Both modes coexist; `ObjectivesConfig.specs` may mix them.

---

## D5 — Anchor / utopia / nadir source

**Dissertation code.** Uses observation-based extremes from the DOE rows
to build the (utopia, nadir) box. This is fast but biased toward DOE
sampling and can miss the true surrogate optimum.

**EAAI 2025 (Eqs. 21–22).** Anchors come from per-objective surrogate
optimization with a hyperspherical region constraint
`xᵀx ≤ ρ², ρ = 2^(k/4)` for spherical CCDs.

**Publication-branch decision.**
- Default for the article track: **surrogate-based anchors**
  (`min_x F̂_k(x)` over the surrogate, on box constraints, with optional
  `xᵀx ≤ ρ²` constraint behind a feature flag).
- Default for HPO problems: **box constraints**, since hyperparameters
  have operational bounds. The hyperspherical constraint is a flag,
  not the default.
- Observation-based anchors remain available behind
  `nbi.anchors.source = "observed"`.

---

## D6 — Coded vs uncoded RSM units

**Dissertation tables (Tabela 3).** Reports coded coefficients.
**Dissertation code (`src/doe_xgb/rsm.py`).** Fits the full quadratic in
**uncoded units**.

**Publication-branch decision.**
- Article track defaults to **coded units** (orthogonality, comparable
  effect sizes, better numerical conditioning).
- Both coded and uncoded coefficient CSVs are written per replica, so
  reviewers can match the dissertation tables and the article tables.
- A `dissertation-parity` config preserves uncoded behavior for
  bit-comparable historical reproduction.

---

## D7 — Final-candidate selection rule

**Dissertation text (Eq. 4.20–4.21).** Decision via utility, knee point,
or distance-to-utopia.
**Dissertation code (`src/doe_xgb/benchmarks.py:80–90`).** Selects by
`max(Accuracy_Mean)`, even for the "proposed" multi-objective method.

**Publication-branch decision.**
- Article track default selection rule for the proposed method:
  `distance_to_utopia` on canonicalized objectives.
- Benchmarks keep `max_accuracy` so dissertation tables still match.
- `selection_strategy` is a configurable enum; full Pareto front is
  always persisted regardless of which rule is applied.

---

## D8 — Factor count

**Dissertation final text.** Uses three VRFs (Quality, Cost,
Robustness/Stability). This is partly a textual correction added by the
advisor at the very end and not fully re-implemented.
**Dissertation code.** Collapses non-time factors into a single
`Score_Quality`, effectively running 2-VRF.

**Publication-branch decision.**
- Article-track baseline: `factor_model.mode = "fixed", n_factors = 3`.
- Dissertation-parity track: replicates the 2-VRF behavior exactly.
- Framework demos default to `mode = "auto"` with
  `cumvar_threshold = 0.85` and Kaiser eigen > 1.
- A `mode = "manual"` lets the user pre-declare constructs (Quality /
  Cost / Robustness / Fairness / Reliability / Economic / …) which is
  closer to how engineering DOE practitioners think.
- A `mode = "none"` fallback skips factor analysis and uses raw
  objectives directly.

---

## D9 — Quasi-normal direction

**Default.** Das & Dennis convention: `n = −Φ · 1 / ‖Φ · 1‖`.
**Diagnostic option.** True orthogonal complement of the CHIM via
Gram–Schmidt, exposed as `nbi.quasi_normal = "gram_schmidt"`.

---

## D10 — Conditional post-optimization (MBPA)

**EAAI 2025 (§2.5–§2.6, Algorithm 4).** A second-stage NBI on the simplex
of weights, with mixture response surfaces (Scheffé canonical
polynomials) over `(GD, S(w))` and an elliptical interior constraint to
avoid degenerate vertex weights.

**Publication-branch decision.**
- Implemented in `src/doe_xgb/post_optimization.py`.
- Default trigger: **conditional**, gated by frontier-quality
  diagnostics (normalized objective range, average pairwise distance,
  unique non-dominated count, weight concentration, curvature, spread).
- When triggered: writes `frontier_quality.json`,
  `post_optimization_diagnostics.json`,
  `post_optimization_mixture_fit.csv`, and
  `post_optimization_refined_candidate.csv`.
- When skipped: still writes `frontier_quality.json` so the decision is
  auditable.

---

## D11 — Per-fold metric persistence

**Dissertation code.** Aggregates fold metrics to the mean and discards
per-fold values, losing the natural source of a stability/robustness
construct.

**Publication-branch decision.**
- Persist per-fold metrics in a long-format CSV per replica.
- Optionally compute Brier score / ECE for probabilistic calibration
  (off by default; on for the article configs).

---

## D12 — Dataset and design provenance

**Publication-branch decision.**
- `data/source/CHECKSUMS.txt` records SHA-256 hashes of expected
  datasets.
- The Minitab design CSV is committed under `data/design/` with a sidecar
  metadata JSON describing generator, kind, factors, run count, center
  points, bounds, and SHA-256.
- A `scripts/fetch_magic_dataset.py` downloads UCI MAGIC and verifies its
  checksum before saving. The dataset itself is not committed.

---

## D13 — Determinism for the article tables

XGBoost's `tree_method="hist"` plus `n_jobs > 1` is non-deterministic by
construction. Article-track configs default to `tree_method = "exact"`
and `n_jobs = 1` for the canonical headline table. Other configs may
trade determinism for speed.

---

## D14 — Mixture model basis is not ordinary RSM

**Publication-branch decision.**
The Scheffé canonical mixture polynomial used during MBPA
post-optimization is *not* the same model class as the ordinary
process-quadratic RSM. It is fitted by a separate model family
(`MixtureScheffeModel`) that respects the simplex constraint
`Σ wᵢ = 1`, drops the intercept, and replaces pure quadratic terms with
Scheffé canonical cross-products. Backward elimination is **disabled**
for mixture models because the canonical interpretation depends on the
full term set.
