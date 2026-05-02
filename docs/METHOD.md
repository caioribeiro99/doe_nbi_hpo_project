# Method overview (article track)

The article-track method on this branch is a generalization of the
dissertation pipeline, anchored in the EAAI 2025 article cited in
`CITATION.cff`. The pipeline has six stages.

## 1. Design of Experiments

A face-centered central composite design (CCDFC) covers the
hyperparameter space of XGBoost. The design is loaded as a Minitab CSV
under `data/design/` with a sidecar metadata JSON. Alternative designs
can be generated through the `DesignProvider` abstraction
(see `docs/DOE_DECISION_TREE.md`).

## 2. CV evaluation

Each design row is evaluated with stratified k-fold CV. We persist
per-fold metrics (accuracy, precision, recall, specificity, optional
Brier and ECE) and per-fold timing. The mean fit-plus-predict time per
fold is the cost proxy.

## 3. Factor analysis with Varimax rotation

Standardized DOE responses are decomposed via PCA followed by Varimax
rotation. The factor count is configurable: auto, fixed, manual
constructs, or none (see `docs/METHODOLOGY_DECISIONS.md` D8).

The article-track default is `mode = "fixed", n_factors = 3` matching
the three-construct VRF interpretation in the EAAI 2025 article.

## 4. Surrogate fitting (response surfaces)

Each VRF (or each raw objective, depending on configuration) gets a
quadratic response surface fitted by OLS with optional backward
elimination (α = 0.05) under a strong-hierarchy constraint. The model
family is selected from the design's metadata via
`select_model_family(design)`. Process-variable RSM is used for
CCD/BB/factorial designs; Scheffé canonical mixture models are used
for simplex-lattice designs.

## 5. N-objective NBI

For an arbitrary number of objectives `q ≥ 2`:

1. Compute anchors `xₖ* = argmin F̂ₖ(x)` for each k = 1, …, q on the
   surrogates.
2. Build the payoff matrix `Φ_{kj} = F̂ₖ(xⱼ*)`. Diagonal of Φ is the
   utopia point. Column-wise max gives a pseudo-nadir (Miettinen).
3. Quasi-normal direction `n = −Φ · 1 / ‖Φ · 1‖` (Das–Dennis).
4. For each weight vector β on the simplex:
   solve  `max t  s.t.  F̂(x) = utopia + Φβ + t · n,  x ∈ Ω`
   via SLSQP, with `t ≥ 0`, vector equalities, and box bounds.
5. Decode each `x*(β)` to hyperparameters and confirm in real CV.

The simplex weights are generated from a Scheffé Simplex-Lattice {q, m},
giving N_sub = C(q+m−1, m) subproblems.

## 6. Conditional post-optimization (MBPA)

If frontier diagnostics indicate a flat / compressed / weakly
informative front, a second-stage MBPA is triggered:

1. Fit Scheffé canonical mixture surrogates over the simplex of weights
   for `GD(w)` (generalized distance to utopia, Mahalanobis-style) and
   `S(w) = −Σ wᵢ ln wᵢ` (Shannon entropy).
2. Run an inner NBI on `(GD, S)` minimization/maximization respectively,
   subject to an elliptical interior constraint that prevents
   degenerate vertex weights.
3. Decode the refined `w*` back into refined hyperparameters.

If the frontier is informative, MBPA is skipped and the decision is
recorded transparently in `frontier_quality.json`.
