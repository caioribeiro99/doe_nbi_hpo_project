# DOE selection decision tree

When a new HPO problem arrives, work top-down:

1. **How many factors / hyperparameters?**
   - 1–3 small space, smooth response, tight budget → consider a 3-level
     full factorial or a small CCD.
   - 4–8 (the dissertation regime, k = 7) → CCD / CCDFC / Box-Behnken.
   - 9–15 → start with a fractional factorial of resolution V or VI to
     screen, then refine in a smaller CCD.
   - 16+ → start with LHS / Sobol screening + tree-importance, then
     pick a CCD on the surviving subset.

2. **Are extreme corners safe / meaningful?**
   - Yes → CCD circumscribed (axial alpha = (2^k)^{1/4}).
   - No (extreme corners trigger pathological model behavior) →
     **CCD face-centered (CCDFC)**, the dissertation choice; axial
     points sit on the box face. Prefer this for HPO since
     hyperparameter combos at the box corners are usually still valid
     for the model.
   - The space is regular but no axial points are tolerated → Box-Behnken.

3. **Are factors discrete / conditional / categorical?**
   - Pure continuous → CCD/BB.
   - Mostly continuous with a few integers (`max_depth`, `n_estimators`)
     → CCDFC + integer rounding (the dissertation regime; documented in
     the article-track as `FactorMeta(type='int')`).
   - Strong conditional structure (XGBoost vs LightGBM-only knobs)
     → run separate per-algorithm designs and combine in a meta-NBI.

4. **Replicating center points?**
   - Yes (recommended for pure-error estimation in RSM) →
     `n_center >= 3`. Dissertation used 4.
   - No (pure deterministic surrogate; no need for ANOVA) → 0.

5. **Mixture / NBI weights?**
   - Always use `simplex_lattice {q, m}` for primary NBI weights and
     `simplex_lattice` (with denser m) for the inner MBPA stage.
   - Never use a CCDFC on weights -- they live on the simplex, not the
     box.

6. **Validate before running.**
   - Call `DesignProvider.validate_for_model(artifact, family, order)`.
     A `mixture_scheffe` model on a CCDFC is rejected; an LHS with
     `process_quadratic` is allowed but warned.

This decision tree is intentionally pragmatic. For exotic cases
(constrained mixtures, D-optimal candidate sets, nested designs) prefer
external generation in Minitab or JMP and load via
`DesignKind.EXTERNAL_CSV`.
