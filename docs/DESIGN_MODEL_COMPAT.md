# Design / surrogate-model compatibility

The article-track keeps a strict separation between the design family
that produced the data and the surrogate-model family that consumes it.
This page is the reference table.

| Design kind | Recommended family | Order(s) | Notes |
|---|---|---|---|
| `external_csv` (Minitab CCDFC) | `process_quadratic` | linear / 2fi / **quadratic** | Validate via metadata coding. |
| `ccd_face_centered` | `process_quadratic` | quadratic | Default for the dissertation. |
| `ccd_circumscribed` | `process_quadratic` | quadratic | Larger axial alpha => wider region. |
| `ccd_inscribed` | `process_quadratic` | quadratic | Stays inside the box. |
| `box_behnken` | `process_quadratic` | quadratic | Avoid corner points; needs 3+ factors. |
| `full_factorial` (2-level) | `linear_or_2fi` | 2fi | No pure quadratic terms. |
| `fractional_factorial` | `linear_or_2fi` | linear or 2fi | Resolution-dependent. |
| `lhs` / `sobol` | `process_quadratic` *(warned)* | quadratic | No inferential support; exploratory. |
| `simplex_lattice` | **`mixture_scheffe`** | linear / **quadratic** / special_cubic / cubic | NEVER process_quadratic. |
| `simplex_centroid` | `mixture_scheffe` | quadratic | Cornell-style. |
| `d_optimal` | family from spec | from spec | `DesignProvider.D_OPTIMAL` raises NotImplementedError; pre-generate the candidate set externally. |

## Guard rails

`DesignProvider.validate_for_model(artifact, family=..., order=...)`:

- raises an **error** if `mixture_scheffe` is requested on a non-mixture
  design;
- raises an **error** if `process_quadratic` is requested on a simplex
  design;
- emits a **warning** if `process_quadratic` is fitted on LHS / Sobol
  (no pure error, weak orthogonality);
- emits a **warning** if the design matrix is rank-deficient or has a
  large condition number.

## Why the separation matters

A Scheffé canonical mixture polynomial has no intercept, no pure
quadratic terms, and respects the simplex constraint
`Σ wᵢ = 1`. Fitting a "process quadratic" on simplex data silently
ignores the constraint and produces meaningless coefficients. See
`docs/METHODOLOGY_DECISIONS.md` D14.
