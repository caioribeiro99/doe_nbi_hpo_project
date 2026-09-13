# PCA / Varimax identity audit

**Status:** complete. **Scope:** the objective-reduction stage in
`src/doe_xgb/factor_analysis.py` at tag `v0.1.0-dissertation` (commit `67d9fe5`).
**Data:** the 88-run MAGIC design reproduced in `audits/provenance/doe_results.csv`, produced by
executing the frozen pipeline itself.
**Machine-readable record:** `audits/pca_varimax_audit.json`.
**Reproduce:** `/tmp/venv-diss/bin/python scripts/pca_varimax_identity_audit.py`
(see `audits/provenance/` for why an era-appropriate interpreter is needed).

Five questions. Two findings are material, one is a naming problem with no measured consequence
here, and two are confirmations of things the author had already recorded.

---

## Q1. Are the reported "loadings" loadings? **No — they are eigenvectors.**

The frozen code takes

```python
pca = PCA(n_components=n_factors_eff, random_state=0)
scores   = pca.fit_transform(Z)
loadings = pca.components_.T
```

`pca.components_` holds the principal axes, whose rows are unit-norm eigenvectors of the correlation
matrix. In factor-analytic usage the loading of a variable on a component is its correlation with
that component, which for standardized data is the eigenvector entry scaled by the square root of
the eigenvalue.

Measured on the MAGIC design:

| Quantity | Factor 1 | Factor 2 | Factor 3 |
|---|---|---|---|
| Column norm of the matrix the code calls "loadings" | 1.000 | 1.000 | 1.000 |
| Eigenvalue | 3.454 | 0.871 | 0.660 |
| Square root of the eigenvalue, that is the true column norm | 1.859 | 0.933 | 0.812 |

The two matrices differ by a factor of up to 2.29 between columns. Any interpretation that reads the
printed numbers as correlations between a metric and a factor is reading the wrong matrix.

## Q2. Does rotating eigenvectors change the rotation? **In principle yes; on this data, no.**

Varimax maximizes the variance of the squared loadings. That criterion is not invariant to rescaling
the columns, so rotating unit-norm eigenvectors is not the same operation as rotating loadings: it
gives a component carrying 13% of the variance the same weight in the criterion as one carrying 69%.

Measured, the difference is small here:

| Quantity | Value |
|---|---|
| Angle between corresponding rotation-matrix columns | 1.19°, 1.44°, 0.99° |
| Frobenius norm of the difference between the two rotations | 0.0368 |
| Metric-to-factor assignment under either convention | identical |

Under both conventions precision and specificity load on Factor 1, time on Factor 2, and accuracy
and recall on Factor 3.

**Reported against interest:** this is a naming and reproducibility problem, not a demonstrated
error in the dissertation's conclusions. The factor structure the dissertation interprets is the one
a correct implementation would also produce, on this dataset. The reason to fix it is that nothing
guarantees the agreement holds on another dataset, and a reader cannot check the claim from the
published numbers because the printed matrix is not the one the method names.

## Q3. How many objectives does the optimizer see? **Three factors extracted, two optimized.**

`n_factors = 3` and `force_time_factor = True` give a three-factor extraction. Then
`combine_quality_factors = True` collapses every non-time factor into a single `Score_Quality`, so
the surrogate and the optimizer work on `(Score_Quality, Score_Cost)`.

This confirms `docs/METHODOLOGY_DECISIONS.md` **D8**, which the author wrote before this audit: the
final dissertation text describes three Varimax-rotated factors and the code runs two. The audit
adds only the mechanism.

## Q4. What does the quality aggregation do? **It equal-weights components of very unequal variance, and that changes the ranking.**

`Score_Quality` is the unweighted mean of the z-scored scores of the two non-time factors. Z-scoring
to unit variance first is what makes the weighting equal: it discards the explained-variance
ordering that motivated extracting the components at all.

| Factor | Share of explained variance | Weight it receives in `Score_Quality` |
|---|---|---|
| Factor 1 | 0.693 | 0.500 |
| Factor 3 | 0.132 | 0.500 |

Comparing the composite the code builds against one weighted by explained-variance share
(0.840 / 0.160):

| Comparison | Value |
|---|---|
| Spearman correlation between the two composites | 0.374 |
| Kendall tau | 0.358 |
| Design rows shared by the two top-ten lists | 1 of 10 |
| Best design row | differs (row 55 versus row 2) |

**This is the material finding.** The quality objective is not determined by the factor extraction
alone. An undeclared aggregation choice, made by a default argument, moves the quality ranking of
the design enough to change which design point looks best. Nothing in the pipeline reports the
choice, and no sensitivity analysis varies it.

This does not show that the dissertation's conclusions are wrong. It shows that a reader cannot tell
from the dissertation whether they are robust to this choice, and neither could we without running
it.

## Q5. Is the EAAI 2025 FMSE wrapper implemented? **No.**

`docs/METHODOLOGY_DECISIONS.md` **D4** describes the EAAI 2025 formulation, in which each
Varimax-rotated factor objective is rewritten as a target-seeking quadratic loss plus a
factor-variance penalty, `VRF_i(x) = [F̃_i(x) − T_i]² + σ²_{F̃_i}`.

The frozen factor stage emits sign-oriented factor scores directly, with sign fixed by the sign of
the time loading and by the mean sign of the quality loadings. There is no target, no squared
deviation and no variance penalty. The dissertation's objectives are raw oriented scores.

Confirms D4's framing. Paper 2 must not describe the dissertation objectives as FMSE objectives.

---

## What Paper 2 may write

Supported:

- "The dissertation's objective reduction is principal component analysis with a Varimax rotation
  applied to the eigenvector matrix, followed by an unweighted average of z-scored component scores
  for quality and a sign-flipped z-score of the time component for cost."
- "Three components are extracted and two objectives are optimized."
- "On the reproduced MAGIC design the aggregation choice changes the quality ranking materially
  (Spearman 0.374 against a variance-weighted composite)."

Not supported, and must not be written:

- That the dissertation's factor interpretation is wrong. On this data it is the same interpretation
  a correct implementation gives.
- That the rotation error changed any published conclusion. We measured it and it did not, here.
- Anything calling the dissertation objectives FMSE or target-seeking.

## Consequence for the protocol

1. Every arm in Paper 2 must share **one** objective-reduction stage, specified explicitly, so the
   arms differ only in the optimizer. The aggregation weighting is part of that specification.
2. The weighting must be a pre-registered, reported choice with a stated sensitivity check, not a
   default argument.
3. The loading matrix reported in Paper 2 is the scaled one, and the text says which it is.

Tracked in `protocol/EXPERIMENT_PROTOCOL.md`.
