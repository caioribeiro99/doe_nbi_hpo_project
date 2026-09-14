# External validation design: formal verification

**Status: verified and frozen.** Run `python scripts/verify_external_validation_design.py`; it
writes `audits/external_validation_verification.json` and exits non-zero on any failure.

**The 78 points are frozen.** They were chosen because a random external set cannot validate a
surface fitted to a factorial design (`PROTOCOL_AMENDMENTS.md` amendment 4), not because of the R²
they produced. They are **not** to be re-tuned against Stage A's R², and the Latin-hypercube and
arcsine results are engineering diagnostics explaining why the specification changed, not
confirmatory evidence about anything.

## The design

| Component | Runs | Definition |
|---|---|---|
| Complementary half fraction | 64 | the 2⁷ corners whose seven-factor sign product is −1; the fitting design uses +1 |
| Axial runs | 14 | ±0.5 in coded units on each factor in turn, that is half the fitting design's axial distance |
| **Total** | **78** | per replication per dataset |

## The seven conditions

| # | Condition | Result |
|---|---|---|
| 1 | No external run appears in the fitting design | **0 shared runs** |
| 2 | Fitting and validation fractions have opposite defining relation | design sign product +1, external −1 |
| 3 | The complementary fraction has the same factorial resolution | **VII and VII** |
| 4 | Every configuration is feasible under the declared bounds and types | no violations; integer realization displaces `max_depth` and `n_estimators` by at most 0.5, recorded per candidate under Phase B8 |
| 5 | Axial runs are distinct from both fractions and from the design's own axial runs | 0 overlap with corners, 0 with the design's axial runs, radius 0.5 against the design's 1.0 |
| 6 | The external set can identify the terms of the fitted response surface | rank 36 of 36 terms with the axial runs; **29 of 36 from corners alone**, which is precisely why the axial runs are in the specification |
| 7 | No validation response can reach any fitting stage | static call-graph check finds 8 fitting calls, all on the design frame, 0 on the validation frame |

Condition 3 is stronger than it needs to be and worth stating: resolution VII means main effects are
clear of two- and three-factor interactions on both halves, so the external set is not merely
disjoint but statistically equivalent in what it can estimate.

Condition 6 is the reason the construction is not simply "the other half fraction". On a two-level
set every squared coordinate equals one, so the quadratic terms collapse into the intercept and
rank falls to 29. The 14 axial runs restore full rank.

## Condition 7 in the campaign

The static check covers the screening code. The campaign runner installs a runtime tripwire instead:
the validation responses are held in an object that raises if read by any stage other than the
external-diagnostic computation. The stages that must never see them are the response
standardization, the principal component extraction, the Varimax rotation, the factor orientation,
the response-surface fit, the backward selection, the anchor construction and every optimizer.

## Diagnostics retained

External **R²**, **RMSE** and **Spearman** per response, plus the external set's own response
spread and the design-to-external spread ratio, so a reader can see whether R² is interpretable
rather than having to assume it. Stage A's own numbers showed what happens when it is not: R² of
−23.99 arising from a near-zero denominator while rank correlation on the same data was 0.589.

The gate remains **external R² ≥ 0.5 and Spearman ≥ 0.9** per response. Not moved.
