# Pilot Stage A: what the screening found, including about itself

**Scope.** Stage A of the pilot, per `protocol/EXPERIMENT_PROTOCOL.md` §12: the 88 design rows and a
held-out set on one partition of each candidate dataset, the decided factor stage, and the four
screening measurements. No arm ran.

**Headline.** Stage A did its job, but most of what it found is about **the protocol's own
measurement machinery**, not about the datasets. Four corrections were needed before any screening
number could be trusted, and one of them was a bug in this workspace's code rather than in anything
inherited. They are reported in the order they were found, with the evidence that forced each.

---

## Finding 1 — the factor model must be fitted once and applied, not refitted

**What was wrong.** The first version of the screening fitted the factor stage separately on the
design set and on the held-out set, then scored a surface fitted against the first using targets
defined by the second. Each fit has its own standardization, its own rotation and its own sign
orientation, so the two composites live in different coordinate systems and the comparison is
meaningless.

**How it showed.** External R² for the quality surface came out at −1.116 on Adult and −1.523 on
Bank Marketing, with rank correlations of 0.015 and −0.057 — that is, essentially no relationship at
all between a surface and the quantity it was fitted to predict.

**The fix.** `FactorModel.fit` on the design, `.transform` applied to held-out points. This is also
what the surrogate gate requires in the campaign: the method only ever sees the design.

**Guard.** `tests/methodology/test_protocol_factor_stage.py` asserts that applying a fitted model and
refitting give measurably different results, so the distinction cannot quietly collapse again.

## Finding 2 — the cost response must be log-transformed

**What was wrong.** `protocol/EXPERIMENT_PROTOCOL.md` §6.1 replaced the dissertation's wall-clock cost
objective with total leaf count, and in doing so dropped the log transform the dissertation applied to
its own cost response (`time_transform="log1p"` is the frozen default in `factor_analysis.py`).

Leaf count spans roughly three orders of magnitude across the design box — the measured range ratios
are 591 to 4,273. A quadratic response surface fitted to that raw cannot describe it.

**Measured, in-sample R² of the cost surface:**

| Dataset | raw leaf count | log-transformed |
|---|---|---|
| MAGIC | 0.601 | **0.957** |
| Spambase | 0.645 | **0.947** |
| Adult | 0.499 | **0.948** |
| Bank Marketing | 0.487 | **0.949** |

Without the transform, backward elimination at α = 0.05 removed *every* term on Adult and Bank
Marketing, leaving an intercept-only cost surface. With it, the surfaces keep 15 to 18 terms and
behave.

**The fix.** Each response now declares its transform alongside its direction, and neither is
inferred. The cost response declares `log1p`.

## Finding 3 — factor signs must be oriented deterministically

**What was wrong.** A principal component's sign is arbitrary and Varimax does not fix it. With no
orientation rule, the quality composite's sign is arbitrary too, so its measured correlation with the
cost factor — which is screening criterion 1, the objective conflict — can come out either way.

**How it showed.** On Bank Marketing, changing the log-loss transform, a change that does not alter
which configurations are good, flipped the measured conflict from **−0.662 to +0.688**. A screening
rule that can be reversed by a benign respecification is not a screening rule.

**The fix.** Each factor is oriented so that the response loading most heavily on it has a positive
loading. Every response is already canonicalized to minimization, so a larger score means a worse
configuration on every factor, and the sign of the conflict measurement means what it says.

**Guard.** Two tests: every factor's dominant loading is positive, and the conflict sign survives the
respecification that previously flipped it.

## Finding 4 — a uniform held-out set does not validate this surrogate

**The most consequential finding, and it invalidates a parameter inherited from Paper 1.**

`protocol/EXPERIMENT_PROTOCOL.md` §7 specified the surrogate gate's external set as 100 Latin-hypercube
points over the hyperparameter box, with Paper 1's thresholds. That construction does not transfer.

In seven dimensions, uniform sampling places essentially no mass near the box corners. Every drawn
configuration is a reasonable one, so the held-out set never probes the region where the surface has
to be trusted, and it carries far less response spread than the design it is meant to validate:

| Dataset | quality spread, design | quality spread, uniform held-out | ratio |
|---|---|---|---|
| MAGIC | 0.654 | 0.330 | 2.0 |
| Spambase | 0.894 | 0.091 | **9.8** |
| Adult | 0.788 | 0.206 | 3.8 |
| Bank Marketing | 0.957 | 0.357 | 2.7 |

On Spambase the design's quality values range over 4.17 and the held-out set's over 0.57. The worst
consequence is arithmetic: R² is `1 − SS_res / SS_tot`, and with `SS_tot` near zero any prediction
error produces a large negative number. The measured −23.99 on Spambase is not evidence that the
surface is bad; it is evidence that the denominator is tiny. Rank correlation on the same data is
0.589, which is unremarkable rather than catastrophic.

This did not arise in Paper 1 because its decision space was a simplex of ensemble weights, where
uniform sampling does reach degenerate corners. A box in seven dimensions is a different geometry.

**The fix.** The external set is drawn as a **spanning** set: half a Latin hypercube plus half an
arcsine-marginal sample, whose Beta(0.5, 0.5) coordinates concentrate near the ends of each range. It
spans the response range the design spans, which is the condition an external set must meet before
R² against it means anything. Both constructions are kept on disk, because the comparison is the
evidence for this finding.

---

## Consequences for the protocol

| # | Change | Section |
|---|---|---|
| 1 | The factor model is fitted on the design and applied to held-out points | §6.3 |
| 2 | Each response declares a transform; the cost response declares `log1p` | §6.2, §6.3 |
| 3 | Factor signs are oriented by the dominant loading, not left arbitrary | §6.3 |
| 4 | The external set is the spanning construction, not a uniform Latin hypercube | §7 |
| 5 | The gate reports the external set's own response spread beside R², so a reader can see whether R² is interpretable | §7 |

Finding 4 also means the inherited gate thresholds must be re-examined against the spanning set
rather than transferred. §7 already forbids retuning a threshold after seeing campaign results; this
is a pre-campaign measurement and is the last point at which the threshold can honestly be set.
