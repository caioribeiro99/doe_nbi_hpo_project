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

## Finding 4b — orienting a factor by its largest loading inverts the quality composite

Found while checking Finding 3's fix against the data, and the most dangerous of the five, because it
produced a confident, wrong screening verdict rather than an obviously broken number.

**The rule that failed.** Orient each factor so the response loading most heavily on it has a
positive loading. It is the obvious rule and it reads as principled.

**Why it fails here.** Within the quality block, specificity trades off against accuracy, recall and
the area under the curve across the decision threshold, so it loads with the opposite sign. On three
of the four candidate datasets, specificity is the dominant loading on the leading quality factor:

| Dataset | dominant loading on the leading quality factor | mean loading over the quality block |
|---|---|---|
| Adult | Specificity, +0.928 | **−0.312** |
| Spambase | Specificity, +0.922 | **−0.411** |

The rule therefore pointed the quality composite at specificity-badness, which runs opposite to
overall quality. The measured objective conflict came out **positive** on Spambase (+0.319), Adult
(+0.512) and Bank Marketing (+0.696) — that is, "no trade-off, these objectives agree" — on datasets
whose raw metrics plainly disagree:

| Dataset | Spearman(log leaves, accuracy) | Spearman(log leaves, area under the curve) | Spearman(log leaves, log loss) |
|---|---|---|---|
| MAGIC | +0.293 | +0.281 | −0.172 |
| Spambase | +0.402 | +0.513 | −0.340 |
| Adult | +0.389 | +0.500 | −0.368 |
| Bank Marketing | +0.414 | +0.496 | −0.386 |

Bigger models are more accurate and better calibrated on every dataset, so quality and cost trade off
on every dataset. Screening criterion 1 would have rejected three of four datasets for having no
conflict, and the reason would have been a sign convention.

**The fix.** Orient each factor by the **mean loading over the responses in its own role block** —
quality factors over the quality responses, the cost factor over the cost response. This is the rule
the dissertation used, and it is the right one.

**Guard, and the general lesson.** The screening now computes the same conflict a second way, from
the canonicalized responses directly with no factor stage involved, and **refuses to report** if the
two disagree in sign. A derived quantity that can silently invert needs an independent check, not a
more careful derivation.

| Dataset | conflict, factor composite | conflict, raw responses |
|---|---|---|
| MAGIC | −0.284 | −0.251 |
| Spambase | −0.428 | −0.462 |
| Adult | −0.562 | −0.398 |
| Bank Marketing | −0.688 | −0.410 |

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

**The first fix did not work, and that is the informative part.** A *spanning* set was tried: half a
Latin hypercube plus half an arcsine-marginal sample, whose Beta(0.5, 0.5) coordinates concentrate
near the ends of each range. It barely helped — the spread ratios moved only to 2.00, 7.27, 2.91 and
2.44.

The reason is structural and worth stating as a general point. A design's response range comes from
specific corner *combinations*, and with seven independent coordinates the probability of landing
near the same end on all of them is about one in 128. **No scheme with independent coordinates
reproduces a factorial design's response spread.** The external set has to be structured too.

**The fix that worked.** The 88-run design is a face-centred central composite: a 64-run half fraction
of the 2⁷ factorial defined by the generator "product of all seven signs = +1", plus 14 axial and 10
centre runs. Its **complementary half fraction** — the 64 corners whose sign product is −1 — is
disjoint from it by construction, has identical corner structure and therefore identical response
spread, and is itself a resolution-VII design. Fourteen axial runs at half the design's axial
distance are added, because on a two-level set every squared coordinate equals one and the quadratic
terms collapse into the intercept, so corners alone cannot test curvature.

**78 evaluations rather than 100, so it is both cheaper and correct.**

Measured, the same surfaces against each construction:

| Dataset | uniform R²_quality | spanning R²_quality | **complement R²_quality** | complement Spearman |
|---|---|---|---|---|
| MAGIC | +0.461 | +0.468 | **+0.775** | 0.916 |
| Spambase | −23.99 | −12.74 | **+0.953** | 0.847 |
| Adult | −2.68 | −0.92 | **+0.911** | 0.925 |
| Bank Marketing | −0.89 | −0.71 | **+0.917** | 0.915 |

The surrogate was never the problem. Every negative number above was the denominator.

All three constructions are kept on disk, because the comparison is the evidence for this finding.

---

## Consequences for the protocol

| # | Change | Section |
|---|---|---|
| 1 | The factor model is fitted on the design and applied to held-out points | §6.3 |
| 2 | Each response declares a transform; the cost response declares `log1p` | §6.2, §6.3 |
| 3 | Factor signs are oriented by the mean loading over the factor's own role block | §6.3 |
| 4 | The external set is the design's complementary half fraction plus axial runs at half the axial distance, 78 points, not a random sample | §7 |
| 5 | The gate reports the external set's own response spread beside R², so a reader can see whether R² is interpretable | §7 |
| 6 | The screening measures objective conflict a second way, from the responses directly, and refuses to report if the two disagree in sign | §6.3 |
| 7 | `B_surrogate_validation` becomes 78, not 100, changing every budget derived from it | §8, §3 |

---

## Screening verdict on the panel

All four measurements, taken with the corrected machinery:

| Dataset | conflict | curvature | R² quality | Spearman quality | R² cost | Spearman cost | cost range | s/eval |
|---|---|---|---|---|---|---|---|---|
| MAGIC | −0.284 | 0.438 | 0.775 | 0.916 | 0.917 | 0.940 | 2,590 | 1.69 |
| Spambase | −0.428 | 0.222 | 0.953 | 0.847 | 0.942 | 0.965 | 591 | 1.02 |
| Adult | −0.562 | 0.288 | 0.911 | 0.925 | 0.935 | 0.955 | 3,402 | 1.57 |
| Bank Marketing | −0.688 | 0.144 | 0.917 | 0.915 | 0.942 | 0.958 | 4,273 | 1.31 |

**This verdict was computed on the superseded factor algebra and CRITERION 1 DOES NOT SURVIVE
CORRECTION.** The row of latent conflict values above — −0.284, −0.428, −0.562, −0.688 — came from
the factor construction the protocol has since withdrawn. Recomputed on the corrected model
(`audits/reference_factor_models/`), the same quantity is **−0.201, +0.310, +0.103 and +0.001**: on
no dataset is it "clearly negative", which is what criterion 1 requires.

The reason is structural and is recorded in amendment 19. The quality composite is a weighted sum of
rotated quality factors and the cost objective is another factor of the same orthogonal basis, so
`Pearson(quality, cost)` is **zero by construction** — measured between 1e−17 and 5e−16 on all four
166-point reference sets. Criterion 1 asks whether two variables that are linearly uncorrelated by
construction are "clearly negatively" associated; its Spearman is rank-nonlinearity residual and has
no stable sign. **The instrument is broken, not the panel.**

The conflict between the **raw responses** and the leaf-count cost — which is what "these objectives
trade off" means physically — is clearly negative on every dataset: **−0.223, −0.500, −0.440,
−0.431**. Criteria 2, 3 and 4 are unaffected by the factor correction and are unchanged.

**What follows from this is NOT decided here.** Restating a frozen screening criterion after seeing
the numbers it produces is exactly the move the pilot/confirmatory boundary exists to prevent, and it
is not the author's to make silently. It is recorded as an open item in `PROTOCOL_AMENDMENTS.md` and
the protocol is not frozen until it is resolved.

**One observation to carry into the manuscript, against interest.** The gate as specified — external
R² ≥ 0.5 and Spearman ≥ 0.9 — now passes 7 of the 8 dataset-by-response cells, failing only
Spambase's quality surface at Spearman 0.847. A gate that almost never fires cannot do what Paper 1's
gate did, which was to identify unusable surfaces. That is not a reason to move the threshold after
the fact; it is a reason to say plainly that on this problem the surrogate is adequate nearly
everywhere, and that the gate is therefore a check rather than a discriminator.

## Measured cost, and what it implies for the campaign

Per-evaluation cost, eight threads, Apple M4 Max: 1.02 to 1.69 seconds, mean **1.40 s**.

With `B_surrogate_validation = 78`, at two objectives:

| Arm | `B_total_solution` |
|---|---|
| HISTORICAL-WS | 108 |
| WS-S | 186 |
| NBI-S | 186 |
| NBI-R | 386 |

Comparator budget 386. Evaluations actually performed per replication per dataset: 446 for the arms,
1,930 for the five comparators, 2,376 in total. Over 30 replications and 4 datasets that is
**285,120 evaluations, about 111 hours, or 4.6 days run serially.**

Three consequences, all decided here rather than after the campaign starts:

1. **The campaign runs at two objectives, not three.** At three objectives the same accounting gives
   357,120 evaluations and 5.8 days, over the ceiling. This closes the open question of whether to
   run at three objectives, with a measurement rather than a preference.
2. **The unmatched NSGA-II run is scoped down.** At ten times the matched budget on every replication
   it would cost 180 hours by itself. It runs instead on **one replication per dataset**, about 6
   hours, reported separately and labelled as a single-replication check on whether a starved
   population explains the matched result.
3. **The serial projection is the pessimistic one.** It assumes no parallelism across replications.
   Paper 1 faced the same arithmetic and found that running independent replications across eight
   processes with single-threaded evaluators cut a projected 106 hours to 15.5 actual. Stage B must
   measure the parallel throughput before the campaign launches; the 4.6-day figure is the number to
   beat, not the number to expect.
