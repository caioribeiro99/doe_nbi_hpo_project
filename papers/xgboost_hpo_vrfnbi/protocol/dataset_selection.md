# Dataset pre-selection

**Purpose.** Choose a panel that can discriminate between the four arms, before the protocol is
frozen, and say what each member is there to do.

**Constraint from Paper 1.** Four datasets produced a cross-dataset *pattern*, not a tested claim,
and the manuscript had to say so. The same limit applies here. The dataset is the unit of
generalization; nothing is pooled across datasets.

---

## What a discriminating dataset looks like here

The four arms differ in reference construction, scalarization geometry and anchor provenance. A
dataset separates them only if it makes at least one of those matter. That requires:

1. **Genuine objective conflict.** If quality and cost are not in tension over the hyperparameter
   box, every arm returns nearly the same point and the comparison is empty. Measured before the
   campaign, on the 88 design rows, as the Spearman correlation between the **raw quality responses**
   and the **raw leaf-count complexity response** — the canonical
   `doe_xgb.campaign.factor_model.raw_conflict`, with no factor stage on either side. Conflict needs
   a clearly negative value. **The latent quality-factor versus latent cost-factor correlation is
   NOT this criterion and is permanently withdrawn** (amendment 20): the two latent axes come from
   one orthogonal rotated basis, so their linear association is zero by construction on the fitting
   sample and the rank statistic has no consistent direction across datasets.
2. **A front with curvature.** A linear front is the case where weighted sum and NBI agree by
   construction (`audits/NBI_GEOMETRY_AUDIT.md`). Datasets whose fronts are near-linear cannot
   detect the geometry factor. Measured on the design rows' non-dominated set.
3. **A surrogate that is neither perfect nor useless.** Paper 1 found surrogate adequacy to be both
   dataset- and metric-dependent, and that this dependence drove the headline result. A panel where
   every surface passes, or none does, loses that axis.
4. **Enough cost range to matter.** If the most and least expensive configurations differ by a few
   percent, the cost objective is noise.

## The proposed panel

| Dataset | Registry id | Role | Why |
|---|---|---|---|
| MAGIC Gamma Telescope | `magic` | **historical continuity** | the dissertation's own dataset; the only member that lets HISTORICAL-WS-asrun be compared with the dissertation's reported results |
| Adult / Census Income | `adult` | scale and mixed types | ~49k rows, eight categorical and six numeric columns, imbalanced; the cost objective has real range here |
| Spambase | `spambase` | small and dense | ~4.6k rows, all numeric; cheap, so it carries the pilot and the sensitivity checks |
| Bank Marketing | `bank_marketing` | imbalance with categoricals | independent of Adult in feature construction, similar in burden |

All four are in the article-track registry (`src/doe_xgb/datasets/registry.py`), all report
`available` in `data/source/AVAILABILITY_CHECK.md`, all are UCI or OpenML mirrors under CC BY 4.0,
and each carries a manifest with a SHA-256 for the raw file. MAGIC is already cached and its
checksum verified byte for byte (`audits/provenance/`); the other three are not yet cached and must
be fetched and verified before the pilot.

**The replacement rule, as frozen.** A dataset that fails criterion 1 or 2 is replaced from the
registry before the full campaign, and the replacement is recorded with the measurement that caused
it.

## Panel status — SETTLED before the confirmatory campaign

Measured on the corrected factor model, before any arm ran. Authoritative artifact:
`audits/final_panel_screening.json`. (`audits/panel_rescreen_corrected.json` is a **superseded**
intermediate recomputation and must not be cited.)

| dataset | C1 raw conflict | C2 front / curvature | C3 | C4 | role |
|---|---:|---|:--:|:--:|---|
| MAGIC | −0.251 | 7 points, 0.173 | met | met | primary geometry confirmatory |
| Adult | −0.398 | 6 points, 0.199 | met | met | primary geometry confirmatory |
| Bank Marketing | −0.410 | 6 points, 0.198 | met | met | primary geometry confirmatory |
| **Spambase** | −0.462 | **2 points, undefined** | met | met | **boundary geometry control** |

**Criterion 1** passes on all four under the canonical raw-response measurement (amendment 20).

**Criterion 2: Spambase fails**, and the criterion was not weakened to avoid it — its threshold is
unmoved, its definition unrewritten, and no alternative formulation was sought. A two-point
non-dominated set has no interior: every scalarization returns the same two extreme points, so there
is no interior front geometry for the weighted-sum-versus-NBI contrast to separate.

**The replacement rule was considered and deliberately not exercised** (amendment 22). Spambase is
**retained and executed in full** — same arms, same budgets, same seeds, same evaluation machinery,
30 replications — and reclassified prospectively as a **boundary geometry control**. **No replacement
dataset was selected.** It is excluded only from the primary inferential family, and the
interpretation of its result is fixed in advance in `EXPERIMENT_PROTOCOL.md` §11.1: a null geometry
difference there is not evidence against the mechanism, because the screening established before
execution that the required interior geometry is absent.

## Candidates held in reserve

`phishing`, `credit_card_default`, `mushroom`, `wine_quality`, `german_credit`, `pima_diabetes`,
`breast_cancer`. `dry_bean` is multiclass and out of scope for this paper.

`credit_card_default` is deliberately **not** in the primary panel despite being a natural fit:
Paper 1 uses the same UCI dataset (id 350), and reusing it would make the two papers share a
dataset, which invites a self-overlap question for no scientific gain.

## The unresolved historical dataset

The Paper-2 mandate names **`pepper_species`** as a historical dataset alongside MAGIC.

**It could not be located.** A search of the whole repository for `pepper` and `pimenta` across
Python, Markdown, YAML and JSON returns nothing. It is not in the article-track registry, not in
`docs/DATASETS.md`, not in the dissertation-parity config, and not in the frozen dissertation code,
which takes a single `--dataset` path per run and whose committed config points at MAGIC.
`article/DISSERTATION_TO_ARTICLE_MAP.md` says the dissertation used "1+2" datasets without naming
the other two.

**What is needed:** the dissertation's results chapter, which is not on this machine (see
`protocol/original_thesis_protocol.md` §13 for where it is), or the dataset file itself.

**Decision:** the panel above does not depend on it. If `pepper_species` is recovered and is public,
it joins the panel as a second continuity dataset. If it is not public it cannot be used, since the
protocol requires every dataset to be independently obtainable with a published checksum. Either way
this is not blocking, and it is tracked in `protocol/EXPERIMENT_PROTOCOL.md`.

## Pre-campaign screening, to be run in the pilot

For each candidate, on one partition, from the 88 design rows alone:

| Measurement | Threshold to stay in the panel |
|---|---|
| Spearman between the **raw quality responses** and the **raw leaf-count cost** | clearly negative (amendment 20: the latent-composite estimator is withdrawn, the construct is kept) |
| curvature of the non-dominated set of design rows | detectably non-linear |
| external R² of each surrogate on held-out compositions | strictly between the trivial and the perfect |
| ratio of the most to the least expensive configuration | large enough that cost differences exceed run-to-run variation |

The thresholds are stated as numbers in `protocol/EXPERIMENT_PROTOCOL.md` before the pilot runs, not
after, so that they cannot be chosen to keep a dataset that the measurement rejects.
