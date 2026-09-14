# Cost objective: what may and may not be claimed

## What the optimizer optimizes

**Total leaf count of the fitted ensemble**, summed over trees, averaged over the cross-validation
folds, `log1p`-transformed before the factor stage. Frozen in `protocol/EXPERIMENT_PROTOCOL.md`
§6.1 and §6.3. It does not change after results are seen.

## Why it is not training time

The dissertation's cost objective was mean per-fold wall-clock training time. It is not reproducible
from seeds: re-executing the frozen pipeline under two environments gave a quality surface agreeing
to six decimals and a cost surface that did not, and two fits of one configuration differ by 1.3%
(median) in measured time. An objective that changes between runs of the same seed cannot anchor a
paired comparison across 30 replications.

## The association, and its limit

Leaf count tracks measured training time at **Spearman ≈ 0.86** in the pre-campaign calibration
experiment (`scripts/cost_objective_selection.py`, `audits/cost_objective_selection.json`). No
deterministic proxy did better, and elaborating a closed-form proxy with the sampling
hyperparameters made it markedly worse, down to 0.47.

0.86 is a strong association and it is not identity.

## Terminology, binding

**Use:**

- "deterministic model-complexity proxy"
- "leaf-count complexity objective"
- "model complexity, measured as total leaf count"

**Do not use, unqualified:**

- "training time"
- "wall-clock cost"
- "computational time"
- "computational cost"

When computational relevance is discussed, the sentence must say that measured training time is
reported as a **secondary audit variable** and give the measured association rather than implying
equivalence.

Enforced by `scripts/check_claim_blacklist.py`.

## What is persisted

For **every** real evaluation, both:

| Quantity | Role |
|---|---|
| total leaf count | the frozen optimization objective |
| measured fit and evaluation wall-clock time | secondary empirical response, and the campaign's own cost measure |

Both go in every candidate record and every design record. The campaign reports the leaf-count to
wall-clock association over the full campaign, at 30 replications and four datasets, which is the
first time that association is measured at scale rather than on one partition.

## What must not happen

Switching the optimization objective to wall-clock time after seeing results, in either direction.
The objective is frozen. If the full-campaign association turns out weaker than 0.86, that is a
reported finding about the proxy, not a reason to re-run with a different objective.
