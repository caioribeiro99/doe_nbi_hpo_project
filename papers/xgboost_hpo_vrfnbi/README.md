# Paper 2 — multiobjective XGBoost hyperparameter optimization

Workspace for the follow-up to the first author's master's dissertation. Branch
`paper/xgboost-hpo-vrfnbi-trustworthiness`, cut from the article track
(`repo-publication-readiness`).

**Status: pre-campaign.** The protocol is decided and ready to freeze. No campaign has run.

---

## Read in this order

| # | Document | What it settles |
|---|---|---|
| 1 | `audits/METHODOLOGICAL_IDENTITY_AUDIT.md` | the dissertation optimizer is weighted-sum scalarization, not NBI, and its weight grid is asymmetric |
| 2 | `audits/PCA_VARIMAX_IDENTITY_AUDIT.md` | what the factor stage computes, and that its aggregation weighting moves the objective |
| 3 | `audits/NBI_GEOMETRY_AUDIT.md` | a defect in the article-track NBI that made non-convex fronts unreachable, found and fixed |
| 4 | `audits/provenance/README.md` | the dissertation pipeline re-executed, and the one objective that does not reproduce |
| 5 | `protocol/original_thesis_protocol.md` | the historical protocol reconstructed from the frozen code |
| 6 | `research_lineage.md` | what is inherited, adapted and new, and from whom |
| 7 | `literature_review.md` | the novelty boundary, from verified records |
| 8 | `novelty_matrix.md` | every claimable element with its status, and the claim blacklist |
| 9 | `baseline_gap_assessment.md` | which comparators the result needs to be believed |
| 10 | `protocol/method_arms.md` | the four arms and what each contrast identifies |
| 11 | `protocol/budget_accounting.md` | the evaluation ledger |
| 12 | `protocol/dataset_selection.md` | the panel and the screening it must pass |
| 13 | `protocol/EXPERIMENT_PROTOCOL.md` | the protocol |
| 14 | `protocol/protocol_adversarial_review.md` | four adversarial roles, eleven must-fix findings, all resolved |

## The short version

The dissertation's optimizer is a normalized weighted sum, not Normal Boundary Intersection, and the
author had already recorded that in `docs/METHODOLOGY_DECISIONS.md` D1 before this audit. That single
fact drives the design: a canonical NBI differs from the historical method in **two** ways at once,
the scalarization geometry and the reference construction, so a two-arm comparison identifies
neither. Hence four arms — HISTORICAL-WS, WS-S, NBI-S, NBI-R — forming a two-by-two on those factors
plus anchor provenance.

What the paper may claim is not a new combination of methods. Every ingredient is prior art, some of
it the authors' own. What is left is the controlled decomposition and whatever it measures, and the
framing for all three possible outcomes is written down in advance in
`protocol/protocol_adversarial_review.md`.

## Scripts

| Script | Purpose | Interpreter |
|---|---|---|
| `scripts/methodological_identity_audit.py` | asserts, from the frozen source, which optimizer the dissertation ran | current |
| `scripts/provenance_reproduce_dissertation.py` | re-executes the frozen DoE, factor and surface stages | **era venv** |
| `scripts/pca_varimax_identity_audit.py` | answers five questions about the factor stage by measurement | **era venv** |
| `scripts/cost_objective_selection.py` | chooses the cost objective by measurement | current |
| `scripts/check_claim_blacklist.py` | fails a build that asserts a blacklisted claim | current |

Two scripts need an interpreter contemporary with the dissertation, because the frozen code cannot
run on pandas 3:

```
uv venv --python 3.11 /tmp/venv-diss
VIRTUAL_ENV=/tmp/venv-diss uv pip install 'numpy<2' 'pandas<3' 'scikit-learn<2' \
    'xgboost>=2.0' 'scipy>=1.11' 'statsmodels>=0.14' tqdm
```

`provenance_reproduce_dissertation.py` refuses to run on pandas 3 and prints this recipe rather than
patching a historical artifact.

## Tests

`tests/methodology/test_nbi_geometry_validation.py`, 12 tests, including a closed-form non-convex
Pareto front on which weighted sum provably returns only anchors and NBI must return the interior
solution. One test asserts that the old `t >= 0` restriction still fails that case, so the reason for
the current default cannot be lost.

## Rules that bind this workspace

- No historical artifact is edited. The frozen dissertation code is extracted to a scratch directory
  and read or executed there.
- No citation is used that has not been resolved against Crossref, arXiv or the publisher's record.
- The claim blacklist in `novelty_matrix.md` is enforced by `scripts/check_claim_blacklist.py`, not
  by a checklist.
- The discrepancy between the dissertation's text and its code is reported plainly and once. It is
  not sensationalized, and it was recorded by the author before it was audited.
- "Pereira et al. (2025) formulate the method using canonical NBI" is permitted. "Their code
  implements canonical NBI" is not: no implementation other than the two audited here was examined.
- Raw datasets and heavy experiment caches stay unversioned.
