# Self-overlap assessment (XGBoost HPO manuscript)

Prepared 2026-10-11 for revision R1 (internal action F9-E), on the model of
`papers/surrogate_nbi_ensemble/self_overlap_assessment.md`. It compares the present manuscript with the authors' own
related work, dimension by dimension, and is reconciled with the Declarations (Section 11). Items marked
**to confirm** could not be established from the repository or the public record and are listed for the authors.

The three related works:

- **Pereira et al. (2025)**, *Engineering Applications of Artificial Intelligence* 162:112510: a hybrid multivariate
  Normal Boundary Intersection with mixture-design post-optimization, applied to hard turning of AISI H13 steel.
- **The companion manuscript** (`papers/surrogate_nbi_ensemble`, Paper 1): surrogate-assisted NBI for weighting a
  five-member classifier ensemble on four financial-services datasets.
- **The first author's master's dissertation** (Ribeiro 2026, UNIFEI, handle 123456789/4372, open access): DoE, response
  surfaces, factor analysis and NBI terminology for XGBoost hyperparameter optimization; archived code at tag
  `v0.1.0-dissertation`.

## 1. Dimension by dimension

| Dimension | Pereira et al. (2025) | Paper 1 | Dissertation | Overlap with the present manuscript |
|---|---|---|---|---|
| Research question | post-Pareto selection over the weight simplex | when surrogate-assisted ensemble weighting is trustworthy | multiobjective XGBoost tuning with DoE and NBI terminology | Low for Pereira and Paper 1. The dissertation's method is the object of study here; the question (which of three mechanisms moves the returned front) is new. |
| Datasets | one manufacturing case study | Santander, BNP Paribas, Porto Seguro, UCI credit | MAGIC plus two further datasets, unnamed on the repository record (one balanced, one highly imbalanced) | MAGIC is shared with the dissertation and declared. No dataset is shared with Paper 1 or Pereira et al. **To confirm:** whether either of the dissertation's further datasets is Adult, Bank Marketing or Spambase. |
| Decision space and bounds | three machining parameters | five ensemble weights on a simplex | the same seven XGBoost hyperparameters and bounds | Shared with the dissertation and declared (Section 5.2, Table 2). |
| Experimental design | 19-run central composite design | 66-run mixture design | the same 88-run face-centred central composite design | Shared with the dissertation and declared (Section 5.3). |
| Objective construction | Varimax-rotated factor scores | ROC-AUC, log-loss and inference cost, no factor stage | PCA/Varimax latent objectives | The latent-objective construction descends from the dissertation and is declared; its specification here (six quality responses plus log leaf count, rotated-share weights, orientation rule) is this study's. |
| Optimizer | NBI with mixture-design post-optimization | NBI variants on the simplex; NSGA-II baseline | normalized weighted-sum scalarization described with NBI terminology | The archived solver is called unmodified as one arm (HISTORICAL-WS-asrun); canonical NBI is this study's own implementation (`src/doe_xgb/nbi_core.py`). |
| Reliability gate | none | external R2 >= 0.5 and Spearman >= 0.9 | none | Thresholds inherited from Paper 1 as a transfer, not a calibration; declared in Section 5.3. |
| Code | not in this repository | `src/mixens` | `v0.1.0-dissertation` | The archived solver is executed as archived; every other component of the campaign is this study's code. |
| Prose | not reused | not reused | not reused, per the Declarations | **To confirm:** no line-by-line comparison with the dissertation PDF was made for this assessment. |
| Figures and tables | none reused | none reused | none reused; the bounds are restated in Table 2 | Declared in the overlap statement. |
| Results | none used | none used | none used; no claim is made about the dissertation's reported results | None. |

## 2. Reconciliation with the Declarations

The overlap statement (Section 11) names exactly the shared elements above: the archived implementation, the
latent-objective construction, MAGIC, the seven hyperparameters and their bounds, and the 88-run design. It states
that the other three datasets were selected for this study, which holds whatever the dissertation's further datasets
were; the open question is whether one of them coincides with Adult, Bank Marketing or Spambase, in which case it must
also be named as shared. The dissertation reference (title, handle, status) was verified against the UNIFEI
repository record on 2026-10-11.
