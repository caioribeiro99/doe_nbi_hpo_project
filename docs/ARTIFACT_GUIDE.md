# Reviewer / artifact guide

If you are a reviewer of a future paper based on this code, start here.

## What lives where

| Branch / tag | Purpose | Status |
|---|---|---|
| `main` | Dissertation baseline as defended at UNIFEI. | Frozen, immutable. |
| tag `v0.1.0-dissertation` | Snapshot of `main` at dissertation submission. | Tagged. |
| `repo-publication-readiness` | Article-track scientific evolution. | Active. |
| `feature/usecase_fairness*` | Out-of-scope fairness extension. | Excluded from this audit. |

The article-track branch may legitimately diverge from the dissertation
text. Every divergence has an entry in `docs/METHODOLOGY_DECISIONS.md`
with citation and rationale.

## Where the methodological correctness lives

- `src/doe_xgb/nbi_core.py` — true N-objective NBI (Das–Dennis 1998;
  Pereira et al., 2025).
- `src/doe_xgb/scalarization.py` — legacy weighted-sum baseline; *not*
  NBI. Kept for ablation only.
- `src/doe_xgb/objectives.py` — explicit `ObjectiveSpec` with
  required direction.
- `src/doe_xgb/factor_model.py` — flexible factor decomposition (auto /
  fixed / manual / none).
- `src/doe_xgb/design/` — `DesignProvider` for external Minitab import,
  CCD/BB/LHS/Sobol/D-optimal generation, and simplex-lattice for NBI
  weights.
- `src/doe_xgb/model_families.py` — design-aware surrogates (process
  quadratic vs Scheffé mixture).
- `src/doe_xgb/post_optimization.py` — conditional MBPA.

## Recommended review order

1. `docs/METHODOLOGY_DECISIONS.md` — the running log of differences vs
   the dissertation. Has citations.
2. `docs/METHOD.md` — the article-track method in plain language.
3. `src/doe_xgb/nbi_core.py` — the headline correction.
4. `tests/methodology/` — pinning tests for NBI / FA / DOE / direction
   handling.
5. `configs/article_3vrf_xgb_magic.yaml` — the headline article config.
6. `examples/dissertation_xgboost_magic/` — minimal example to run.

## Reproducing the paper tables

See `docs/REPRODUCIBILITY.md`. In short:

```bash
make install-dev
make data
make repro-full
make tables
```

## Questions?

Open an issue on
[GitHub](https://github.com/caioribeiro99/doe_nbi_hpo_project/issues).
