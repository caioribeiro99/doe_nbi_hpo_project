# Internal smoke / profiling panel (NOT the doctoral benchmark)

This directory holds the **12-dataset internal panel** that the
repository accumulated through earlier commits (see `data/source/*` and
`src/doe_xgb/datasets/registry.py`). It is **not** part of the
doctoral benchmark.

The doctoral benchmark proper is the **OpenML-CC18 suite**
(72 tasks; `benchmarks/doctoral/openml_cc18/`).

## Purpose of this panel

- **Smoke tests** for the article-track pipeline
  (`scripts/run_article_true_nbi_magic_smoke.py`,
  `scripts/run_v1_binary_smoke.py`,
  `scripts/run_v1_binary_3alg_smoke.py`).
- **Runtime profiling** to anchor cost-estimator multipliers
  (`scripts/profile_v1_full_dataset_runtime.py`).
- **Dissertation-continuity examples**: MAGIC and Credit Card Default
  reproduce the dissertation's headline analyses.
- **Development fixtures** for unit and integration tests.

## Files

- `datasets.csv` — the 12 entries with `include=True,
  loader_status=registered` (their `manifest.json` files live under
  `data/source/<id>/`).

## What this panel does NOT do

- Drive the doctoral campaign.
- Provide headline scientific results for the doctorate / paper.
- Replace any OpenML-CC18 task.

## Migration note

`benchmarks/doctoral_82/` is **deprecated** (Commit 25). The 12 entries
in this panel are the same ones that lived there. The 70-row "pending"
slot list has been retired together with the "82" framing.
