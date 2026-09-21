# Proposed Zenodo archive — manifest and audit

Generated from the tracked tree at commit `c7c16be736588939c45ee4c910e1c359075bad98`. **326 files, 3.39 MB.**

The GitHub-release-to-Zenodo route would archive the whole repository, which also holds the previous paper's doctoral benchmark. This manifest is the curated alternative; every exclusion below is a stated decision.

## Included

| group | files | size | purpose | releasable |
|---|---|---|---|---|
| `src/` | 71 | 495 KB | campaign, arms, scoring and seeding implementation | yes |
| `papers/xgboost_hpo_vrfnbi/protocol/` | 12 | 144 KB | the protocol as frozen before the campaign | yes |
| `papers/xgboost_hpo_vrfnbi/scripts/` | 36 | 342 KB | analysis, audit and build scripts | yes |
| `papers/xgboost_hpo_vrfnbi/analysis/` | 9 | 499 KB | aggregated analysis artifacts every reported number is read from | yes |
| `papers/xgboost_hpo_vrfnbi/audits/` | 50 | 495 KB | screening, factor-model and identity audits | yes |
| `papers/xgboost_hpo_vrfnbi/manuscript/sections/` | 16 | 113 KB | manuscript source | yes |
| `papers/xgboost_hpo_vrfnbi/manuscript/figures/` | 6 | 276 KB | figures, generated from the artifacts | yes |
| `data/design/` | 4 | 6 KB | the 88-run design matrix and its metadata | yes |
| `data/source/` | 16 | 23 KB | dataset manifests and SHA-256 checksums — NOT the datasets | yes |
| `configs/` | 6 | 18 KB | frozen configuration | yes |
| `tests/` | 76 | 840 KB | the regression suite, including the reproducibility guards | yes |
| `docs/` | 19 | 205 KB | methodology decisions and engineering notes | yes |
| repository metadata | 5 | 15 KB | `pyproject.toml`, `README.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json` | yes |

## Excluded, and why

| path | reason |
|---|---|
| `jobs/` | 34.8 MB of OpenML CC-18 job-queue databases from the PREVIOUS paper's doctoral benchmark; irrelevant to this study |
| `experiments/` | stage-run summaries from the previous paper; they carry the author's home directory in 545 absolute paths |
| `benchmarks/` | task lists for the previous paper's benchmark |
| `article/` | the previous paper's LaTeX draft |
| `notebooks/` | exploratory, not part of the frozen pipeline |
| `examples/` | demonstration scripts for the previous paper |
| `papers/xgboost_hpo_vrfnbi/submission/` | journal correspondence — cover letter, title page, vitae, checklist. Submission strategy does not belong in a public reproducibility archive |
| `papers/xgboost_hpo_vrfnbi/vendor/` | third-party MathJax; MIT-licensed and redistributable, but it is a build dependency rather than research output |
| `papers/xgboost_hpo_vrfnbi/manuscript/Paper2_*` | compiled PDFs; the published article is the canonical copy |

## Restricted-material audit

Patterns searched: `/Users/`, `/home/`, `kaggle.json`, `token`, `password`, `secret`, `api_key`, `credential`, `.env`.

| pattern | file | occurrences |
|---|---|---|
| `/Users/` | `papers/xgboost_hpo_vrfnbi/protocol/original_thesis_protocol.md` | 1 |

`token` and `.env` match 63 times across 16 files, all ordinary prose — "numeric token", "tokenize", and the README's instructions for copying `.env.example`. No credential file is tracked.


## Datasets

`data/source/` carries manifests and SHA-256 checksums only. **No dataset is redistributed.** The four analysed datasets are public and are fetched by checksum-verified loaders; the archive reproduces the pipeline, not the data.

