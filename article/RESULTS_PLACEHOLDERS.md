# Results placeholders

This file lists every table and figure the article expects, with the
exact aggregation source and the section it ships in. Fill these in
**only** after the v1 campaign aggregates cleanly. Do not invent
numbers.

## Tables

| Tag | Section | Title | Source CSV (planned) |
|---|---|---|---|
| `tab:datasets` | 4 | Dataset panel summary | `article/tables/datasets.csv` (hand-curated; see `EXPERIMENT_PLAN_V1.md`). |
| `tab:hpo-spaces` | 4 | Per-algorithm hyperparameter search spaces | `article/tables/hpo_spaces.csv` (hand-curated). |
| `tab:doe-design` | 4 (appendix) | CCDFC matrix excerpt + metadata | `data/design/hyperparameter_design.csv` (committed) + `*.metadata.json`. |
| `tab:performance` | 5 | Predictive performance per (dataset, alg, method) | `article/tables/performance.csv` <- aggregator on per-replica `confirmation_summary.csv`. |
| `tab:cost` | 5 | Computational cost per (dataset, alg, method) | `article/tables/cost.csv` <- aggregator on per-replica `confirmation_summary.csv`. |
| `tab:nbi-residuals` | 5 | NBI sub-problem residual statistics | `article/tables/nbi_residuals.csv` <- aggregator on per-replica `nbi_subproblem_diagnostics.csv`. |
| `tab:vrf-loadings` | 5 | Rotated factor loadings per dataset | `article/tables/vrf_loadings.csv` <- aggregator on per-replica `factor_loadings.csv` and `factor_diagnostics.json`. |
| `tab:mbpa` | 5 | MBPA trigger summary | `article/tables/mbpa_summary.csv` <- aggregator on per-replica `frontier_quality.json` and `post_optimization_diagnostics.json`. |
| `tab:friedman` | 5 | Friedman ranks + Nemenyi pairs | `article/tables/friedman_nemenyi.csv` <- cross-dataset stats notebook. |
| `tab:ablation` | 5 | Legacy weighted-sum vs true NBI ablation | `article/tables/ablation.csv` <- separate runs of `--method legacy_scalarization` vs `--method nbi`. |
| `tab:artifacts` | 5 | Reproducibility artifact summary | `article/tables/artifacts.csv` <- count + sha256 audit on `experiments/`. |

## Figures

| Tag | Section | Title | Source |
|---|---|---|---|
| `fig:pipeline` | 3 | Article-track pipeline overview | hand-drawn; export `article/figures/pipeline.pdf`. |
| `fig:nbi-vs-ws` | 5.4 | True NBI vs weighted-sum on a curved 2D Pareto front | `notebooks/02_nbi_vs_weighted_sum_demo.ipynb`. |
| `fig:pareto-<dataset>-<alg>` | 5.1 | Per-(dataset, algorithm) frontier plot | aggregator notebook to be added. |
| `fig:cd-diagram` | 5.3 | Critical-difference diagram across datasets | cross-dataset stats notebook. |
| `fig:vrf-cluster` | 5.5 | Varimax rotated factor loadings (heatmap) | per-dataset variant; example notebook to be added. |
| `fig:mbpa-trigger` | 5.6 | MBPA trigger frequency vs frontier-quality | per-dataset summary notebook. |
| `fig:cost-box` | 5.2 | Boxplot of mean time per fold by method | aggregator notebook. |
| `fig:residual-hist` | 5.4 | NBI sub-problem residual histogram | aggregator notebook. |

## Discipline

- Every numeric claim in the prose must point at a row in one of the
  CSVs above. No hand-typed numbers.
- Every figure must list its provenance file in the LaTeX caption
  (e.g., `Source: \texttt{article/tables/cost.csv}`).
- Tables that describe the protocol (datasets, hyperparameter spaces,
  DoE matrix) are hand-curated but must match the YAML configs.

## Versioning

Every aggregated CSV/PDF must include a header row or footer giving:

- the GitHub release tag of the campaign;
- the SHA-256 of the input dataset(s);
- the SHA-256 of the design CSV;
- the `tree_method` and `n_jobs` used.

The aggregation script writes these in a `_provenance` JSON sibling
file.
