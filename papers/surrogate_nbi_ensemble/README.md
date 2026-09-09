# Manuscript workspace: surrogate-assisted NBI for classifier ensemble weighting

Working title: *When Is Surrogate-Assisted Multiobjective Ensemble Weighting Trustworthy? A Replicated Study of
Mixture Designs, Scheffé Surfaces and Normal Boundary Intersection on the Classifier-Weight Simplex.*

Branch `paper/surrogate-nbi-ensemble-weighting`, cut from the frozen R = 30 state (tag `pco213-postwork-r30`,
commit `b3ed050`). The computational study is complete and frozen; nothing in this directory modifies it.

## Build

```bash
tectonic -X compile main.tex --outdir build     # or: pdflatex main; bibtex main; pdflatex main x2
python build_assets.py                          # regenerates all figures and tables from frozen artifacts
```

`build_assets.py` reads only `reports/pco213_postwork_benchmark/` and `experiments/pco213_postwork_benchmark/` and
writes only into `figures/` and `tables/` here. Figures require the unversioned raw artifacts (829 MB); the LaTeX
tables and the six statistics-derived figures need only the committed CSVs.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Manuscript root; `sections/00`–`09` are `\input` in order |
| `sections/` | Abstract, introduction, related work, methodology, protocol, results, discussion, limitations, conclusion, reproducibility |
| `references.bib` | 179 verified entries, deduplicated by DOI across seven literature clusters |
| `figures/` | 8 manuscript figures, `.pdf` and `.png` |
| `tables/` | 7 LaTeX tables plus 3 derived CSVs |
| `build_assets.py` | Regenerates every figure and table from the frozen artifacts |

## Supporting documents

Read these before editing the manuscript. They constrain what it may claim.

| Document | What it settles |
|---|---|
| `research_lineage.md` | What is INHERITED (4 components), ADAPTED (7) and NEW (7). **The DoE–RSM–NBI-over-ensemble-weights construction is prior work and may not be claimed.** Lists forbidden phrasings. |
| `claims_and_evidence.md` | Every claim C1–C14 mapped to its source CSV, statistic, interval, effect size, figure, limitation and evidence tier, plus the nine-item claim blacklist. |
| `novelty_matrix.md` | Verdicts on 17 elements: KNOWN / PARTIAL / NO DIRECT MATCH FOUND / OWN PRIOR WORK. |
| `literature_review.md` | Cluster-by-cluster review and the comparison table; ten closest prior works. |
| `self_overlap_assessment.md` | 13-dimension comparison against the 2025 EAAI predecessor; citation wording; residual risks. |
| `baseline_gap_assessment.md` | Whether NSGA-II is required. Verdict USEFUL→LIKELY REQUIRED, with measured cost (1.3 h / 18.4 h). |
| `venue_analysis.md` | 12 title candidates scored; 5 venues ranked; prepared answers to 7 reviewer concerns. |
| `supplementary_plan.md` | S1–S12 allocation of the remaining 27 figures and all statistics CSVs, including a corrections log. |

## Three rules

1. **Do not claim the pipeline.** Mixture design over combination weights + polynomial surrogate + NBI is published
   prior work, including for model ensembles (Bacci 2019; Moreira 2021; Rocha 2025; Pereira 2025). The contribution is
   the evaluation architecture and the findings.
2. **Every quantitative sentence traces to `claims_and_evidence.md`.** If a number is not there, it does not go in.
3. **The experiment is frozen.** No result may be recomputed, and no protocol element changed, at tag
   `pco213-postwork-r30`. Additive work (for instance an NSGA-II arm) writes to a new directory and is tagged
   separately.

## State

Complete first draft: all nine sections written, 8 figures, 7 tables, 86 citations resolving against 179 verified
references, compiles clean with no undefined citations or references.

**Next actions before submission**, in order: (1) compress the body from about 16,700 words toward the 10,000-word
target, mainly in Results; (2) read the two unread lineage papers named in `research_lineage.md`; (3) decide the
NSGA-II arm per `baseline_gap_assessment.md`; (4) confirm authorship and funding; (5) produce the supplementary PDF
per `supplementary_plan.md`.
