# Changelog

All notable changes to this repository are tracked here.
Format: human-readable; loosely follows Keep a Changelog. Versioning follows
the branch / tag policy described in `docs/ARTIFACT_GUIDE.md`.

## [v0.1.0-dissertation] — 2026-05-02

Tag of `main` at the state defended in the master's dissertation (UNIFEI).
This is the immutable historical baseline. No further commits should land on
`main` outside of trivial repo hygiene.

Highlights of this state:
- DOE + PCA/Varimax + RSM (uncoded units, full quadratic + backward elimination
  with α = 0.05) + a beta-grid scalarized search documented as "NBI-like".
- 30-replica protocol on UCI MAGIC Gamma Telescope (canonical) plus two
  external validation datasets.
- Benchmarks: coarse grid, random, scikit-optimize Bayes, Hyperopt TPE — all
  matched to the same evaluation budget.

## [Unreleased] — branch `repo-publication-readiness`

The article-track evolution. Goal: a publishable, fully reproducible artifact
suitable as supplementary material for a future A1-level journal article.

The methodological compass for this branch is the EAAI 2025 article
"A hybrid multivariate normal boundary intersection approach with
post-optimization assisted by mixture design of experiments"
(Pereira, Tertuliano Ribeiro, et al., 2025;
DOI: 10.1016/j.engappai.2025.112510).

See `docs/METHODOLOGY_DECISIONS.md` for the running log of every divergence
between this branch and the closed dissertation text, and the rationale.
