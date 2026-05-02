# Dissertation -> Article mapping

This map records what each article section reuses from, departs from,
or replaces relative to the closed dissertation
([Ribeiro 2026](../docs/METHODOLOGY_DECISIONS.md)). The dissertation is
the **historical baseline**; the article-track repository is the
**evolution**. The article must not pretend the two implementations
are identical.

Source PDFs live under
`/Users/caiotertuliano/Library/CloudStorage/OneDrive-Pessoal/BackUp PC/Documentos/CAIO/Material Didático/UNIFEI/Doutorado/Artigos/Artigo da Dissertação/Tradução da Dissertação/`:

- `Dissertacao_Caio_01_introduction_en.pdf`
- `Dissertacao_Caio_02_theoretical_framework_en.pdf`
- `Dissertacao_Caio_03_methodology_en.pdf`
- `Dissertacao_Caio_04_results_discussion_en.pdf`
- `Dissertacao_Caio_05_conclusions_en.pdf`
- `Dissertacao_Caio_06_referencias_p165-p172.pdf`

## Section-by-section mapping

| Article section | Dissertation source | Reuse / departure |
|---|---|---|
| `00_abstract.tex` | Resumo / Abstract (front matter) | **Rewrite.** Drop XGBoost-only framing; emphasize three GBDT families and the explicit methodological refinements. |
| `01_introduction.tex` | Ch.1 Introduction (`Dissertacao_Caio_01_introduction_en.pdf`) | **Adapt + extend.** Keep the supervised-learning framing, the multi-objective motivation, and the four-method benchmark family. Add: explicit acknowledgement that the dissertation "NBI" was a weighted-sum scalarization (METHODOLOGY_DECISIONS D1); explicit positioning relative to Pereira et al. (2025) EAAI. |
| `02_related_work.tex` | Ch.2 Theoretical Framework | **Condense heavily.** A journal article is not a dissertation literature review. Keep one paragraph per topic: HPO baselines; multi-objective HPO; multivariate NBI / VRF; mixture DoE for post-optimization. Cite the dissertation for the long survey. |
| `03_method.tex` | Ch.3 Methodology + new article-track modules | **Rewrite.** The article method is the article-track method, not the dissertation method. Cover: ObjectiveSpec, FactorModel modes, DesignProvider, design-aware surrogates, **true N-objective NBI** (replaces the dissertation's weighted-sum), conditional MBPA. Keep the dissertation's CCDFC + Varimax + RSM machinery as the building blocks, since those are unchanged. |
| `04_experimental_design.tex` | Sect.3.5 -- 3.13 (CV protocol, replicas, deterministic mode) | **Adapt.** Same protocol skeleton (5-fold CV, R replicas, fairness-by-evaluations budget). Differences: 12 datasets vs 1+2 in the dissertation; 3 GBDT algorithms vs XGBoost only; multiclass support; calibration metrics added; explicit cost-estimator gate. |
| `05_results.tex` | Ch.4 Results | **Replace.** All numbers must come from the v1 campaign on the new panel. Only the qualitative claims (NBI residuals are small; the proposed method dominates on cost) carry over as hypotheses to verify. |
| `06_discussion.tex` | Ch.4 Section 4.8 multivariate analysis + Ch.5 Conclusions sect. 5.1 -- 5.2 | **Adapt.** Reuse the discussion structure (qualitative vs cost-quality; threats to validity) and add the MBPA trigger discussion. |
| `07_conclusion.tex` | Ch.5 Conclusions | **Rewrite.** Article-track wording; future-work bullets reduced to three. |
| `references.bib` | Ch.6 Referencias | **Curate.** Drop the dissertation-specific references that the article does not actually use. Keep Das & Dennis 1998, Pereira et al. 2025, Chen & Guestrin 2016, Ke et al. 2017, Prokhorenkova et al. 2018, Bergstra & Bengio 2012, Bergstra et al. 2011 TPE, Shahriari et al. 2016 BO survey, Demsar 2006, Montgomery 2017, Myers et al. 2016, Cornell 2002, Scheffe 1958, Kaiser 1958. |

## Figures and tables

| Asset | Source | Action |
|---|---|---|
| Pipeline flowchart (Fig.\ 4 in the dissertation) | `Dissertacao_Caio_03_methodology_en.pdf` | **Regenerate.** The article-track pipeline has new stages (ObjectiveSpec, design-aware surrogates, MBPA). Draw a fresh diagram. |
| CCDFC matrix excerpt (Fig.\ 6) | Ch.4 | **Reuse** as a table; cite the canonical CSV at `data/design/hyperparameter_design.csv`. |
| Pareto-effects per response (Figs.\ 7--11) | Ch.4 | **Regenerate** for each (dataset, algorithm). |
| Varimax factor-cluster diagram (Fig.\ 12) | Ch.4 | **Regenerate** for the v1 panel; the article expects three constructs but the FactorModel mode is configurable. |
| Boxplots of cost (Figs.\ 26--28) | Ch.4 | **Regenerate** with three GBDT algorithms in the rows. |
| Pareto / NBI Pareto-set figures (Figs.\ 21--25) | Ch.4 | **Regenerate** with the true N-objective NBI; the legacy weighted-sum result is included as ablation. |
| RSM surfaces (Figs.\ 36--40) | Ch.4 | **Regenerate** in coded units (article default); the dissertation showed uncoded units (METHODOLOGY_DECISIONS D6). |

## Claims still supported by the dissertation

These can be cited as prior evidence in the introduction / discussion
without re-running the experiment:

- DoE + RSM + Varimax-rotated factor scores produces a parsimonious,
  interpretable cost-quality decomposition for XGBoost on MAGIC.
- A 30-replica protocol stabilizes the multivariate distance measures
  used to compare methods.
- CCDFC with 88 runs gives sufficient coverage for a quadratic RSM on
  $k{=}7$ XGBoost factors with $\alpha{=}0.05$ backward elimination.

## Claims that REQUIRE new experiments

- Any quantitative comparison between **true** $N$-objective NBI and
  the dissertation's weighted-sum scalarization.
- All numbers for LightGBM and CatBoost.
- All multiclass numbers (the dissertation was binary-only).
- Any frontier-quality and MBPA-trigger statistics.
- All calibration metrics (Brier, ECE) -- the dissertation did not
  report them.

## Article-track divergences (binding)

These divergences are documented in `../docs/METHODOLOGY_DECISIONS.md`
and must be honored in the article narrative:

| ID | Topic | Article-track |
|---|---|---|
| D1 | "NBI" labeling | True NBI replaces weighted sum; legacy preserved as ablation. |
| D2 | $N$-objective generality | Math kernel works for $q \ge 2$; no hardcoded $q$. |
| D3 | Direction handling | Required, explicit; no name-based inference. |
| D4 | FMSE wrapping | Default for VRF objectives; raw mode for direct ML metrics. |
| D5 | Anchor source | Surrogate-based per-objective optimization; box constraints by default for HPO. |
| D6 | RSM coding | Coded units default for article runs; uncoded available for dissertation parity. |
| D7 | Selection rule | `distance_to_utopia` for the proposed method; `max_accuracy` for benchmarks. |
| D8 | Factor count | Configurable (auto / fixed / manual / none); article default = fixed=3. |
| D9 | Quasi-normal | Das-Dennis $-\Phi \mathbf{1} / \|\Phi \mathbf{1}\|$ default. |
| D10 | MBPA | Conditional by default; logged whether triggered or skipped. |
| D11 | Per-fold metrics | Persisted in long-format CSV. |
| D12 | Dataset / design provenance | SHA-256 + downloader; design CSV committed. |
| D13 | Determinism | `tree_method=exact, n_jobs=1` for headline tables. |
| D14 | Mixture-model basis | Scheff\'e canonical; backward elimination disabled. |
