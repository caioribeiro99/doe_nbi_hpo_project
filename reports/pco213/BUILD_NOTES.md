# BUILD NOTES — Relatório final PCO213

## Template e formato

- O material da disciplina (diretório OneDrive do autor) foi copiado para `source_materials/`
  (2 PDFs de descrição/avaliação, feedback de artigo e `template_relatrio.zip`).
- O template extraído (`template/template_relatório/`) é **LaTeX IEEE conference**
  (`IEEEtran.cls` + `Article_PCO213.tex` exemplo). O formato do template foi adotado
  como fonte de verdade: o artigo é editado diretamente em LaTeX
  (`article/Article_PCO213.tex`), **sem** Quarto — desnecessário aqui, já que o template
  é nativamente LaTeX.
- Limite de páginas: descrição do projeto exige ≤6; o PDF final tem **6 páginas**.
- Requisitos do feedback da disciplina atendidos: diagrama geral (Fig. 1), fundamentação
  teórica (14 referências), sem código no corpo (equações (1)–(4) + fluxograma +
  pseudocódigo Algorithm 1), toda figura citada no texto, 3ª pessoa/voz passiva.

## Toolchain

- **TinyTeX** instalado user-local via `quarto install tinytex` (sem sudo), em
  `~/Library/TinyTeX`; pacotes adicionais: `babel-portuges`, `algorithms`, `float`
  (via `tlmgr install`, após `tlmgr update --self`).
- Compilação: `pdflatex -interaction=nonstopmode Article_PCO213.tex` (2 passadas),
  com `PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`.
- **DOCX não gerado**: o template da disciplina é LaTeX; conversão para DOCX degradaria
  a fidelidade ao template e não é exigida.

## Reprodutibilidade dos números

- Todos os números do artigo vêm de `results_macros.tex`, `table_base_models.tex` e
  `table_holdout.tex`, **gerados automaticamente** por
  `scripts/pco213_make_article_macros.py` a partir dos artefatos de
  `experiments/pco213/santander/` — nunca editar esses três arquivos à mão.
- Cadeia completa de rebuild:
  1. `.venv-pco213/bin/python scripts/pco213_run_santander_study.py --mode full_optional`
     (estágios data→figures; ~16 min no M4 Max);
  2. executar `notebooks/pco213/01_santander_mixture_study.ipynb` (gera EDA +
     `eda_summary.json`);
  3. `.venv-pco213/bin/python scripts/pco213_make_article_macros.py`;
  4. `pdflatex` (2×) em `reports/pco213/article/`.

## Limitações registradas

- As figuras do artigo são cópias renomeadas de `figures/pco213/` (mapeamento no
  script de macros).
- `source_materials/` contém material interno da disciplina — não redistribuir fora
  do contexto acadêmico.
