# IMPLEMENTATION PLAN — Projeto Final PCO213
## Otimização de ensembles de classificadores por arranjo de misturas e metamodelagem de superfície de resposta — estudo no Santander

**Branch:** `pco213-classification-mixture-ensemble` (a partir da `main`, 2026-07-06)
**Documentos-base:** [`FEASIBILITY_PCO213_FINAL_PROJECT.md`](FEASIBILITY_PCO213_FINAL_PROJECT.md) (viabilidade metodológica) · [`KAGGLE_DATASET_STRATEGY.md`](KAGGLE_DATASET_STRATEGY.md) (estratégia de dataset, rev. 2 — escopo final de 1 dataset)
**Histórico de revisão:** 2026-07-06 rev. 1 — UCI→benchmark Kaggle 3 datasets; **2026-07-06 rev. 2 — escopo final: estudo principal em 1 dataset (Santander), runtime ≤2 h no M4 Max 36 GB, por restrição de runtime e horas humanas.** BNP Paribas, Porto Seguro, Give Me Some Credit e UCI Credit Card Default movidos para future work/fallback. Slides fora do escopo (feitos separadamente depois).

---

## 1. Estratégia de repositório

- Branch deste repositório; base = `main` (imutável, sem os módulos de mistura — porte por cópia da `origin/repo-publication-readiness`, com atribuição). `src/doe_xgb/` intocado; tudo novo autocontido em `src/mixens/`. Sem PR para `main`.
- **Dados nunca commitados** (licença Kaggle + tamanho): `.gitignore` bloqueia `data/pco213/raw/*`, `data/pco213/processed/*`, `experiments/pco213/*` (exceto `.gitkeep`); só scripts de download/instruções.
- **Não portar** `post_optimization.py` (bug conhecido na origem).

## 2. Escopo final congelado

| Item | Decisão |
|---|---|
| Dataset | **Santander Customer Transaction Prediction** (único; demais em future work/fallback) |
| Tarefa | Classificação binária |
| Runtime alvo | **≤2 h no M4 Max 36 GB** (hard stop lógico: estimativa >2 h → reduzir amostra ou trocar kNN por ExtraTrees) |
| Validação | Holdout externo 80/20 estratificado + **`RepeatedStratifiedKFold(n_splits=5, n_repeats=2)`** = 10 rodadas OOF (não usar 10 seeds × 5 folds — explode o runtime) |
| Modos de execução | `fast` (amostra estratificada ≤60k) · `final_2h` (≤100k se couber em 2 h) · `full_optional` (dataset completo só se estimado <2 h) — escolhidos por **microbenchmark** (10–20k linhas, 1 fold, tempo por modelo); uso de amostra é pré-registrado no relatório |
| Modelos-base (M=5) | LogisticRegression · GaussianNB · **kNN OU ExtraTrees** (microbenchmark + diversidade; correlação de erros OOF como critério) · **RandomForest leve** (árvores moderadas, `min_samples_leaf`) · **XGBoost OU LightGBM (um só GBDT** — default XGBoost, já validado em arm64**)**. Sem zoo só-GBDT (colinearidade mata a interpretação dos β de Scheffé). Sem tuning extenso; configs leves fixas documentadas |
| Métrica primária | ROC-AUC |
| Métricas secundárias | log-loss (clipping ε=10⁻³ nos componentes), Brier (sanidade — quadrático exato em w), F1 com limiar escolhido na OOF, acurácia balanceada, tempo de treino/predição |
| Baselines | melhor modelo individual · média uniforme das probabilidades · stacking com LogisticRegression sobre OOF · **SLSQP direto** · mistura+Scheffé (método proposto) |
| Método | design de mistura no simplex (vértices + lattice {5,2} + centroide + axiais) → avaliação sobre OOF → Scheffé quadrático → ótimo do metamodelo vs SLSQP direto → confirmação única no holdout; validação externa do metamodelo com pontos Dirichlet |
| Inferência | t corrigido para CV repetida (Nadeau–Bengio) sobre as 10 rodadas OOF, com limitações declaradas; holdout = estimativa pontual final; sem Friedman |
| Relatório | Formato de artigo, **template da disciplina** como fonte de verdade (material local copiado para `reports/pco213/source_materials/`); Quarto permitido se respeitar o template; ≤6 páginas preferencialmente; sem código no corpo; **sem slides neste escopo** |

Condições metodológicas inegociáveis (FEASIBILITY): honestidade computacional na introdução; nunca p-valores de pseudo-réplicas como generalização; stacking na mesma OOF; clipping/limiar/Jensen declarados a priori.

## 3. Dados

1. **Aquisição:** Kaggle CLI se instalada+autenticada; senão, tentar mirror público verificável; senão, **documentar e instruir download manual** para `data/pco213/raw/santander/` (o pipeline detecta e orienta).
2. **Validação de invariantes antes de rodar:** existência de `train.csv`; coluna `target`; 200 features `var_*`; taxa de positivos ≈10,05%; ausência de missing.
3. `ID_code` descartado; deduplicação com contagem reportada; features de contagem/"mágica" **proibidas** (transdutivas).

## 4. Arquitetura

```
src/mixens/                      # autocontido; nada de import doe_xgb
├── __init__.py
├── data.py                      # aquisição (CLI/mirror/manual) + invariantes + amostragem por modo
├── base_models.py               # zoo M=5 (pipelines fold-safe) + OOF 5×2 + Q holdout + clipping + timings
├── mixture_design.py            # [PORTADO simplex.py] + Dirichlet
├── scheffe.py                   # [PORTADO MixtureScheffeModel+FitReport] + validação externa
├── ensemble_eval.py             # métrica(w | OOF) + baselines (single/uniforme/stacking/SLSQP)
├── optimize.py                  # SLSQP Σw=1 (metamodelo e direto)
└── plots.py                     # ternários baricêntricos, pred-vs-obs, coeficientes, comparação
scripts/
├── pco213_run_santander_study.py  # --mode fast|final_2h|full_optional --max-runtime-minutes 120
│                                  # --sample-size --random-state --output-dir experiments/pco213/santander
└── (microbenchmark embutido no runner, estágio 0)
notebooks/pco213/01_santander_mixture_study.ipynb   # EDA + leitura dos resultados salvos; reexecutável
reports/pco213/{source_materials,template,article}/ + BUILD_NOTES.md
dist/pco213_final/ + dist/PCO213_final_entrega.zip  # entrega (sem dados brutos)
```

## 5. Plano de commits

| # | Commit | Status |
|---|---|---|
| 1 | `chore: scaffold PCO213 classification mixture ensemble project` | ✅ `5d2bc4a` |
| 2 | `docs: reduce PCO213 scope to Santander mixture ensemble study` | ← agora |
| 3 | `feat(pco213): port simplex and Scheffé modules with tests` | próximo |
| 4 | `feat(pco213): add Santander data loader and runtime benchmark` | |
| 5 | `feat(pco213): add base model zoo and OOF matrices` | |
| 6 | `feat(pco213): evaluate mixture design and baselines` | |
| 7 | `feat(pco213): fit Scheffé metamodel and optimize ensemble weights` | |
| 8 | `docs(pco213): add final report and deliverable bundle` | |

## 6. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Runtime >2 h | Microbenchmark antes do pipeline; modos com amostra estratificada pré-registrada; hard stop lógico; kNN→ExtraTrees se preciso |
| Dados indisponíveis (CLI ausente/sem token) | Mirror verificado por invariantes; senão instruções manuais claras + pausa; fallbacks (Give Me Some Credit, UCI CCD) só por decisão do autor |
| EDA de features anônimas | EDA estrutural (correlação≈identidade, distribuições por classe, duplicatas, AUC por feature) |
| NB≈XGB só em ranking → w* pode concentrar no vértice XGB em log-loss | Análise principal = divergência w*(AUC) vs w*(log-loss); resultado interessante em qualquer geometria |
| Colinearidade no zoo (ExtraTrees~RF; 2 GBDTs) | Um GBDT só; decisão kNN vs ExtraTrees pondera correlação de erros OOF; correlações reportadas |
| Vazamento | OOF estrita; scaler/limiar na OOF; contagens transdutivas proibidas; holdout tocado uma vez |
| "Ferramenta demais" | Artigo abre com EDA + comparação dos 5 modelos; mistura como metodologia; ênfase na proposta |
| Template do artigo incompatível com Quarto | Formato do template = fonte de verdade; LaTeX direto se preciso; limitações em BUILD_NOTES.md |
