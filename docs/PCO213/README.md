# PCO213 — Otimização de ensembles de classificadores por arranjo de misturas (estudo no Santander)

Projeto final da disciplina **PCO213 — Machine Learning e Data Mining** (UNIFEI). Desenvolvido na branch `pco213-classification-mixture-ensemble` deste repositório, criada a partir da `main` (baseline da dissertação). **Não é parte da dissertação nem do artigo** — é um trabalho de disciplina que reutiliza, com atribuição, módulos matemáticos do ecossistema do repo.

## Objetivo

**Estudo principal em um dataset real de competição tabular** — Santander Customer Transaction Prediction (Kaggle, 2019; 200k×200 numéricas anônimas, 10,05% positivos, métrica oficial ROC-AUC): dado um conjunto de M=5 classificadores-base heterogêneos, verificar se um ensemble com pesos tratados como componentes de mistura supera baselines internos honestos e, principalmente, se o **metamodelo de Scheffé permite interpretar a superfície de desempenho do ensemble**:

p_ens(x) = w₁·p₁(x) + … + w₅·p₅(x),  com wᵢ ≥ 0 e Σwᵢ = 1

usando um **arranjo de misturas** para planejar quais vetores de pesos avaliar, predições out-of-fold (holdout 80/20 externo + `RepeatedStratifiedKFold` 5×2 = 10 rodadas OOF) para medi-los sem vazamento, e o **polinômio canônico de Scheffé** como metamodelo da superfície.

**Avisos de escopo:** objetivo ≠ vencer leaderboard Kaggle (test oficial sem rótulos; scores públicos só como contexto de literatura); runtime alvo ≤2 h no M4 Max (modos de execução com amostragem estratificada pré-registrada, escolhidos por microbenchmark); sem tuning extenso; **slides/apresentação fora deste escopo** (serão feitos separadamente).

## Hipótese metodológica

A otimização direta dos pesos é computacionalmente trivial (sobre a matriz out-of-fold, avaliar qualquer w custa um produto matriz-vetor, e log-loss é convexa em w). A contribuição não é achar o ótimo mais rápido — é o **modelo estatístico interpretável da superfície de desempenho**: os coeficientes lineares do Scheffé medem a contribuição individual de cada classificador, e os coeficientes cruzados β_ij medem a **complementaridade (diversidade útil) entre pares de classificadores**, com validação externa (pontos Dirichlet), contornos ternários e regiões quase-ótimas. A otimização direta (SLSQP no simplex) entra como **validador** do metamodelo. No Santander, a análise central é a divergência entre o w* ótimo em AUC e o w* ótimo em log-loss (o GaussianNB é competitivo em ranking, mas superconfiante em probabilidade).

## Por que pesos de ensemble formam um problema de mistura

Os pesos satisfazem exatamente as restrições canônicas de mistura (não-negatividade e soma 1): o domínio experimental é o **simplex**, não uma caixa. Nesse domínio, fatorial/CCD e RSM quadrática comum produzem coeficientes sem sentido (a restrição de soma torna a matriz de design singular com intercepto); o formalismo correto é o de **arranjos de mistura** (simplex-lattice, centroide, pontos axiais) com o **polinômio canônico de Scheffé** (sem intercepto, sem quadrados puros). Os "componentes" são os classificadores: o vértice puro wᵢ=1 é usar só o classificador i, e o voto uniforme w=1/5 é o centroide do simplex — um ponto do próprio design.

## Escopo congelado (detalhes em `IMPLEMENTATION_PLAN.md` e `KAGGLE_DATASET_STRATEGY.md`)

Santander único · holdout 80/20 + RepeatedStratifiedKFold 5×2 · zoo M=5: LogReg, GaussianNB, kNN-ou-ExtraTrees (decisão por microbenchmark + diversidade), RandomForest leve, um GBDT (XGBoost default) · design {5,2}+centroide+axiais + validação Dirichlet · Scheffé quadrático · baselines: melhor modelo, média uniforme, stacking logístico, SLSQP direto · métrica primária ROC-AUC; secundárias log-loss/Brier/F1(limiar na OOF)/acurácia balanceada/tempos · future work/fallback: BNP Paribas, Porto Seguro, Give Me Some Credit, UCI Credit Card Default.

## Artefatos deste escopo

- `scripts/pco213_run_santander_study.py` — pipeline completo com `--mode fast|final_2h|full_optional`, microbenchmark e hard stop de runtime;
- `notebooks/pco213/01_santander_mixture_study.ipynb` — EDA estrutural + leitura reexecutável dos resultados salvos;
- `src/mixens/` — pacote autocontido (dados, zoo OOF, design de mistura, Scheffé, otimização, figuras) com testes em `tests/mixens/`;
- `experiments/pco213/santander/` — OOF `.npz`, resultados CSV/JSON (não versionados; dados brutos **nunca** entram no repo);
- `reports/pco213/` — artigo no **template da disciplina** (source em `source_materials/`, build notes em `BUILD_NOTES.md`; Quarto permitido se respeitar o template);
- `dist/pco213_final/` + `dist/PCO213_final_entrega.zip` — bundle de entrega (sem dados brutos; com instruções de download).

## Relação com o restante do repositório

| Diretório | Relação |
|---|---|
| `src/doe_xgb/` | baseline da dissertação — **intocado** |
| `src/mixens/` | pacote novo do PCO213; cópias atribuídas de `simplex.py` e `MixtureScheffeModel` (da branch `repo-publication-readiness`, ausentes na `main`) |
| `docs/PCO213/` | viabilidade (`FEASIBILITY_PCO213_FINAL_PROJECT.md`), estratégia de dataset (`KAGGLE_DATASET_STRATEGY.md`), plano (`IMPLEMENTATION_PLAN.md`), este README |
| `data/pco213/`, `experiments/pco213/`, `notebooks/pco213/`, `reports/pco213/`, `figures/pco213/`, `dist/` | áreas exclusivas do projeto |

Relação metodológica com a dissertação: **transposição, não reciclagem** — lá, Scheffé/simplex operam sobre pesos de escalarização de objetivos (MBPA/NBI); aqui, sobre pesos de componentes de um ensemble de classificadores em dados de competição.
