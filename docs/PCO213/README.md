# PCO213 — Otimização de ensembles de classificadores por arranjo de misturas

Projeto final da disciplina **PCO213 — Machine Learning e Data Mining** (UNIFEI). Desenvolvido na branch `pco213-classification-mixture-ensemble` deste repositório, criada a partir da `main` (baseline da dissertação). **Não é parte da dissertação nem do artigo** — é um trabalho de disciplina que reutiliza, com atribuição, módulos matemáticos do ecossistema do repo.

## Objetivo

Dado um conjunto de M classificadores-base heterogêneos, encontrar e — principalmente — **caracterizar estatisticamente** a combinação convexa ótima de suas probabilidades previstas:

p_ens(x) = w₁·p₁(x) + … + w_M·p_M(x),  com wᵢ ≥ 0 e Σwᵢ = 1

usando um **arranjo de misturas** para planejar quais vetores de pesos avaliar, validação cruzada (predições out-of-fold) para medi-los sem vazamento, e um **metamodelo de superfície de resposta de Scheffé** para modelar o desempenho do ensemble como função dos pesos.

## Hipótese metodológica

A otimização direta dos pesos é computacionalmente trivial (sobre a matriz out-of-fold, avaliar qualquer w custa um produto matriz-vetor, e log-loss é convexa em w). A contribuição não é achar o ótimo mais rápido — é o **modelo estatístico interpretável da superfície de desempenho**: os coeficientes lineares do Scheffé medem a contribuição individual de cada classificador, e os coeficientes cruzados β_ij medem a **complementaridade (diversidade útil) entre pares de classificadores**, com intervalos de estabilidade entre réplicas, contornos ternários e regiões quase-ótimas — informação que um otimizador pontual não fornece. A otimização direta (SLSQP no simplex, varredura Dirichlet densa) entra como **validador** do metamodelo.

## Por que pesos de ensemble formam um problema de mistura

Os pesos satisfazem exatamente as restrições canônicas de mistura (não-negatividade e soma 1): o domínio experimental é o **simplex**, não uma caixa. Nesse domínio, fatorial/CCD e RSM quadrática comum produzem coeficientes sem sentido (a restrição de soma torna a matriz de design singular com intercepto); o formalismo correto é o de **arranjos de mistura** (simplex-lattice, simplex-centroide, pontos axiais) com o **polinômio canônico de Scheffé** (sem intercepto, sem quadrados puros). Os "componentes" da mistura são os classificadores: o vértice puro wᵢ=1 é usar só o classificador i, e o voto uniforme w=1/M é o centroide do simplex — um ponto do próprio design.

## Escopo (congelado — detalhes em `IMPLEMENTATION_PLAN.md`)

Classificação binária · UCI Default of Credit Card Clients (ID 350) · zoo M=5 (LogReg, GaussianNB, kNN, RandomForest, XGBoost) · design {5,2}+centroide+axiais (21 corridas) + 25 pontos Dirichlet de validação · Scheffé quadrático por réplica (10 seeds) · holdout 80/20 + 5-fold OOF · baselines: melhor modelo, voto uniforme, stacking logístico, mistura otimizada.

## Artefatos que serão produzidos

- `notebooks/pco213/01_eda.ipynb` — análise exploratória e decisões de pré-processamento;
- `src/mixens/` — pacote Python autocontido do projeto (dados, zoo OOF, design de mistura, Scheffé, otimização, figuras), com testes em `tests/mixens/`;
- `experiments/pco213/` — matrizes OOF cacheadas e resultados (não versionados);
- `figures/pco213/` — 7 figuras finais (EDA, boxplots dos modelos-base, contornos ternários, predito-vs-observado, forest plot dos coeficientes, fronteira log-loss×custo, curvas de calibração);
- `reports/pco213/` — relatório em formato de artigo (sem código no corpo: equações, fluxograma, pseudocódigo) e slides.

## Relação com o restante do repositório

| Diretório | Relação |
|---|---|
| `src/doe_xgb/` | baseline da dissertação — **intocado** por este projeto |
| `src/mixens/` | pacote novo do PCO213; inclui cópias atribuídas de `simplex.py` e do `MixtureScheffeModel` (originários da branch `repo-publication-readiness`, ausentes na `main`) |
| `docs/PCO213/` | toda a documentação do projeto: análise de viabilidade (`FEASIBILITY_PCO213_FINAL_PROJECT.md`), plano de implementação (`IMPLEMENTATION_PLAN.md`) e este README |
| `data/pco213/`, `experiments/pco213/`, `notebooks/pco213/`, `reports/pco213/`, `figures/pco213/` | áreas de trabalho exclusivas do projeto, separadas por sufixo/subpasta para nunca colidir com artefatos da dissertação |

A relação metodológica com a dissertação é de **transposição, não reciclagem**: lá, Scheffé/simplex operam sobre pesos de escalarização de objetivos no pós-processamento do NBI (MBPA); aqui, sobre pesos de componentes de um ensemble de classificadores — problema, dataset e pergunta científica distintos.
