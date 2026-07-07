# KAGGLE DATASET STRATEGY — Estudo principal em 1 dataset (Santander) para o PCO213

**Data:** 2026-07-06 (rev. 2: escopo reduzido de 3 datasets para 1)
**Branch:** `pco213-classification-mixture-ensemble`
**Processo:** investigação multiagente com verificação web (1 pesquisador por competição, fontes citadas, benchmarks de runtime executados nesta máquina), avaliação de metodologia/runtime e revisão adversarial — seguida de **decisão final do autor (2026-07-06)** reduzindo o escopo para um único dataset.

---

## 1. Resumo da direção final

O projeto é um **estudo principal em um único dataset de competição Kaggle antiga** — **Santander Customer Transaction Prediction (2019)** — avaliando se um ensemble de classificadores com pesos tratados como componentes de mistura supera baselines internos honestos e, principalmente, se o **metamodelo de Scheffé permite interpretar a superfície de desempenho do ensemble** (contribuições individuais + complementaridades par a par).

**Motivo da redução (registrado para auditoria):** o plano de benchmark com 3 datasets (Santander + BNP Paribas + Porto Seguro) foi analisado e considerado tecnicamente viável em máquina (~30–40 h de computação), mas a revisão adversarial demonstrou que o orçamento de **horas humanas** (~75–120 h: 3 EDAs, 2 adaptadores de pré-processamento, artigo cobrindo 3 datasets) excedia o disponível para um aluno que trabalha em tempo integral (~80–110 h), sem folga alguma. A decisão final restringe ainda mais: **runtime máximo alvo de 2 h no M4 Max (36 GB)** para o pipeline completo, garantindo iteração rápida e reprodutibilidade na entrega. A análise dos 3 datasets fica preservada neste documento como base do "future work".

**Fora do escopo (explícito):** slides/apresentação (serão feitos separadamente depois — nenhum artefato de apresentação faz parte deste plano); vitória em leaderboard Kaggle (test sets oficiais não têm rótulo; scores públicos só como contexto de literatura, com não-comparabilidade declarada); tuning extenso de hiperparâmetros (configurações leves, fixas e documentadas).

## 2. Por que Santander como dataset único

Fatos verificados (2026-07-06, fontes nos registros do processo):

| Fato | Valor |
|---|---|
| Slug | `santander-customer-transaction-prediction` (2019, ~8,8k equipes) |
| Acesso hoje | SIM — página+dados no ar (login + aceite de regras); mirror público conhecido |
| train.csv | **200.000 × 200** features numéricas anônimas (`var_0..var_199`) + `ID_code` + `target`, ~288 MB |
| test.csv | 200.000 linhas **sem rótulo** (~metade sintética) — descartado; toda avaliação sai do train.csv |
| Positivos | **10,049%** (20.098) |
| Métrica oficial | **ROC-AUC** |
| Missing / categóricas | **zero / zero** — superfície mínima de pré-processamento e de vazamento |

Razões da escolha:

1. **Melhor cenário público contra colapso de vértice:** as 200 features são quase independentes entre si, e abordagens Naive-Bayes-por-feature atingem AUC ~0,899 vs ~0,90 de GBDTs (kernels públicos de C. Deotte e outros) — GaussianNB genuinamente competitivo com XGBoost é raríssimo, e é exatamente a matéria-prima de que a superfície de mistura precisa;
2. **Dados limpos** (zero missing, zero categóricas): o pipeline fold-safe inteiro é validável sem adaptadores de encoding/imputação — menor risco de vazamento e de horas humanas;
3. **Evidência histórica de blend heterogêneo:** a solução vencedora foi um blend NN+LightGBM; diversidade de família importou no topo;
4. **Runtime medido nesta máquina** (benchmarks reais na escala de fold 128k×200): XGB hist ~3,3 s/fit; RF-100 ~55 s/fit (o gargalo); kNN brute predict ~2,3 s/fold via BLAS; LR 10–30 s — o estudo completo cabe com folga na meta de 2 h, mesmo com o dataset cheio;
5. **História honesta para o artigo:** competição famosa, com "mágica" documentada (features de contagem transdutivas — **proibidas** no nosso protocolo) e writeups públicos citáveis.

**Riscos específicos e mitigação:** (i) EDA de features anônimas → EDA estrutural pré-planejada (correlação ≈ identidade — achado notável em si, distribuições por classe, duplicatas, AUC discriminativa por feature); (ii) NB≈XGB vale em métrica de *ranking* — em log-loss o NB superconfiante é punido, então o resultado interessante é a divergência entre w* ótimo em AUC e w* ótimo em log-loss (análise principal); (iii) kNN em 200 dims é estatisticamente fraco → decisão kNN vs ExtraTrees por microbenchmark + diversidade (ver plano).

## 3. Métricas

- **Primária:** ROC-AUC (métrica oficial da competição — reporte direto, sem proxy).
- **Secundárias:** log-loss (probabilidades de componentes clipadas em ε=10⁻³), Brier score (quadrático exato em `w` — papel de sanidade do pipeline, declarado), F1 com limiar escolhido na OOF (nunca no holdout), acurácia balanceada, tempo de treino/predição por modelo.

## 4. Protocolo experimental (pré-registrado)

1. Toda avaliação supervisionada sai do `train.csv`; `ID_code` descartado; deduplicação com contagem reportada; invariantes validados antes de rodar (200.000 linhas; coluna `target`; 200 features; ~10,05% positivos; zero missing);
2. **Split externo:** holdout 80/20 estratificado (semente fixa), intocado até a confirmação final;
3. **CV interna:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=2)` → **10 rodadas OOF totais** (decisão de runtime: substitui as 10 seeds × 5 folds do plano anterior, que multiplicaria o custo por 5); cada modelo fitado 1× por fold dentro de `Pipeline` com toda transformação dependente de dados; matrizes OOF `P` por repetição, cacheadas em `.npz`;
4. **Amostragem pré-registrada por modo de execução** (decidido por microbenchmark antes do pipeline): `fast` = amostra estratificada ≤60k linhas; `final_2h` = ≤100k linhas se o microbenchmark indicar que cabe em 2 h; `full_optional` = dataset completo apenas se estimado <2 h. Se um modo com amostra for usado, isso é pré-registrado no relatório. **Hard stop lógico:** estimativa >2 h → reduzir amostra ou trocar kNN por ExtraTrees;
5. **Design de mistura (M=5):** vértices + simplex-lattice {5,2} + centroide + pontos axiais (21 corridas) + pontos Dirichlet de validação externa; avaliado sobre a OOF (custo ~zero); Scheffé quadrático ajustado por repetição; ótimo do metamodelo comparado ao **SLSQP direto sobre a OOF** e confirmado no holdout;
6. **Baselines (mesma OOF):** melhor modelo individual, média uniforme, stacking com LogisticRegression, SLSQP direto;
7. **Inferência honesta:** com 1 holdout + 5×2 CV, as comparações formais usam o t corrigido para CV repetida (correção de Nadeau–Bengio, ρ=n_test/n_train) sobre as 10 rodadas OOF, rotulado com suas limitações; o holdout dá a estimativa pontual final (aplicado uma vez); sem Friedman, sem claims de significância fora desse quadro.

## 5. Zoológico de modelos (M=5)

LogisticRegression · GaussianNB · **kNN OU ExtraTrees** (decisão crítica por microbenchmark: kNN é barato nesta máquina — 2,3 s/fold medidos — mas estatisticamente fraco em 200 dims; ExtraTrees é rápido porém quase-gêmeo do RandomForest, o que enfraquece a identificabilidade dos coeficientes de Scheffé; a decisão pondera runtime E diversidade, com correlação de erros OOF como critério) · **RandomForest leve** (número moderado de árvores + `min_samples_leaf`) · **XGBoost OU LightGBM — apenas um GBDT** (XGBoost é o default: já instalado e validado em arm64; dois GBDTs no zoo são vetados por colinearidade). Sem busca de hiperparâmetros; configurações leves, fixas e documentadas.

## 6. Future work / fallbacks (fora do escopo atual)

A análise completa de viabilidade dos demais datasets (rev. 1 deste documento) fica registrada como trabalho futuro e plano de contingência:

- **BNP Paribas Cardif Claims (2016)** — 114,3k×131, 33,5% células NaN, métrica oficial log-loss (= a perda da mistura); mirror OpenML 46856 sem login; seria a 1ª replicação num benchmark futuro;
- **Porto Seguro Safe Driver (2017)** — 595k×57, 3,65% positivos, Gini=2·AUC−1; exigiria cap de 200k linhas; maior risco de colapso em vértice; 2ª replicação futura;
- **Give Me Some Credit (2011)** — 150k×10, AUC; fallback nº 1 se o Santander se tornar inacessível;
- **UCI Credit Card Default** — fallback permanente, viabilidade já estudada no FEASIBILITY, EDA interpretável, dados no ecossistema do repo.

Se o Santander não puder ser baixado (Kaggle CLI ausente/sem token e mirrors indisponíveis), o fallback é acionado **por decisão do autor**, não automaticamente.

## 7. Relatório final

O relatório seguirá **o template da disciplina** (material em diretório local do autor; copiado para `reports/pco213/source_materials/`). **Quarto poderá ser usado como sistema de geração do artigo, desde que o resultado respeite o template** (LaTeX/Word/outro — o formato do template é a fonte de verdade). Limite de páginas do enunciado respeitado (preferencialmente ≤6). Sem código no corpo; equações, fluxograma e pseudocódigo. Slides **não** fazem parte deste escopo.

## 8. Recomendação final

# **GO — escopo reduzido congelado (1 dataset, ≤2 h de runtime)**

Estudo principal no Santander com protocolo do §4, zoo do §5, entregáveis = artigo no template da disciplina + notebook/código reprodutível + bundle de entrega. Upside do benchmark multi-dataset preservado como future work documentado (§6).
