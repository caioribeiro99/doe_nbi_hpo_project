# FEASIBILITY — Projeto Final PCO213
## Otimização de ensembles de classificadores por arranjo de misturas e metamodelagem de superfície de resposta

**Data:** 2026-07-04
**Base analisada:** `~/Projects/doe_nbi_hpo_project` (branch `repo-publication-readiness`)
**Processo:** análise multiagente (3 leitores do repositório → 3 avaliadores independentes de metodologia/datasets/risco → 1 revisão adversarial que verificou as citações de arquivo/linha no disco). Este documento é a síntese consolidada, já com as correções da revisão adversarial incorporadas.
**Status (2026-07-06):** decisão de estratégia de repositório SUBSTITUÍDA — o desenvolvimento será feito na branch `pco213-classification-mixture-ensemble` deste próprio repositório (criada a partir da `main`), e não em repo standalone como recomendado no §12/§13. Ver `docs/PCO213/IMPLEMENTATION_PLAN.md` para a arquitetura adaptada. As citações de arquivo/linha deste documento referem-se à branch `repo-publication-readiness`; a `main` (baseline da dissertação) NÃO contém `simplex.py`, `model_families.py`, `nbi_core.py`, `design/` nem `tests/` — esses módulos serão portados para a branch nova como parte da implementação. O restante da análise (metodologia, dataset, protocolo, riscos) permanece válido.

---

## 1. Resumo executivo

**Decisão: GO — com uma especificação única congelada (§13) e quatro condições inegociáveis.**

A ideia é rara no melhor sentido: a geometria do problema (pesos de ensemble no simplex) casa **exatamente** com o formalismo que você já domina e já tem implementado (arranjo simplex, polinômio canônico de Scheffé, NBI). O repositório da dissertação contém ~70% da maquinaria matemática pronta e testada, o custo computacional total é de ~1–2 h de CPU num MacBook, e o tema é genuinamente distinto da dissertação (lá: pesos de escalarização de objetivos de HPO; aqui: pesos de componentes de um ensemble de classificadores — zero menções a "ensemble/stacking/blend" no repo).

Os riscos que reprovariam o trabalho não são de implementação — são de **enquadramento e de estatística**, e todos têm correção barata se adotada desde o desenho:

1. **Honestidade computacional (fatal se escondida).** Sobre a matriz de predições out-of-fold, avaliar qualquer vetor de pesos é quase grátis, e log-loss é convexa em `w` (SLSQP resolve o ótimo global exato em milissegundos). O metamodelo de Scheffé **não se justifica por custo** — justifica-se como *modelo estatístico interpretável da superfície de desempenho* (contribuições individuais + complementaridades par a par com incerteza), com a otimização direta incluída como comparador de validação. Isso precisa estar escrito na introdução do artigo.
2. **Inferência estatística (fatal se feita errada).** Réplicas de seed compartilham os mesmos dados: Wilcoxon entre réplicas mede robustez a particionamento, **não** generalização; Friedman/Nemenyi com 1–2 datasets não tem sentido. Afirmações comparativas formais exigem splits externos repetidos com o t corrigido de Nadeau–Bengio — ou linguagem sem significância.
3. **Vazamento.** Protocolo OOF + holdout intocado + stacking ajustado na mesma OOF como competidor obrigatório (§9).
4. **Armadilhas técnicas específicas** já mapeadas: clipping das probabilidades dos componentes (log-loss infinita nos vértices de kNN/NB), vacuidade do Brier (é exatamente quadrático em `w` — o Scheffé quadrático o reproduz com R²=1 por identidade algébrica), design com pontos interiores, zoo sem gêmeos GBDT.

Com a spec congelada do §13, o projeto cumpre todos os requisitos da disciplina com folga, tem um marco submetível já na semana 3, e um diferencial (NBI 2-objetivo) que nenhum outro grupo terá.

---

## 2. O que o repo atual já oferece

Reuso direto (copiar arquivos, nunca `import doe_xgb`):

| Módulo | O que faz | Estado para o novo projeto |
|---|---|---|
| `src/doe_xgb/simplex.py` (163 L) | `generate_simplex_lattice(q,m)` (`:33`), `generate_simplex_centroid` (`:63`), vértices extremos com bounds (`:86`), `validate_weights` (`:131`) — tudo dimensão-agnóstico | **Reuso as-is** (é literalmente o gerador de arranjos de mistura) |
| `MixtureScheffeModel` (`src/doe_xgb/model_families.py:324`) | Polinômio canônico de Scheffé nas 4 ordens (linear/quadrático/cúbico-especial/cúbico completo, `:40`), OLS statsmodels **sem intercepto**, backward elimination desabilitada de propósito (`:343-344`, decisão D14 dos docs), `FitReport` com R², R²-adj, número de condição, rank e p-valores | **Reuso as-is** + acréscimos externos (PRESS/validação externa; ver §3) |
| `src/doe_xgb/design/diagnostics.py` (44 L) | rank/condicionamento da matriz de design | Reuso as-is |
| `src/doe_xgb/nbi_core.py` (351 L) | NBI N-objetivo genérico sobre `Callable[[np.ndarray], float]` (`:34`), zero acoplamento ao HPO | Opcional (stretch multiobjetivo); simplex via reformulação em q−1 variáveis livres, já documentada em `post_optimization.py:126-131` |
| `src/doe_xgb/selection.py`, `objectives.py` | Seleção multiobjetivo (utopia-distance, knee, TOPSIS) | Opcional, reuso as-is |
| `tests/unit/test_simplex.py` (12 testes), `test_model_families.py` (11 testes) | Cobertura dos dois módulos centrais | Copiar junto — cobertura "de graça" |
| Datasets | **11 datasets UCI já baixados com checksum e script de fetch** — incluindo Credit Card Default (XLS bruto + CSV processado) e Bank Marketing | Risco de engenharia de dados ≈ zero |
| Docs normativos | `DOE_DECISION_TREE.md` item 5 já codifica a regra do projeto: *"pesos vivem no simplex, nunca CCDFC"*; `DESIGN_MODEL_COMPAT.md`: simplex → `mixture_scheffe`, "NEVER process_quadratic" | Fundamentação metodológica pronta e citável |
| Protocolo comparativo | `COMPARATIVE_PROTOCOL.md`: orçamento pareado de avaliações entre método e baselines | Gabarito adaptável |

**Achado estrutural importante:** o MBPA da dissertação (`post_optimization.py`) já ajusta superfícies de Scheffé sobre um simplex de pesos e otimiza sobre elas — o "loop" conceitual do novo projeto já existe, aplicado a outro objeto. A transposição pesos-de-escalarização → pesos-de-ensemble é a contribuição nova.

**Orientação do repo:** classificação binária + HPO + RSM + NBI. Não é um repo de ensembles — o que é bom: o tema novo não recicla resultado da dissertação.

## 3. Lacunas do repo atual

Todas pequenas; total estimado de código novo < 500 linhas:

1. **Fábrica de classificadores-base sklearn** — o repo só tem GBDTs (`methods/_gbdt_factory.py`: xgboost/lightgbm/catboost; caminho principal é XGBoost hardcoded em `evaluation.py:100`). LogReg/NB/kNN/RF não existem. (~60 L, padrão de closures já dado pela fábrica existente.)
2. **Avaliador OOF** — gerar matriz de probabilidades out-of-fold `P (n × M)` treinando cada modelo 1× por fold, e avaliar `métrica(w) = métrica(P·w)` para qualquer `w`. (~100–150 L; o pipeline atual re-treina um XGB por ponto do design — não serve e não deve ser tocado.)
3. **Amostragem Dirichlet** — inexistente em `simplex.py`; necessária para pontos de validação externa e para o baseline de busca densa. (~10 L.)
4. **Diagnósticos de metamodelo ausentes do `FitReport`** — sem lack-of-fit, PRESS ou R²-preditivo. Resolver por validação externa (pontos Dirichlet fora do design) + ajuste por réplica (§8). (~40 L.)
5. **Plot ternário** — nenhuma lib no ambiente; helper baricêntrico em matplotlib puro (~40 L) evita dependência nova.
6. **`simplex_centroid` não está plugado no `provider.build()`** — mas a função existe em `simplex.py:63`; chamar direto.
7. **Métricas binárias** — `metrics.py` só tem accuracy/precision/recall/specificity no caminho binário (AUC/Brier/ECE existem só no multiclasse). Usar `sklearn.metrics` direto no avaliador novo.
8. **Não copiar:** `post_optimization.py` (bug confirmado em `:166-170` — `run_nbi` recebe candidatos em vez de callables e sempre cai no `except`), `factor_analysis.py` (hardcoded à dissertação), `rsm.py`/`scalarization.py`/`nbi.py` legados, benchmarks de HPO.
9. **Protocolo de validação do repo não serve de modelo num ponto:** é CV-only, sem holdout (grep por `train_test_split` vazio), e a confirmação NBI reusa a MESMA CV com o mesmo seed (`run_replica.py:284-293`). O projeto da disciplina precisa de holdout externo — não herdar esse desenho.

---

## 4. Viabilidade da ideia de ensemble por arranjo de misturas

**Estatisticamente adequada — com três ressalvas que devem virar texto do artigo, não notas de rodapé.**

A adequação formal é impecável: pesos de soft-voting satisfazem exatamente `w_i ≥ 0, Σw_i = 1`; o simplex é o domínio experimental natural; o polinômio de Scheffé é a base canônica para superfícies nesse domínio (Cornell, 2002). Nenhuma adaptação forçada.

**Ressalva 1 — componentes não são ingredientes físicos.** O vértice puro é "usar só o classificador i"; o termo `β_ij·w_i·w_j` mede o desvio do blend em relação à interpolação linear dos desempenhos — ou seja, **β_ij é uma medida direta de complementaridade entre os classificadores i e j**, conectando o metamodelo à decomposição de ambiguidade de Krogh & Vedelsky e à literatura de diversidade em ensembles. Esse é o argumento interpretativo central do artigo. Dicionário de leitura correto (a revisão adversarial pegou o erro na primeira versão): |β_ij| grande na direção benéfica = complementaridade; **β_ij ≈ 0 = redundância** (não β_ij negativo).

**Ressalva 2 — sinal conhecido a priori para perdas convexas (Jensen).** Log-loss e Brier são convexas em `w`: a perda do blend nunca excede a interpolação linear das perdas. Logo, para essas métricas, "detectar sinergia" é estruturalmente garantido — não é hipótese falseável. A pergunta científica correta é *quão forte* é a complementaridade de cada par e *quais pares dominam*; para AUC (não convexa em `w`) o sinal volta a ser livre. Declarar isso no artigo é obrigatório; agrava-se com o **Brier, que é exatamente quadrático em `w`** — o Scheffé quadrático o reproduz com R²=1 por identidade algébrica. Uso correto: log-loss e AUC como respostas de modelagem genuína; a exatidão do Brier vira **teste de sanidade do pipeline** (recuperação numérica dos coeficientes verdadeiros), dito explicitamente — vira demonstração de entendimento em vez de armadilha de arguição.

**Ressalva 3 — estrutura de erro não clássica (a mais importante).** Dada a matriz OOF fixa, `métrica(w)` é determinística: erro puro clássico = 0, e a decomposição lack-of-fit/pure-error de curso de misturas não se aplica. Réplicas vêm de re-splits da CV, e os pontos do design dentro de uma réplica são fortemente correlacionados — os p-valores OLS do `FitReport` **não são honestos** sobre réplicas empilhadas. Solução correta e barata: **ajustar o Scheffé por réplica** (R ajustes) e reportar a distribuição dos coeficientes entre réplicas como *intervalos de estabilidade a particionamento* (rótulo exato; não são ICs amostrais da superfície populacional). Enquadrar como metamodelagem de experimento computacional determinístico.

**A objeção que decide a nota — "por que não otimizar direto?"** Sobre a OOF, avaliar `w` custa um produto matriz-vetor; log-loss é convexa → SLSQP no simplex dá o ótimo global exato em milissegundos; AUC não é convexa mas uma varredura Dirichlet densa (10⁴–10⁵ pontos) resolve na prática. **O metamodelo não se justifica por custo, e esconder isso é falha fatal.** Enquadramento sobrevivente (validado pela revisão adversarial como genuíno, não maquiagem — desde que o artigo entregue o que só o Scheffé dá):

> *"A otimização direta dos pesos é computacionalmente trivial; a contribuição deste trabalho é a caracterização estatística da superfície de desempenho do ensemble — quantificando contribuições e complementaridades entre classificadores com incerteza — usando a otimização direta como referência de validação do metamodelo."*

Entregas exclusivas do Scheffé: coeficientes com significado + intervalos de estabilidade, teste de ordem (linear vs quadrático vs cúbico especial), contornos ternários, **regiões quase-ótimas** em vez de ponto único. E o resultado esperado (superfície achatada perto do ótimo, porque os `p_i` são correlacionados) é *achado reportável* — explica mecanisticamente por que voto uniforme é difícil de bater.

**Metamodelos alternativos considerados e rejeitados como principal:** RSM quadrática comum com restrição de soma — proibida pelos próprios docs do repo ("coeficientes sem sentido" no simplex); GP/Random-Forest surrogate — joga fora a interpretabilidade, que é a única justificativa do metamodelo aqui (aceitável só como nota de robustez).

## 5. Classificação vs. regressão: recomendação

**Classificação binária (alternativa A), sem hesitação.**

| Critério | Classificação | Regressão |
|---|---|---|
| Naturalidade da mistura | Combinação convexa de **probabilidades** é um objeto canônico (opinion pooling); vértices = classificadores individuais | Média ponderada de predições funciona, mas sem a camada probabilística |
| Riqueza de métricas | AUC/PR-AUC/log-loss/Brier divergem sob desbalanceamento → sustenta a história multi-métrica e o eixo multiobjetivo | RMSE/MAE/R² fortemente correlacionadas entre si → discussão pobre |
| Eixo de calibração | Existe (curvas de calibração, Brier) — bônus de discussão | Não existe |
| Datasets prontos no repo | 11 binários baixados com checksum | Nenhum |
| Aderência à disciplina | EDA de desbalanceamento motiva a escolha de métricas (conexão EDA→método que professores valorizam) | Menos ganchos |

A versão regressão não é inviável — é estritamente menos interessante em todos os eixos, e exigiria engenharia de dados nova.

## 6. Dataset recomendado e justificativa

**Primário: UCI Default of Credit Card Clients (ID 350)** — 30.000 × 23, 22,1% de default (6.636), já em disco (`data/source/credit_card_default/`, XLS bruto + CSV processado + checksum + fetch script).

Por quê (em ordem de peso):
1. **De-risca a hipótese existencial do projeto.** O fracasso possível é o ótimo colapsar num vértice (um modelo domina → mistura vira formalidade). O paper de origem do dataset (Yeh & Lien, 2009, *ESWA*) é literalmente uma comparação de 6 famílias de classificadores com desempenhos próximos e ranking dependente da métrica — o desacordo entre famílias está documentado **na fonte primária do dataset**. Gancho narrativo pronto: *"Yeh & Lien compararam seis classificadores para escolher o melhor; nós os tratamos como componentes de uma mistura."*
2. **Material genuíno para cada critério de nota:** EDA real (correlação BILL_AMT1-6, assimetria monetária), pré-processamento com decisões defensáveis (categorias não documentadas EDUCATION=0/5/6 e MARRIAGE=0, codificação dos ordinais PAY_0…PAY_6, escalonamento para kNN/LogReg), teto honesto (AUC ~0,77–0,78 na literatura — ninguém "gabarita").
3. **22% de desbalanceamento é o ponto doce:** métricas divergem sem exigir SMOTE/reponderação de classe.
4. Citação limpa (UCI ID 350, DOI, paper canônico).

**Secundário/robustez (opcional, semana 8): UCI Bank Marketing (ID 222)** — 41.188 × 20, 11,3% positivos, também em disco. Estressa o método sob desbalanceamento mais forte e mix categórico pesado; a exclusão documentada de `duration` (vazamento pós-desfecho clássico) rende pontos de metodologia. **Atenção às duas cópias no repo:** `data/source/bank/bank-additional-full.csv` já vem **sem** `duration`; a cópia canônica com `duration` presente é `data/source/bank_marketing/raw/`. Fonte única = `bank_marketing/raw`, com a remoção de `duration` feita (e documentada) no seu próprio `data.py`.

**Rejeitados:** MAGIC (dataset-manchete da dissertação — marcado como "dissertation continuity" no próprio repo; ótica de reciclagem na mesma instituição, dado simulado por Monte Carlo, zero história de pré-processamento); AI4I 2020 (sintético — fere "dataset real" —, vazamento estrutural: o alvo é o OR das colunas de modo de falha presentes no arquivo, 3,4% de positivos com dominância de árvores → colapso de vértice quase certo); Adult (clichê, e a cópia do repo é só o split de treino, 30.162 linhas); Spambase (features pré-engenheiradas, quase balanceado, teto alto); Kaggle Telco Churn (licença IBM ambígua, sem citação canônica/DOI — pior que qualquer UCI para relatório em formato de artigo).

## 7. Metodologia proposta

**Componentes (M=5, heterogeneidade máxima de viés, sem gêmeos):** LogisticRegression, GaussianNB, kNN, RandomForest, XGBoost. A revisão adversarial vetou o zoo com XGBoost+HistGradientBoosting (GBDTs quase idênticos → resposta quase constante na direção de troca entre eles, w* não-identificável, ICs enormes). **SVM-RBF proibido** (O(n²–n³) em ~24k linhas de treino ≈ 17 h só nele).

**Gate de diversidade (antes de rodar o arranjo):** matriz de correlação de erros OOF / Q-statistic par a par. Regra declarada a priori: se algum par tiver correlação de erros > 0,95, trocar um componente. Se (contra a literatura) um modelo dominar, o dataset secundário assume como primário — pipelines idênticos a partir da OOF.

**Pipeline (equações e fluxograma no artigo; sem código no corpo):**
1. Split externo estratificado 80/20 (seed fixa) — teste intocado até o fim.
2. K-fold estratificado (K=5) no treino: cada modelo-base treinado 1× por fold → matriz OOF `P (n_train × 5)`. Probabilidades dos **componentes** clipadas em `[ε, 1−ε]`, ε=10⁻³ declarado a priori (kNN/NB produzem 0/1 exatos → log-loss infinita nos vértices; clipar o blend quebraria a linearidade em `w`). Sensibilidade ε=10⁻³ vs 10⁻⁶ reportada.
3. **Arranjo de misturas:** lattice {5,2} (15 pontos) + centroide global (1) + 5 axiais `x_i=(M+1)/2M=0,6` (21 corridas) — o quadrático tem 15 termos → 6 gl; os axiais + centroide dão cobertura interior (o lattice {5,2} sozinho só tem vértices e arestas, e o ótimo realista vive no interior). +25 pontos Dirichlet(1,…,1) **de validação externa** (nunca usados no ajuste). O voto uniforme `w=1/5` é o centroide — ponto do design, de graça.
4. **Metamodelo:** Scheffé quadrático (cúbico especial como teste de ordem), ajustado **por réplica** (R=10 seeds de CV); coeficientes reportados como mediana + intervalos de estabilidade percentílicos entre réplicas; número de condição da matriz de design reportado (Scheffé quadrático tem colinearidade moderada conhecida entre termos lineares e cruzados — não alegar "bem condicionado por construção").
5. **Otimização:** (a) SLSQP com `Σw=1` sobre o metamodelo; (b) SLSQP direto sobre a OOF (log-loss — ótimo global exato); (c) varredura Dirichlet densa (AUC). Gap entre (a) e (b)/(c) no teste = validação do metamodelo. Regiões quase-ótimas (nível ≥ 99% do ótimo) reportadas nos ternários.
6. **Confirmação:** refit dos 5 modelos no treino completo; `w*` agregado entre réplicas (mediana projetada no simplex) aplicado ao teste. (Frase-padrão no artigo sobre o deslocamento OOF→refit, prática herdada do stacking.)
7. **Multiobjetivo (stretch, uma seção):** log-loss × custo de inferência `Σw_i·c_i` (linear em `w` → âncoras NBI triviais). NBI com ~10 subproblemas via reformulação em q−1 variáveis livres; justificado honestamente pelo espaçamento uniforme da fronteira (não por necessidade — com ambos objetivos convexos, weighted-sum também recupera a fronteira); fallback: varredura Dirichlet + filtro não-dominado, que sozinho sustenta a figura. Pares rejeitados: AUC×log-loss (correlacionadas, fronteira degenerada), ECE (estimador frágil por binning).

## 8. Baselines e métricas

Todos sob folds/seeds idênticos (mesma OOF por réplica):

| Baseline | Papel |
|---|---|
| Melhor modelo único (selecionado no OOF) | Piso |
| Voto uniforme `w=1/5` | Centroide do design — grátis |
| **Stacking: LogReg sobre a mesma OOF** | **Competidor direto, obrigatório.** Capacidade estritamente maior (pesos livres + intercepto) — em amostra, stacking ≥ mistura por construção. O artigo antecipa isso: a restrição de simplex é regularização com interpretação direta; se perder no teste, reporta-se. Omitir seria a segunda falha fatal. |
| SLSQP direto no simplex (log-loss) | Ótimo exato — valida o metamodelo |
| Dirichlet densa 10⁴ pontos (AUC) | Ótimo direto não-suave + mede a planura da superfície |

**Métricas:** ROC-AUC (primária de discriminação), PR-AUC, log-loss (primária de otimização, sobre componentes clipados), Brier (sanidade do pipeline, §4), acurácia balanceada. **F1 só com política de limiar pré-registrada** (mesma regra para todos os métodos, ex.: limiar que maximiza F1 no OOF de cada método) — ou cortar F1: com 22% de positivos e limiar 0,5, F1 mede calibração, não discriminação. Custo: tempo médio de inferência por modelo → `Σw_i·c_i`.

**Comparação estatística (corrigida pela revisão adversarial — o achado fatal):**
- Wilcoxon pareado entre as R réplicas de seed = **análise de estabilidade a particionamento**, rotulada como tal — nunca como teste de generalização (réplicas compartilham treino, folds sobrepostos e o mesmo holdout).
- Para qualquer afirmação comparativa formal: **~10 splits externos repetidos 80/20 com o t corrigido de Nadeau–Bengio** (custo ~10× fits ≈ poucas horas — viável e entra no lugar do 2º dataset, não além dele). Alternativa: só estimativas pontuais + gap OOF→teste, sem linguagem de significância.
- **Friedman/Nemenyi: eliminado** (exige muitos datasets; com 1–2 é sem sentido — Demšar).

**Critérios de sucesso declarados a priori (não dependem de "vencer"):**
- S1: Scheffé quadrático com R²-adj ≥ 0,8 no ajuste **e** RMSE de validação externa (25 pontos Dirichlet) pequeno em relação à amplitude da resposta no simplex (não usar R² externo isolado: em superfície achatada ele é mal-posto e sai negativo sem culpa do metamodelo);
- S2: intervalos de estabilidade dos β_ij em **AUC** (métrica de sinal livre — em log-loss a sinergia é garantida por Jensen, §4) identificando quais pares têm complementaridade forte vs. redundância (β_ij ≈ 0);
- S3: predição do metamodelo no `w*` dentro do intervalo observado na confirmação;
- S4: comparação honesta no holdout conforme o protocolo acima — empate com voting/stacking é achado (região ótima plana contendo o uniforme), com análise de equivalência prática.

## 9. Como evitar data leakage

Protocolo inegociável (a revisão adversarial o validou como fundamentalmente sólido):

1. Holdout externo 80/20 estratificado fixado **antes de tudo**; nenhuma decisão o toca.
2. OOF interna: `P` construída com predições exclusivamente out-of-fold (modelo do fold k prediz só o fold k).
3. **Toda seleção de `w`** (design, metamodelo, SLSQP, stacking) usa exclusivamente `P`. Nunca otimizar/escolher no teste.
4. Calibração (se usada): ajustada **dentro** dos folds (sub-split do treino do fold). Calibrar na OOF inteira e depois selecionar `w` na mesma OOF é vazamento de segundo nível. Se complicar, cortar do escopo — o clipping de componentes (§7.2) já resolve o problema numérico.
5. Comparar métodos na própria OOF favorece quem seleciona nela (mistura, stacking, SLSQP) contra quem não seleciona nada (uniforme, melhor single). Toda comparação final é no holdout; o gap OOF→teste de cada método é reportado como diagnóstico de superajuste de seleção.
6. Disciplina do holdout reconciliada por escrito (incoerência apontada pela revisão): o teste é avaliado nas R réplicas **sem seleção** (só leitura); a inferência sobre essas R leituras segue as regras do §8 (estabilidade, não generalização); a estimativa-manchete é a do `w*` agregado aplicado uma vez.
7. Anti-exemplos documentados no próprio repo/datasets: não herdar a confirmação com mesma CV/mesmo seed (`run_replica.py:284-293`); no Bank, `duration` é removida com justificativa.

## 10. Plano de implementação por etapas (8 semanas; com 6, cortar W8 e o NBI)

| Sem. | Entrega | Definition of done |
|---|---|---|
| 1 | Repo novo + dados + EDA | CI 1-job verde; módulos copiados com testes passando; `01_eda.ipynb` renderiza do zero; `data.py` determinístico; **spec do §13 congelada por escrito no README** |
| 2 | Zoo + OOF | 250 fits cacheados em `.npz`; Tabela 1 (5 modelos × métricas em CV) pronta; gate de diversidade avaliado; runtime registrado |
| 3 | Misturas + baselines — **marco: projeto mínimo submetível** | Design de 21 pontos avaliado sobre OOF; uniforme/melhor-single/stacking calculados; requisitos mínimos da disciplina 100% cumpridos |
| 4 | Scheffé + ternários + **esqueleto do artigo** | Metamodelo por réplica com tabela de coeficientes + validação externa; Figs. 3–5; teste `w=e_i ⇒ métrica do modelo i` verde; seções 1–3 do artigo esboçadas (não deixar para a semana 7) |
| 5 | Otimização + confirmação | SLSQP (metamodelo e direto) + Dirichlet densa; gap reportado; confirmação no holdout; *(stretch: NBI q−1 vars)* |
| 6 | Estatística + figuras finais | 10 splits Nadeau–Bengio rodados; estabilidade entre seeds rotulada; 7 figuras finais versionadas; S1–S4 avaliados |
| 7 | Artigo completo | Draft sem código no corpo (equações 1–5, fluxograma, 2 pseudocódigos); números conferidos contra `experiments/` |
| 8 | Slides + buffer | ~15 slides ensaiados; buffer para polimento (Bank como robustez **só** se sobrar tempo) |

Esforço total estimado: ~75–100 h (artigo: orçar 25–30 h, não 16–20). Runtime: 250 fits ≈ 1–2 h; +10 splits externos ≈ mais algumas horas; avaliação de misturas ≈ segundos (700 avaliações × ~15 ms).

## 11. Riscos e mitigação

| # | Risco | Sev. | Mitigação (embutida na spec) |
|---|---|---|---|
| 1 | Inferência inválida (Wilcoxon/Friedman sobre pseudo-réplicas como generalização) | **Fatal** | §8: rótulo de estabilidade + Nadeau–Bengio para claims formais; Friedman eliminado |
| 2 | "Por que não otimizar direto?" sem resposta no texto | **Fatal** | §4: enquadramento interpretativo na introdução + SLSQP/Dirichlet como comparadores |
| 3 | Empate com voting/stacking (provável) | Alto | Critérios S1–S4 a priori; empate = achado mecanisticamente explicado (superfície plana) |
| 4 | Log-loss infinita nos vértices (kNN/NB com prob. 0/1) | Alto | Clipping de **componentes** com ε pré-declarado + sensibilidade |
| 5 | Zoo com gêmeos GBDT → w* não-identificável | Alto | Zoo LR/NB/kNN/RF/XGB + gate de diversidade com regra de troca |
| 6 | Vacuidade do Brier (R²=1 por identidade) | Médio | Brier = sanidade do pipeline, declarado; modelagem genuína em log-loss/AUC |
| 7 | Design sem pontos interiores (lattice puro) | Médio | Centroide + axiais no design; validação Dirichlet interior |
| 8 | Colapso de vértice (um modelo domina) | Médio | Dataset primário com desacordo documentado na literatura; secundário como plano B |
| 9 | Scope creep do repo da dissertação | Médio | Repo novo; lista fechada de ~700 L copiadas; proibições explícitas (FA/Varimax, MBPA, benchmarks HPO, 12 datasets, 30 réplicas) |
| 10 | "Ferramenta demais, proposta de menos" | Médio | Relatório abre com EDA + comparação dos 5 classificadores (que sozinha cumpre "≥2 algoritmos"); mistura entra como Metodologia; NBI é apêndice; "DOE" não aparece no título antes de "ensemble" |
| 11 | NBI no simplex (única incerteza técnica) | Baixo | Stretch isolado na W5 com fallback garantido (SLSQP + Dirichlet + filtro não-dominado) |
| 12 | Dependências | Baixo | Tudo essencial já roda em arm64 (xgboost 2.1.4 + libomp OK); venv novo com python3.11; ternário em matplotlib puro |
| 13 | Cronograma (artigo subestimado) | Médio | Esqueleto do artigo na W4; marco submetível na W3; cortes atingem só stretch goals |

## 12. Estrutura sugerida do repositório

**Repo novo standalone** (`mixture-ensemble-pco213`) — nunca fork/branch do repo da dissertação (main imutável, 9,5k linhas 90% HPO-específicas, péssima UX de avaliação). Cada arquivo copiado com docstring de atribuição *"Adapted from doe_nbi_hpo_project (Ribeiro, 2026), MIT License"*.

```
mixture-ensemble-pco213/
├── README.md                  # o quê, como reproduzir em 3 comandos, figura-vitrine, spec congelada
├── requirements.txt           # ~8 linhas
├── pyproject.toml             # nome, ruff, pytest
├── configs/experiment.yaml    # dataset, folds, réplicas, design, seeds, ε de clipping
├── data/{raw,processed}/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
├── src/mixens/
│   ├── data.py                # load + pré-processamento + split externo
│   ├── base_models.py         # zoo (5 clf) + geração/caching OOF (.npz) + clipping
│   ├── mixture_design.py      # [COPIADO simplex.py] + Dirichlet
│   ├── ensemble_eval.py       # métrica(w | OOF) + baselines voting/stacking
│   ├── scheffe.py             # [COPIADO MixtureScheffeModel] + validação externa + coef-table
│   ├── optimize.py            # SLSQP Σw=1; Pareto (Dirichlet + não-dominado); NBI opcional
│   └── plots.py               # ternários baricêntricos, Pareto, boxplots, calibração
├── scripts/
│   ├── run_experiment.py      # OOF → design → aval → Scheffé → otim → confirmação
│   └── make_figures.py
├── experiments/               # oof/*.npz, results/*.csv (nomes determinísticos por seed)
├── reports/{article,figures}/
└── tests/                     # test_mixture_design, test_scheffe [adaptados do repo] +
                               # test_ensemble_eval (w=e_i ⇒ modelo i), test_optimize
```

Artefatos finais: `01_eda.ipynb`; pipeline reprodutível; artigo (Resumo · Introdução · Fundamentação (1 página de misturas, não um capítulo) · Metodologia (fluxograma + eqs + pseudocódigo + protocolo a priori) · Dados/EDA · Resultados · Discussão · Conclusão); ~15 slides; 7 figuras — (1) EDA/desbalanceamento, (2) boxplots dos modelos-base, (3) **contornos ternários de AUC em sub-misturas de 3 modelos com pontos do design sobrepostos — a figura-vitrine**, (4) predito-vs-observado + validação externa, (5) forest plot dos β com intervalos de estabilidade, (6) fronteira log-loss × custo com uniforme/single/stacking/ótimo marcados, (7) curvas de calibração.

## 13. Decisão final

# **GO — com escopo congelado**

**Especificação única (resolve as divergências entre os pareceres; congelar no README na semana 1):**

| Item | Decisão congelada |
|---|---|
| Tarefa | Classificação binária |
| Dataset primário | UCI Credit Card Default (ID 350) — 30.000×23, 22,1% |
| Dataset secundário | UCI Bank Marketing (ID 222), fonte `bank_marketing/raw` — **só** como robustez de semana 8 |
| Zoo (M=5) | LogReg, GaussianNB, kNN, RandomForest, XGBoost (sem SVM-RBF, sem 2º GBDT) |
| Design | Lattice {5,2} (15) + centroide global + 5 axiais = **21 corridas**; +25 Dirichlet de validação externa |
| Metamodelo | Scheffé quadrático (cúbico especial como teste de ordem), ajuste **por réplica**, R=10 seeds |
| Perda de otimização | log-loss (componentes clipados, ε=10⁻³); AUC como resposta de sinal livre; Brier = sanidade |
| CV | Holdout 80/20 + StratifiedKFold K=5 interno; OOF estrita |
| Baselines | melhor single, uniforme, **stacking LogReg na mesma OOF**, SLSQP direto, Dirichlet densa |
| Inferência | Estabilidade entre seeds (rotulada) + 10 splits Nadeau–Bengio para claims formais; sem Friedman |
| Multiobjetivo | Stretch: log-loss × custo linear, NBI q−1 vars, fallback Dirichlet+não-dominado |
| Repo | Novo, standalone, ~700 L copiadas com atribuição |

**Condições do GO (sem elas, NO-GO):**
1. O enquadramento de honestidade computacional (§4) escrito na introdução do artigo;
2. O protocolo de inferência corrigido (§8) — nunca imprimir p-valores de pseudo-réplicas como evidência de generalização;
3. Stacking na mesma OOF entre os baselines;
4. Clipping/limiar/Jensen declarados a priori na Metodologia.

Veredito da revisão adversarial, na íntegra: *"o caso GO sobrevive à tentativa de refutação"* — condicionado exatamente aos itens acima, todos com correção barata **antes** da semana 1. Com eles, é um projeto acima da média da disciplina; sem eles, cai na primeira arguição competente.

## 14. Próximos passos concretos

1. **Congelar a spec do §13** (decidir apenas: manter GaussianNB ou reduzir para M=4 sem ele — recomendo manter; NB é o componente de viés mais distinto e seu colapso esperado em pesos baixos é história para a discussão).
2. Criar o repo `mixture-ensemble-pco213` com a árvore do §12; venv python3.11; copiar `simplex.py`, `MixtureScheffeModel`+`FitReport`, `design/diagnostics.py` e os testes correspondentes, com cabeçalhos de atribuição.
3. Copiar `data/source/credit_card_default/` (raw XLS + processed) para `data/raw/`; registrar checksum.
4. `01_eda.ipynb`: EDA + decisões de pré-processamento documentadas (EDUCATION/MARRIAGE não documentados, ordinais PAY_*, escalonamento por modelo).
5. Implementar `base_models.py` (zoo + OOF cacheada + clipping) e rodar o gate de diversidade — **primeira validação empírica da hipótese existencial do projeto** (esperado: correlações de erro < 0,95 entre famílias).
6. A partir daí, seguir o plano semanal do §10.

Nada disso foi implementado ainda, conforme solicitado — este documento é o único artefato produzido.
