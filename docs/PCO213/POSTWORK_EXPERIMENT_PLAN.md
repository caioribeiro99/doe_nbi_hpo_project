# POSTWORK EXPERIMENT PLAN — DoE + RSM + NBI sobre pesos de ensemble

**Data:** 2026-07-07
**Branch:** `pco213-classification-mixture-ensemble`
**Status:** plano de pesquisa pós-entrega. O trabalho final da disciplina (commits `5d2bc4a..9b1d59b`, `dist/PCO213_final_entrega.zip`) está **congelado e não é alterado** por este plano. Nenhum experimento novo foi executado ainda.

---

## 1. Diferença entre o trabalho entregue e o pós-trabalho

| | **Entregue (PCO213)** | **Pós-trabalho (pesquisa)** |
|---|---|---|
| Método | Arranjo de misturas + metamodelo de Scheffé, **mono-objetivo por vez** | **Método completo da dissertação: DoE + metamodelagem de superfície de resposta + otimização multiobjetivo guiada por NBI** |
| Otimização | SLSQP no metamodelo vs. SLSQP direto, uma métrica de cada vez | NBI sobre os metamodelos das múltiplas respostas → **fronteira Pareto interpretável** |
| Respostas | AUC, log-loss, Brier (analisadas separadamente) | AUC, log-loss, Brier, PR-AUC, **custo de inferência** e **N_eff**, tratadas como sistema multiobjetivo |
| Replicações | 2 repetições da CV (restrição de runtime da disciplina) | R≥5, com papel inferencial explícito (§7) |
| Pergunta | "a mistura supera baselines? o metamodelo é interpretável?" | Pergunta revisada (§2) sobre a **fronteira** discriminação × calibração × custo |
| Datasets | Santander apenas | Escalonamento mínimo → intermediário → completo (§8) |

O entregue é o *estudo de viabilidade executado* do pós-trabalho: OOF, arranjo, Scheffé, protocolo anti-vazamento e infra de comparação já existem e são reutilizados sem alteração conceitual.

## 2. Pergunta científica revisada

> **"Pode-se tratar o espaço de pesos de um ensemble probabilístico como um domínio de mistura, ajustar metamodelos de resposta para múltiplas métricas e usar NBI para obter uma fronteira Pareto interpretável entre discriminação, calibração e custo?"**

Os **fatores do DoE não são hiperparâmetros** dos modelos-base — são os **pesos do ensemble**, com as restrições canônicas de mistura `w_i ≥ 0, Σw_i = 1`. Não se trata de HPO: nenhum modelo-base é reajustado durante a otimização; o espaço de decisão é o simplex dos pesos e as respostas são funções determinísticas de `w` dada a matriz OOF.

## 3. Por que o método completo é DoE + metamodelagem + NBI

A assinatura metodológica do mestrado nunca foi "arranjo + metamodelo" isolados: é o encadeamento **planejar (DoE) → modelar (RSM) → otimizar multiobjetivo (NBI) → confirmar**. No domínio dos pesos, cada elo tem papel próprio:

1. **DoE no simplex** dá um plano de avaliação auditável com suporte correto para o polinômio canônico (vértices, arestas, interior) e graus de liberdade para diagnóstico;
2. **Metamodelos de Scheffé por resposta** entregam o que nenhum otimizador pontual entrega: coeficientes interpretáveis (β_i = contribuição; β_ij = complementaridade), regiões quase-ótimas e superfícies contínuas sobre as quais o NBI opera de forma barata e suave;
3. **NBI** transforma o conjunto de metamodelos numa **fronteira Pareto com espaçamento uniforme** — o diferencial clássico do NBI sobre weighted-sum, que aglomera pontos em regiões convexas e falha em regiões não convexas da fronteira.

**Declaração de honestidade (herdada do trabalho entregue, inegociável):** para **objetivo único**, especialmente log-loss (convexa nos pesos), o **SLSQP direto sobre a OOF é o oráculo natural** — encontra o ótimo global exato em milissegundos. O método **não alega** que Scheffé/NBI seja mais eficiente que SLSQP nesse caso, nem que "economiza avaliações" (avaliar `w` na OOF é um produto matriz-vetor). O valor do método está em: modelar a superfície, interpretar coeficientes e complementaridades, mapear regiões quase-ótimas, construir fronteiras Pareto multiobjetivo e explicar o trade-off discriminação × calibração × custo. O SLSQP entra como **benchmark/oráculo single-objective**; a fronteira por varredura Dirichlet densa entra como **referência de Pareto "verdadeiro"**.

## 4. Por que HPO/Grid/Optuna/BO não são os benchmarks principais

- **Objeto errado:** Grid/Optuna/BO/SMAC são buscadores para funções caras de avaliar em espaços-caixa de hiperparâmetros. Aqui os fatores são pesos num **simplex**, a avaliação é **quase gratuita** (OOF pré-computada) e não há re-treino por ponto — a premissa de custo que justifica BO não existe;
- **Pergunta errada:** esses métodos respondem "onde está o ótimo?" com o mínimo de avaliações; a pergunta do pós-trabalho é "**como é a superfície** e qual é a **fronteira** entre objetivos?";
- **Comparação não informativa:** com avaliação grátis, qualquer buscador razoável encontra o ótimo mono-objetivo; vencê-los não diz nada sobre o mérito do método.

**Benchmarks corretos para *ensemble weighting*** (todos sobre a MESMA OOF, mesmos folds):
melhor modelo individual · voto uniforme (centroide do design) · **stacking logístico** (competidor direto com capacidade maior) · **SLSQP mono-objetivo** (oráculo log-loss; para AUC, varredura densa) · **random Dirichlet search com o mesmo orçamento de pontos do design** (higiene de protocolo: o DoE deve informar mais que o mesmo número de pontos aleatórios) · **Dirichlet denso + filtro de não dominância** (aproximação do Pareto real) · **NSGA-II opcional** (só se não inflar o escopo; população×gerações casada ao orçamento).

## 5. Como o NBI entra no método

O NBI opera **sobre os metamodelos** (funções suaves e baratas), no espaço dos pesos:

1. **Normalização dos objetivos** pelos pontos de utopia/nadir extraídos da matriz payoff (escalas de AUC ~0,88, log-loss ~0,21 e custo em ms são incomensuráveis);
2. **Âncoras:** ótimo individual de cada metamodelo no simplex (SLSQP multi-partida com Σw=1). Para o custo, linear em w, a âncora é trivial (vértice do modelo mais barato);
3. **Matriz payoff Φ:** valor de todos os objetivos em cada âncora → define a CHIM (convex hull of individual minima);
4. **Subproblemas NBI:** para cada β num reticulado do simplex dos objetivos (ex.: {k,10} para k objetivos), maximizar o avanço t ao longo da quase-normal à CHIM, sujeito a `Φβ + t·n̂ = F(w) − F*` e `w ∈ Δ^{M−1}`;
5. **Resolução no espaço dos pesos:** reutilizar o `nbi_core.py` da branch `repo-publication-readiness` (genérico sobre `Callable[[np.ndarray], float]`), portado com atribuição. Como seu espaço de decisão é uma caixa com restrições de desigualdade, aplicar a reformulação já documentada na origem: otimizar em **M−1 variáveis livres** com `w_M = 1 − Σ` e restrição `w_M ≥ 0`. **Não portar `post_optimization.py`** (bug conhecido em `:166-170` na origem);
6. **Candidatos Pareto:** os `w` resultantes são reavaliados **nas métricas reais** (OOF) — o metamodelo propõe, a OOF confirma; filtro final de não dominância sobre valores reais.

**Ressalva declarada:** com objetivos convexos (log-loss × custo linear), weighted-sum também recupera a fronteira — o argumento do NBI é o **espaçamento uniforme** dos pontos e a robustez a regiões não convexas (esperadas quando AUC, não convexa em w, entra no sistema), nunca "necessidade".

## 6. Etapas do método pós-trabalho

### Etapa A — Modelos-base e OOF
- Treinar o zoo heterogêneo congelado (LR, GNB, kNN, RF, XGB) com as configurações fixas do trabalho entregue;
- Gerar predições OOF (`RepeatedStratifiedKFold`, K=5, R repetições — R por plano do §8) e matriz de holdout `Q` com refit único por split;
- **Medir custo de inferência por modelo** `c_i` (ms/1000 predições, mediana de ≥5 medições no holdout) → custo do blend = `Σ w_i c_i` (linear em w; âncora NBI trivial). Registrar também custo de treino como contexto.

### Etapa B — DoE no simplex
- Simplex-lattice {5,2} (15 pontos) + **centroide global** + **5 pontos axiais** (x_i=(M+1)/2M) = 21 corridas (design do trabalho entregue, reutilizado);
- Aumento opcional para o cúbico especial: +C(5,3)=10 centroides de face ternária ({5,3}-subset) se a validação externa indicar curvatura de 3ª ordem;
- **25–40 pontos Dirichlet(1,…,1) de validação externa**, nunca usados no ajuste.

### Etapa C — Respostas
Para cada vetor de pesos `w` (design + validação), calcular sobre a OOF:
- **ROC-AUC** (maximizar; discriminação);
- **log-loss** (minimizar; calibração — componentes clipados em ε=10⁻³, política herdada);
- **Brier** (minimizar; quadrático exato em w → papel permanente de teste de sanidade do pipeline, não de resposta de modelagem);
- **PR-AUC** (reportar sempre em datasets desbalanceados; objetivo NBI apenas se substituir AUC, nunca junto — quase colineares);
- **custo de inferência ponderado** `Σ w_i c_i` (minimizar);
- **número efetivo de componentes** `N_eff = 1/Σw_i²` (complexidade do ensemble; exatamente quadrático em w — como o Brier, o metamodelo o reproduz por identidade; usável como objetivo/registro sem metamodelagem "de mérito").

### Etapa D — Metamodelagem
- Ajustar Scheffé **linear, quadrático e cúbico especial** para cada resposta modelável (AUC, log-loss, PR-AUC; custo e N_eff têm forma fechada — declarar);
- **Comparar ordens por validação externa** (RMSE absoluto e relativo à amplitude nos pontos Dirichlet; R² externo apenas quando a amplitude o sustenta — lição do trabalho entregue: superfície de AUC com amplitude 0,016 torna R² externo mal-posto);
- **Escolher a ordem com parcimônia:** menor ordem cujo RMSE externo não seja significativamente pior que o da ordem seguinte (regra pré-registrada);
- Analisar β_i e β_ij por repetição (faixas de estabilidade) e no ajuste agrupado; número de condição da matriz canônica reportado.

### Etapa E — NBI
Conforme §5: normalizar → âncoras → payoff → subproblemas (β-reticulado, 10–15 pontos por par de objetivos; {3,10}≈66 para o trio) → candidatos no espaço de pesos → **validação dos candidatos nas métricas reais OOF** → filtro de não dominância → seleção de pontos operacionais (joelho/TOPSIS, reutilizando `selection.py` da origem) → **confirmação única no holdout** dos pontos selecionados.

Sistemas de objetivos (em ordem de prioridade):
1. **Trio principal: AUC (max) × log-loss (min) × custo (min)** — discriminação, calibração e custo; assinatura de 3 objetivos da dissertação;
2. Par limpo: log-loss × custo (convexo + linear; fronteira de leitura imediata);
3. Par de tensão: AUC × log-loss (registrar risco de fronteira degenerada por correlação; a tensão GNB-calibração observada no trabalho entregue sugere que não será totalmente degenerada — verificar antes de prometer).

### Etapa F — Comparação
Contra o NBI/metamodelo, sob os MESMOS folds/OOF e orçamentos declarados:
- melhor modelo individual; voto uniforme; stacking logístico;
- SLSQP mono-objetivo para log-loss (oráculo exato);
- otimização mono-objetivo para AUC via varredura densa (AUC não é suave/convexa em w);
- **random Dirichlet search com o mesmo nº de pontos do design** (21+aumento) — o DoE precisa vencer o aleatório de mesmo orçamento em qualidade de metamodelo e de fronteira;
- **Dirichlet denso (10⁴–10⁵) + filtro Pareto = fronteira de referência**: métricas GD/IGD e spacing entre a fronteira NBI e a de referência;
- NSGA-II opcional (mesmo nº de avaliações reais que o Dirichlet denso, ou orçamento casado ao NBI — decidir e declarar a priori; cortar se inflar o escopo).

## 7. Papel das replicações

Replicações (R repetições da CV, e opcionalmente S splits externos repetidos) **não servem para melhorar score médio** — servem para medir estabilidade e incerteza de TODOS os objetos que o método produz:

1. **Estabilidade dos pesos Pareto:** dispersão de cada w* selecionado (joelho/extremos) entre repetições — por componente e em N_eff;
2. **Estabilidade dos coeficientes de Scheffé:** faixas percentílicas de β_i, β_ij entre repetições (rotuladas como estabilidade a repartição, não incerteza amostral plena — lição herdada);
3. **Estabilidade da fronteira:** GD/IGD entre fronteiras de repetições distintas; envelope da fronteira em vez de curva única;
4. **Estabilidade dos pares complementares:** o ranking dos β_ij (ex.: GNB–kNN dominante sob AUC no trabalho entregue) se mantém entre repetições?
5. **Variância do gap NBI×Pareto direto:** o gap médio e sua dispersão entre repetições — um gap pequeno e estável é o critério de fidelidade do metamodelo/NBI;
6. **Consistência contra baselines:** os sinais das comparações (t corrigido para CV repetida, ρ=0,25) devem manter direção entre repetições; afirmações formais continuam restritas ao quadro de inferência do trabalho entregue.

## 8. Planos mínimo, intermediário e completo

| Plano | Datasets | Replicações | Objetivos | Computação estimada* |
|---|---|---|---|---|
| **Mínimo** | Santander (reuso da infra executada) | R=5 repetições (vs. 2 da entrega); 1 split externo | Trio principal + par log-loss×custo | ~25 fits/repetição ≈ 25 min de OOF novo + NBI em segundos |
| **Intermediário** | Santander + **Give Me Some Credit ou UCI Credit Card Default** (baratos; UCI adiciona EDA interpretável) | R=5 por dataset; 3–5 splits externos no menor dataset | idem + comparação de β_ij entre datasets | +30–60 min |
| **Completo** | Santander; **BNP Paribas** (log-loss oficial, missing pesado); **Porto Seguro (cap 200k) ou Give Me Some Credit** (desbalanceamento severo); **UCI Credit Card Default** (âncora interpretável) | R=5–10; 10 splits externos onde a inferência formal for reivindicada | sistema completo + análise cruzada de fronteiras por regime de dados | ~1–2 dias de lotes noturnos (aritmética validada nos estudos anteriores) |

\* Base: OOF do Santander custa ~5 min/repetição (medido); avaliação de pesos e NBI custam segundos por operarem sobre metamodelos/OOF.

## 9. Critérios de sucesso (pré-registrados)

O método é bem-sucedido se:

- **S1 (fidelidade do metamodelo):** prediz bem pontos externos do simplex — RMSE externo pequeno em relação à amplitude da resposta (relatado por resposta; R² externo só onde a amplitude o sustenta);
- **S2 (fidelidade da fronteira):** a fronteira NBI aproxima a fronteira por Dirichlet denso — GD/IGD pequenos e estáveis entre repetições;
- **S3 (trade-offs reais):** os pontos Pareto oferecem trade-offs mensuráveis entre AUC, log-loss e custo (fronteira não degenerada em pelo menos um par de objetivos);
- **S4 (interpretabilidade):** os coeficientes de Scheffé são interpretáveis e estáveis entre repetições (ranking dos β_ij preservado);
- **S5 (competitividade):** os pesos sugeridos são competitivos contra stacking, voto uniforme e SLSQP mono-objetivo nas métricas correspondentes;
- **S6 (valor explicativo no empate):** quando não houver ganho preditivo, o metamodelo deve **explicar** o resultado — região quase-ótima plana contendo o centroide (caso AUC do trabalho entregue) ou colapso para vértice — transformando o empate em achado, não em falha.

## 10. Restrições e próximos passos

- **Nada foi executado**: este documento é apenas o plano; nenhum experimento novo, nenhuma alteração na entrega final nem no `dist/PCO213_final_entrega.zip`;
- Implementação futura (quando autorizada): portar `nbi_core.py` (com atribuição, reformulação M−1 vars) e opcionalmente `selection.py` para `src/mixens/`; estender `pco213_run_santander_study.py` com estágios `cost`, `nbi` e `pareto`; medir `c_i`; rodar o plano mínimo;
- Publicação-alvo natural: nota curta/artigo de workshop conectando DOE de misturas, decomposição de ambiguidade (Krogh–Vedelsby) e NBI — diferencial: fronteira Pareto *interpretável* sobre pesos de ensemble, com o repositório e o estudo PCO213 como base empírica.
