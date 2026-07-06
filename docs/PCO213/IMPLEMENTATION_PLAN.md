# IMPLEMENTATION PLAN — Projeto Final PCO213
## Otimização de ensembles de classificadores por arranjo de misturas e metamodelagem de superfície de resposta

**Branch de desenvolvimento:** `pco213-classification-mixture-ensemble` (criada a partir da `main` em 2026-07-06)
**Documento-base:** [`FEASIBILITY_PCO213_FINAL_PROJECT.md`](FEASIBILITY_PCO213_FINAL_PROJECT.md) (veredito GO, com 4 condições)
**Estado atual:** apenas scaffold — nenhum pipeline implementado, nenhum dataset baixado, nenhum experimento executado.

---

## 1. Estratégia de repositório (decisão)

O desenvolvimento será feito **como branch deste repositório** (`doe_nbi_hpo_project`), **não** como repo separado — decisão do autor em 2026-07-06, substituindo a recomendação de repo standalone do relatório de viabilidade (§12).

Consequências práticas:

- **Base = `main`** (baseline imutável da dissertação, tag `v0.1.0-dissertation`). A `main` **não** contém os módulos reutilizáveis identificados na análise de viabilidade (`simplex.py`, `model_families.py` com o `MixtureScheffeModel`, `nbi_core.py`, `design/diagnostics.py`, `tests/unit/`) — eles existem apenas na branch `repo-publication-readiness`. Serão **portados por cópia** para esta branch (commit 2, §9), com cabeçalho de atribuição indicando branch e commit de origem.
- **Nenhum módulo existente da `main` será alterado** (`src/doe_xgb/` da era da dissertação fica intocado). Todo código novo vive num pacote irmão `src/mixens/`, autocontido.
- Esta branch é de longa duração e **não** gera PR para `main` (política do repo: `main` imutável). Ela é o entregável da disciplina; a avaliação parte de `docs/PCO213/` + `src/mixens/`.
- `git checkout repo-publication-readiness -- <path>` é o mecanismo de porte (traz o arquivo sem trocar de branch); os arquivos portados são então movidos/renomeados para `src/mixens/` e ajustados (imports, docstring de atribuição).

## 2. Escopo congelado

| Item | Decisão |
|---|---|
| Tarefa | **Classificação binária** |
| Dataset primário | UCI Default of Credit Card Clients (ID 350) — 30.000×23, 22,1% default |
| Dataset secundário (opcional, só se sobrar tempo) | UCI Bank Marketing (ID 222), fonte canônica `bank-additional-full.csv` **com** `duration` (removida e documentada no pré-processamento) |
| Modelos-base (M=5) | LogisticRegression, GaussianNB, kNN, RandomForest, XGBoost |
| Design de mistura | Simplex-lattice {5,2} (15 pts) + centroide global + 5 axiais = **21 corridas**; +25 pontos Dirichlet de validação externa (fora do ajuste) |
| Metamodelo | Scheffé quadrático (cúbico especial como teste de ordem), ajustado **por réplica** (R=10 seeds de CV) |
| Validação | Holdout externo 80/20 estratificado + StratifiedKFold K=5 interno; matriz OOF estrita |
| Perda de otimização | log-loss sobre probabilidades de componentes clipadas (ε=10⁻³, declarado a priori) |
| Inferência | Estabilidade entre seeds (rotulada como tal) + 10 splits externos com t corrigido de Nadeau–Bengio para afirmações formais; **sem Friedman** |
| Multiobjetivo (stretch) | log-loss × custo de inferência `Σw_i·c_i`; NBI em q−1 variáveis livres; fallback: varredura Dirichlet + filtro não-dominado |

**Condições do GO (do relatório de viabilidade — inegociáveis):**
1. Enquadramento de honestidade computacional na introdução do artigo (o metamodelo é inferência interpretável, não necessidade computacional; SLSQP/Dirichlet densa validam o ótimo);
2. Nunca imprimir p-valores de pseudo-réplicas como evidência de generalização;
3. Stacking sobre a mesma OOF entre os baselines;
4. Clipping, política de limiar e o resultado de Jensen declarados a priori na Metodologia.

## 3. Classificação vs. regressão

**Classificação binária.** Justificativa (detalhes no feasibility §5): combinação convexa de probabilidades é objeto canônico (opinion pooling) com vértices = classificadores individuais; o desbalanceamento faz ROC-AUC/F1/balanced accuracy/log-loss divergirem (sustenta a discussão multi-métrica e o eixo multiobjetivo); existe eixo de calibração (inexistente em regressão); 11 datasets binários já validados no ecossistema do repo. A versão regressão não é inviável — é estritamente menos interessante em todos os eixos.

## 4. Dataset recomendado e justificativa

**Primário: UCI Default of Credit Card Clients (ID 350).** O paper de origem (Yeh & Lien, 2009, *Expert Systems with Applications*) é uma comparação de 6 famílias de classificadores com desempenhos próximos e ranking dependente da métrica — o desacordo entre famílias (hipótese existencial do projeto: sem ele, o ótimo colapsa num vértice) está documentado na fonte primária. EDA e pré-processamento genuínos (categorias não documentadas EDUCATION=0/5/6 e MARRIAGE=0; ordinais PAY_0…PAY_6; escalas monetárias assimétricas; correlação BILL_AMT1-6). Desbalanceamento de 22,1% é o ponto doce: métricas divergem sem exigir SMOTE. Teto honesto (AUC ~0,77–0,78). Citação limpa (UCI ID 350, DOI).

Rejeitados (feasibility §6): MAGIC (dataset-manchete da dissertação — reciclagem, simulado, sem pré-processamento), AI4I (sintético + vazamento estrutural + 3,4% positivos), Adult (clichê), Spambase (pré-engenheirado), Kaggle Telco (licença/citação frágeis).

**Nota de execução:** o download será refeito **nesta branch** via script próprio em `src/mixens/`/`scripts` (commit 3), gravando em `data/pco213/raw/` com checksum — não referenciar `data/source/` de outra branch.

## 5. Baselines obrigatórios

Todos sob folds/seeds idênticos (mesma matriz OOF por réplica); comparação final apenas no holdout:

1. **Melhor modelo individual** (selecionado na OOF, mesma regra de seleção dos demais);
2. **Média simples / voting uniforme** `w = 1/5` (é o centroide do design — ponto grátis);
3. **Stacking com Logistic Regression** ajustada sobre a mesma OOF — competidor direto, capacidade estritamente maior (pesos livres + intercepto); omiti-lo seria falha fatal;
4. **Ensemble otimizado por arranjo de misturas** (ótimo do metamodelo de Scheffé).

Comparadores de validação do metamodelo (condição do GO, não "baselines" no sentido da disciplina):
- SLSQP direto sobre a OOF com `Σw=1` (log-loss é convexa em `w` → ótimo global exato);
- Varredura Dirichlet densa (10⁴ pontos) para AUC.

## 6. Modelos-base candidatos

| Modelo | Papel no zoo | Observações |
|---|---|---|
| LogisticRegression | viés linear | escalonamento no pipeline próprio |
| GaussianNB | viés generativo/independência | satura probabilidades em 0/1 → clipping obrigatório |
| k-NN | viés local/instância | probabilidades em múltiplos de 1/k (inclui 0 e 1 exatos) → clipping; escalonamento |
| RandomForest | bagging de árvores | — |
| XGBoost | boosting | **dependência viável confirmada**: xgboost 2.1.4 importa e roda em arm64 com libomp do Homebrew (verificado na análise de viabilidade); instalar no venv novo |

Regras: **sem SVM-RBF** (proibitivo em ~24k linhas de treino); **sem segundo GBDT** (gêmeos tornam w* não-identificável). **Gate de diversidade** antes do arranjo: correlação de erros OOF / Q-statistic par a par; se algum par > 0,95, trocar um componente (regra declarada a priori).

## 7. Métricas

| Métrica | Papel | Cuidados |
|---|---|---|
| ROC-AUC | primária de discriminação (resposta de sinal livre para o Scheffé) | não convexa em `w` — otimização via varredura densa |
| log-loss | primária de otimização | **clipping das probabilidades dos COMPONENTES** em `[ε, 1−ε]`, ε=10⁻³ declarado a priori, sensibilidade ε=10⁻³ vs 10⁻⁶ reportada; nunca clipar o blend (quebra a linearidade em `w`) |
| F1-score | métrica operacional | **só com política de limiar pré-registrada**: mesmo procedimento para todos os métodos (limiar que maximiza F1 na OOF de cada método); nunca limiar 0,5 com 22% de positivos |
| Balanced accuracy | métrica operacional | mesma política de limiar do F1 |
| Precision / Recall | leitura operacional do trade-off | reportadas no limiar pré-registrado |
| Brier | **sanidade do pipeline** | é exatamente quadrático em `w` → Scheffé quadrático o reproduz com R²=1 por identidade; usar como teste de recuperação de coeficientes, declarado no texto — não como validação do metamodelo |
| Tempo computacional | objetivo secundário (stretch multiobjetivo) | custo de inferência do blend é `Σw_i·c_i` — linear em `w` |

## 8. Arquitetura proposta (adaptada à decisão de branch)

```
doe_nbi_hpo_project/                      (branch pco213-classification-mixture-ensemble)
├── src/
│   ├── doe_xgb/                          # INTOCADO (baseline da dissertação)
│   └── mixens/                           # pacote novo, autocontido — todo o projeto PCO213
│       ├── __init__.py
│       ├── data.py                       # download UCI 350 + pré-processamento + split externo
│       ├── base_models.py                # zoo M=5 + geração/caching OOF (.npz) + clipping
│       ├── mixture_design.py             # [PORTADO simplex.py @ repo-publication-readiness] + Dirichlet
│       ├── scheffe.py                    # [PORTADO MixtureScheffeModel+FitReport] + validação externa
│       ├── ensemble_eval.py              # métrica(w | OOF) + baselines voting/stacking
│       ├── optimize.py                   # SLSQP Σw=1; Pareto (Dirichlet + não-dominado); NBI opcional
│       └── plots.py                      # ternários baricêntricos, Pareto, boxplots, calibração
├── scripts/
│   ├── pco213_run_experiment.py          # OOF → design → aval → Scheffé → otim → confirmação
│   └── pco213_make_figures.py            # regenera figuras a partir de experiments/pco213/
├── tests/mixens/                         # testes portados (simplex, Scheffé) + novos
│   │                                     #   (w=e_i ⇒ métrica do modelo i; ótimo ∈ simplex)
├── docs/PCO213/                          # este diretório: feasibility, plano, README
├── notebooks/pco213/                     # 01_eda.ipynb, 02_results.ipynb
├── data/pco213/{raw,processed}/          # dataset (raw entra no .gitignore quando baixado)
├── experiments/pco213/                   # saídas: oof/*.npz, results/*.csv (já ignorado pelo git)
├── reports/pco213/                       # artigo (formato de disciplina) + tabelas
└── figures/pco213/                       # PNGs/PDFs finais numerados como no artigo
```

Convenções: prefixo `pco213_` nos scripts para não colidir com os da dissertação; arquivos portados com docstring *"Ported from doe_nbi_hpo_project @ repo-publication-readiness (<commit>), adapted for PCO213"*; seeds e configuração em um `configs/pco213_experiment.yaml`.

## 9. Etapas de implementação (mapeadas em commits)

Cronograma-alvo: 8 semanas (com 6, cortar a etapa 8-stretch e o dataset secundário). Marco de segurança: **ao fim da etapa 4 o projeto já cumpre todos os requisitos mínimos da disciplina.**

| # | Commit sugerido | Conteúdo | Definition of done |
|---|---|---|---|
| 1 | `chore: scaffold PCO213 classification mixture ensemble project` | este scaffold (docs + estrutura de diretórios) | branch criada; docs revisados |
| 2 | `feat(pco213): port simplex and Scheffé modules with tests` | `mixture_design.py`, `scheffe.py`, `tests/mixens/` portados de `repo-publication-readiness`; `src/mixens/__init__.py`; venv python3.11 + `requirements` mínimo | `pytest tests/mixens` verde no venv novo |
| 3 | `feat(pco213): data loading and preprocessing for UCI credit card default` | `data.py` (download com checksum → `data/pco213/raw/`, pré-processamento documentado, split externo 80/20); regra de `.gitignore` para `data/pco213/raw/*` | `data/pco213/processed/` determinístico por seed |
| 4 | `feat(pco213): EDA notebook` | `notebooks/pco213/01_eda.ipynb` | renderiza do zero; decisões de pré-processamento justificadas |
| 5 | `feat(pco213): base model zoo and OOF probability matrices` | `base_models.py` (5 modelos × 5 folds × 10 réplicas, cache `.npz`, clipping); gate de diversidade | 250 fits cacheados; Tabela 1 (modelos × métricas em CV); correlações de erro reportadas |
| 6 | `feat(pco213): mixture design evaluation and mandatory baselines` | `ensemble_eval.py` + design de 21 pontos + uniforme/melhor-single/stacking | **marco: projeto mínimo submetível** |
| 7 | `feat(pco213): Scheffé metamodel per replica with diagnostics and ternary plots` | `scheffe.py` (validação externa em 25 pts Dirichlet, tabela de coeficientes, intervalos de estabilidade), `plots.py` | teste `w=e_i ⇒ métrica do modelo i` verde; Figs. 3–5 |
| 8 | `feat(pco213): weight optimization and holdout confirmation` | `optimize.py` (SLSQP metamodelo + SLSQP direto + Dirichlet densa; gap reportado); confirmação no holdout; *(stretch: NBI q−1 vars; Pareto log-loss × custo)* | gap metamodelo↔direto no teste documentado |
| 9 | `feat(pco213): statistical comparison and final figures` | 10 splits Nadeau–Bengio; estabilidade entre seeds rotulada; 7 figuras finais em `figures/pco213/` | critérios S1–S4 do feasibility avaliados |
| 10 | `docs(pco213): article-format report and slides` | `reports/pco213/` (artigo sem código no corpo: equações, fluxograma, 2 pseudocódigos) + ~15 slides | números conferidos contra `experiments/pco213/` |

Esqueleto do artigo começa junto com o commit 7 (não deixar para o fim); orçar 25–30 h para o texto.

## 10. Riscos e mitigação

| # | Risco | Sev. | Mitigação |
|---|---|---|---|
| 1 | Inferência inválida (p-valores de pseudo-réplicas como generalização) | **Fatal** | §2: estabilidade rotulada + Nadeau–Bengio; sem Friedman |
| 2 | "Por que não otimizar direto?" sem resposta | **Fatal** | Enquadramento na introdução + SLSQP/Dirichlet como validadores (condição 1 do GO) |
| 3 | Empate com voting/stacking (provável) | Alto | Critérios de sucesso a priori (S1–S4); empate = achado (superfície plana explicada mecanisticamente) |
| 4 | log-loss infinita nos vértices (kNN/GaussianNB com prob. 0/1) | Alto | Clipping de componentes ε=10⁻³ + análise de sensibilidade |
| 5 | Colapso de vértice (um modelo domina) | Médio | Dataset com desacordo documentado na literatura; gate de diversidade; Bank Marketing como plano B |
| 6 | Vacuidade do Brier (R²=1 por identidade) | Médio | Brier = sanidade declarada; modelagem genuína em log-loss/AUC |
| 7 | **Branch base sem os módulos** (main não tem simplex/Scheffé/nbi_core) | Médio | Porte explícito no commit 2 via `git checkout repo-publication-readiness -- <path>`; testes portados junto |
| 8 | Scope creep da dissertação (9,5k linhas, FA/Varimax, MBPA, benchmarks HPO) | Médio | Lista fechada de porte (~700 L); `doe_xgb/` intocado; **não copiar `post_optimization.py`** (bug conhecido nas linhas 166–170 da outra branch) |
| 9 | "Ferramenta demais, proposta de menos" | Médio | Relatório abre com EDA + comparação dos 5 classificadores; mistura entra como Metodologia; NBI é apêndice |
| 10 | NBI no simplex (única incerteza técnica) | Baixo | Stretch isolado com fallback garantido (SLSQP + Dirichlet + filtro não-dominado) |
| 11 | Cronograma (artigo subestimado) | Médio | Esqueleto do artigo a partir do commit 7; marco submetível no commit 6; cortes atingem só stretch |
| 12 | Conflito com a política de branches do repo | Baixo | Branch longa e autônoma; nunca PR para `main`; rebase/merge de `main` desnecessário (main é estática) |

## 11. Próximos commits sugeridos

1. `chore: scaffold PCO213 classification mixture ensemble project` ← **este (staged, aguardando confirmação)**
2. `feat(pco213): port simplex and Scheffé modules with tests`
3. `feat(pco213): data loading and preprocessing for UCI credit card default`
4. `feat(pco213): EDA notebook`
5. `feat(pco213): base model zoo and OOF probability matrices`
6. `feat(pco213): mixture design evaluation and mandatory baselines`
7. `feat(pco213): Scheffé metamodel per replica with diagnostics and ternary plots`
8. `feat(pco213): weight optimization and holdout confirmation`
9. `feat(pco213): statistical comparison and final figures`
10. `docs(pco213): article-format report and slides`
