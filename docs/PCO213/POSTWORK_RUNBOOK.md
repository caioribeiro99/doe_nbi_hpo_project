# POSTWORK RUNBOOK — como rodar o NBI sobre pesos de ensemble

**Data:** 2026-07-07 · **Branch:** `pco213-classification-mixture-ensemble`
**Plano de pesquisa:** [`POSTWORK_EXPERIMENT_PLAN.md`](POSTWORK_EXPERIMENT_PLAN.md)

## 1. Objetivo da extensão

Completar a assinatura metodológica **DoE + metamodelagem RSM + NBI multiobjetivo** sobre os pesos do ensemble: a partir dos metamodelos de Scheffé já ajustados na entrega, gerar uma **fronteira Pareto interpretável** entre discriminação (ROC-AUC), calibração (log-loss) e custo de inferência, e validá-la contra uma fronteira de referência por busca Dirichlet densa.

## 2. Diferença entre a entrega final e o pós-trabalho

- A **entrega da disciplina está congelada** (commits até `9b1d59b`, `dist/PCO213_final_entrega.zip`) e não é tocada pelos novos estágios;
- Os estágios pós-trabalho (`cost`, `nbi`, `pareto`) **apenas leem** os artefatos de `experiments/pco213/santander/` e **escrevem** em `experiments/pco213_postwork/santander/` (diretório separado, também fora do git);
- `--stage all` continua executando somente o pipeline original da entrega; o pós-trabalho roda com `--stage postwork_all` (ou estágio a estágio).

**Aviso metodológico permanente:** para objetivo único, o **SLSQP direto sobre a OOF é o oráculo** — log-loss é convexa nos pesos e o ótimo global sai em milissegundos. O NBI **não** compete com ele: serve para a **fronteira multiobjetivo** e para a interpretação (âncoras, payoff, trade-offs, espaçamento uniforme dos pontos).

## 3. Comandos

Todos a partir da raiz do repo, com o venv da entrega (`.venv-pco213`).

### Testes (rápido, sem dados)
```bash
.venv-pco213/bin/python -m pytest tests/mixens        # 55 testes, ~2 s
```

### Ver o plano sem executar nada (dry-run)
```bash
.venv-pco213/bin/python scripts/pco213_run_santander_study.py --stage postwork_all --dry-run
```

### Rodar só o NBI usando os artefatos existentes da entrega
```bash
.venv-pco213/bin/python scripts/pco213_run_santander_study.py --stage cost
.venv-pco213/bin/python scripts/pco213_run_santander_study.py --stage nbi
# opções: --nbi-points 28  --objectives auc,logloss  (>=2 dentre auc,logloss,cost)
```

### Plano mínimo completo (NBI + fronteira de referência + métricas de gap)
```bash
.venv-pco213/bin/python scripts/pco213_run_santander_study.py --stage postwork_all
# fronteira de referência maior (mais lenta): --pareto-dirichlet-points 20000
```

### Pré-requisito (só se os artefatos da entrega não existirem nesta máquina)
```bash
.venv-pco213/bin/python scripts/pco213_run_santander_study.py --stage all --mode full_optional
# ~16 min no M4 Max; baixa o dado automaticamente (mirror público) se ausente
```

## 4. Estimativas de runtime (M4 Max)

| Comando | Custo |
|---|---|
| `pytest tests/mixens` | ~2 s |
| `--stage cost` | <1 s (deriva dos timings salvos; **nunca** reajusta modelos) |
| `--stage nbi` (15 subproblemas, 3 objetivos) | segundos (opera sobre metamodelos + 1 avaliação OOF por candidato) |
| `--stage pareto` (default 5.000 pontos) | ~3–6 min (AUC real sobre a OOF agrupada de 320k linhas domina; ~1 min por 1.000 pontos) |
| `--stage postwork_all` | ~4–7 min no total |

Nada disso re-executa OOF/treino de modelos (o guard `--skip-heavy` é o default e os estágios não têm caminho de código para disparar fits).

## 5. Saídas esperadas (em `experiments/pco213_postwork/santander/`)

| Arquivo | Conteúdo | Como interpretar |
|---|---|---|
| `inference_costs.json` | custo por modelo em **ms por 1.000 predições**, derivado dos timings de predição da OOF (`method: oof_timings_fallback` — os modelos não são persistidos; para medição direta, reexecutar `--stage oof`) | espere kNN ≫ RF > XGB > LR ≈ GNB; o custo do blend é `Σ wᵢcᵢ` (linear nos pesos) |
| `nbi_candidates.csv` | 1 linha por subproblema NBI: `beta_*` (posição na CHIM), `w_<modelo>` (pesos no simplex), `t` (avanço na quase-normal), `residual_norm`, `success`, e as métricas **reais** OOF (`real_roc_auc`, `real_log_loss`, `real_cost_ms_1k`) | os `w` são os candidatos Pareto propostos; `success=True` e `residual_norm < 1e-3` indicam subproblema bem resolvido; as colunas `real_*` são a revalidação honesta (o metamodelo propõe, a OOF confirma) |
| `nbi_summary.json` | âncoras (em pesos e em métricas reais), matriz payoff, utopia/pseudo-nadir, direção quase-normal, contagens | as âncoras são os ótimos individuais de cada objetivo; payoff/utopia definem a normalização (o NBI é sensível a escala) |
| `pareto_reference.csv` | pontos Dirichlet densos + vértices + centroide avaliados nas métricas reais, com a coluna `non_dominated` | as linhas `non_dominated=True` formam a **fronteira de referência** ("Pareto verdadeiro" empírico) |
| `pareto_metrics.json` | **GD**, **IGD** e **spacing** entre a fronteira NBI e a referência (objetivos normalizados pela escala da referência) | GD baixo = fronteira NBI *próxima* da real; IGD baixo = boa *cobertura*; spacing menor no NBI que na referência = o argumento clássico do NBI (pontos uniformemente espaçados). Critério S2 do plano: GD/IGD pequenos e estáveis |

## 6. Depois do plano mínimo

Próximos incrementos (ver `POSTWORK_EXPERIMENT_PLAN.md` §8): aumentar réplicas (R=5) para medir estabilidade da fronteira e dos β_ij; figura ternária com a fronteira projetada; comparadores de orçamento pareado (random Dirichlet com 21 pontos); e replicação em segundo dataset (Give Me Some Credit ou UCI Credit Card Default). Nenhum desses passos está implementado ainda — este runbook cobre exatamente o que existe no código hoje.
