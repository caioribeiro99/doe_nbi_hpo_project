# DOE + FA/PCA + RSM + NBI for Multi-Objective HPO (XGBoost)

This repository provides a **reproducible end-to-end pipeline** for **multi-objective hyperparameter tuning** of XGBoost (binary classification) using:

1. **DOE execution** (design matrix CSV) to evaluate hyperparameter configurations with stratified CV  
2. **Factor Analysis via PCA + Varimax rotation** to build two orthogonal objective scores:
   - **Score_Quality** (maximize)
   - **Score_Cost** (minimize; derived from `Time_MeanFold`)
3. **RSM (quadratic) + backward elimination** to fit surrogate response surfaces
4. **NBI-like candidate generation** using a **β grid** (e.g., step = 0.02)
5. **Confirmation run** + benchmark optimizers (**coarse grid / random / bayes / hyperopt (TPE)**) under a **fairness-by-evaluations** budget

Outputs are written under:

`experiments/<dataset_stem>/<design_stem>/replica_XX/`

---

## Quickstart

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> macOS note (Apple Silicon): XGBoost may require OpenMP:

```bash
brew install libomp
```

### 3) Optional: configure defaults via `.env`

Copy `.env.example` to `.env` and edit paths/parameters as needed.

---

## Inputs

- Dataset: put under `data/source/` (xlsx/csv/parquet)
- DOE design CSV: put under `data/design/`

Dataset requirement:

- A target column named `y` by default (override with `--target` or `TARGET_COL`).
- For the MAGIC Gamma Telescope example, the pipeline maps `{g,h}` → `{0,1}` inside `scripts/run_replica.py`.

---

## Run

### Run a single replica (end-to-end)

```bash
python scripts/run_replica.py \
  --dataset data/source/telescope2.xlsx \
  --design  data/design/hyperparameter_design.csv \
  --replica 1 \
  --seed 42
```

Key useful flags:

- `--n-splits` (default 5)
- `--n-jobs` (default -1)
- `--beta-step` (default 0.02)
- `--nbi-eval-k` (0 = evaluate all unique NBI candidates; otherwise evaluate top-k)
- `--nbi-n-starts` (multi-starts for the NBI optimizer)
- `--budget` (benchmark budget; `0` means auto-match fairness)

---

### Run 30 replicas (automatic experiment runner)

**Recommended (explicit args):**

```bash
python scripts/run_experiment.py \
  --dataset data/source/telescope2.xlsx \
  --design  data/design/hyperparameter_design.csv \
  --n-replicas 30 \
  --seed-base 42
```

This will:

- generate 30 seeds from `seed-base`
- run `scripts/run_replica.py` for replicas `01..30`
- write a global manifest + a run list JSON under:

`experiments/<dataset_stem>/<design_stem>/experiment_manifest.json`  
`experiments/<dataset_stem>/<design_stem>/replica_runs.json`

**If you already set `DATASET_PATH` and `DESIGN_PATH` in `.env`, you can run:**

```bash
python scripts/run_experiment.py --n-replicas 30 --seed-base 42
```

---

## Budget and fairness

**Key point:** each “evaluation” is one hyperparameter configuration evaluated with **k-fold stratified CV**.

- DOE stage consumes `DOE_RUNS` evaluations (e.g., 88)
- NBI confirmation consumes `NBI_EVAL_K` evaluations (e.g., 50)

So the fairness-matched budget is:

`TOTAL_EVALS_DOE_NBI = DOE_RUNS + NBI_EVAL_K`

Benchmarks are run with the **same evaluation budget**.

### How fairness is implemented

If `BENCHMARK_BUDGET <= 0` (or `--budget 0`), the experiment runner passes `--budget 0` to each replica, and the replica auto-matches budgets to `DOE_RUNS + NBI_EVAL_K`.

If you want to force a budget:

```bash
python scripts/run_experiment.py --budget 200
```

---

## Key experimental decisions (frozen)

- Replica seed affects both:
  - CV split: `StratifiedKFold(shuffle=True, random_state=seed)`
  - XGBoost: `random_state=seed`
- Time metric: **mean time per fold** (fit + predict), measured with `time.perf_counter()`
- Parallelism: XGBoost `n_jobs = -1` applied to all methods (fairness)
- Integers: `max_depth`, `n_estimators` are rounded to int
- FA: z-score standardization + PCA + **Varimax**
- RSM: quadratic + backward elimination (α = 0.05)
- Benchmarks: coarse grid / random / bayes / hyperopt (TPE)

---

## Environment variables (optional)

`run_experiment.py` reads `.env` and supports these variables as defaults:

- `DATASET_PATH` (dataset path)
- `DESIGN_PATH` (DOE design CSV path)
- `OUT_ROOT` (default: `experiments`)
- `TARGET_COL` (default: `y`)
- `N_REPLICAS` (default: `30`)
- `SEED_BASE` (default: `42`)
- `BENCHMARK_BUDGET` (default: `0` → auto-fairness)
- `NBI_BETA_STEP` (default: `0.02`)
- `NBI_EVAL_K` (default: `0` in CLI; if set > 0 evaluates top-k)
- `NBI_N_STARTS` (default: `10`)

---

## Notes

- CSV outputs use `sep=';'` and `decimal=','` (Excel/Minitab friendly in pt-BR locale).
- Outputs are not meant to be committed (use `.gitignore` to keep `experiments/` out).
