# DOE + RSM + NBI Pipeline (XGBoost)

This repository provides a **reproducible, end-to-end pipeline** to run:

1. **DOE execution** (e.g., CCD face-centered exported from Minitab) to evaluate XGBoost hyperparameters with CV  
2. **Factor Analysis** (z-score + PCA/Factor extraction + Varimax rotation) to obtain **orthogonal factor scores**
3. **RSM (quadratic) + backward elimination** (α = 0.05, hierarchical)
4. **NBI-like multiobjective optimization** (beta weight grid) to generate candidate hyperparameter sets
5. **Confirmation run** + benchmark optimizers (**Grid / Random / Bayesian / Hyperopt (TPE)**)

The code was modularized from the original Jupyter workflow (replicas differed only by seed).

---

## Quickstart

### Option A — Recommended: run the setup script

**macOS / Linux (bash or zsh):**

```bash
./setup.sh
```

**Windows PowerShell:**

```powershell
./setup.ps1
```

### Option B — Manual setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux (bash or zsh)
# .venv\Scripts\activate  # Windows (PowerShell)

pip install -r requirements.txt
```

> macOS note (Apple Silicon): XGBoost may require OpenMP:

```bash
brew install libomp
```

---

## Inputs

Put your files here:

- Dataset (xlsx/csv/parquet): `data/source/`
- DOE design CSV exported from Minitab: `data/design/`

Dataset requirement:

- A target column named `y` by default (override with `--target` or `TARGET_COL` in `.env`).
- For the MAGIC Gamma Telescope example, the code maps `{g,h}` to `{0,1}`.

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

### Run the full experiment (N replicas)

```bash
python scripts/run_experiment.py
```

By default it reads `.env` (optional) and runs `N_REPLICAS` replicas.

Outputs will be saved under:
`experiments/<dataset_name>/<design_name>/replica_XX/`

---

## Budget and fairness

**Key point:** every "evaluation" is one hyperparameter configuration evaluated with **5-fold CV**.

- DOE stage already consumes `DOE_RUNS` evaluations (e.g., 88 for the CCD you generated).
- Confirmation stage evaluates at most `NBI_EVAL_K` NBI candidates.

So the total evaluations used by DOE+NBI is approximately:

```bash
TOTAL_EVALS_DOE_NBI = DOE_RUNS + NBI_EVAL_K
```

Benchmarks must be compared with the same evaluation budget to be fair.

### How fairness is implemented (default)

If `BUDGET=0` (or `--budget 0`), the pipeline **auto-matches** benchmark budgets to:
`DOE_RUNS + NBI_EVAL_K`.

You can force a specific budget with `--budget <int>` (or `.env:BUDGET`).

---

## Key experimental decisions (frozen)

- Replica seed affects **both**:
  - CV split: `StratifiedKFold(shuffle=True, random_state=seed)`
  - XGBoost: `random_state=seed`
- Time metric: **mean time per fold** (fit + predict), measured with `time.perf_counter()`
- Parallelism: XGBoost `n_jobs = -1` (applied to all methods)
- Integers: `max_depth = int(round(...))`, `n_estimators = int(round(...))`
- FA: z-score standardization + PCA/Factor extraction + **Varimax rotation**
- RSM: quadratic + **backward elimination** with α = 0.05, keeping hierarchy
- Benchmarks: Grid / Random / Bayesian / Hyperopt (TPE) within the same bounds as the DOE

---

## Environment variables

Copy `.env.example` to `.env` and edit if you want defaults:

- `N_REPLICAS` (default: 30)
- `SEED_BASE` (default: 42)
- `NBI_BETA_STEP` (default: 0.02)
- `NBI_EVAL_K` (default: 20)
- `BUDGET` (default: 0 = auto-match)

---

## Notes

- CSV outputs use `sep=';'` and `decimal=','` (Minitab/Excel friendly in pt-BR locale).
- File names are in English by design; content may include Portuguese field names for compatibility.
