# DOE + RSM + NBI Pipeline (XGBoost)

This repository is a **reproducible pipeline** to run:
1. **DOE execution** (e.g., CCD face-centered) to evaluate XGBoost hyperparameters with CV
2. **Factor Analysis (ML + Varimax)** to obtain rotated factor scores (Quality vs Cost)
3. **RSM (quadratic) + backward elimination** (α = 0.05, hierarchical)
4. **NBI-like multiobjective optimization** (beta grid) to generate candidate hyperparameter sets
5. **Confirmation run** + benchmark optimizers (Grid / Random / Bayes / Hyperopt)

The code was modularized from the original Jupyter workflow (replicas differed only by seed).

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

> macOS note (Apple Silicon): XGBoost may require OpenMP:
```bash
brew install libomp
```

### 2) Put your inputs
- Dataset (xlsx/csv/parquet): `data/source/`
- DOE design CSV exported from Minitab: `data/design/`

### 3) Run one replica end-to-end (example)
```bash
python scripts/run_replica.py \
  --dataset data/source/telescope2.xlsx \
  --design  data/design/hyperparameter_design.csv \
  --replica 1
```

Outputs will be saved under:
`experiments/<dataset_name>/<design_name>/replica_01/`

## Key experimental decisions (frozen)

- Replica seed affects **both**:
  - CV split: `StratifiedKFold(random_state=seed)`
  - XGBoost: `random_state=seed`
- Time metric: **mean time per fold** (fit + predict), measured with `time.perf_counter()`
- Parallelism: XGBoost `n_jobs = -1` (applied to all methods)
- Integers: `max_depth = int(round(...))`, `n_estimators = int(round(...))`
- Factor Analysis: **Maximum Likelihood + Varimax**, 2 factors
- RSM: quadratic + **backward elimination** with α = 0.05, keeping hierarchy
- Benchmark fairness: same budget `BUDGET = 40` evaluations (default)

## Scripts

- `scripts/run_doe.py` – run DOE evaluation and save `doe_results.csv`
- `scripts/run_fa_rsm.py` – run FA + fit RSM models and save coefficients
- `scripts/run_nbi.py` – generate NBI candidates from RSM coefficients
- `scripts/run_confirmation.py` – evaluate NBI candidates + benchmark methods
- `scripts/run_replica.py` – run all steps in sequence for one replica

## Notes
- CSV outputs use `sep=';'` and `decimal=','` (Minitab/Excel friendly in pt-BR locale).
- All file names are in English by design; content may contain Portuguese field names for compatibility.
