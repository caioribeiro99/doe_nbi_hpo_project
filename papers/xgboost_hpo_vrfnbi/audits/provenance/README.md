# Provenance reproduction

The dissertation DoE, factor-analysis and response-surface stages, re-executed from the code frozen
at tag `v0.1.0-dissertation` (commit `67d9fe5`) on the MAGIC gamma telescope data and the
version-controlled 88-run design.

Reproduce:

```
uv venv --python 3.11 /tmp/venv-diss
VIRTUAL_ENV=/tmp/venv-diss uv pip install 'numpy<2' 'pandas<3' 'scikit-learn<2' \
    'xgboost>=2.0' 'scipy>=1.11' 'statsmodels>=0.14' tqdm
/tmp/venv-diss/bin/python ../../scripts/provenance_reproduce_dissertation.py --no-reuse-doe
```

## Why an era-appropriate interpreter

The dissertation pinned `pandas>=2.1`. Its factor stage writes into the array returned by
`np.asarray(df.loc[:, metrics])`. Under pandas 2.x that array is writable; under pandas 3
copy-on-write it is read-only and the frozen code raises `ValueError: assignment destination is
read-only`. The frozen source was not patched. The script refuses to run on pandas 3 and prints the
environment recipe instead.

## What was verified

| Item | Result |
|---|---|
| `magic04.data` SHA-256 against the repository manifest | matches byte for byte |
| MAGIC rows, features, positive-class prevalence | 19,020 / 10 / 0.3516 |
| Design runs and SHA-256 | 88, recorded in `provenance_run.json` |
| `Score_Quality` surface | R² 0.8590, adjusted 0.8248, 18 terms |
| `Score_Cost` surface | R² 0.9015, adjusted 0.8873, 12 terms |
| DoE stage wall clock | 4.6 minutes for 88 runs, 8 threads, Apple M4 Max |

## One result that does not reproduce exactly, and why

The same script was run twice, once under the current environment (numpy 2.4.6 / pandas 3.0.5) and
once under the era environment (numpy 1.26.4 / pandas 2.3.3):

| Response | Current environment | Era environment |
|---|---|---|
| `Score_Quality` R² | 0.8590107 | 0.8590104 |
| `Score_Cost` R² | 0.9026830 | 0.9015419 |

The quality surface agrees to six decimals; the cost surface does not. The cause is not numerical
drift in the fit: `Time_MeanFold` is wall-clock training time, so the cost response is measured
anew on every run and is not a deterministic function of the design.

**This is a finding, not a nuisance.** One of the two objectives the dissertation optimizes is not
reproducible from the seeds, because it is a timing measurement. Any replication of the cost-quality
trade-off inherits that, and Paper 2's cost objective has to address it. See
`../../protocol/budget_accounting.md`.
