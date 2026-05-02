# Article-track v1 datasets

The article-track first paper uses **twelve** public tabular datasets,
defined in the registry at `src/doe_xgb/datasets/registry.py`.

## CLI

```bash
# List the registry.
doe-xgb datasets list

# Probe canonical sources and (optionally) write a Markdown + JSON
# availability report.
doe-xgb datasets check-availability \
    --timeout 8 \
    --out-md  data/source/AVAILABILITY_CHECK.md \
    --out-json data/source/dataset_registry.json

# Inspect one entry's metadata.
doe-xgb datasets inspect --dataset-id breast_cancer

# Smoke-load one or all entries to verify shapes (skips entries whose
# local cache is missing).
doe-xgb datasets smoke --dataset-id breast_cancer
doe-xgb datasets smoke --all
```

## Registry summary

| ID | Name | Source | Task | Notes |
|---|---|---|---|---|
| `magic` | MAGIC Gamma Telescope | UCI | binary | dissertation continuity |
| `breast_cancer` | Breast Cancer Wisc.\ Diag. | sklearn | binary | shipped with sklearn |
| `pima_diabetes` | Pima Indians Diabetes | OpenML id 37 | binary | UCI original retracted |
| `spambase` | Spambase | UCI | binary | classical |
| `adult` | Adult / Census Income | UCI / OpenML 1590 | binary | imbalanced; categoricals |
| `bank_marketing` | Bank Marketing | UCI / OpenML 1461 | binary | imbalanced; categoricals |
| `credit_card_default` | Credit Card Default | UCI / OpenML 42477 | binary | XLS source |
| `german_credit` | German Credit | UCI / OpenML 31 | binary | small; categorical |
| `wine_quality` | Wine Quality (binarised) | UCI / OpenML 287 | binary | red+white merged; quality >= 6 |
| `dry_bean` | Dry Bean | UCI / OpenML 43466 | multiclass (7) | multiclass test |
| `mushroom` | Mushroom | UCI / OpenML 24 | binary | categorical-only |
| `phishing` | Phishing Websites | UCI 327 / OpenML 4534 | binary | engineered features |

Each entry carries: source URL, OpenML id, target column, target
transformation, declared categorical / numeric column lists, missing
value policy, recommended metrics, calibration enable flag, expected
computational burden, license note, citation key, v1 inclusion flag,
fallback id, and free-form notes. See
`src/doe_xgb/datasets/metadata.py` for the dataclass.

## Downloaders

Each non-sklearn entry has a downloader script under `scripts/`:

| ID | Script | Source kind |
|---|---|---|
| `magic` | `scripts/fetch_magic_dataset.py` | UCI .data |
| `pima_diabetes` | `scripts/fetch_pima_diabetes_dataset.py` | OpenML id 37 |
| `spambase` | `scripts/fetch_spambase_dataset.py` | UCI .data |
| `adult` | `scripts/fetch_adult_dataset.py` | UCI .data |
| `bank_marketing` | `scripts/fetch_bank_marketing_dataset.py` | UCI ZIP -> CSV |
| `credit_card_default` | `scripts/fetch_credit_card_default_dataset.py` | UCI XLS (OpenML 42477 fallback) |
| `german_credit` | `scripts/fetch_german_credit_dataset.py` | UCI .data |
| `wine_quality` | `scripts/fetch_wine_quality_dataset.py` | UCI .csv (red+white merged, binarised) |
| `dry_bean` | `scripts/fetch_dry_bean_dataset.py` | UCI ZIP -> XLSX -> CSV |
| `mushroom` | `scripts/fetch_mushroom_dataset.py` | UCI .data |
| `phishing` | `scripts/fetch_phishing_dataset.py` | UCI ARFF -> CSV |

`breast_cancer` does **not** have a downloader because it ships with
`scikit-learn` (`sklearn.datasets.load_breast_cancer`).

### One-shot

```bash
make data            # = doe-xgb datasets fetch --all
make data-checksums  # = doe-xgb datasets verify-checksums
```

### Single dataset

```bash
python scripts/fetch_magic_dataset.py            # respects manifest
python scripts/fetch_magic_dataset.py --force    # re-download
python scripts/fetch_magic_dataset.py --no-network  # process raw only
```

### Idempotence and checksums

Each script writes:

```
data/source/<id>/raw/<original-file>
data/source/<id>/processed/<id>.csv
data/source/<id>/manifest.json
```

The manifest lists every file with SHA-256 + byte size. The same hashes
are mirrored into `data/source/CHECKSUMS.txt`, segmented by dataset id
(`# >>> <id>` ... `# <<< <id>` markers). Re-running a downloader whose
manifest matches the on-disk files is a no-op; pass `--force` to
re-download.

`doe-xgb datasets verify-checksums [--dataset-id <id>]` re-hashes every
file referenced in the manifests and reports True/False per dataset.

## Three-algorithm binary smoke (Commit 21)

`scripts/run_v1_binary_3alg_smoke.py` extends the Commit 20 smoke
across the three article-track GBDT families:

```bash
python scripts/run_v1_binary_3alg_smoke.py --max-rows 1000
```

Per-(dataset, algorithm) it loads the cached dataset, applies the
algorithm-specific preprocessing, fits one safe hyperparameter point
under 2-fold stratified CV, and writes
`experiments/_v1_smoke/binary_3alg_smoke_output.json`. The script
fails fast if any pair returns missing keys, accuracy below 0.50, or
non-finite runtime.

### Algorithm-specific notes

- **XGBoost** (`tree_method="hist"`, `n_jobs=1`,
  `eval_metric="logloss"`): non-numeric columns are converted to
  deterministic integer category codes via `pd.Categorical(...).codes`.
- **LightGBM** (`verbose=-1`, `n_jobs=1`): same encoding as XGBoost.
  A residual sklearn `UserWarning` about feature names is suppressed
  inside the smoke.
- **CatBoost** (`thread_count=1`, `verbose=False`,
  `allow_writing_files=False`, `bootstrap_type="Bernoulli"`): tries
  native categorical handling first by passing the DataFrame as-is
  with `cat_features=<indices of non-numeric columns>` and casting
  string columns to `str`. Falls back to the same encoded-int-codes
  representation if the native path raises; the chosen path is
  recorded under `preprocessing_mode` in the output JSON
  (`catboost_native_categorical` or
  `catboost_fallback_encoded_int_codes`).

The smoke does **not** run DOE / RSM / NBI / MBPA. Total runtime is
~1.7 s on Apple Silicon.

## Tiny binary smoke (Commit 20)

`scripts/run_v1_binary_smoke.py` exercises the load -> evaluate path
end-to-end on the three small fetched datasets (`german_credit`,
`pima_diabetes`, `spambase`), runs `evaluate_xgb_cv` once each at a
single safe hyperparameter point with 2-fold CV, and asserts that
the dissertation-era binary keys are populated. Total runtime is
~1.3 seconds on Apple Silicon at `--max-rows 1000`.

```bash
python scripts/fetch_german_credit_dataset.py
python scripts/fetch_pima_diabetes_dataset.py
python scripts/fetch_spambase_dataset.py
python -m doe_xgb.cli datasets verify-checksums --dataset-id german_credit
python -m doe_xgb.cli datasets verify-checksums --dataset-id pima_diabetes
python -m doe_xgb.cli datasets verify-checksums --dataset-id spambase
python scripts/run_v1_binary_smoke.py --max-rows 1000
```

The smoke script writes `experiments/_v1_smoke/binary_smoke_output.json`
and fails fast if any dataset returns multiclass metrics or any of
`Accuracy_Mean / Precision_Mean / Recall_Mean / Specificity_Mean` is
missing. It does **not** run DOE / RSM / NBI / MBPA, and it does
**not** load Dry Bean.

## Loaders

Each `load_*` function under `src/doe_xgb/datasets/loaders.py` returns
a `LoadedDataset(X: pd.DataFrame, y: pd.Series, metadata:
DatasetMetadata)`. `metadata.n_rows`, `metadata.n_features`, and
`metadata.class_distribution` are populated at load time; static
fields (target column, categorical / numeric lists, license note) are
not modified.

Behavior:

- The `breast_cancer` loader uses `sklearn.datasets.load_breast_cancer`
  and works offline.
- Every other loader looks for a cached file under
  `data/source/<id>/`. If absent, it raises
  `DatasetUnavailableError(metadata=...)` with the canonical URL so
  the caller can fetch it.
- Loaders never auto-download. They never split train/test. They
  separate the target column cleanly. They do not one-hot encode --
  that is a model-specific concern.

## Wine Quality, Dry Bean, and other formulation choices

- **Wine Quality**: red + white merged; the headline target is
  `quality >= 6` (binary). A multiclass / ordinal variant is deferred
  to a follow-up paper because it would require re-validating
  multiclass NBI scoring across the full panel.
- **Dry Bean**: kept as multiclass (7 classes). The article reports
  `roc_auc_ovr`, `f1_macro`, and balanced accuracy on this dataset.
- **Phishing**: UCI 327 (`Training Dataset.arff`) is the canonical
  source; values in {-1, 0, 1} are mapped to integers untouched.
- **Adult / Mushroom**: native categorical handling for CatBoost; the
  XGBoost / LightGBM pipelines apply one-hot or ordinal encoding
  inside the model-specific preprocessing stage, **not** inside the
  loader.

## Substitutions

Allowed only after a documented availability or licence failure:

- **Higgs (small)** could replace MAGIC.
- **Telco Customer Churn** could replace Bank Marketing.
- **Covertype (binarised)** could replace Spambase.

The fallback id is the `fallback_dataset_id` field of the registry
entry.
