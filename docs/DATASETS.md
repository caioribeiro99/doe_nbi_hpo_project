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
