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
