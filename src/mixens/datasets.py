"""Dataset registry for the PCO213 post-work multi-dataset benchmark.

Four binary-classification datasets with heterogeneous structure:

- ``santander``: Santander Customer Transaction Prediction (Kaggle 2019);
  200 anonymous numeric features, no missing values, 10.05% positives.
- ``bnp``: BNP Paribas Cardif Claims Management (Kaggle 2016); 131 mixed
  features (19 categorical, incl. one with ~18k levels), ~34% missing,
  76.1% positives (target = claim suitable for accelerated approval).
- ``porto``: Porto Seguro Safe Driver Prediction (Kaggle 2017); 57 features
  (14 categorical ``*_cat``, 17 binary ``*_bin``, rest ordinal/continuous;
  missing coded as -1), 3.64% positives; capped to 200k rows by a fixed,
  seeded stratified subsample (documented in the manifest).
- ``uci_credit``: UCI Default of Credit Card Clients (Yeh & Lien 2009);
  23 features (3 small categoricals), 22.1% positives.

Raw files live under ``data/pco213/raw/<name>/`` and are never committed.
Kaggle competition test labels are never used. Each loader validates the
official row/column/prevalence invariants before returning.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROW_CAP_SEED = 20260904


@dataclass
class DatasetSpec:
    name: str
    source: str
    acquisition_command: str
    train_file: str
    sha256: str
    target: str
    n_rows_raw: int
    n_rows_used: int
    n_features: int
    prevalence: float
    missing_fraction: float
    numeric_cols: list[str]
    categorical_cols: list[str]
    binary_cols: list[str]
    preprocessing: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    row_cap: int | None = None
    row_cap_seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blob := f.read(chunk):
            h.update(blob)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BNP_CATEGORICAL = ["v3", "v22", "v24", "v30", "v31", "v47", "v52", "v56", "v66", "v71",
                   "v74", "v75", "v79", "v91", "v107", "v110", "v112", "v113", "v125"]

REGISTRY: dict[str, dict] = {
    "santander": {
        "dir": "santander",
        "train_file": "train.csv",
        "source": "Kaggle competition santander-customer-transaction-prediction "
                  "(public mirror lakshmi25npathi/santander-customer-transaction-prediction-dataset)",
        "acquisition_command": "python scripts/pco213_run_santander_study.py --stage data",
        "target": "target",
        "id_col": "ID_code",
        "expected_rows": 200_000,
        "expected_features": 200,
        "expected_prevalence": (0.1005, 0.003),
    },
    "bnp": {
        "dir": "bnp",
        "train_file": "train.csv",
        "source": "Kaggle competition bnp-paribas-cardif-claims-management; file obtained from the public "
                  "dataset mirror hjimbean/kaggle-classification-autofe-benchmark "
                  "(data/bnp-paribas-cardif-claims-management/train.csv); same 114,321 x 133 layout, string-coded "
                  "categoricals (v22 with 18,210 levels) and 76.12% positives as the official file",
        "acquisition_command": "kaggle datasets download hjimbean/kaggle-classification-autofe-benchmark "
                               "-f data/bnp-paribas-cardif-claims-management/train.csv",
        "target": "target",
        "id_col": "ID",
        "expected_rows": 114_321,
        "expected_features": 131,
        "expected_prevalence": (0.7612, 0.003),
    },
    "porto": {
        "dir": "porto",
        "train_file": "train.csv",
        "source": "Kaggle competition porto-seguro-safe-driver-prediction; file obtained from the public "
                  "dataset mirror pushero/porto-seguros-safe-driver-prediction-train-data (byte-identical "
                  "size to the official train.csv, 115,852,544 bytes)",
        "acquisition_command": "kaggle datasets download pushero/porto-seguros-safe-driver-prediction-train-data",
        "target": "target",
        "id_col": "id",
        "expected_rows": 595_212,
        "expected_features": 57,
        "expected_prevalence": (0.0364, 0.002),
        "row_cap": 200_000,
    },
    "uci_credit": {
        "dir": "uci_credit",
        "train_file": "default of credit card clients.xls",
        "source": "UCI Machine Learning Repository #350, Default of Credit Card Clients (Yeh & Lien, 2009)",
        "acquisition_command": "curl -L -o uci_credit.zip "
                               "'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'",
        "target": "default payment next month",
        "id_col": "ID",
        "expected_rows": 30_000,
        "expected_features": 23,
        "expected_prevalence": (0.2212, 0.003),
    },
}

DATASET_NAMES = list(REGISTRY)


def raw_path(raw_root: str | Path, name: str) -> Path:
    r = REGISTRY[name]
    return Path(raw_root) / r["dir"] / r["train_file"]


def _read_raw(name: str, path: Path) -> pd.DataFrame:
    if name == "uci_credit":
        return pd.read_excel(path, header=1)
    return pd.read_csv(path)


def _column_roles(name: str, df: pd.DataFrame, target: str, id_col: str) -> tuple[list[str], list[str], list[str]]:
    feats = [c for c in df.columns if c not in (target, id_col)]
    if name == "santander":
        return feats, [], []
    if name == "bnp":
        cats = [c for c in BNP_CATEGORICAL if c in feats]
        num = [c for c in feats if c not in cats]
        return num, cats, []
    if name == "porto":
        cats = [c for c in feats if c.endswith("_cat")]
        bins = [c for c in feats if c.endswith("_bin")]
        num = [c for c in feats if c not in cats and c not in bins]
        return num, cats, bins
    if name == "uci_credit":
        cats = [c for c in ("SEX", "EDUCATION", "MARRIAGE") if c in feats]
        num = [c for c in feats if c not in cats]
        return num, cats, []
    raise KeyError(name)


def load_dataset(name: str, raw_root: str | Path) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load, validate invariants, apply the documented preprocessing that is
    independent of any split (type casting, missing-value coding, fixed row
    cap), and return (X, y, spec). Fold-dependent preprocessing (imputation,
    scaling, encoding) is NOT applied here — see ``mixens.bench_models``."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {DATASET_NAMES}")
    r = REGISTRY[name]
    path = raw_path(raw_root, name)
    if not path.exists():
        raise FileNotFoundError(f"{name}: raw file missing at {path}; acquire with: {r['acquisition_command']}")
    sha = sha256_of(path)
    df = _read_raw(name, path)
    target, id_col = r["target"], r["id_col"]
    if target not in df.columns:
        raise ValueError(f"{name}: target column {target!r} not found")
    n_raw = len(df)
    if n_raw != r["expected_rows"]:
        raise ValueError(f"{name}: expected {r['expected_rows']} rows, got {n_raw}")
    num, cats, bins = _column_roles(name, df, target, id_col)
    n_feat = len(num) + len(cats) + len(bins)
    if n_feat != r["expected_features"]:
        raise ValueError(f"{name}: expected {r['expected_features']} features, got {n_feat}")
    prev = float(df[target].mean())
    lo, tol = r["expected_prevalence"]
    if abs(prev - lo) > tol:
        raise ValueError(f"{name}: prevalence {prev:.4f} outside {lo}±{tol}")

    notes: list[str] = []
    X = df[num + cats + bins].copy()
    y = df[target].astype(np.int8)

    if name == "porto":
        # -1 codes missing values in the official data: make it NaN for numeric
        # columns (imputed fold-wise) and keep it as its own level for categoricals.
        for c in num:
            X.loc[X[c] == -1, c] = np.nan
        notes.append("Porto: -1 → NaN for numeric columns (fold-wise median imputation + missing indicator); "
                     "-1 kept as a level for *_cat columns")
    if name == "bnp":
        notes.append("BNP: 19 categorical columns from the official competition list (strings), treated as "
                     "nominal with missing as its own level; v22 (18k levels) is frequency-capped by the encoder")
    if name == "uci_credit":
        notes.append("UCI: SEX/EDUCATION/MARRIAGE treated as nominal; PAY_0..PAY_6 kept as ordinal numeric")

    missing_fraction = float(X.isna().mean().mean())
    n_used = n_raw
    row_cap = r.get("row_cap")
    if row_cap and row_cap < n_raw:
        X, _, y, _ = train_test_split(X, y, train_size=row_cap, stratify=y, random_state=ROW_CAP_SEED)
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        n_used = len(X)
        notes.append(f"stratified row cap {row_cap} (seed {ROW_CAP_SEED}), fixed across replications")

    for c in cats:
        X[c] = X[c].astype("object")
    for c in num + bins:
        X[c] = X[c].astype(np.float32)

    spec = DatasetSpec(
        name=name, source=r["source"], acquisition_command=r["acquisition_command"],
        train_file=str(path), sha256=sha, target=target, n_rows_raw=n_raw, n_rows_used=n_used,
        n_features=n_feat, prevalence=float(y.mean()), missing_fraction=missing_fraction,
        numeric_cols=num, categorical_cols=cats, binary_cols=bins,
        preprocessing={
            "numeric": "fold-wise median imputation (+ missing indicator when NA present); "
                       "StandardScaler for lr/knn; raw for gnb/rf/xgb",
            "categorical": "fold-wise OneHotEncoder(min_frequency=0.005, infrequent grouped, unknown→infrequent) "
                           "for lr/gnb/knn; OrdinalEncoder(unknown→-1) for rf/xgb; missing as its own level",
            "binary": "passed through as 0/1",
        },
        notes=notes, row_cap=row_cap, row_cap_seed=ROW_CAP_SEED if row_cap else None,
    )
    return X, y, spec


def acquire(name: str, raw_root: str | Path) -> Path:
    """Ensure the raw file exists (Santander via the existing loader's cascade;
    others must already be present — their acquisition commands are recorded)."""
    path = raw_path(raw_root, name)
    if path.exists():
        return path
    if name == "santander":
        from mixens import data as data_mod
        data_mod.acquire(path.parent)
        return path
    raise FileNotFoundError(f"{name}: raw file missing at {path}; acquire with: "
                            f"{REGISTRY[name]['acquisition_command']}")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


__all__ = ["DATASET_NAMES", "DatasetSpec", "REGISTRY", "acquire", "git_commit", "load_dataset", "raw_path", "sha256_of"]
