"""Santander Customer Transaction data acquisition, validation and splits.

Acquisition cascade (pre-registered in docs/PCO213):
1. local ``train.csv`` already present under ``data/pco213/raw/santander/``;
2. official Kaggle CLI (``kaggle competitions download``) if installed and
   authenticated — requires accepting the competition rules on the website;
3. anonymous download of the public Kaggle *dataset* mirror
   ``lakshmi25npathi/santander-customer-transaction-prediction-dataset``
   (a community redistribution of the competition files); the file is then
   validated against the official invariants below before any use;
4. otherwise: raise with clear manual-download instructions.

Raw data is NEVER committed (see .gitignore). Official source citation:
Kaggle, "Santander Customer Transaction Prediction", 2019,
https://www.kaggle.com/competitions/santander-customer-transaction-prediction
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COMPETITION_SLUG = "santander-customer-transaction-prediction"
MIRROR_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "lakshmi25npathi/santander-customer-transaction-prediction-dataset"
)

# Official invariants (verified against competition documentation).
EXPECTED_ROWS = 200_000
EXPECTED_FEATURES = 200
EXPECTED_POS_RATE = 0.10049
POS_RATE_TOL = 0.002
TARGET_COL = "target"
ID_COL = "ID_code"

SAMPLE_CAPS = {"fast": 60_000, "final_2h": 100_000, "full_optional": None}

MANUAL_INSTRUCTIONS = f"""
Santander train.csv not found and automatic acquisition failed.

Manual download (either option):
  a) Official (requires Kaggle account + accepting the competition rules):
     https://www.kaggle.com/competitions/{COMPETITION_SLUG}/data
  b) Public mirror dataset:
     https://www.kaggle.com/datasets/lakshmi25npathi/santander-customer-transaction-prediction-dataset

Then place the file at:
  data/pco213/raw/santander/train.csv
and re-run. The loader validates it against the official invariants
({EXPECTED_ROWS} rows, {EXPECTED_FEATURES} var_* features, ~10.05% positives).
"""


@dataclass(frozen=True)
class DataMeta:
    source: str
    sha256: str
    n_rows_raw: int
    n_duplicates_dropped: int
    n_rows: int
    n_features: int
    pos_rate: float
    mode: str
    sample_size: int | None
    random_state: int


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blob := f.read(chunk):
            h.update(blob)
    return h.hexdigest()


def _kaggle_cli_available() -> bool:
    exe = shutil.which("kaggle")
    if exe is None:
        return False
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _acquire_via_cli(raw_dir: Path) -> bool:
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG,
             "-f", "train.csv", "-p", str(raw_dir)],
            check=True, capture_output=True, timeout=1800,
        )
    except Exception:
        return False
    zips = list(raw_dir.glob("train.csv.zip")) + list(raw_dir.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith("train.csv"):
                    zf.extract(name, raw_dir)
                    extracted = raw_dir / name
                    if extracted != raw_dir / "train.csv":
                        extracted.rename(raw_dir / "train.csv")
    return (raw_dir / "train.csv").exists()


def _acquire_via_mirror(raw_dir: Path) -> bool:
    zip_path = raw_dir / "mirror.zip"
    if not zip_path.exists():
        try:
            urllib.request.urlretrieve(MIRROR_URL, zip_path)  # noqa: S310
        except Exception:
            return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = next((n for n in zf.namelist() if n.endswith("train.csv")), None)
            if member is None:
                return False
            with zf.open(member) as src, open(raw_dir / "train.csv", "wb") as dst:
                shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        return False
    return (raw_dir / "train.csv").exists()


def acquire(raw_dir: str | Path) -> tuple[Path, str]:
    """Return (path to train.csv, source label); raise with instructions on failure."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    train = raw_dir / "train.csv"
    if train.exists():
        return train, "local"
    if _kaggle_cli_available() and _acquire_via_cli(raw_dir):
        return train, "kaggle_cli_official"
    if _acquire_via_mirror(raw_dir):
        return train, "kaggle_public_mirror"
    raise FileNotFoundError(MANUAL_INSTRUCTIONS)


def load_santander(
    raw_dir: str | Path,
    *,
    mode: str = "fast",
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, DataMeta]:
    """Load, validate invariants, deduplicate and (optionally) subsample.

    ``sample_size`` overrides the mode cap when given. Sampling is
    stratified on the target and pre-registered in the report.
    """
    if mode not in SAMPLE_CAPS:
        raise ValueError(f"mode must be one of {sorted(SAMPLE_CAPS)}; got {mode!r}")
    train_path, source = acquire(raw_dir)
    sha = _sha256(train_path)

    df = pd.read_csv(train_path)
    n_raw = len(df)

    # --- invariant validation (fail fast on a wrong/corrupted mirror) ---
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"expected columns {ID_COL!r} and {TARGET_COL!r}; got {list(df.columns)[:5]}…")
    feat_cols = [c for c in df.columns if c.startswith("var_")]
    if len(feat_cols) != EXPECTED_FEATURES:
        raise ValueError(f"expected {EXPECTED_FEATURES} var_* features; got {len(feat_cols)}")
    if n_raw != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows; got {n_raw}")
    pos_rate_raw = float(df[TARGET_COL].mean())
    if abs(pos_rate_raw - EXPECTED_POS_RATE) > POS_RATE_TOL:
        raise ValueError(f"positive rate {pos_rate_raw:.5f} outside {EXPECTED_POS_RATE}±{POS_RATE_TOL}")
    if df[feat_cols].isna().any().any():
        raise ValueError("unexpected missing values in var_* features")

    # --- dedup (on features + target, ignoring ID) ---
    before = len(df)
    df = df.drop_duplicates(subset=feat_cols + [TARGET_COL]).reset_index(drop=True)
    n_dupes = before - len(df)

    y = df[TARGET_COL].astype(np.int8)
    X = df[feat_cols].astype(np.float32)

    # --- stratified subsample per execution mode ---
    cap = sample_size if sample_size is not None else SAMPLE_CAPS[mode]
    if cap is not None and cap < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=cap, stratify=y, random_state=random_state
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    meta = DataMeta(
        source=source,
        sha256=sha,
        n_rows_raw=n_raw,
        n_duplicates_dropped=n_dupes,
        n_rows=len(X),
        n_features=len(feat_cols),
        pos_rate=float(y.mean()),
        mode=mode,
        sample_size=cap,
        random_state=random_state,
    )
    return X, y, meta


def external_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified external holdout, fixed before anything else touches the data."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return (
        X_tr.reset_index(drop=True),
        X_te.reset_index(drop=True),
        y_tr.reset_index(drop=True),
        y_te.reset_index(drop=True),
    )


def meta_dict(meta: DataMeta) -> dict:
    return asdict(meta)


__all__ = [
    "COMPETITION_SLUG",
    "DataMeta",
    "acquire",
    "load_santander",
    "external_split",
    "meta_dict",
]
