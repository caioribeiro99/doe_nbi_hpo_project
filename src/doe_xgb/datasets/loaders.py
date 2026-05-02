"""Lazy dataset loaders.

Each ``load_*`` function returns a :class:`LoadedDataset` carrying X, y,
and a populated :class:`DatasetMetadata`. Loaders look for a cached
file under ``data/source/<id>/`` first; if absent they raise
:class:`DatasetUnavailableError` with the canonical URL so the caller
can decide whether to fetch it. They never auto-download.

The ``sklearn``-backed loader (Breast Cancer) is the exception: it
loads in-memory from the ``sklearn`` distribution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .metadata import DatasetMetadata, LoadedDataset
from .registry import REGISTRY


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset's local cache file is missing.

    The exception's ``.metadata`` attribute carries the registry entry
    so callers can show the canonical download URL.
    """

    def __init__(self, message: str, metadata: DatasetMetadata) -> None:
        super().__init__(message)
        self.metadata = metadata


def _data_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3] / "data" / "source"


def _ensure_pandas():
    import pandas as pd  # noqa: PLC0415

    return pd


def _attach(meta: DatasetMetadata, X, y) -> LoadedDataset:
    """Attach n_rows / n_features / class_distribution to ``meta``."""
    counts = y.value_counts().to_dict() if hasattr(y, "value_counts") else {}
    populated = replace(
        meta,
        n_rows=int(len(X)),
        n_features=int(X.shape[1]) if hasattr(X, "shape") else None,
        class_distribution={str(k): int(v) for k, v in counts.items()},
    )
    return LoadedDataset(X=X, y=y, metadata=populated)


def _missing(meta: DatasetMetadata, expected_path: Path) -> DatasetUnavailableError:
    return DatasetUnavailableError(
        f"Local cache for {meta.dataset_id!r} not found at {expected_path}. "
        f"Download from {meta.source_url or 'OpenML'} and place under data/source/{meta.dataset_id}/.",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------


def load_magic() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["magic"]
    candidates = [
        _data_root() / "telescope2.xlsx",
        _data_root() / "magic.csv",
        _data_root() / "magic" / "magic.csv",
        _data_root() / "magic" / "magic04.data",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, header=None if path.suffix == ".data" else 0)
                if path.suffix == ".data":
                    df.columns = list(meta.numeric_columns) + ["class"]
            target_col = meta.target_column or "class"
            if target_col not in df.columns:
                # The xlsx variant uses 'y'.
                if "y" in df.columns:
                    target_col = "y"
            y = df[target_col].copy()
            mapped = y.map({"g": 0, "h": 1})
            if mapped.notna().any():
                y = mapped
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y.astype(int))
    raise _missing(meta, candidates[0])


def load_breast_cancer() -> LoadedDataset:
    from sklearn.datasets import load_breast_cancer as _sk_load  # noqa: PLC0415

    meta = REGISTRY["breast_cancer"]
    bunch = _sk_load(as_frame=True)
    df = bunch.frame
    y = df[bunch.target.name].astype(int)
    X = df.drop(columns=[bunch.target.name])
    return _attach(replace(meta, numeric_columns=tuple(X.columns.tolist())), X, y)


def load_pima_diabetes() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["pima_diabetes"]
    candidates = [
        _data_root() / "pima_diabetes" / "diabetes.csv",
        _data_root() / "pima_diabetes" / "pima.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            target_col = meta.target_column or "class"
            if target_col not in df.columns:
                target_col = df.columns[-1]
            y = df[target_col].copy()
            if y.dtype == object:
                y = y.map({"tested_negative": 0, "tested_positive": 1, "negative": 0, "positive": 1, 0: 0, 1: 1})
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y.astype(int))
    raise _missing(meta, candidates[0])


def load_spambase() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["spambase"]
    candidates = [
        _data_root() / "spambase" / "spambase.data",
        _data_root() / "spambase" / "spambase.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, header=None) if path.suffix == ".data" else pd.read_csv(path)
            if path.suffix == ".data":
                df.columns = [f"feat_{i}" for i in range(df.shape[1] - 1)] + ["is_spam"]
            target_col = meta.target_column or "is_spam"
            y = df[target_col].astype(int)
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_adult() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["adult"]
    candidates = [
        _data_root() / "adult" / "adult.csv",
        _data_root() / "adult" / "adult.data",
    ]
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".data":
                df = pd.read_csv(path, header=None, names=columns,
                                 skipinitialspace=True, na_values="?")
            else:
                df = pd.read_csv(path, na_values="?")
            df = df.dropna()
            target_col = meta.target_column or "income"
            y = (df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
                 .map({">50K": 1, "<=50K": 0}).astype(int))
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_bank_marketing() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["bank_marketing"]
    candidates = [
        _data_root() / "bank" / "bank-additional-full.csv",
        _data_root() / "bank_marketing" / "bank-additional-full.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep=";")
            target_col = meta.target_column or "y"
            y = (df[target_col].astype(str).str.lower().map({"yes": 1, "no": 0}).astype(int))
            X = df.drop(columns=[target_col])
            X = X[~X.eq("unknown").any(axis=1)]
            y = y.loc[X.index]
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_credit_card_default() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["credit_card_default"]
    candidates = [
        _data_root() / "credit_card_default.csv",
        _data_root() / "credit_card_default" / "credit_card_default.csv",
        _data_root() / "credit_card_default" / "default of credit card clients.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            target_col = meta.target_column or "default_payment_next_month"
            if target_col not in df.columns:
                # Some mirrors use "default payment next month" (with spaces).
                spaced = "default payment next month"
                if spaced in df.columns:
                    df = df.rename(columns={spaced: target_col})
            y = df[target_col].astype(int)
            X = df.drop(columns=[target_col, "ID"], errors="ignore")
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_german_credit() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["german_credit"]
    candidates = [
        _data_root() / "german_credit" / "german.data",
        _data_root() / "german_credit" / "german.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".data":
                df = pd.read_csv(path, sep=r"\s+", header=None)
                df.columns = list(meta.categorical_columns + meta.numeric_columns) + ["risk"]
            else:
                df = pd.read_csv(path)
            target_col = meta.target_column or "risk"
            y = (df[target_col].astype(int).map({1: 0, 2: 1, 0: 0}).fillna(0).astype(int))
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_wine_quality() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["wine_quality"]
    red = _data_root() / "wine_quality" / "winequality-red.csv"
    white = _data_root() / "wine_quality" / "winequality-white.csv"
    if red.exists() or white.exists():
        frames = []
        if red.exists():
            frames.append(pd.read_csv(red, sep=";").assign(is_red=1))
        if white.exists():
            frames.append(pd.read_csv(white, sep=";").assign(is_red=0))
        df = pd.concat(frames, ignore_index=True)
        target_col = meta.target_column or "quality"
        y = (df[target_col] >= 6).astype(int)
        X = df.drop(columns=[target_col])
        return _attach(meta, X, y)
    raise _missing(meta, red)


def load_dry_bean() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["dry_bean"]
    candidates = [
        _data_root() / "dry_bean" / "Dry_Bean_Dataset.xlsx",
        _data_root() / "dry_bean" / "dry_bean.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            target_col = meta.target_column or "Class"
            y_raw = df[target_col].astype(str)
            classes = sorted(y_raw.unique().tolist())
            mapping = {c: i for i, c in enumerate(classes)}
            y = y_raw.map(mapping).astype(int)
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_mushroom() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["mushroom"]
    candidates = [
        _data_root() / "mushroom" / "agaricus-lepiota.data",
        _data_root() / "mushroom" / "mushroom.csv",
    ]
    columns = ["class"] + list(meta.categorical_columns)
    for path in candidates:
        if path.exists():
            if path.suffix == ".data":
                df = pd.read_csv(path, header=None, names=columns)
            else:
                df = pd.read_csv(path)
            target_col = meta.target_column or "class"
            y = df[target_col].map({"p": 1, "e": 0}).astype(int)
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


def load_phishing() -> LoadedDataset:
    pd = _ensure_pandas()
    meta = REGISTRY["phishing"]
    candidates = [
        _data_root() / "phishing" / "phishing.csv",
        _data_root() / "phishing" / "Training Dataset.arff",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".arff":
                rows = []
                cols: list[str] = []
                with path.open() as f:
                    in_data = False
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("%"):
                            continue
                        if line.lower().startswith("@attribute"):
                            cols.append(line.split()[1].strip("'\""))
                        elif line.lower().startswith("@data"):
                            in_data = True
                        elif in_data:
                            rows.append([int(x) for x in line.split(",")])
                df = pd.DataFrame(rows, columns=cols)
            else:
                df = pd.read_csv(path)
            target_col = meta.target_column or "Result"
            y = df[target_col].map({-1: 0, 0: 0, 1: 1}).astype(int)
            X = df.drop(columns=[target_col])
            return _attach(meta, X, y)
    raise _missing(meta, candidates[0])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_LOADERS: dict[str, Any] = {
    "magic": load_magic,
    "breast_cancer": load_breast_cancer,
    "pima_diabetes": load_pima_diabetes,
    "spambase": load_spambase,
    "adult": load_adult,
    "bank_marketing": load_bank_marketing,
    "credit_card_default": load_credit_card_default,
    "german_credit": load_german_credit,
    "wine_quality": load_wine_quality,
    "dry_bean": load_dry_bean,
    "mushroom": load_mushroom,
    "phishing": load_phishing,
}


def load(dataset_id: str) -> LoadedDataset:
    """Dispatch to the right loader."""
    if dataset_id not in _LOADERS:
        raise KeyError(
            f"unknown dataset_id {dataset_id!r}; choose from {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[dataset_id]()


__all__ = [
    "DatasetUnavailableError",
    "load",
    "load_magic",
    "load_breast_cancer",
    "load_pima_diabetes",
    "load_spambase",
    "load_adult",
    "load_bank_marketing",
    "load_credit_card_default",
    "load_german_credit",
    "load_wine_quality",
    "load_dry_bean",
    "load_mushroom",
    "load_phishing",
]
