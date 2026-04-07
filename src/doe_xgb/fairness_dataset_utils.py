from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .io_utils import _read_csv_flexible


_SUPPORTED_PROTECTED_MODES = {
    "age_ge_25",
    "age_ge_30",
    "sex_male_is_1",
    "sex_female_is_1",
    "binary_one_is_privileged",
    "binary_zero_is_privileged",
    "median_ge_is_privileged",
}


def _drop_unknown_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows where any object/string column equals 'unknown' or '?' (case-insensitive)."""
    obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not obj_cols:
        return df, 0

    unknown_mask = pd.Series(False, index=df.index)
    for c in obj_cols:
        s = df[c].astype(str).str.strip().str.lower()
        unknown_mask = unknown_mask | s.eq("unknown") | s.eq("?")

    removed = int(unknown_mask.sum())
    if removed > 0:
        df = df.loc[~unknown_mask].copy()
    return df, removed


def _drop_known_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known duplicate-value columns seen in public fairness datasets.

    Current use case:
      - german_credit.csv often contains both `x13` and `age` with identical values.
    """
    out = df.copy()
    if "x13" in out.columns and "age" in out.columns:
        try:
            a = pd.to_numeric(out["x13"], errors="coerce")
            b = pd.to_numeric(out["age"], errors="coerce")
            if a.equals(b):
                out = out.drop(columns=["x13"])
        except Exception:
            pass
    return out


def _coerce_target_binary(series: pd.Series, target_positive: Optional[str] = None) -> pd.Series:
    """Convert a target column to {0,1}."""
    s = series.copy()

    if target_positive is not None:
        tp_raw = str(target_positive).strip()
        s_num = pd.to_numeric(s, errors="coerce")
        try:
            tp_num = float(tp_raw)
            mask = s_num == tp_num
            if mask.notna().any():
                return mask.fillna(False).astype(int)
        except Exception:
            pass
        return s.astype(str).str.strip().eq(tp_raw).astype(int)

    s_str = s.astype(str).str.strip().str.lower()
    if set(s_str.dropna().unique()).issubset({"0", "1", "false", "true", "no", "yes", "n", "y", "good", "bad", ">50k", "<=50k", ">50k.", "<=50k."}):
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "n": 0,
            "y": 1,
            "bad": 0,
            "good": 1,
            "<=50k": 0,
            ">50k": 1,
            "<=50k.": 0,
            ">50k.": 1,
        }
        return s_str.map(mapping).astype(int)

    s_num = pd.to_numeric(s, errors="coerce")
    uniq_num = set(s_num.dropna().astype(int).unique().tolist())
    if uniq_num.issubset({0, 1}):
        return s_num.fillna(0).astype(int)

    raise ValueError("Could not infer a binary target automatically. Pass --target-positive explicitly.")


def _protected_from_mode(series: pd.Series, mode: str) -> pd.Series:
    mode = str(mode)
    if mode not in _SUPPORTED_PROTECTED_MODES:
        raise ValueError(f"Unsupported protected attribute mode: {mode}")

    s_num = pd.to_numeric(series, errors="coerce")
    s_str = series.astype(str).str.strip().str.lower()

    if mode == "age_ge_25":
        return (s_num >= 25).fillna(False).astype(int)
    if mode == "age_ge_30":
        return (s_num >= 30).fillna(False).astype(int)
    if mode == "sex_male_is_1":
        if s_num.notna().any():
            return (s_num == 1).fillna(False).astype(int)
        return s_str.eq("male").astype(int)
    if mode == "sex_female_is_1":
        if s_num.notna().any():
            return (s_num == 2).fillna(False).astype(int)
        return s_str.eq("female").astype(int)
    if mode == "binary_one_is_privileged":
        if s_num.notna().any():
            return (s_num == 1).fillna(False).astype(int)
        return s_str.eq("1").astype(int)
    if mode == "binary_zero_is_privileged":
        if s_num.notna().any():
            return (s_num == 0).fillna(False).astype(int)
        return s_str.eq("0").astype(int)
    if mode == "median_ge_is_privileged":
        med = float(s_num.median())
        return (s_num >= med).fillna(False).astype(int)

    raise ValueError(f"Unhandled protected attribute mode: {mode}")


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, drop_first=False)


def load_bank_dataset(path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load UCI Bank Marketing using the fairness preprocessing used in the repo."""
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' in bank dataset.")
    if "age" not in df.columns:
        raise ValueError("Expected column 'age' in bank dataset.")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    df, _ = _drop_unknown_rows(df)

    y = (df["y"].astype(str).str.lower() == "yes").astype(int)
    protected = (pd.to_numeric(df["age"], errors="coerce") >= 25).fillna(False).astype(int)

    X = df.drop(columns=["y"]).copy()
    X = _prepare_features(X)
    return X, y, protected


def load_credit_card_default_dataset(
    path: str | Path,
    *,
    protected_attr_mode: str = "sex_male_is_1",
    target_positive: str = "1",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the UCI Credit Card Default dataset already stored in the repo."""
    path = Path(path)
    df = _read_csv_flexible(path)
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' in credit card default dataset.")
    if "SEX" not in df.columns:
        raise ValueError("Expected column 'SEX' in credit card default dataset.")

    y = _coerce_target_binary(df["y"], target_positive=str(target_positive))
    protected = _protected_from_mode(df["SEX"], protected_attr_mode)

    X = df.drop(columns=["y"]).copy()
    X = _prepare_features(X)
    return X, y, protected


def load_generic_fairness_dataset(
    path: str | Path,
    *,
    target_col: str = "y",
    target_positive: Optional[str] = None,
    protected_col: Optional[str] = None,
    protected_attr_mode: str = "binary_one_is_privileged",
    drop_unknown_rows: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generic CSV/XLSX/parquet loader for fairness experiments."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = _read_csv_flexible(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format for fairness experiment: {path.suffix}")

    df = _drop_known_duplicate_columns(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if protected_col is None:
        raise ValueError("You must pass --protected-col when --dataset-kind=generic")
    if protected_col not in df.columns:
        raise ValueError(f"Protected attribute column not found: {protected_col}")

    if drop_unknown_rows:
        df, _ = _drop_unknown_rows(df)

    y = _coerce_target_binary(df[target_col], target_positive=target_positive)
    protected = _protected_from_mode(df[protected_col], protected_attr_mode)

    X = df.drop(columns=[target_col]).copy()
    X = _prepare_features(X)
    return X, y, protected
