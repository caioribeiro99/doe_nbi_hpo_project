from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import pandas as pd


def _resolve_target_column(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    *,
    fallback_candidates: Iterable[str] = ("y", "target", "label", "class"),
) -> str:
    """Resolve the target column name.

    Priority:
      1) Explicit `target_col` (if present)
      2) Common names: y, target, label, class
      3) Last column in the dataframe
    """
    if target_col and target_col in df.columns:
        return str(target_col)
    for cand in fallback_candidates:
        if cand in df.columns:
            return str(cand)
    return str(df.columns[-1])


def load_dataset(
    path: str | Path,
    target_col: Optional[str] = "y",
    target_map: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from xlsx/csv/parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv"}:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";", decimal=",")
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    resolved_target = _resolve_target_column(df, target_col)

    y = df[resolved_target].copy()
    if target_map is not None:
        mapped = y.map(target_map)
        # If mapping fails (all NaN), keep original labels.
        if mapped.notna().any():
            y = mapped

    X = df.drop(columns=[resolved_target])
    return X, y


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV trying to be compatible with pt-BR exports (sep=';' decimal=',')."""
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if first_line.count(";") > first_line.count(","):
        return pd.read_csv(path, sep=";", decimal=",")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";", decimal=",")


def load_design(path: str | Path) -> pd.DataFrame:
    """Load DOE design exported from Minitab (or any CSV)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Design file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Design file must be .csv (exported from Minitab or similar).")

    df = _read_csv_flexible(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_csv_ptbr(df: pd.DataFrame, path: str | Path) -> None:
    """Save CSV in pt-BR friendly format (sep=';' decimal=',')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", decimal=",")
