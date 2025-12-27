from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def load_dataset(path: str | Path, target_col: str = "y", target_map: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from xlsx/csv/parquet.

    Parameters
    ----------
    path:
        File path.
    target_col:
        Name of the target column.
    target_map:
        Optional mapping applied to target values, e.g. {'g': 0, 'h': 1}.

    Returns
    -------
    X, y
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv"}:
        # try common CSV formats
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";", decimal=",")
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    y = df[target_col].copy()
    if target_map is not None:
        y = y.map(target_map)
    X = df.drop(columns=[target_col])
    return X, y


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV trying to be compatible with pt-BR exports (sep=';' decimal=',')."""
    # Heuristic: if header contains ';' assume semicolon
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if first_line.count(";") > first_line.count(","):
        return pd.read_csv(path, sep=";", decimal=",")
    # else try comma
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";", decimal=",")


def load_design(path: str | Path) -> pd.DataFrame:
    """Load DOE design exported from Minitab (or any CSV).

    Expected columns include the hyperparameter names plus optional fields (StdOrder, RunOrder, PtType, Blocks).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Design file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Design file must be .csv (exported from Minitab or similar).")

    df = _read_csv_flexible(path)

    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_csv_ptbr(df: pd.DataFrame, path: str | Path) -> None:
    """Save CSV in pt-BR friendly format (sep=';' decimal=',')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", decimal=",")
