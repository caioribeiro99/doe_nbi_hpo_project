#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_ucimlrepo():
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'ucimlrepo'. Install with: pip install ucimlrepo"
        ) from e
    return fetch_ucirepo


def _copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def _coerce_binary_target(series: pd.Series, positive_label: str) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    pos = str(positive_label).strip().lower()
    return s.eq(pos).astype(int)


def fetch_south_german(dst: Path) -> None:
    fetch_ucirepo = _require_ucimlrepo()
    data = fetch_ucirepo(id=522)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y_name = str(getattr(y, "name", "credit_risk"))
    df = X.copy()
    df["y"] = _coerce_binary_target(pd.Series(y), positive_label="good")

    # Harmonize likely age column names.
    if "alter" in df.columns and "age" not in df.columns:
        df = df.rename(columns={"alter": "age"})
    if "credit_risk" in df.columns:
        df = df.drop(columns=["credit_risk"])
    if y_name in df.columns and y_name != "y":
        df = df.drop(columns=[y_name])

    df.to_csv(dst, index=False)


def fetch_adult(dst: Path) -> None:
    fetch_ucirepo = _require_ucimlrepo()
    data = fetch_ucirepo(id=20)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df["y"] = _coerce_binary_target(pd.Series(y), positive_label=">50k")
    df.to_csv(dst, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch/copy the four fairness bases used in the R30 suite.")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "source" / "fairness"))
    p.add_argument("--skip-adult", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy repo-local datasets into a fairness subfolder for convenience.
    _copy_if_exists(
        REPO_ROOT / "data" / "source" / "bank" / "bank-additional-full.csv",
        out_dir / "bank-additional-full.csv",
    )
    _copy_if_exists(
        REPO_ROOT / "data" / "source" / "credit_card_default.csv",
        out_dir / "credit_card_default.csv",
    )

    fetch_south_german(out_dir / "south_german_credit.csv")
    if not args.skip_adult:
        fetch_adult(out_dir / "adult.csv")

    print("✅ Fairness bases prepared under:", out_dir)
    for path in sorted(out_dir.glob("*.csv")):
        print(" -", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
