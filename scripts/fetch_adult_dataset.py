#!/usr/bin/env python
"""Fetch the Adult / Census Income dataset (UCI 2)."""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
RAW_NAME = "adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, header=None, names=COLUMNS, skipinitialspace=True, na_values="?")
    df = df.dropna()
    df["income"] = (
        df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
        .map({">50K": 1, "<=50K": 0}).astype(int)
    )
    out = processed_dir / "adult.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="adult",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="'?' rows dropped; income mapped to {0,1}; categoricals preserved as strings.",
    )


if __name__ == "__main__":
    sys.exit(main())
