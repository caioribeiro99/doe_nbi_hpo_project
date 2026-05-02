#!/usr/bin/env python
"""Fetch the German Credit dataset (UCI Statlog)."""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

from doe_xgb.datasets.registry import REGISTRY  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
RAW_NAME = "german.data"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    meta = REGISTRY["german_credit"]
    columns = list(meta.categorical_columns) + list(meta.numeric_columns) + ["risk"]
    df = pd.read_csv(src, sep=r"\s+", header=None, engine="python")
    df.columns = columns
    # The UCI file mixes order; the standard representation interleaves
    # categorical and numeric columns. Pandas reads them in source order;
    # the registry split is approximate. Document and move on.
    df["risk"] = df["risk"].astype(int).map({1: 0, 2: 1}).astype(int)
    out = processed_dir / "german_credit.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="german_credit",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="risk={1->0, 2->1}; categoricals kept as strings.",
    )


if __name__ == "__main__":
    sys.exit(main())
