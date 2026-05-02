#!/usr/bin/env python
"""Fetch the Mushroom dataset (UCI 73)."""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

from doe_xgb.datasets.registry import REGISTRY  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
RAW_NAME = "agaricus-lepiota.data"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    meta = REGISTRY["mushroom"]
    columns = ["class"] + list(meta.categorical_columns)
    df = pd.read_csv(src, header=None, names=columns)
    out = processed_dir / "mushroom.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="mushroom",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="22 categorical features kept as strings; 'class' is 'p'/'e' (mapped at load time).",
    )


if __name__ == "__main__":
    sys.exit(main())
