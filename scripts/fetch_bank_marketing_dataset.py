#!/usr/bin/env python
"""Fetch the Bank Marketing dataset (UCI 222) -- bank-additional-full.

The UCI ZIP nests the relevant CSV under bank-additional/. We extract
into raw/ and emit a normalized CSV under processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

from doe_xgb.datasets.download import extract_zip  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
RAW_NAME = "bank-additional.zip"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    zpath = raw_dir / RAW_NAME
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    extract_zip(zpath, raw_dir)
    candidates = [
        raw_dir / "bank-additional-full.csv",
        raw_dir / "bank-additional.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("bank-additional-full.csv not found inside the ZIP")
    df = pd.read_csv(src, sep=";")
    df["y"] = df["y"].astype(str).str.lower().map({"yes": 1, "no": 0}).astype(int)
    out = processed_dir / "bank_marketing.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="bank_marketing",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="Uses bank-additional-full; categoricals preserved; y mapped to {0,1}.",
    )


if __name__ == "__main__":
    sys.exit(main())
