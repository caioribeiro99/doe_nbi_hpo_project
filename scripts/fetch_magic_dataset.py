#!/usr/bin/env python
"""Fetch the MAGIC Gamma Telescope dataset (UCI 159).

Replaces the legacy fetcher with the new ``manifest + checksum``
contract. Writes:

  data/source/magic/raw/magic04.data
  data/source/magic/processed/magic.csv
  data/source/magic/manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
)
RAW_NAME = "magic04.data"
COLUMNS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class",
]


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, header=None, names=COLUMNS)
    df["class"] = df["class"].map({"g": 0, "h": 1}).astype(int)
    out = processed_dir / "magic.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="magic",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="Targets {g,h} mapped to {0,1}.",
    )


if __name__ == "__main__":
    sys.exit(main())
