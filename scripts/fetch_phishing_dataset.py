#!/usr/bin/env python
"""Fetch the Phishing Websites dataset (UCI 327).

The canonical source is an ARFF file. We download it and emit a CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
)
RAW_NAME = "Training Dataset.arff"


def _parse_arff(src: Path):
    import pandas as pd  # noqa: PLC0415

    cols: list[str] = []
    rows: list[list[int]] = []
    in_data = False
    with src.open() as f:
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
    return pd.DataFrame(rows, columns=cols)


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    df = _parse_arff(src)
    out = processed_dir / "phishing.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="phishing",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="ARFF source converted to CSV; values in {-1, 0, 1} preserved.",
    )


if __name__ == "__main__":
    sys.exit(main())
