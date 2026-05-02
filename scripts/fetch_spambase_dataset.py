#!/usr/bin/env python
"""Fetch the Spambase dataset (UCI 94)."""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
RAW_NAME = "spambase.data"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, header=None)
    df.columns = [f"feat_{i}" for i in range(df.shape[1] - 1)] + ["is_spam"]
    out = processed_dir / "spambase.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="spambase",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="57 numeric features renamed feat_0..feat_56; target 'is_spam'.",
    )


if __name__ == "__main__":
    sys.exit(main())
