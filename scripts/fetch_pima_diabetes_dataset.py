#!/usr/bin/env python
"""Fetch the Pima Indians Diabetes dataset via OpenML id 37.

The original UCI dataset was retracted; OpenML id 37 is the canonical
mirror. Uses ``sklearn.datasets.fetch_openml`` lazily.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

OPENML_ID = 37


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    from sklearn.datasets import fetch_openml  # noqa: PLC0415

    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    target_col = bunch.target.name
    df[target_col] = (
        df[target_col].astype(str).str.lower()
        .map({"tested_negative": 0, "tested_positive": 1, "negative": 0, "positive": 1})
        .astype(int)
    )
    df = df.rename(columns={target_col: "class"})
    raw_csv = raw_dir / "pima_openml_37.csv"
    df.to_csv(raw_csv, index=False)
    out = processed_dir / "pima_diabetes.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="pima_diabetes",
        raw_url=None,            # OpenML fetcher writes raw_dir itself
        raw_filename=None,
        process_fn=process,
        notes="OpenML id 37 (UCI original retracted).",
    )


if __name__ == "__main__":
    sys.exit(main())
