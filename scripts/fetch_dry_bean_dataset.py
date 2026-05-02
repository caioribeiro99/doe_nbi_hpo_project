#!/usr/bin/env python
"""Fetch the Dry Bean dataset (UCI 602).

The UCI source is a ZIP containing an XLSX. We extract and convert to
CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

from doe_xgb.datasets.download import extract_zip  # noqa: E402

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
RAW_NAME = "DryBeanDataset.zip"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    zpath = raw_dir / RAW_NAME
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    extract_zip(zpath, raw_dir)
    xlsx = raw_dir / "Dry_Bean_Dataset.xlsx"
    if not xlsx.exists():
        for cand in raw_dir.rglob("*.xlsx"):
            xlsx = cand
            break
    if not xlsx.exists():
        raise FileNotFoundError("no Dry_Bean_Dataset.xlsx inside the ZIP")
    df = pd.read_excel(xlsx)
    out = processed_dir / "dry_bean.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="dry_bean",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="Multiclass (7 classes); 'Class' kept as string for label encoding at load time.",
    )


if __name__ == "__main__":
    sys.exit(main())
