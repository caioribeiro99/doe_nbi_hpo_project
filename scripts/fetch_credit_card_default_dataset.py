#!/usr/bin/env python
"""Fetch the Default of Credit Card Clients dataset (UCI 350).

The canonical UCI source is an XLS file. We download it and convert to
CSV. ``pandas.read_excel`` requires ``xlrd`` for legacy XLS; we fall
back to OpenML id 42477 if the local environment lacks XLS support.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
RAW_NAME = "default_of_credit_card_clients.xls"


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    src = raw_dir / RAW_NAME
    if not src.exists():
        raise FileNotFoundError(src)
    try:
        df = pd.read_excel(src, header=1)
    except (ImportError, ValueError) as e:
        # Fall back to OpenML mirror id 42477.
        from sklearn.datasets import fetch_openml  # noqa: PLC0415
        print(f"XLS read failed ({e}); falling back to OpenML id 42477.")
        bunch = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = bunch.frame.copy()
    rename = {
        "default payment next month": "default_payment_next_month",
        "default.payment.next.month": "default_payment_next_month",
        # OpenML id 42477 ships features 'x1'..'x23' and target 'y'.
        "y": "default_payment_next_month",
        "Y": "default_payment_next_month",
    }
    for old, new in rename.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df["default_payment_next_month"] = df["default_payment_next_month"].astype(int)
    out = processed_dir / "credit_card_default.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="credit_card_default",
        raw_url=URL,
        raw_filename=RAW_NAME,
        process_fn=process,
        notes="UCI XLS or OpenML id 42477 fallback; ID column dropped.",
    )


if __name__ == "__main__":
    sys.exit(main())
