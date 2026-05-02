#!/usr/bin/env python
"""Fetch the Wine Quality dataset (red + white) and binarise.

Article-track v1 binarises ``quality >= 6`` and merges red+white with
an ``is_red`` flag. The original CSVs are preserved under raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _dataset_fetch_base import run_fetcher  # noqa: E402

from doe_xgb.datasets.download import download_url  # noqa: E402

RED = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
WHITE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
)


def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
    import pandas as pd  # noqa: PLC0415

    red_path = raw_dir / "winequality-red.csv"
    white_path = raw_dir / "winequality-white.csv"
    # The base helper only downloads the URL passed to run_fetcher;
    # the second URL is fetched here.
    if not white_path.exists():
        try:
            download_url(WHITE, white_path)
        except Exception as e:  # pragma: no cover - network exotica
            print(f"warning: white wine download failed: {e}")
    frames = []
    if red_path.exists():
        df_r = pd.read_csv(red_path, sep=";")
        df_r["is_red"] = 1
        frames.append(df_r)
    if white_path.exists():
        df_w = pd.read_csv(white_path, sep=";")
        df_w["is_red"] = 0
        frames.append(df_w)
    if not frames:
        raise FileNotFoundError("neither red nor white wine CSV found")
    df = pd.concat(frames, ignore_index=True)
    df["target_high_quality"] = (df["quality"] >= 6).astype(int)
    out = processed_dir / "wine_quality.csv"
    df.to_csv(out, index=False)
    return [out]


def main() -> int:
    return run_fetcher(
        dataset_id="wine_quality",
        raw_url=RED,
        raw_filename="winequality-red.csv",
        process_fn=process,
        notes="Red+white merged with is_red flag; binary target_high_quality = (quality >= 6).",
    )


if __name__ == "__main__":
    sys.exit(main())
