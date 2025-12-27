#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from doe_xgb.seeds import generate_seeds


def main():
    p = argparse.ArgumentParser(description="Generate replica seeds")
    p.add_argument("--initial_seed", type=int, default=42)
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    seeds = generate_seeds(args.initial_seed, args.n)
    print(seeds)


if __name__ == "__main__":
    main()
