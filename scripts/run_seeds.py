#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from doe_xgb.seeds import generate_seeds


def main() -> None:
    p = argparse.ArgumentParser(description="Generate replica seeds")

    p.add_argument(
        "--n",
        type=int,
        default=int(os.getenv("N_REPLICAS", 30)),
        help="Number of replicas (CLI > env:N_REPLICAS > default)",
    )

    p.add_argument(
        "--seed-base",
        type=int,
        default=int(os.getenv("SEED_BASE", 42)),
        help="Initial seed (CLI > env:SEED_BASE > default)",
    )

    args = p.parse_args()

    seeds = generate_seeds(
        initial_seed=args.seed_base,
        n_replicas=args.n,
    )

    for s in seeds:
        print(s)


if __name__ == "__main__":
    main()
