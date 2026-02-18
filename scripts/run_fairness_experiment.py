#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run fairness pipeline for multiple replicas (wrapper around run_fairness_replica.py)."
    )
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--seed0", type=int, default=42)

    # Everything else is forwarded to run_fairness_replica.py
    args, rest = p.parse_known_args()

    replicas = int(args.replicas)
    seed0 = int(args.seed0)

    if replicas <= 0:
        raise SystemExit("--replicas must be >= 1")

    for i in range(replicas):
        replica = i + 1
        seed = seed0 + i

        print(f"\n=== Replica {replica}/{replicas} | seed={seed} ===", flush=True)

        cmd: List[str] = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "run_fairness_replica.py"),
            "--replica",
            str(replica),
            "--seed",
            str(seed),
            *rest,
        ]

        print("CMD:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)

    print("\n✅ All replicas finished.", flush=True)


if __name__ == "__main__":
    main()
