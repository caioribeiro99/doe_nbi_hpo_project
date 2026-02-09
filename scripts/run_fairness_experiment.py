#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLICA_SCRIPT = REPO_ROOT / "scripts" / "run_fairness_replica.py"


def _remove_arg_pair(args: List[str], flag: str) -> List[str]:
    """Remove occurrences of `flag value` from args if present."""
    out: List[str] = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == flag:
            skip_next = True
            continue
        out.append(a)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run multiple fairness replicas. "
            "This script only owns --replicas and --seed0; "
            "all other args are forwarded to run_fairness_replica.py."
        )
    )
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--seed0", type=int, default=42)

    # Key trick: accept and forward all unknown args
    args, passthrough = p.parse_known_args()

    if not REPLICA_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {REPLICA_SCRIPT}")

    # Safety: remove --replica/--seed if user included them (we will set per replica)
    passthrough = _remove_arg_pair(passthrough, "--replica")
    passthrough = _remove_arg_pair(passthrough, "--seed")

    for r in range(1, int(args.replicas) + 1):
        seed = int(args.seed0) + r - 1

        cmd: List[str] = [
            sys.executable,
            "-u",  # IMPORTANT: unbuffered output so you see progress prints immediately
            str(REPLICA_SCRIPT),
            "--replica",
            str(r),
            "--seed",
            str(seed),
        ] + passthrough

        print(f"\n=== Replica {r}/{args.replicas} | seed={seed} ===")
        print("CMD:", " ".join(cmd))
        subprocess.check_call(cmd)

    print("\n✅ Fairness experiment finished")


if __name__ == "__main__":
    main()
