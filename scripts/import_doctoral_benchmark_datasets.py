#!/usr/bin/env python
"""Import / validate the doctoral-benchmark dataset registry CSV.

Behavior:
- ``--csv path``: read the supplied CSV (the new / external one).
- ``--validate-only``: parse the CSV, validate uniqueness, and exit.
- ``--out path``: write the canonicalized + merged registry to
  ``--out``. By default the merge target is
  ``benchmarks/doctoral_82/datasets.csv``.
- ``--overwrite``: on conflict, the incoming row wins. Default is to
  preserve existing rows.
- ``--openml-suite-id``: TODO; not implemented yet. Will populate the
  registry from an OpenML study (e.g., suite 99 / CC18) once the
  selection_policy procedure is automated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.doctoral_benchmark.registry import (  # noqa: E402
    RegistryError,
    load_registry_csv,
    merge_registries,
    validate_registry,
    write_registry_csv,
)


DEFAULT_TARGET = REPO / "benchmarks" / "doctoral_82" / "datasets.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=False,
                        help="Input CSV to validate / merge.")
    parser.add_argument("--out", type=Path, default=DEFAULT_TARGET,
                        help="Output registry path.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate the input CSV and exit; do not merge or write.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Incoming rows win on dataset_id conflict.")
    parser.add_argument("--openml-suite-id", type=int, default=None,
                        help="TODO: not implemented; left as a forward marker.")
    args = parser.parse_args(argv)

    if args.openml_suite_id is not None:
        print("openml-suite-id import is not yet implemented; "
              "use --csv with a curated CSV for now.", file=sys.stderr)
        return 2

    if args.csv is None and args.validate_only:
        target = args.out
        if not target.exists():
            print(f"validate-only: registry not found at {target}", file=sys.stderr)
            return 1
        rows = load_registry_csv(target)
        try:
            validate_registry(rows)
        except RegistryError as e:
            print(f"validation failed: {e}", file=sys.stderr)
            return 1
        print(f"validate-only: {target} -- {len(rows)} rows OK.")
        return 0

    if args.csv is None:
        parser.error("either --csv or --validate-only (with default registry) is required")

    incoming = load_registry_csv(args.csv)
    try:
        validate_registry(incoming)
    except RegistryError as e:
        print(f"incoming CSV failed validation: {e}", file=sys.stderr)
        return 1

    if args.validate_only:
        print(f"validate-only: {args.csv} -- {len(incoming)} rows OK.")
        return 0

    if args.out.exists():
        base = load_registry_csv(args.out)
    else:
        base = []

    merged = merge_registries(base, incoming, overwrite_existing=args.overwrite)
    write_registry_csv(merged, args.out)
    n_added = len({r.dataset_id for r in merged}) - len({r.dataset_id for r in base})
    print(f"merged {len(incoming)} incoming rows into {args.out}; "
          f"+{max(0, n_added)} new dataset_id (total = {len(merged)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
