#!/usr/bin/env python
"""Audit the capability of every CC18 method adapter.

Reads ``benchmarks/doctoral/openml_cc18/method_matrix.csv``, skips
``literature_only`` rows, imports each remaining adapter, calls
``import_check()``, and writes a JSON + Markdown report under
``experiments/_capability_audit/``.

The report is the gating artifact between protocol freeze and the
actual benchmark run: a method whose required packages are missing
cannot be claimed by the local runner. The report does NOT execute
any HPO; it only inspects what is importable.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.methods import ADAPTERS, CapabilityStatus  # noqa: E402

DEFAULT_MATRIX = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"
DEFAULT_OUT_DIR = REPO / "experiments/_capability_audit"


def _load_matrix(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def audit(matrix_path: Path) -> dict:
    rows = _load_matrix(matrix_path)
    benchmarked: list[dict[str, str]] = []
    literature: list[str] = []
    for r in rows:
        if r["primary_or_ablation"] == "literature_only":
            literature.append(r["method_id"])
        else:
            benchmarked.append(r)

    statuses: list[CapabilityStatus] = []
    adapters_missing: list[str] = []
    for r in benchmarked:
        mid = r["method_id"]
        cls = ADAPTERS.get(mid)
        if cls is None:
            adapters_missing.append(mid)
            continue
        try:
            statuses.append(cls().import_check())
        except Exception as exc:  # noqa: BLE001
            statuses.append(CapabilityStatus(
                method_id=mid,
                adapter_import_ok=False,
                required_packages=(),
                missing_packages=(),
                package_versions={},
                supports_binary=False,
                supports_multiclass=False,
                supports_xgboost=False,
                supports_lightgbm=False,
                supports_catboost=False,
                run_status="stub_only",
                notes=f"import_check failed: {type(exc).__name__}: {exc}"[:300],
            ))

    stub_only = [s.method_id for s in statuses if s.run_status == "stub_only"]
    dispatch_only = [s.method_id for s in statuses if s.run_status == "dispatch_only"]
    smoke_ready = [s.method_id for s in statuses if s.run_status == "smoke_ready"]
    full_ready = [s.method_id for s in statuses if s.run_status == "full_ready"]

    pkg_versions: dict[str, str | None] = {}
    pkg_missing: set[str] = set()
    for s in statuses:
        for name, ver in s.package_versions.items():
            if ver is not None:
                pkg_versions[name] = ver
            elif name not in pkg_versions:
                pkg_versions[name] = None
        for m in s.missing_packages:
            pkg_missing.add(m)

    blockers: list[str] = []
    for s in statuses:
        if s.missing_packages:
            blockers.append(
                f"{s.method_id}: missing {','.join(s.missing_packages)}"
            )
    if adapters_missing:
        blockers.append(f"adapters_missing={adapters_missing}")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "n_methods_in_matrix": len(rows),
        "n_benchmarked": len(benchmarked),
        "n_literature_only": len(literature),
        "literature_only": literature,
        "adapters_found": [s.method_id for s in statuses],
        "adapters_missing": adapters_missing,
        "stub_only": stub_only,
        "dispatch_only": dispatch_only,
        "smoke_ready": smoke_ready,
        "full_ready": full_ready,
        "package_versions": dict(sorted(pkg_versions.items())),
        "missing_packages_overall": sorted(pkg_missing),
        "blockers_before_stage0": blockers,
        "per_method": [s.to_dict() for s in statuses],
    }


def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cc18_capability_report.json"
    out_md = out_dir / "cc18_capability_report.md"

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines: list[str] = []
    lines.append("# OpenML-CC18 method capability audit\n")
    lines.append(f"- generated_at: `{report['generated_at']}`")
    lines.append(f"- python: `{report['platform']['python']}` "
                 f"({report['platform']['system']}/{report['platform']['machine']})")
    lines.append(f"- methods in matrix: {report['n_methods_in_matrix']}")
    lines.append(f"- benchmarked: {report['n_benchmarked']}")
    lines.append(f"- literature_only (skipped): {report['n_literature_only']}")
    lines.append("")

    if report["blockers_before_stage0"]:
        lines.append("## Blockers before stage 0\n")
        for b in report["blockers_before_stage0"]:
            lines.append(f"- {b}")
    else:
        lines.append("## Blockers before stage 0\n\n_(none)_")
    lines.append("")

    lines.append("## Adapter run-status\n")
    for label, key in (
        ("full_ready", "full_ready"),
        ("smoke_ready", "smoke_ready"),
        ("dispatch_only", "dispatch_only"),
        ("stub_only", "stub_only"),
    ):
        items = report[key]
        lines.append(f"- **{label}** ({len(items)}): "
                     + (", ".join(f"`{m}`" for m in items) or "_(none)_"))
    lines.append("")

    lines.append("## Per-method\n")
    lines.append("| method | import_ok | run_status | required | missing | notes |")
    lines.append("|---|---|---|---|---|---|")
    for s in report["per_method"]:
        req = ",".join(s["required_packages"]) or "—"
        miss = ",".join(s["missing_packages"]) or "—"
        notes = (s["notes"] or "").replace("|", "\\|")[:80]
        lines.append(
            f"| `{s['method_id']}` | {s['adapter_import_ok']} | "
            f"`{s['run_status']}` | {req} | {miss} | {notes} |"
        )
    lines.append("")

    lines.append("## Package versions\n")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, ver in report["package_versions"].items():
        lines.append(f"| `{name}` | {ver if ver else '_missing_'} |")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_json, out_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Make sure the importlib metadata cache picks up freshly installed extras.
    importlib.invalidate_caches()

    report = audit(args.method_matrix)
    json_p, md_p = write_reports(report, args.out_dir)
    if not args.quiet:
        print(f"benchmarked={report['n_benchmarked']}  "
              f"adapters_found={len(report['adapters_found'])}  "
              f"adapters_missing={len(report['adapters_missing'])}")
        print(f"stub_only={len(report['stub_only'])}  "
              f"dispatch_only={len(report['dispatch_only'])}  "
              f"smoke_ready={len(report['smoke_ready'])}  "
              f"full_ready={len(report['full_ready'])}")
        if report["missing_packages_overall"]:
            print(f"missing_packages={report['missing_packages_overall']}")
        print(f"json: {json_p}")
        print(f"md:   {md_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
