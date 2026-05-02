#!/usr/bin/env python
"""Cost-calibrated sizing report for the article-track v1 campaign.

Reads ``experiments/_cost_calibration/{xgboost,lightgbm,catboost}.json``,
takes the worst-case ``avg_seconds_per_fit`` across the GBDT trio, runs
``estimate_cost`` on four scope variants under three local profiles
plus a candidate cloud profile, manually aggregates a combined two-Mac
scenario, and writes a JSON + Markdown report under
``experiments/_cost_calibration/``.

This script does NOT run the experiment.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.cost_estimator import (  # noqa: E402
    BenchmarkSpec,
    CloudProfile,
    LocalProfile,
    PRESETS,
    estimate_cost,
)


CAL_DIR = REPO / "experiments" / "_cost_calibration"
ARTICLE_TRIO = ("xgboost", "lightgbm", "catboost")
SCENARIOS = (
    "article_v1_12_datasets_3_algorithms_10_replicas",
    "article_v1_12_datasets_3_algorithms_5_replicas",
    "article_v1_8_datasets_3_algorithms_10_replicas",
    "article_v1_8_datasets_3_algorithms_5_replicas",
)


def _load_calibrations() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in ("xgboost.json", "lightgbm.json", "catboost.json"):
        p = CAL_DIR / f
        if not p.exists():
            continue
        out[p.stem] = json.loads(p.read_text())["timings_per_algorithm"]
    return out


def _worst_case_seconds(calibrations: dict[str, dict[str, float]]) -> tuple[str, float]:
    best: tuple[str, float] = ("", 0.0)
    for runs in calibrations.values():
        for algo in ARTICLE_TRIO:
            v = runs.get(algo)
            if v is None:
                continue
            if v > best[1]:
                best = (algo, float(v))
    return best


def _profile_caio_working() -> LocalProfile:
    return LocalProfile(
        max_workers_when_idle=6,
        max_workers_while_working=2,
        hours_idle_per_day=8.0,
        hours_working_per_day=6.0,
        reserve_cores_for_user=2,
        efficiency_factor=0.70,
        model_n_jobs=1,
        warn_if_wall_days_above=14.0,
    )


def _profile_caio_overnight() -> LocalProfile:
    return LocalProfile(
        max_workers_when_idle=6,
        max_workers_while_working=0,
        hours_idle_per_day=14.0,
        hours_working_per_day=0.0,
        reserve_cores_for_user=0,
        efficiency_factor=0.70,
        model_n_jobs=1,
        warn_if_wall_days_above=14.0,
    )


def _profile_dedicated_mac(efficiency: float) -> LocalProfile:
    return LocalProfile(
        max_workers_when_idle=10,
        max_workers_while_working=10,
        hours_idle_per_day=24.0,
        hours_working_per_day=0.0,
        reserve_cores_for_user=0,
        efficiency_factor=efficiency,
        model_n_jobs=1,
        warn_if_wall_days_above=14.0,
    )


def _cloud() -> CloudProfile:
    return CloudProfile(
        workers=32,
        instance_hourly_price_per_worker_usd=0.10,
        efficiency_factor=0.85,
        max_concurrent_jobs=32,
    )


def _combined_two_mac_wall_days(cpu_hours: float, efficiency: float) -> float:
    """Manual aggregation: Caio Mac (6 effective workers, 24h/day) +
    dedicated Mac (10 workers, 24h/day) -> 16 workers, 24h/day,
    times the chosen efficiency_factor."""
    daily_cpu_hours = (6 + 10) * 24.0 * efficiency
    return cpu_hours / daily_cpu_hours if daily_cpu_hours > 0 else float("inf")


def _scenario_spec(name: str, avg_sec: float) -> BenchmarkSpec:
    base = PRESETS[name]
    return BenchmarkSpec(
        n_datasets=base.n_datasets,
        n_algorithms=base.n_algorithms,
        n_replicas=base.n_replicas,
        n_folds=base.n_folds,
        doe_evaluations=base.doe_evaluations,
        nbi_candidates=base.nbi_candidates,
        benchmark_evaluations=base.benchmark_evaluations,
        n_optimization_methods=base.n_optimization_methods,
        avg_seconds_per_fit=avg_sec,
        overhead_factor=base.overhead_factor,
    )


def main() -> int:
    calibrations = _load_calibrations()
    if not calibrations:
        print("No calibration files found. Run estimate-cost --calibrate first.", file=sys.stderr)
        return 1

    worst_algo, worst_sec_calibrated = _worst_case_seconds(calibrations)

    # The synthetic calibration uses 1500 rows x 12 features at default
    # hyperparameters; real article-track DOE rows include heavier
    # configurations (n_estimators up to 700, max_depth up to 18). We
    # report a planning sweep at three multipliers so the reader can
    # see optimistic / realistic / pessimistic projections.
    sweep_multipliers = (1.0, 4.0, 8.0)

    cloud = _cloud()
    profiles = {
        "caio_mac_while_working": _profile_caio_working(),
        "caio_mac_overnight": _profile_caio_overnight(),
        "dedicated_mac_eff_065": _profile_dedicated_mac(0.65),
        "dedicated_mac_eff_070": _profile_dedicated_mac(0.70),
        "dedicated_mac_eff_075": _profile_dedicated_mac(0.75),
    }

    report: dict = {
        "calibration": {
            "raw_files": sorted(p.name for p in CAL_DIR.glob("*.json")),
            "timings_per_algorithm_min_across_runs": {
                algo: round(min(runs.get(algo, float("inf"))
                                for runs in calibrations.values()), 4)
                for algo in ARTICLE_TRIO
            },
            "worst_case_calibrated": {
                "algorithm": worst_algo,
                "seconds_per_fit": round(worst_sec_calibrated, 4),
            },
        },
        "planning_sweep_seconds_per_fit": [
            round(worst_sec_calibrated * m, 4) for m in sweep_multipliers
        ],
        "scenarios": {},
    }

    for sweep_m in sweep_multipliers:
        avg_sec = worst_sec_calibrated * sweep_m
        for scenario in SCENARIOS:
            spec = _scenario_spec(scenario, avg_sec)
            scen_block = report["scenarios"].setdefault(scenario, {})
            sweep_block = scen_block.setdefault(f"x{sweep_m:g}_avg_sec_{round(avg_sec,3)}", {})
            for prof_name, prof in profiles.items():
                est = estimate_cost(spec, local=prof, cloud=cloud)
                sweep_block[prof_name] = {
                    "cpu_hours": round(est.cpu_hours, 1),
                    "local_wall_days": round(est.local_wall.wall_days, 2),
                    "local_wall_hours": round(est.local_wall.wall_hours, 1),
                    "warnings": est.warnings,
                }
            # Combined two-Mac aggregation (manual; estimator does not yet model multi-machine).
            est_for_cpu = estimate_cost(spec, local=profiles["dedicated_mac_eff_070"], cloud=cloud)
            sweep_block["combined_two_macs"] = {
                "total_workers": 16,
                "available_hours_per_day": 24,
                "cpu_hours": round(est_for_cpu.cpu_hours, 1),
                "wall_days_eff_065": round(_combined_two_mac_wall_days(est_for_cpu.cpu_hours, 0.65), 2),
                "wall_days_eff_070": round(_combined_two_mac_wall_days(est_for_cpu.cpu_hours, 0.70), 2),
                "wall_days_eff_075": round(_combined_two_mac_wall_days(est_for_cpu.cpu_hours, 0.75), 2),
            }
            sweep_block["cloud_32_workers"] = {
                "wall_hours": round(est_for_cpu.cloud_wall.wall_hours, 1),
                "wall_days": round(est_for_cpu.cloud_wall.wall_days, 2),
                "cost_usd": round(est_for_cpu.cloud_wall.cost_usd, 2),
                "effective_workers": est_for_cpu.cloud_wall.effective_workers,
                "instance_hourly_price_per_worker_usd": cloud.instance_hourly_price_per_worker_usd,
            }
            sweep_block["totals"] = {
                "total_model_fits": est_for_cpu.total_model_fits,
                "total_fold_fits": est_for_cpu.total_fold_fits,
                "storage_mb": round(est_for_cpu.storage.total_storage_mb, 1),
            }

    out_json = CAL_DIR / "article_v1_cost_estimates.json"
    out_md = CAL_DIR / "article_v1_cost_estimates.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary -----------------------------------------------------
    lines: list[str] = []
    lines.append("# Article v1 cost estimates\n\n")
    lines.append("Generated by `scripts/article_v1_sizing.py`.\n\n")
    lines.append("## Calibration\n\n")
    lines.append(f"- Calibration files: {', '.join(report['calibration']['raw_files'])}\n")
    lines.append("- Best-of-N seconds per fit (synthetic 1500 rows x 12 feats, default HP, n_jobs=1):\n\n")
    lines.append("| Algorithm | Seconds/fit |\n|---|---:|\n")
    for algo, v in report["calibration"]["timings_per_algorithm_min_across_runs"].items():
        lines.append(f"| {algo} | {v:.3f} |\n")
    lines.append(
        f"\nWorst-case calibrated: **{worst_algo} = {worst_sec_calibrated:.3f} s/fit**.\n"
    )
    lines.append("\n## Planning sweep\n\n")
    lines.append(
        "Real article-track DOE rows include heavier configurations than the\n"
        "calibration micro-benchmark (n_estimators up to 700, max_depth up to 18).\n"
        "We report three sweep multipliers so the reader can choose:\n\n"
    )
    lines.append("| Sweep | Seconds/fit | Note |\n|---|---:|---|\n")
    for m, s in zip(sweep_multipliers, report["planning_sweep_seconds_per_fit"], strict=True):
        note = {1.0: "optimistic (raw calibration)",
                4.0: "realistic (4x inflation for full DOE/CV)",
                8.0: "pessimistic (8x inflation)"}.get(m, "")
        lines.append(f"| x{m:g} | {s:.3f} | {note} |\n")

    lines.append("\n## Scenarios\n\n")
    for scenario in SCENARIOS:
        lines.append(f"### {scenario}\n\n")
        for sweep_key, block in report["scenarios"][scenario].items():
            lines.append(f"**{sweep_key}**\n\n")
            t = block["totals"]
            lines.append(
                f"- total_model_fits: {t['total_model_fits']:,}; "
                f"total_fold_fits: {t['total_fold_fits']:,}; "
                f"storage: {t['storage_mb']} MB\n"
            )
            lines.append("| Profile | CPU-hours | Wall-days | Wall-hours | Notes |\n")
            lines.append("|---|---:|---:|---:|---|\n")
            for prof_name in (
                "caio_mac_while_working", "caio_mac_overnight",
                "dedicated_mac_eff_065", "dedicated_mac_eff_070", "dedicated_mac_eff_075",
            ):
                p = block[prof_name]
                warn = "⚠️ " + p["warnings"][0] if p["warnings"] else ""
                lines.append(
                    f"| {prof_name} | {p['cpu_hours']} | {p['local_wall_days']} | "
                    f"{p['local_wall_hours']} | {warn} |\n"
                )
            cm = block["combined_two_macs"]
            lines.append(
                f"| combined_two_macs (16w x 24h) | {cm['cpu_hours']} | "
                f"{cm['wall_days_eff_065']} / {cm['wall_days_eff_070']} / {cm['wall_days_eff_075']} | "
                f"-- | eff 0.65 / 0.70 / 0.75 |\n"
            )
            cl = block["cloud_32_workers"]
            lines.append(
                f"| cloud (32w @ $0.10/h, eff 0.85) | {cm['cpu_hours']} | "
                f"{cl['wall_days']} | {cl['wall_hours']} | "
                f"**${cl['cost_usd']}** |\n\n"
            )
    out_md.write_text("".join(lines), encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
