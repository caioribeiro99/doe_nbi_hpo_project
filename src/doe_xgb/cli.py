"""Console entry point: ``doe-xgb`` / ``python -m doe_xgb.cli``.

Subcommands
-----------
- ``run --config <path>``  validates a YAML config and reports its
  resolved structure. Long pipeline execution is intentionally NOT
  performed here (in the publication-readiness branch this CLI is a
  smoke / validation harness; the heavy orchestration still lives in
  ``scripts/run_replica.py`` and ``scripts/run_experiment.py``).
- ``validate --config <path>``  runs only Pydantic v2 validation.
- ``smoke``  runs a tiny synthetic NBI pipeline (q=2 quadratics) end
  to end; useful in CI.
- ``info``  prints package version, branch policy, and a pointer to
  the article-track docs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _cmd_validate(args: argparse.Namespace) -> int:
    from .config_schema import load_config

    cfg = load_config(args.config)
    print(f"OK: {args.config}")
    print(f"  experiment.name = {cfg.experiment.name}")
    print(f"  factor_model.mode = {cfg.factor_model.mode}")
    print(f"  nbi.weights.method = {cfg.nbi.weights.method} (q={cfg.nbi.weights.q}, m={cfg.nbi.weights.m})")
    print(f"  primary objectives  = {len([s for s in cfg.objectives.specs if s.role.value == 'primary_nbi'])}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .config_schema import load_config

    cfg = load_config(args.config)
    print(f"Loaded {args.config}.")
    print("This CLI does not run long experiments. Use scripts/run_experiment.py for that.")
    print(f"  factor_model.mode={cfg.factor_model.mode}, nbi.q={cfg.nbi.weights.q}, "
          f"post_opt={cfg.post_optimization.enabled}")
    return 0


def _cmd_smoke(_args: argparse.Namespace) -> int:
    """Tiny synthetic NBI pipeline. Verifies the math kernel runs end-to-end."""
    from .nbi_core import NBIConfig, run_nbi
    from .simplex import generate_simplex_lattice

    q = 3
    targets = np.eye(q)

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(targets[i]) for i in range(q)]
    bounds = np.array([[-1.5, 1.5]] * q)
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=0, maxiter=300)
    weights = generate_simplex_lattice(q, 4)
    run = run_nbi(surrogates, weights, cfg)
    residuals = [c.residual_norm for c in run.candidates]
    out = {
        "n_candidates": len(run.candidates),
        "max_residual": float(max(residuals)),
        "mean_residual": float(sum(residuals) / len(residuals)),
        "anchors_utopia": run.anchors.utopia.tolist(),
    }
    print(json.dumps(out, indent=2))
    return 0 if max(residuals) < 1e-2 else 1


def _cmd_datasets(args: argparse.Namespace) -> int:
    from .datasets import (
        REGISTRY,
        DatasetUnavailableError,
        check_all,
        get_metadata,
        list_dataset_ids,
        load,
        write_availability_report,
    )

    if args.dataset_action == "list":
        for did in list_dataset_ids():
            meta = REGISTRY[did]
            star = "[v1]" if meta.include_in_v1 else "    "
            print(f"  {star} {did:24s} {meta.task_type:10s} {meta.source_type:8s} {meta.display_name}")
        return 0

    if args.dataset_action == "check-availability":
        results = check_all(timeout=args.timeout)
        print(f"Probed {len(results)} datasets:")
        for r in results:
            print(f"  - {r.dataset_id:24s} {r.status:14s} http={r.http_status}")
        if args.out_md is not None:
            write_availability_report(
                results,
                out_md=args.out_md,
                out_json=args.out_json or args.out_md.with_suffix(".json"),
            )
            print(f"Report written to {args.out_md}")
        return 0

    if args.dataset_action == "inspect":
        if not args.dataset_id:
            raise SystemExit("inspect requires --dataset-id")
        meta = get_metadata(args.dataset_id)
        from dataclasses import asdict

        print(json.dumps(asdict(meta), indent=2, default=str))
        return 0

    if args.dataset_action == "smoke":
        ids = list_dataset_ids() if args.all else [args.dataset_id]
        if not args.all and not args.dataset_id:
            raise SystemExit("smoke requires --dataset-id or --all")
        successes, failures = [], []
        for did in ids:
            try:
                ds = load(did)
                successes.append(
                    {
                        "id": did,
                        "rows": ds.metadata.n_rows,
                        "features": ds.metadata.n_features,
                        "task": ds.metadata.task_type,
                        "classes": ds.metadata.class_distribution,
                    }
                )
            except DatasetUnavailableError as e:
                failures.append({"id": did, "reason": str(e)})
            except Exception as e:  # pragma: no cover - defensive
                failures.append({"id": did, "reason": f"unexpected: {e}"})
        print(json.dumps({"loaded": successes, "failures": failures}, indent=2))
        return 0 if not failures else 1

    raise SystemExit(f"unknown datasets action: {args.dataset_action}")


def _cmd_estimate_cost(args: argparse.Namespace) -> int:
    from .cost_estimator import (
        BenchmarkSpec,
        CloudProfile,
        LocalProfile,
        calibrate,
        estimate_cost,
        get_preset,
        list_presets,
    )

    if args.list_presets:
        for name in list_presets():
            print(name)
        return 0

    if args.preset:
        spec = get_preset(args.preset)
    else:
        spec = BenchmarkSpec(
            n_datasets=args.datasets,
            n_algorithms=args.algorithms,
            n_replicas=args.replicas,
            n_folds=args.folds,
            doe_evaluations=args.doe_evaluations,
            nbi_candidates=args.nbi_candidates,
            benchmark_evaluations=args.benchmark_evaluations,
            n_optimization_methods=args.n_methods,
            avg_seconds_per_fit=args.avg_seconds_per_fit,
            overhead_factor=args.overhead_factor,
        )

    avg_sec = args.avg_seconds_per_fit
    if args.calibrate:
        cal = calibrate(output=args.calibration_output)
        if args.algorithm in cal.timings_per_algorithm:
            avg_sec = cal.timings_per_algorithm[args.algorithm]
        elif cal.timings_per_algorithm:
            avg_sec = max(cal.timings_per_algorithm.values())
        if args.preset:
            spec = BenchmarkSpec(
                n_datasets=spec.n_datasets,
                n_algorithms=spec.n_algorithms,
                n_replicas=spec.n_replicas,
                n_folds=spec.n_folds,
                doe_evaluations=spec.doe_evaluations,
                nbi_candidates=spec.nbi_candidates,
                benchmark_evaluations=spec.benchmark_evaluations,
                n_optimization_methods=spec.n_optimization_methods,
                avg_seconds_per_fit=avg_sec,
                overhead_factor=spec.overhead_factor,
            )

    local = LocalProfile(
        max_workers_when_idle=args.max_workers_when_idle,
        max_workers_while_working=args.max_workers_while_working,
        hours_idle_per_day=args.hours_idle_per_day,
        hours_working_per_day=args.hours_working_per_day,
        reserve_cores_for_user=args.reserve_cores_for_user,
        efficiency_factor=args.local_efficiency_factor,
        model_n_jobs=args.model_n_jobs,
        checkpoint_frequency_replicas=args.local_checkpoint_frequency,
        warn_if_wall_days_above=args.warn_if_wall_days_above,
    )

    cloud = CloudProfile(
        workers=args.cloud_workers,
        instance_hourly_price_per_worker_usd=args.cloud_price_per_hour,
        efficiency_factor=args.cloud_efficiency_factor,
        max_concurrent_jobs=args.max_concurrent_jobs,
        checkpoint_frequency_replicas=args.cloud_checkpoint_frequency,
    )

    estimate = estimate_cost(spec, local=local, cloud=cloud)
    out = estimate.to_dict()
    print(json.dumps(out, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    print(
        "doe-xgb (article track)\n"
        "Branch policy: main = dissertation baseline (frozen);\n"
        "               repo-publication-readiness = article-track evolution.\n"
        "Methodological compass: Pereira et al. 2025\n"
        "  https://doi.org/10.1016/j.engappai.2025.112510\n"
        "See docs/ARTIFACT_GUIDE.md and docs/METHODOLOGY_DECISIONS.md for the full picture."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doe-xgb")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate a YAML config.")
    p_validate.add_argument("--config", type=Path, required=True)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Inspect a YAML config (no heavy run).")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.set_defaults(func=_cmd_run)

    p_smoke = sub.add_parser("smoke", help="Run a tiny synthetic NBI pipeline.")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_info = sub.add_parser("info", help="Print branch / methodology pointers.")
    p_info.set_defaults(func=_cmd_info)

    p_ds = sub.add_parser("datasets", help="Inspect / probe / load v1 datasets.")
    ds_sub = p_ds.add_subparsers(dest="dataset_action", required=True)
    ds_sub.add_parser("list", help="List the dataset registry.")
    p_ds_chk = ds_sub.add_parser(
        "check-availability",
        help="HEAD-probe the canonical URLs and write a Markdown + JSON report.",
    )
    p_ds_chk.add_argument("--timeout", type=float, default=5.0)
    p_ds_chk.add_argument("--out-md", type=Path, default=None,
                          help="Path to write the Markdown report (default: stdout only).")
    p_ds_chk.add_argument("--out-json", type=Path, default=None,
                          help="Path to write the JSON report.")
    p_ds_ins = ds_sub.add_parser("inspect", help="Print one dataset's metadata.")
    p_ds_ins.add_argument("--dataset-id", type=str, required=True)
    p_ds_smk = ds_sub.add_parser("smoke", help="Load one or all datasets to verify shapes.")
    p_ds_smk.add_argument("--dataset-id", type=str, default=None)
    p_ds_smk.add_argument("--all", action="store_true")
    p_ds.set_defaults(func=_cmd_datasets)

    p_cost = sub.add_parser(
        "estimate-cost",
        help="Estimate experiment cost (local + cloud); does NOT run the benchmark.",
    )
    p_cost.add_argument("--list-presets", action="store_true")
    p_cost.add_argument("--preset", type=str, default=None)
    p_cost.add_argument("--datasets", type=int, default=1)
    p_cost.add_argument("--algorithms", type=int, default=1)
    p_cost.add_argument("--replicas", type=int, default=30)
    p_cost.add_argument("--folds", type=int, default=5)
    p_cost.add_argument("--doe-evaluations", type=int, default=88)
    p_cost.add_argument("--nbi-candidates", type=int, default=50)
    p_cost.add_argument("--benchmark-evaluations", type=int, default=138)
    p_cost.add_argument("--n-methods", type=int, default=4)
    p_cost.add_argument("--avg-seconds-per-fit", type=float, default=0.5)
    p_cost.add_argument("--overhead-factor", type=float, default=1.10)
    # Local
    p_cost.add_argument("--max-workers-when-idle", type=int, default=8)
    p_cost.add_argument("--max-workers-while-working", type=int, default=2)
    p_cost.add_argument("--hours-idle-per-day", type=float, default=10.0)
    p_cost.add_argument("--hours-working-per-day", type=float, default=6.0)
    p_cost.add_argument("--reserve-cores-for-user", type=int, default=2)
    p_cost.add_argument("--local-efficiency-factor", type=float, default=0.70)
    p_cost.add_argument("--model-n-jobs", type=int, default=1)
    p_cost.add_argument("--local-checkpoint-frequency", type=int, default=5)
    p_cost.add_argument("--warn-if-wall-days-above", type=float, default=14.0)
    # Cloud
    p_cost.add_argument("--cloud-workers", type=int, default=32)
    p_cost.add_argument("--cloud-price-per-hour", type=float, default=0.10)
    p_cost.add_argument("--cloud-efficiency-factor", type=float, default=0.85)
    p_cost.add_argument("--max-concurrent-jobs", type=int, default=32)
    p_cost.add_argument("--cloud-checkpoint-frequency", type=int, default=10)
    # Calibration / output
    p_cost.add_argument("--calibrate", action="store_true",
                        help="Run a tiny synthetic fit per available algorithm to measure avg seconds per fit.")
    p_cost.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help="Path to write cost_estimate_calibration.json (only with --calibrate).",
    )
    p_cost.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "catboost", "histgb"],
        help="Algorithm to use for the calibrated avg_seconds_per_fit.",
    )
    p_cost.add_argument("--output", type=Path, default=None,
                        help="Path to write the JSON cost estimate.")
    p_cost.set_defaults(func=_cmd_estimate_cost)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
