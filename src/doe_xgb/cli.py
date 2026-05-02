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
from pathlib import Path
from typing import Sequence

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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
