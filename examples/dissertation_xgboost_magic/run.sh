#!/usr/bin/env bash
# Convenience wrapper for the dissertation example.
# See README.md in this directory for the full discussion.
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_ROOT"

echo ">>> Validating dissertation-baseline config"
$PYTHON -m doe_xgb.cli validate --config configs/dissertation_baseline_xgb_magic.yaml

echo ">>> Validating article-track 3-VRF config"
$PYTHON -m doe_xgb.cli validate --config configs/article_3vrf_xgb_magic.yaml

echo ">>> Running smoke pipeline (synthetic q=3 NBI)"
$PYTHON -m doe_xgb.cli smoke

echo "Done. To run a real replica, invoke scripts/run_replica.py with a real --dataset path."
