#!/usr/bin/env bash
# Prepare the dedicated MacBook Pro environment for the OpenML-CC18
# batch_00 synthetic canary gate.
#
# Strategy (graceful):
#   1. Require Python >= 3.10 (3.12 preferred).
#   2. Create / activate .venv (idempotent).
#   3. Upgrade pip + install the *minimum* canary environment:
#        pip install -e ".[gbdt,doctoral,dev]"
#        pip install "optuna>=3.5"
#   4. Attempt the broader extras:
#        pip install -e ".[hpo_baselines]"
#      Failures here (SMAC / DEHB / pymoo wheels not yet available
#      on this Python version) are recorded as warnings; they do
#      NOT block batch_00. Batch_00 only needs:
#        xgboost, lightgbm, catboost, optuna, sklearn, openml, doe_xgb.
#   5. Print a versioned summary so the operator can paste it into
#      a sign-off line.
#
# This script never trains a model, never downloads a dataset, and
# never starts a benchmark.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python_bin() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"; return
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"; return
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"; return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"; return
  fi
  echo "python"
}

PYBIN="$(python_bin)"
PYVER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")"
PYMAJOR="${PYVER%.*}"; PYMINOR="${PYVER#*.}"
if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]; }; then
  echo "ERROR: Python >= 3.10 is required; found $PYBIN -> $PYVER" >&2
  echo "Install python@3.12 (Homebrew) or pyenv install 3.12.x and retry." >&2
  exit 2
fi
echo "Using $PYBIN ($PYVER)"

# Create / refresh .venv.
if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating .venv with $PYBIN"
  "$PYBIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup] active venv: $(python -c 'import sys; print(sys.executable)')"

python -m pip install --upgrade pip wheel setuptools

echo "[setup] installing minimum canary environment"
python -m pip install -e ".[gbdt,doctoral,dev]"
python -m pip install "optuna>=3.5"

echo "[setup] attempting broader hpo_baselines extras (failures tolerated)"
HPO_OK=0
if python -m pip install -e ".[hpo_baselines]"; then
  HPO_OK=1
else
  echo "[setup] WARNING: .[hpo_baselines] install failed; SMAC/DEHB/pymoo"
  echo "[setup]          will be flagged as missing in the audit. batch_00"
  echo "[setup]          gate does NOT depend on these packages."
fi

echo
echo "================ environment fingerprint ================"
python - <<'PY'
import importlib
import importlib.metadata as m
import platform
import sys

print(f"hostname:       {platform.node()}")
print(f"uname:          {platform.platform()}")
print(f"python:         {sys.version.split()[0]} ({sys.executable})")

REQUIRED = ("xgboost", "lightgbm", "catboost", "optuna", "sklearn", "openml", "doe_xgb")
OPTIONAL = ("smac", "pymoo", "dehb")

def show(name):
    try:
        importlib.import_module(name)
        try:
            ver = m.version(name)
        except m.PackageNotFoundError:
            ver = "imported (version unknown)"
        return ver
    except Exception as exc:
        return f"MISSING ({type(exc).__name__})"

print()
print("required for batch_00 gate:")
missing_req = []
for n in REQUIRED:
    v = show(n)
    print(f"  {n:<10} {v}")
    if v.startswith("MISSING"):
        missing_req.append(n)
print()
print("optional (not required by batch_00):")
for n in OPTIONAL:
    print(f"  {n:<10} {show(n)}")
print()
if missing_req:
    print(f"setup result: FAILED — required packages missing: {missing_req}")
    sys.exit(3)
print("setup result: OK — required packages present; ready for batch_00")
PY
SETUP_RC=$?

echo
echo "Next steps:"
echo "  1) python scripts/audit_method_capabilities.py"
echo "  2) python scripts/run_batch_00_synthetic_canary.py"
echo
echo "(Both are read-only with respect to the committed SQLite shards.)"
exit $SETUP_RC
