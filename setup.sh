#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# doe_nbi_hpo_project setup script
#
# Usage:
#   ./setup.sh                 # creates .venv + installs requirements
#   ./setup.sh --no-venv       # installs into current python environment
#   ./setup.sh --python python3.11
#   ./setup.sh --no-libomp     # skips brew install libomp on macOS
#
# Notes:
# - On macOS Apple Silicon, XGBoost may require libomp via Homebrew.
# - This script is designed for macOS/Linux shells (bash/zsh).
# ---------------------------------------

VENV_DIR=".venv"
PYTHON_BIN="python3"
USE_VENV=1
INSTALL_LIBOMP=1

print_help() {
  cat <<'EOF'
Usage:
  ./setup.sh [options]

Options:
  --no-venv            Install dependencies into the current Python environment (no virtualenv).
  --python <path>      Python executable to use (default: python3).
  --venv-dir <dir>     Virtual environment directory (default: .venv).
  --no-libomp          Skip installing libomp on macOS (Homebrew).
  -h, --help           Show this help.

Examples:
  ./setup.sh
  ./setup.sh --python python3.11
  ./setup.sh --no-venv --python /opt/homebrew/opt/python@3.11/bin/python3.11
EOF
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv)
      USE_VENV=0
      shift
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      if [[ -z "$PYTHON_BIN" ]]; then
        echo "ERROR: --python requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      if [[ -z "$VENV_DIR" ]]; then
        echo "ERROR: --venv-dir requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --no-libomp)
      INSTALL_LIBOMP=0
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

echo "==> doe_nbi_hpo_project setup"
echo "==> Python executable: ${PYTHON_BIN}"

# --- Basic checks ---
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" --version

# --- macOS libomp for XGBoost (optional) ---
if [[ "$(uname -s)" == "Darwin" && "${INSTALL_LIBOMP}" -eq 1 ]]; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> macOS detected. Checking libomp (often required by XGBoost)..."
    if brew list libomp >/dev/null 2>&1; then
      echo "==> libomp already installed."
    else
      echo "==> Installing libomp via Homebrew..."
      brew install libomp
    fi
  else
    echo "==> macOS detected but Homebrew not found."
    echo "    If XGBoost fails to import due to libomp, install Homebrew and run: brew install libomp"
  fi
fi

# --- Create/activate venv if requested ---
if [[ "${USE_VENV}" -eq 1 ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    echo "==> Virtual environment already exists at ${VENV_DIR}"
  fi

  echo "==> Activating virtual environment"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  echo "==> Using venv python: $(python -c 'import sys; print(sys.executable)')"
else
  echo "==> --no-venv selected. Installing into current Python environment."
  echo "==> Using python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
fi

# --- Install deps ---
echo "==> Upgrading pip"
python -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
  echo "==> Installing requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in current directory." >&2
  exit 1
fi

# --- Smoke test ---
echo "==> Running quick import test"
python - <<'PY'
import sys
print("Python:", sys.version)
import numpy, pandas
import sklearn
import xgboost
print("OK: numpy/pandas/sklearn/xgboost imported")
PY

echo
echo "✅ Setup complete."
if [[ "${USE_VENV}" -eq 1 ]]; then
  echo "To activate later: source ${VENV_DIR}/bin/activate"
fi
