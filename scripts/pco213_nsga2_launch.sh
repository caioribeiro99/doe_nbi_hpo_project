#!/usr/bin/env bash
# Detached, resumable launch of the evaluation-matched NSGA-II baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=pco213nsga2
LOG=experiments/pco213_postwork_nsga2/nsga2.log
mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already running"; exit 0
fi
tmux new-session -d -s "$SESSION" \
  ".venv-pco213/bin/python scripts/pco213_run_nsga2_baseline.py $* 2>&1 | tee -a $LOG"
echo "launched tmux session $SESSION; log: $LOG"
