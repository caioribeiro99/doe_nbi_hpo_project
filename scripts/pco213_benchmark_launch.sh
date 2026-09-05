#!/usr/bin/env bash
# Launch (or resume) the PCO213 post-work benchmark in a detached tmux session.
# Usage: scripts/pco213_benchmark_launch.sh [extra runner args]
# Attach: tmux attach -t pco213bench   |   Log: experiments/pco213_postwork_benchmark/benchmark.log
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=pco213bench
ROOT=experiments/pco213_postwork_benchmark
mkdir -p "$ROOT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already running (pid $(cat $ROOT/benchmark.pid 2>/dev/null || echo ?))"; exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd $(pwd) && .venv-pco213/bin/python scripts/pco213_run_postwork_benchmark.py --run-all --resume $* >> $ROOT/stdout.log 2>&1"
sleep 2
echo "launched tmux session $SESSION; pid file: $ROOT/benchmark.pid; log: $ROOT/benchmark.log"
