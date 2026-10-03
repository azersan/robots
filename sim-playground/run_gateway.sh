#!/usr/bin/env bash
# Start the Sim Gateway inside WSL (kills any previous instance first).
#   bash run_gateway.sh [--scene anna_pl] [--port 8642] [--paused]
# From Windows PowerShell:
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/<you>/OneDrive/Documents/GitHub/robots/sim-playground/run_gateway.sh
set -euo pipefail
VENV="${SIMPG_VENV:-$HOME/venvs/simpg}"
cd "$(dirname "${BASH_SOURCE[0]}")"
pkill -f "simpg.gateway" 2>/dev/null && sleep 1 || true
export PYTHONUNBUFFERED=1
# mujoco_warp prints a line-search notice on hard contact frames; it's harmless noise.
"$VENV/bin/python" -m simpg.gateway "$@" 2>&1 | grep --line-buffered -v "linesearch iterations"
