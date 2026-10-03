#!/usr/bin/env bash
# WSL side of view_driveway.sh: run the driveway sim with the web viewer on :8080, looping.
#   sim/run_viewer.sh [scan-name]
set -euo pipefail
# A previous session that lost its SSH connection can leave a sim holding port 8080.
pkill -f "python sim/driveway.py" 2>/dev/null && sleep 2 || true
cd ~/robots
. .venv/bin/activate
exec python sim/driveway.py ~/scans/"${1:-driveway-2026-10-03-1746.scan}" --viewer viser --port 8080 --loop
