#!/usr/bin/env bash
# Run the driveway sim on the NZXT (WSL, RTX 5090) and watch it in this Mac's browser.
#   sim/view_driveway.sh [scan-name]        then open http://localhost:8080      Ctrl-C to stop
# The sim runs in the foreground of this SSH session: WSL shuts a distro down when no Windows-side session is
# attached, so a detached background process would not survive.
set -euo pipefail
SCAN="${1:-driveway-2026-10-03-1746.scan}"
PORT="${PORT:-8080}"
(sleep 25 && open "http://localhost:${PORT}") >/dev/null 2>&1 &
# 127.0.0.1, not localhost: Windows resolves localhost to IPv6, which WSL does not forward.
# No quotes in the remote command: the NZXT's SSH shell is Windows cmd, which doesn't understand single quotes.
exec ssh -t -L "${PORT}:127.0.0.1:8080" nzxt wsl -d Ubuntu-24.04 -- /home/tony/robots/sim/run_viewer.sh "${SCAN}"
