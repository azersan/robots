#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu. Run from anywhere:
#   bash /mnt/c/Users/<you>/OneDrive/Documents/GitHub/robots/sim-playground/setup_wsl.sh
#
# The venv lives on the Linux filesystem (~/venvs/simpg) because running
# Python out of /mnt/c (and OneDrive) is very slow.
set -euo pipefail

VENV="${SIMPG_VENV:-$HOME/venvs/simpg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$HERE/requirements.txt"

"$VENV/bin/python" - <<'EOF'
import newton, warp as wp
wp.init()
print("newton", newton.__version__, "| warp", wp.__version__)
print("cuda devices:", wp.get_cuda_devices())
EOF

echo
echo "Done. Activate with:  source $VENV/bin/activate"
