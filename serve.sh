#!/bin/bash
# serve.sh — run the Ariadne Labs site locally.
#
# Usage:
#   ./serve.sh           # serves on http://127.0.0.1:8765
#   PORT=9000 ./serve.sh # custom port
#
# This is for local / cofounder use only. Do NOT expose this to the public
# internet without first replacing the client-side auth (js/auth.js) with
# proper server-side auth.

set -u
cd "$(dirname "$0")"
PORT="${PORT:-8765}"

echo "ariadne labs · serving site on http://127.0.0.1:${PORT}"
echo "  homepage:        http://127.0.0.1:${PORT}/"
echo "  research papers: http://127.0.0.1:${PORT}/research.html"
echo "  investor login:  http://127.0.0.1:${PORT}/login.html"
echo ""
echo "auth: any non-empty username and password will sign in."
echo "  the terminal is effectively public — do not put non-public data"
echo "  in data/fund_state.json on this build."
echo ""

# Refresh the fund-state JSON from current backtest + live signal data
python3 build_fund_state.py 2>&1 | tail -2
echo ""

exec python3 -m http.server "$PORT" --bind 127.0.0.1
