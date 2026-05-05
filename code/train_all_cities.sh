#!/bin/bash
# train_all_cities.sh — orchestrator scaffold for per-city model training.
#
# THIS SCRIPT IS NOT YET A FULL TRAINER.  It documents the planned pipeline
# and currently runs only the steps that have been refactored to be
# city-parameterized.  The remaining steps (03_features, 04_train,
# 05_backtest, 09_daily_extreme_train, 10_decision_dataset_v2,
# 11_trade_backtest, 12_meta_model) are still SFO-coupled.  Refactoring
# them is the next session's work.
#
# Usage:
#   bash code/train_all_cities.sh              # all cities except sfo
#   bash code/train_all_cities.sh --city nyc   # single city
#
# What runs today (already city-parameterized):
#   1. 02b_build_lcd_dataset.py   — LCD → hourly + daily parquet
#   2. 14b_multi_refresh_metar.py — append latest METARs from NWS API
#
# What remains (next session):
#   3. 03_features.py            → city-parameterize, ~2 min × 19 = ~40 min
#   4. 04_train.py               → city-parameterize, ~21 min × 19 = ~7 hr
#   5. 05_backtest.py            → city-parameterize, ~30 sec × 19 = ~10 min
#   6. 09_daily_extreme_train.py → city-parameterize, ~30 min × 19 = ~10 hr
#   7. 10_decision_dataset_v2.py → needs Kalshi historical for each city
#   8. 11_trade_backtest.py      → city-parameterize
#   9. 12_meta_model.py          → city-parameterize
#  10. 13_live_signal.py         → already partially refactored, finish it
#
# Total wall time once refactored: ~17 hours sequential, ~3 hours on 8 cores.
#
# Per-city Kalshi historical analysis (the wake-up-time heatmap idea):
#   Will live in a new script 08_timing_skill.py — see Paper 02 plan.

set -e

ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"

ONE_CITY=""
if [ "$1" = "--city" ] && [ -n "$2" ]; then
    ONE_CITY="$2"
fi

echo "[orchestrator] phase 1: build per-city LCD hourly + daily parquets"
if [ -n "${ONE_CITY}" ]; then
    python3 code/02b_build_lcd_dataset.py --city "${ONE_CITY}"
else
    python3 code/02b_build_lcd_dataset.py --all
fi

echo "[orchestrator] phase 2: append latest live METARs"
if [ -n "${ONE_CITY}" ]; then
    python3 code/14b_multi_refresh_metar.py --city "${ONE_CITY}" --hours 168
else
    python3 code/14b_multi_refresh_metar.py --hours 168
fi

echo "[orchestrator] phase 3: regenerate fund_state.json"
python3 build_fund_state.py

cat <<'EOF'

[orchestrator] DONE — phase 1 + 2 complete.
The multi-city terminal now shows live Kalshi prices + last 72h
observations for every configured city.

Phase 3 (model training) is NOT yet implemented.  Cities continue to
show "model pending" in the city-tab indicator until per-city
quantile models are trained.  See the comment block at the top of
this script for the queued work.

EOF
