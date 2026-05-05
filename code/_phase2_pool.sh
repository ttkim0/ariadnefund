#!/bin/bash
# Phase-2 per-city pipeline: 05_backtest → 09_daily_extreme_train → 13_live_signal.
# Runs ONLY for cities that already have hourly quantile models (i.e., 04_train
# has finished).  Idempotent — skips steps whose outputs already exist.
#
# Usage:  bash code/_phase2_pool.sh nyc
set -e
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"
SLUG="$1"
LOG="logs/phase2_${SLUG}.log"

# Guard: 04_train must have completed (last model exists).
if [ ! -f "models/${SLUG}/qmodel_h72_q95.joblib" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] phase 2 skipped — hourly models not done" | tee -a "${LOG}"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] [${SLUG}] phase 2 start" | tee -a "${LOG}"

# 05 backtest (~30s; tolerate failure on cities with too-short test window)
if [ ! -f "reports/backtest_metrics_${SLUG}.json" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 05_backtest" | tee -a "${LOG}"
    python3 code/05_backtest.py --city "${SLUG}" >> "${LOG}" 2>&1 || \
        echo "  [${SLUG}] backtest failed — continuing" | tee -a "${LOG}"
fi

# 09 daily extreme train (~25-30 min)
if [ ! -f "models/${SLUG}/dxmodel_low_q95.joblib" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 09_daily_extreme_train" | tee -a "${LOG}"
    python3 code/09_daily_extreme_train.py --city "${SLUG}" >> "${LOG}" 2>&1
fi

# 13 live signal (writes reports/live_signals_<slug>.json — needed by terminal)
echo "[$(date '+%H:%M:%S')] [${SLUG}] 13_live_signal" | tee -a "${LOG}"
python3 code/13_live_signal.py --city "${SLUG}" >> "${LOG}" 2>&1 || \
    echo "  [${SLUG}] live_signal failed — continuing" | tee -a "${LOG}"

echo "[$(date '+%H:%M:%S')] [${SLUG}] phase 2 DONE" | tee -a "${LOG}"
