#!/bin/bash
# Run the full per-city training pipeline for one city: 03→04→05→09.
# Designed to be xargs -P parallelized.  Each city runs all four steps
# sequentially in its own process, so when one city's 04 finishes its 05
# can start without waiting for the other parallel cities.
#
# Usage:  bash code/_full_city_pipeline.sh nyc
set -e
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"
SLUG="$1"
LOG="logs/pipeline_${SLUG}.log"

echo "[$(date '+%H:%M:%S')] [${SLUG}] start full pipeline" | tee -a "${LOG}"

# 03 features (idempotent — fast if features exist already)
if [ ! -f "data/${SLUG}_features.parquet" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 03_features" | tee -a "${LOG}"
    python3 code/03_features.py --city "${SLUG}" >> "${LOG}" 2>&1
fi

# 04 train hourly quantile models (49 models)
if [ ! -f "models/${SLUG}/qmodel_h72_q95.joblib" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 04_train (hourly)" | tee -a "${LOG}"
    python3 code/04_train.py --city "${SLUG}" >> "${LOG}" 2>&1
fi

# 05 backtest (~30s)
if [ ! -f "reports/backtest_metrics_${SLUG}.json" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 05_backtest" | tee -a "${LOG}"
    python3 code/05_backtest.py --city "${SLUG}" >> "${LOG}" 2>&1 || \
        echo "  [${SLUG}] backtest failed (likely too-short test window) — continuing" | tee -a "${LOG}"
fi

# 09 daily-extreme train (14 models)
if [ ! -f "models/${SLUG}/dxmodel_low_q95.joblib" ]; then
    echo "[$(date '+%H:%M:%S')] [${SLUG}] 09_daily_extreme_train" | tee -a "${LOG}"
    python3 code/09_daily_extreme_train.py --city "${SLUG}" >> "${LOG}" 2>&1
fi

echo "[$(date '+%H:%M:%S')] [${SLUG}] pipeline DONE" | tee -a "${LOG}"
