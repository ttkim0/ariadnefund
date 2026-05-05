#!/bin/bash
# Helper: train one city's hourly quantile models, log to per-city file.
# Called by xargs -P N for parallelism.
set -e
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"
SLUG="$1"
echo "[$(date '+%H:%M:%S')] start training ${SLUG}"
python3 code/04_train.py --city "${SLUG}" > "logs/train_${SLUG}.log" 2>&1
echo "[$(date '+%H:%M:%S')] done training ${SLUG} (rc=$?)"
