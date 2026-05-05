#!/bin/bash
# Watch for cities that finish 04_train (last hourly quantile model exists)
# and queue them into phase 2 (05_backtest + 09_daily_extreme_train + 13).
# Idempotent — only launches phase 2 once per city.
#
# Runs until either:
#  * all 19 trainable cities (everything except sfo) have phase-2 launched
#  * --max-iters elapsed (default 240 polls × 30s = 2 hours)
#
# Phase-2 jobs are run with limited parallelism (max 3 at once) — enough
# concurrency to make progress, low enough to avoid swamping the box that
# is also still running 04_train for other cities.
set -u
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"

CITIES=(nyc lax mia chi phx den aus atl okc bos sea msp phl las san dal dca msy hou)
LAUNCHED=()
MAX_PARALLEL=3
MAX_ITERS=240   # 240 × 30s = 2 hours

iter=0
while [ ${iter} -lt ${MAX_ITERS} ]; do
    iter=$((iter + 1))
    all_done=1

    for slug in "${CITIES[@]}"; do
        # Already launched
        if [[ " ${LAUNCHED[*]:-} " =~ " ${slug} " ]]; then
            continue
        fi

        all_done=0   # at least one not yet launched

        # Phase-1 (04_train) complete check
        if [ -f "models/${slug}/qmodel_h72_q95.joblib" ]; then
            # Throttle: don't exceed MAX_PARALLEL phase-2 jobs simultaneously
            running=$(pgrep -f "phase2_pool.sh" 2>/dev/null | wc -l | tr -d ' ')
            running2=$(pgrep -f "09_daily_extreme_train" 2>/dev/null | wc -l | tr -d ' ')
            running=$((running + running2))
            if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
                continue   # try next iteration
            fi

            echo "[$(date '+%H:%M:%S')] [watcher] launching phase 2 for ${slug}"
            nohup bash code/_phase2_pool.sh "${slug}" > "logs/phase2_${slug}_launcher.log" 2>&1 &
            LAUNCHED+=("${slug}")
        fi
    done

    if [ "${all_done}" -eq 1 ]; then
        echo "[$(date '+%H:%M:%S')] [watcher] all cities have phase-2 launched"
        break
    fi

    sleep 30
done

echo "[$(date '+%H:%M:%S')] [watcher] exiting after ${iter} polls; launched: ${LAUNCHED[*]:-none}"
