#!/bin/bash
# Run 13_live_signal.py for every city that has finished training.
# A city is "trained" if its daily-extreme metrics file exists.
# Used by refresh_chain.sh to populate live_signals_<slug>.json for all
# trained cities each cycle.  Parallelism kept low to respect Kalshi's
# rate limit (saw 429s above ~3 concurrent fetchers in earlier testing).
set -u
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"

trained=()
for slug in $(python3 code/cities_config.py 2>/dev/null | awk 'NR>1 {print $1}'); do
    # SFO uses legacy reports/daily_extreme_metrics.json
    if [ -f "reports/daily_extreme_metrics_${slug}.json" ] || \
       { [ "${slug}" = "sfo" ] && [ -f "reports/daily_extreme_metrics.json" ]; }; then
        trained+=("${slug}")
    fi
done

if [ ${#trained[@]} -eq 0 ]; then
    echo "[live-all] no trained cities yet"
    exit 0
fi

echo "[live-all] running live signal for: ${trained[*]}"

# Run with limited parallelism (Kalshi 429-rate-limit-friendly)
printf "%s\n" "${trained[@]}" | xargs -P 2 -I {} \
    bash -c 'python3 code/13_live_signal.py --city "$1" > "logs/live_$1.log" 2>&1; echo "[live-all] done $1"' _ {}
