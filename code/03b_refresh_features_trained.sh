#!/bin/bash
# Regenerate features.parquet for every trained city in parallel.
# A city is "trained" if the daily-extreme metrics file exists — that means
# 13_live_signal.py is going to read the features parquet to produce signals,
# so it must be in lockstep with the hourly parquet that 14b just refreshed.
#
# Without this step, features.parquet is frozen at whatever timestamp it had
# when 03_features.py first ran (typically several hours ago at training
# time), and 13_live_signal reissues a stale midnight forecast every cycle.
set -u
ROOT="/Users/terrykim/Documents/SF Weather"
cd "${ROOT}"

trained=()
# Collect all cities whose daily-extreme metrics exist (= ready for live signals).
for slug in $(python3 code/cities_config.py 2>/dev/null | awk 'NR>1 {print $1}'); do
    if [ -f "reports/daily_extreme_metrics_${slug}.json" ] || \
       { [ "${slug}" = "sfo" ] && [ -f "reports/daily_extreme_metrics.json" ]; }; then
        trained+=("${slug}")
    fi
done
[ ${#trained[@]} -eq 0 ] && { echo "[features-refresh] no trained cities yet"; exit 0; }

echo "[features-refresh] regenerating features for: ${trained[*]}"
# 03_features takes ~30s/city; with -P 4 we get ~8 min for 19 cities, fine
# inside the 10-min slow cycle.  03_features.py only writes the parquet; if
# parallel writes step on each other we'll see it here.
printf "%s\n" "${trained[@]}" | xargs -P 4 -I {} \
    bash -c 'python3 code/03_features.py --city "$1" > "logs/features_refresh_$1.log" 2>&1; echo "[features-refresh] done $1"' _ {}
