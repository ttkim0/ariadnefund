#!/bin/bash
# refresh_chain.sh — Hourly auto-refresh of the SFO forecasting + Kalshi system.
# Designed to be invoked by launchd (Mac) or cron.
#
# Steps:
#   1. Pull last 7d of KSFO METARs from NWS → update hourly grid + features
#   2. Regenerate hourly-temperature forecast (forecast.json/md)
#   3. Pull current Kalshi markets, run model, generate live signals
#   4. Regenerate the HTML dashboard
#
# All steps use absolute paths so this runs cleanly from launchd.
# Logs go to logs/refresh.log (rotated by max-size only — see plist).

set -u  # exit on undefined var (don't use -e: we want to continue on per-step failure)

ROOT="/Users/terrykim/Documents/SF Weather"
PY="/usr/bin/python3"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/refresh.log"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT}/data" "${ROOT}/reports" "${ROOT}/models"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" >> "${LOG_FILE}"; }

log "============================================================"
log "refresh_chain.sh starting"
cd "${ROOT}" || { log "FATAL: cd to ${ROOT} failed"; exit 1; }

run_step() {
    local label="$1"; shift
    log "→ ${label}"
    local t0=$(date +%s)
    if "$@" >> "${LOG_FILE}" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        log "  ✓ ${label} (${dt}s)"
        return 0
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        log "  ✗ ${label} FAILED rc=${rc} (${dt}s)"
        return ${rc}
    fi
}

run_step "noaa metar refresh"   "${PY}" code/14_refresh_noaa.py --hours 168
run_step "hourly forecast"      "${PY}" code/06_predict.py
run_step "live kalshi signals"  "${PY}" code/13_live_signal.py
run_step "dashboard"            "${PY}" code/15_dashboard.py

log "refresh_chain.sh done"

# Truncate the log if > 5MB to keep things tidy
if [ -f "${LOG_FILE}" ] && [ "$(stat -f%z "${LOG_FILE}")" -gt 5242880 ]; then
    tail -c 2097152 "${LOG_FILE}" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "${LOG_FILE}"
    log "(log truncated to last 2MB)"
fi
