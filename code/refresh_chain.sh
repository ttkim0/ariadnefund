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

# ─────────────────────────────────────────────────────────────────────────
# Prevent concurrent runs.  With launchd firing every 90s and the slow
# path occasionally taking ~30s, two chains could otherwise race on the
# git push.  flock makes a second instance exit immediately.
# ─────────────────────────────────────────────────────────────────────────
LOCK_FILE="${LOG_DIR}/.refresh.lock"
exec 9>"${LOCK_FILE}"
if ! /usr/bin/perl -e 'use Fcntl ":flock"; flock(STDIN, LOCK_EX|LOCK_NB) or exit 1' <&9; then
    log "another refresh_chain.sh is already running — exiting"
    exit 0
fi

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

# ─────────────────────────────────────────────────────────────────────────
# Two-speed pipeline:
#   • SLOW steps (NOAA + forecast + static dashboard) run at most every
#     SLOW_INTERVAL seconds.  NOAA METARs only update hourly, and the model
#     forecast can't change without new METARs, so running these every cycle
#     is wasted work and just slows the chain down.
#   • FAST steps (Kalshi signals + fund_state.json + git push) run on
#     every invocation.  This is what keeps the public terminal in lockstep
#     with live Kalshi prices.
# ─────────────────────────────────────────────────────────────────────────
SLOW_INTERVAL=600   # 10 min
SLOW_FLAG="${ROOT}/logs/.last_slow_refresh"

# Auto-push function and its constants are defined here (before first call
# in phase A).  Pushes fund_state.json to GitHub iff it actually changed
# AND the throttle window has passed.  Vercel Pro deploys on push.
LAST_PUSH_FILE="${ROOT}/logs/.last_auto_push"
# Vercel Pro caps: 6000/day, 450/hour, 120 per 5 min.  At 30s throttle on a
# 90s launchd interval, max ~40 pushes/hour — well under all caps.
MIN_PUSH_INTERVAL=30

push_data() {
    if ! command -v git >/dev/null 2>&1; then
        log "  ✗ git not on PATH — skipping push"
        return 1
    fi
    cd "${ROOT}" || return 1

    if git diff --quiet -- data/fund_state.json 2>/dev/null; then
        log "  · no fund_state.json change since HEAD — skipping push"
        return 0
    fi

    if [ -f "${LAST_PUSH_FILE}" ]; then
        local now=$(date +%s)
        local last=$(cat "${LAST_PUSH_FILE}" 2>/dev/null || echo 0)
        local age=$(( now - last ))
        if [ "${age}" -lt "${MIN_PUSH_INTERVAL}" ]; then
            log "  · last push was ${age}s ago (<${MIN_PUSH_INTERVAL}s) — skipping push"
            return 0
        fi
    fi

    git add data/fund_state.json >> "${LOG_FILE}" 2>&1 || return 1
    git -c user.name="ariadne-refresh-bot" \
        -c user.email="refresh-bot@ariadnefund.local" \
        commit -m "auto: refresh fund_state $(ts)" >> "${LOG_FILE}" 2>&1 || {
            log "  ✗ git commit failed"; return 1; }
    if git push origin main >> "${LOG_FILE}" 2>&1; then
        date +%s > "${LAST_PUSH_FILE}"
        log "  ✓ pushed → vercel will redeploy in ~30s"
        return 0
    else
        log "  ✗ git push failed (keychain locked? rate-limited?)"
        return 1
    fi
}

now_epoch=$(date +%s)
slow_age=$(( now_epoch - $(cat "${SLOW_FLAG}" 2>/dev/null || echo 0) ))

# ─────────────────────────────────────────────────────────────────────────
# Phase A — FAST PATH (every cycle, runs FIRST so Vercel gets fresh Kalshi
# prices within ~30s of cycle start, regardless of whether slow steps fire).
# Previously the slow path ran first and Vercel could lag 3-4 min during
# slow cycles.  Now Kalshi-fetch + build + push happens up front, then slow
# steps run, then a SECOND build + push captures any model-output changes.
# ─────────────────────────────────────────────────────────────────────────
run_step "multi-city kalshi"    "${PY}" code/16_multi_kalshi_fetch.py
run_step "fund state json (early)" "${PY}" build_fund_state.py
log "→ early auto-push (Kalshi-fresh)"
push_data

# ─────────────────────────────────────────────────────────────────────────
# Phase B — SLOW PATH (every SLOW_INTERVAL, runs AFTER the early push so
# user sees Kalshi prices immediately even when slow steps are scheduled).
# ─────────────────────────────────────────────────────────────────────────
if [ "${slow_age}" -ge "${SLOW_INTERVAL}" ]; then
    log "(slow steps: last ran ${slow_age}s ago, refreshing)"
    run_step "noaa metar refresh (sfo)"   "${PY}" code/14_refresh_noaa.py --hours 168
    run_step "multi-city METAR refresh"   "${PY}" code/14b_multi_refresh_metar.py --hours 24
    run_step "hourly forecast (sfo)"      "${PY}" code/06_predict.py
    run_step "dashboard (sfo)"            "${PY}" code/15_dashboard.py
    # CRITICAL: regenerate features for every trained city BEFORE running
    # live signals — otherwise features.parquet stays frozen at training
    # time and 13_live_signal re-issues a stale midnight forecast.
    run_step "features refresh (trained)" bash code/03b_refresh_features_trained.sh
    # 13_live_signal for every city that has finished training.
    run_step "live signals (all trained)" bash code/13c_live_signal_all_trained.sh
    date +%s > "${SLOW_FLAG}"
else
    log "(slow steps: last ran ${slow_age}s ago, <${SLOW_INTERVAL}s — skipping)"
fi

# ─────────────────────────────────────────────────────────────────────────
# Phase C — FINAL build + push (only if slow steps actually ran and produced
# fresh model output).  Cheap if slow path skipped.
# ─────────────────────────────────────────────────────────────────────────
run_step "live kalshi signals (sfo)"  "${PY}" code/13_live_signal.py
run_step "fund state json (final)"    "${PY}" build_fund_state.py

log "→ final auto-push (post-slow-path, if anything changed)"
push_data

log "refresh_chain.sh done"

# Truncate the log if > 5MB to keep things tidy
if [ -f "${LOG_FILE}" ] && [ "$(stat -f%z "${LOG_FILE}")" -gt 5242880 ]; then
    tail -c 2097152 "${LOG_FILE}" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "${LOG_FILE}"
    log "(log truncated to last 2MB)"
fi
