"""
13_live_signal.py — Live trading signals for current SFO weather markets.

Pulls the currently-open Kalshi events for KXHIGHTSFO and KXLOWTSFO, fetches
each market's current best bid/ask, runs our daily-extreme forecast for each
market's settlement day, and produces a ranked list of trades with edge,
recommended Kelly fraction, and projected EV.

NO authentication required (all market-data endpoints are public). NO trades
are placed — output is informational only.

Reads:
  data/sfo_features.parquet
  reports/daily_extreme_metrics.json
  models/dxmodel_{kind}_q{Q}.joblib  (14 models)
  models/meta_calibrator.joblib       (optional; falls back to raw model prob)

Writes:
  reports/live_signals.json           machine-readable
  reports/live_signals.md             human-readable

Usage:
  python3 code/13_live_signal.py [--issue-now]
  python3 code/13_live_signal.py --issued "2026-04-22T08:00"   (PST)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import get_city  # noqa: E402

ROOT = REPO_ROOT

# Local-standard-time UTC offset by IANA timezone (no DST).  Used to convert
# Kalshi's strike_date (UTC) into the city's local civil day for matching to
# the LCD/feature parquet (which uses local-standard time).
_LOCAL_STD_OFFSET_H = {
    "America/Los_Angeles": 8,
    "America/Denver":      7,
    "America/Phoenix":     7,
    "America/Chicago":     6,
    "America/New_York":    5,
}

API = "https://api.elections.kalshi.com/trade-api/v2"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
KELLY_MULT = 0.25
MAX_FRACTION = 0.05
MIN_EV_PER_CONTRACT = 0.02


def http_get(url: str, retries: int = 3) -> dict:
    last_err = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "kalshi-research/1.0",
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return {"_status": e.code}
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(0.4 * (a + 1))
    raise RuntimeError(f"GET {url}: {last_err}")


def utc_to_local_std(ts, offset_h: int) -> pd.Timestamp:
    """Convert a UTC timestamp to the city's LOCAL STANDARD time (no DST).
    Returns a tz-naive pandas Timestamp.  Matches the time base of LCD-derived
    parquet files."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t - pd.Timedelta(hours=offset_h)


# back-compat alias
def utc_to_pst(ts):
    return utc_to_local_std(ts, 8)


def isotonic(qpred: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(qpred, axis=1)


def cdf_at(qrow: np.ndarray, qs: np.ndarray, x: float) -> float:
    return float(np.interp(x, qrow, qs, left=0.0, right=1.0))


def kelly_fraction(p: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price
    f = (p * b - (1.0 - p)) / b
    return max(0.0, f)


def fetch_open_markets(series_ticker: str) -> list[dict]:
    """Return list of (event, [markets]) for currently open events."""
    out = []
    cursor = None
    while True:
        url = f"{API}/events?series_ticker={series_ticker}&status=open&limit=200"
        if cursor: url += f"&cursor={cursor}"
        d = http_get(url)
        events = d.get("events", []) or []
        for e in events:
            mu = f"{API}/markets?event_ticker={e['event_ticker']}&limit=200"
            md = http_get(mu)
            markets = md.get("markets", []) or []
            out.append((e, markets))
        cursor = d.get("cursor") or None
        if not cursor or not events:
            break
    return out


def fetch_recent_candle(series_ticker: str, market_ticker: str) -> dict | None:
    """Fetch the latest candle within the last 48h. Returns None if none."""
    end = int(time.time())
    start = end - 48 * 3600
    url = (f"{API}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
           f"?start_ts={start}&end_ts={end}&period_interval=60")
    d = http_get(url)
    if d.get("_status") in (404, 400):
        return None
    cs = d.get("candlesticks", []) or []
    return cs[-1] if cs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sfo")
    ap.add_argument("--issued", default=None,
                    help="Issuance timestamp local std time (default: latest feature row)")
    args = ap.parse_args()
    city = get_city(args.city)

    FEAT_PATH    = city.features_path
    DX_META      = REPO_ROOT / "reports" / f"daily_extreme_metrics_{city.slug}.json"
    META_PATH    = city.models_dir / "meta_calibrator.joblib"
    MODEL_DIR    = city.models_dir
    SIGNALS_JSON = city.live_signals_path
    SIGNALS_MD   = REPO_ROOT / "reports" / f"live_signals_{city.slug}.md"

    # SFO legacy paths — fall back to old locations
    if city.slug == "sfo" and not DX_META.exists():
        DX_META = REPO_ROOT / "reports" / "daily_extreme_metrics.json"
    if city.slug == "sfo" and not META_PATH.exists():
        META_PATH = REPO_ROOT / "models" / "meta_calibrator.joblib"
    if city.slug == "sfo":
        # Keep writing the legacy paths too so existing readers keep working
        SIGNALS_JSON = REPO_ROOT / "reports" / "live_signals.json"
        SIGNALS_MD   = REPO_ROOT / "reports" / "live_signals.md"
        if not (city.models_dir / "dxmodel_high_q50.joblib").exists():
            MODEL_DIR = REPO_ROOT / "models"

    series_list = city.kalshi_series()  # [HIGH] and/or [LOW] tickers
    if not series_list:
        print(f"[live:{city.slug}] no Kalshi series configured; skipping")
        return

    if not DX_META.exists():
        print(f"[live:{city.slug}] no daily-extreme metrics — train 09 for this city first; skipping")
        return

    offset_h = _LOCAL_STD_OFFSET_H.get(city.timezone, 8)

    print(f"[live:{city.slug}] loading features + models", flush=True)
    features = pd.read_parquet(FEAT_PATH)
    if "timezone" in features.columns:
        features = features.drop(columns=["timezone"])
    dx_meta = json.loads(DX_META.read_text())
    fcols = dx_meta["feature_cols"]   # includes hours_to_settle at the end

    if args.issued:
        issued = pd.Timestamp(args.issued)
    else:
        valid = features[features["temp_f"].notna()]
        issued = valid["hour"].max()
    print(f"[live:{city.slug}] issuance time (local std): {issued}", flush=True)

    feat_row_base = features.loc[features["hour"].eq(issued)]
    if feat_row_base.empty:
        raise SystemExit(f"No feature row for {issued}")
    feat_row_base = feat_row_base.iloc[0]

    # Fetch open events + markets for this city's Kalshi series only
    print(f"[live:{city.slug}] fetching open Kalshi events for {series_list} ...", flush=True)
    all_events = []
    low_series = city.kalshi.low_series
    for s in series_list:
        ev_markets = fetch_open_markets(s)
        for e, ms in ev_markets:
            e["_series_ticker"] = s
            all_events.append((e, ms))
    print(f"[live:{city.slug}] open events: {len(all_events)}", flush=True)

    # Optional meta-calibrator (only SFO has one trained)
    meta = None
    if META_PATH.exists():
        try:
            meta = joblib.load(META_PATH)
        except Exception:
            meta = None
    print(f"[live:{city.slug}] meta calibrator: {'loaded' if meta else 'not found'}", flush=True)

    qs_arr = np.array(QUANTILES)
    signals = []

    for e, markets in all_events:
        if not markets:
            continue
        kind = "low" if e["_series_ticker"] == low_series else "high"
        # day_D = strike_date in local std - 1d
        sd_local = utc_to_local_std(e["strike_date"], offset_h)
        day_D = (sd_local - pd.Timedelta(days=1)).floor("D")
        hours_to_settle = ((day_D + pd.Timedelta(days=1)) - issued).total_seconds() / 3600.0
        if hours_to_settle <= 0:
            continue
        # Build feature vector
        x = np.empty(len(fcols), dtype="float32")
        for i, c in enumerate(fcols):
            if c == "hours_to_settle":
                x[i] = float(hours_to_settle)
            else:
                v = feat_row_base[c]
                x[i] = np.nan if pd.isna(v) else float(v)

        # Predict 7 quantiles for that kind
        qpred = np.empty((1, len(QUANTILES)))
        for j, q in enumerate(QUANTILES):
            mp = MODEL_DIR / f"dxmodel_{kind}_q{int(q*100):02d}.joblib"
            m = joblib.load(mp)
            qpred[0, j] = float(m.predict(x.reshape(1, -1))[0])
        qpred = isotonic(qpred)
        median = qpred[0, QUANTILES.index(0.50)]

        for mk in markets:
            ticker = mk["ticker"]
            stype = mk.get("strike_type")
            floor = mk.get("floor_strike")
            cap = mk.get("cap_strike")
            yes_bid = mk.get("yes_bid")
            yes_ask = mk.get("yes_ask")

            def cents_to_dollar(v):
                if v is None: return None
                v = float(v) / (100.0 if v > 1.5 else 1.0)
                return v
            yes_bid = cents_to_dollar(yes_bid)
            yes_ask = cents_to_dollar(yes_ask)

            # Fallback: if live bid/ask missing, use the most recent hourly candle.
            if yes_bid is None or yes_ask is None:
                cnd = fetch_recent_candle(e["_series_ticker"], ticker)
                if cnd:
                    yb = cnd.get("yes_bid", {})
                    ya = cnd.get("yes_ask", {})
                    yes_bid = float(yb.get("close_dollars")) if yb.get("close_dollars") is not None else yes_bid
                    yes_ask = float(ya.get("close_dollars")) if ya.get("close_dollars") is not None else yes_ask
            data_status = "live" if (mk.get("yes_bid") is not None) else ("candle" if (yes_bid is not None) else "no_data")

            # Compute model_prob_yes (with NWS integer-rounding adjustment)
            try:
                if stype == "greater":
                    model_p = 1.0 - cdf_at(qpred[0], qs_arr, float(floor) + 0.5)
                elif stype == "less":
                    model_p = cdf_at(qpred[0], qs_arr, float(cap) - 0.5)
                elif stype == "between":
                    model_p = (cdf_at(qpred[0], qs_arr, float(cap) + 0.5)
                               - cdf_at(qpred[0], qs_arr, float(floor) - 0.5))
                    model_p = max(0.0, min(1.0, model_p))
                else:
                    continue
            except Exception:
                continue

            # Compute model_prob for this strike regardless of market data
            try:
                if stype == "greater":
                    model_p = 1.0 - cdf_at(qpred[0], qs_arr, float(floor) + 0.5)
                elif stype == "less":
                    model_p = cdf_at(qpred[0], qs_arr, float(cap) - 0.5)
                elif stype == "between":
                    model_p = (cdf_at(qpred[0], qs_arr, float(cap) + 0.5)
                               - cdf_at(qpred[0], qs_arr, float(floor) - 0.5))
                    model_p = max(0.0, min(1.0, model_p))
                else:
                    continue
            except Exception:
                continue

            if yes_bid is None or yes_ask is None:
                signals.append({
                    "ticker": ticker, "event_ticker": e["event_ticker"],
                    "series_ticker": e["_series_ticker"], "side_label": kind.upper(),
                    "day_D": str(day_D.date()),
                    "hours_to_settle": round(hours_to_settle, 2),
                    "yes_sub_title": mk.get("yes_sub_title"),
                    "strike_type": stype, "floor_strike": floor, "cap_strike": cap,
                    "yes_bid": None, "yes_ask": None, "spread": None,
                    "model_prob_yes": round(model_p, 4),
                    "meta_prob_yes": None, "p_final": round(model_p, 4),
                    "model_median_temp_f": round(float(median), 2),
                    "trade_side": None, "data_status": "no_data",
                })
                continue
            mid = 0.5 * (yes_bid + yes_ask)
            spread = yes_ask - yes_bid

            # Optional meta calibration
            meta_p = None
            if meta is not None:
                fnames = meta["feature_names"]
                fdict = {
                    "logit_model": np.log(np.clip(model_p, 1e-6, 1-1e-6)/np.clip(1-model_p, 1e-6, 1)),
                    "logit_market": np.log(np.clip(mid, 1e-6, 1-1e-6)/np.clip(1-mid, 1e-6, 1)),
                    "disagreement": abs(model_p - mid),
                    "hours_to_settle": hours_to_settle,
                    "spread": spread,
                    "log_oi": np.log1p(mk.get("open_interest") or 0),
                    "log_vol": np.log1p(mk.get("volume") or 0),
                    "st_greater": int(stype == "greater"),
                    "st_less": int(stype == "less"),
                    "st_between": int(stype == "between"),
                }
                xf = np.array([fdict[n] for n in fnames]).reshape(1, -1).astype("float32")
                p_raw = float(meta["pipeline"].predict_proba(xf)[0, 1])
                meta_p = float(meta["isotonic"].transform([p_raw])[0])

            p_final = meta_p if meta_p is not None else model_p

            # Compute EV for both sides after fees
            yes_cost = yes_ask
            no_cost = 1.0 - yes_bid
            yes_fee = max(0.01, 0.07 * yes_cost * (1 - yes_cost))
            no_fee = max(0.01, 0.07 * no_cost * (1 - no_cost))
            yes_ev = p_final - yes_cost - yes_fee
            no_ev = (1.0 - p_final) - no_cost - no_fee

            side = None
            if yes_ev >= MIN_EV_PER_CONTRACT and yes_ev >= no_ev:
                side, cost, p_win = "yes", yes_cost, p_final
            elif no_ev >= MIN_EV_PER_CONTRACT:
                side, cost, p_win = "no", no_cost, 1.0 - p_final
            else:
                # Still log the no-trade for transparency
                side = None

            f_kelly = kelly_fraction(p_win, cost) if side else 0.0
            f_used = min(f_kelly * KELLY_MULT, MAX_FRACTION) if side else 0.0

            signals.append({
                "ticker": ticker,
                "event_ticker": e["event_ticker"],
                "series_ticker": e["_series_ticker"],
                "side_label": kind.upper(),
                "day_D": str(day_D.date()),
                "hours_to_settle": round(hours_to_settle, 2),
                "yes_sub_title": mk.get("yes_sub_title"),
                "strike_type": stype,
                "floor_strike": floor,
                "cap_strike": cap,
                "yes_bid": yes_bid, "yes_ask": yes_ask,
                "spread": spread,
                "model_prob_yes": round(model_p, 4),
                "meta_prob_yes": None if meta_p is None else round(meta_p, 4),
                "p_final": round(p_final, 4),
                "model_median_temp_f": round(float(median), 2),
                "trade_side": side,
                "trade_cost": cost if side else None,
                "trade_ev_per_c": (yes_ev if side == "yes" else no_ev) if side else None,
                "kelly_full": round(f_kelly, 4),
                "kelly_used": round(f_used, 4),
                "open_interest": mk.get("open_interest"),
                "volume": mk.get("volume"),
                "data_status": data_status,
            })

    # Sort: actionable trades first by EV
    signals.sort(key=lambda s: (s["trade_side"] is None, -(s.get("trade_ev_per_c") or -1)))

    SIGNALS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps({
        "city": city.slug,
        "issued_local_std": str(issued),
        "issued_pst": str(issued),  # legacy alias for backwards compat
        "n_signals": len(signals),
        "n_actionable": sum(1 for s in signals if s["trade_side"] is not None),
        "signals": signals,
    }, indent=2))

    # Markdown
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_hours = (now - (issued + pd.Timedelta(hours=offset_h))).total_seconds() / 3600.0
    stale_warning = ""
    if age_hours > 12:
        stale_warning = (f"\n> ⚠️  **STALE DATA**: features are from {age_hours:.0f}h ago. "
                         f"Run code/14b_multi_refresh_metar.py to pull latest METARs.\n")

    lines = [f"# Live Trading Signals — {city.slug.upper()} ({city.icao})\n",
             f"Issued at **{issued} local-std** (UTC-{offset_h}, no DST).",
             f"All Kalshi prices in dollars (1.00 = settles YES).{stale_warning}\n",
             "## Actionable trades (EV ≥ $0.02/contract after fees, sorted by EV)\n"]
    actionable = [s for s in signals if s["trade_side"] is not None]
    if not actionable:
        lines.append("_No actionable opportunities at current prices._\n")
    else:
        lines.append("| Day | Ticker | Bucket | Side | Cost | p_model | p_market_mid | p_final | EV/$ | Kelly used |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for s in actionable:
            mid = 0.5 * (s["yes_bid"] + s["yes_ask"])
            lines.append(f"| {s['day_D']} | {s['ticker']} | {s['yes_sub_title']} | "
                         f"{s['trade_side'].upper()} | {s['trade_cost']:.3f} | "
                         f"{s['model_prob_yes']:.3f} | {mid:.3f} | {s['p_final']:.3f} | "
                         f"{s['trade_ev_per_c']:+.4f} | {s['kelly_used']:.4f} |")

    lines.append("\n## All open markets (full table)\n")
    lines.append("| Day | Ticker | Bucket | Bid | Ask | p_model | p_final | OI | Vol |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in signals:
        lines.append(f"| {s['day_D']} | {s['ticker']} | {s['yes_sub_title']} | "
                     f"{s['yes_bid']:.3f} | {s['yes_ask']:.3f} | "
                     f"{s['model_prob_yes']:.3f} | {s['p_final']:.3f} | "
                     f"{s.get('open_interest') or 0} | {s.get('volume') or 0} |")
    SIGNALS_MD.write_text("\n".join(lines))
    print(f"[live] wrote {SIGNALS_JSON} and {SIGNALS_MD}", flush=True)
    print(f"[live] {len(signals)} signals, {len(actionable)} actionable", flush=True)


if __name__ == "__main__":
    main()
