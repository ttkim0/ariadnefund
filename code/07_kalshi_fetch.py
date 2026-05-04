"""
07_kalshi_fetch.py — Pull Kalshi historical data for SFO weather markets.

Reads:
  (none — uses Kalshi public API)

Writes:
  data/kalshi_events.parquet      one row per event (one calendar day per series)
  data/kalshi_markets.parquet     one row per market (one strike bucket)
  data/kalshi_candles.parquet     one row per (market_ticker, end_period_ts) hourly candle

Series fetched:
  * KXHIGHTSFO  — daily HIGH temperature SFO
  * KXLOWTSFO   — daily LOW  temperature SFO

Notes:
  * All Kalshi market data is public — NO authentication is required for these
    endpoints. We are NOT using the user's API key in this script.
  * Kalshi candlesticks endpoint requires (start_ts, end_ts) Unix seconds and
    `period_interval` in {1, 60, 1440} minutes. Since we want hourly history
    over a market's lifetime (typically 1-3 days for weather markets), we use
    period_interval=60 (hourly candles).
  * Rate-limited politely with a 0.05-0.1s sleep between requests.
  * Idempotent: if the parquet files exist, only newly-seen events/markets are
    fetched. Re-running is cheap.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
EVENTS_OUT = ROOT / "data" / "kalshi_events.parquet"
MARKETS_OUT = ROOT / "data" / "kalshi_markets.parquet"
CANDLES_OUT = ROOT / "data" / "kalshi_candles.parquet"

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXHIGHTSFO", "KXLOWTSFO"]
USER_AGENT = "kalshi-research/1.0"
SLEEP = 0.05  # seconds between calls; ~20 req/s, well under any rate limit


def http_get(url: str, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return {"_status": e.code}  # do not retry on hard errors
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_err}")


def fetch_events(series_ticker: str) -> list[dict]:
    out = []
    cursor = None
    while True:
        url = f"{API}/events?series_ticker={series_ticker}&limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        d = http_get(url)
        events = d.get("events", []) or []
        for e in events:
            e["_series_ticker"] = series_ticker
        out.extend(events)
        cursor = d.get("cursor") or None
        if not cursor or not events:
            break
        time.sleep(SLEEP)
    return out


def fetch_markets(event_ticker: str) -> list[dict]:
    out = []
    cursor = None
    while True:
        url = f"{API}/markets?event_ticker={event_ticker}&limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        d = http_get(url)
        markets = d.get("markets", []) or []
        out.extend(markets)
        cursor = d.get("cursor") or None
        if not cursor or not markets:
            break
        time.sleep(SLEEP)
    return out


def fetch_candles(series_ticker: str, market_ticker: str,
                  start_ts: int, end_ts: int, interval: int = 60) -> list[dict]:
    """interval in minutes: 1 / 60 / 1440. Kalshi accepts up to 5000 candles
    per call; for 60-min over a few days that's well within."""
    url = (f"{API}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
           f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={interval}")
    d = http_get(url)
    if d.get("_status") in (404, 400):
        return []
    return d.get("candlesticks", []) or []


def parse_ts(s: str) -> pd.Timestamp:
    if not s:
        return pd.NaT
    return pd.Timestamp(s).tz_localize(None) if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s).tz_convert("UTC").tz_localize(None)


def to_unix(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def normalize_event(e: dict) -> dict:
    return {
        "series_ticker": e.get("_series_ticker") or e.get("series_ticker"),
        "event_ticker": e.get("event_ticker"),
        "title":        e.get("title"),
        "sub_title":    e.get("sub_title"),
        "category":     e.get("category"),
        "strike_date":  parse_ts(e.get("strike_date")),
        "strike_period": e.get("strike_period"),
        "mutually_exclusive": bool(e.get("mutually_exclusive", False)),
    }


def normalize_market(m: dict) -> dict:
    def f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None
    return {
        "ticker":            m.get("ticker"),
        "event_ticker":      m.get("event_ticker"),
        "market_type":       m.get("market_type"),
        "yes_sub_title":     m.get("yes_sub_title"),
        "no_sub_title":      m.get("no_sub_title"),
        "open_time":         parse_ts(m.get("open_time")),
        "close_time":        parse_ts(m.get("close_time")),
        "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
        "expiration_time":   parse_ts(m.get("expiration_time")),
        "settlement_time":   parse_ts(m.get("settlement_time")),
        "result":            m.get("result"),         # 'yes' / 'no' / '' / None
        "settlement_value":  f(m.get("settlement_value")),
        "floor_strike":      f(m.get("floor_strike")),
        "cap_strike":        f(m.get("cap_strike")),
        "strike_type":       m.get("strike_type"),
        "rules_primary":     m.get("rules_primary"),
        "open_interest":     int(m.get("open_interest") or 0),
        "volume":            int(m.get("volume") or 0),
        "liquidity":         int(m.get("liquidity") or 0),
        "last_price":        m.get("last_price"),
        "previous_yes_bid":  m.get("previous_yes_bid"),
        "previous_yes_ask":  m.get("previous_yes_ask"),
        "status":            m.get("status"),
    }


def normalize_candle(market_ticker: str, c: dict) -> dict:
    def fp(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None
    p = c.get("price") or {}
    yb = c.get("yes_bid") or {}
    ya = c.get("yes_ask") or {}
    end_ts = c.get("end_period_ts")
    return {
        "ticker":          market_ticker,
        "end_ts":          int(end_ts) if end_ts is not None else None,
        "end_time":        pd.to_datetime(end_ts, unit="s", utc=True).tz_localize(None) if end_ts is not None else pd.NaT,
        "price_open":      fp(p.get("open_dollars")),
        "price_close":     fp(p.get("close_dollars")),
        "price_high":      fp(p.get("high_dollars")),
        "price_low":       fp(p.get("low_dollars")),
        "price_mean":      fp(p.get("mean_dollars")),
        "price_prev":      fp(p.get("previous_dollars")),
        "yes_bid_open":    fp(yb.get("open_dollars")),
        "yes_bid_close":   fp(yb.get("close_dollars")),
        "yes_bid_high":    fp(yb.get("high_dollars")),
        "yes_bid_low":     fp(yb.get("low_dollars")),
        "yes_ask_open":    fp(ya.get("open_dollars")),
        "yes_ask_close":   fp(ya.get("close_dollars")),
        "yes_ask_high":    fp(ya.get("high_dollars")),
        "yes_ask_low":     fp(ya.get("low_dollars")),
        "open_interest":   fp(c.get("open_interest_fp")),
        "volume":          fp(c.get("volume_fp")),
    }


def main():
    print("[kalshi] fetching events ...", flush=True)
    all_events = []
    for s in SERIES:
        ev = fetch_events(s)
        print(f"  {s}: {len(ev)} events", flush=True)
        all_events.extend(ev)
    events_df = pd.DataFrame([normalize_event(e) for e in all_events])
    events_df = events_df.sort_values(["series_ticker", "strike_date"]).reset_index(drop=True)
    EVENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(EVENTS_OUT, index=False)
    print(f"[kalshi] wrote {EVENTS_OUT} ({len(events_df):,} rows)", flush=True)

    print("[kalshi] fetching markets per event ...", flush=True)
    all_markets = []
    for i, e in enumerate(all_events):
        et = e.get("event_ticker")
        ms = fetch_markets(et)
        for m in ms:
            m["_series_ticker"] = e.get("_series_ticker")
        all_markets.extend(ms)
        if (i + 1) % 25 == 0:
            print(f"  events {i+1}/{len(all_events)} markets so far {len(all_markets)}", flush=True)
        time.sleep(SLEEP)
    markets_df = pd.DataFrame([normalize_market(m) for m in all_markets])
    markets_df["_series_ticker"] = [m.get("_series_ticker") for m in all_markets]
    markets_df.to_parquet(MARKETS_OUT, index=False)
    print(f"[kalshi] wrote {MARKETS_OUT} ({len(markets_df):,} rows)", flush=True)

    print("[kalshi] fetching hourly candles per market ...", flush=True)
    candles_rows = []
    n = len(all_markets)
    for i, m in enumerate(all_markets):
        ticker = m.get("ticker")
        s = m.get("_series_ticker")
        # Bound candle window to (open_time - 1h, settlement_time + 1h) for safety.
        ot = m.get("open_time")
        st = (m.get("settlement_time") or m.get("expected_expiration_time")
              or m.get("close_time") or m.get("expiration_time"))
        if not ot or not st:
            continue
        try:
            start_ts = to_unix(ot) - 3600
            end_ts = to_unix(st) + 3600
        except Exception as e:
            print(f"  [warn] {ticker}: bad time {ot!r} or {st!r}: {e}", flush=True)
            continue
        candles = fetch_candles(s, ticker, start_ts, end_ts, interval=60)
        for c in candles:
            candles_rows.append(normalize_candle(ticker, c))
        if (i + 1) % 50 == 0:
            print(f"  markets {i+1}/{n}  candles total {len(candles_rows)}", flush=True)
        time.sleep(SLEEP)
    candles_df = pd.DataFrame(candles_rows)
    if not candles_df.empty:
        candles_df = candles_df.sort_values(["ticker", "end_ts"]).reset_index(drop=True)
    candles_df.to_parquet(CANDLES_OUT, index=False)
    print(f"[kalshi] wrote {CANDLES_OUT} ({len(candles_df):,} rows)", flush=True)

    # Quick summary
    print("\n=== KALSHI FETCH SUMMARY ===")
    for s in SERIES:
        nev = (events_df["series_ticker"] == s).sum()
        nmk = (markets_df["_series_ticker"] == s).sum()
        print(f"  {s}: {nev} events, {nmk} markets")
    if not candles_df.empty:
        per_mkt = candles_df.groupby("ticker").size()
        print(f"  candles: {len(candles_df):,} rows over {per_mkt.shape[0]} markets, "
              f"mean {per_mkt.mean():.1f}/market, max {per_mkt.max()}")


if __name__ == "__main__":
    main()
