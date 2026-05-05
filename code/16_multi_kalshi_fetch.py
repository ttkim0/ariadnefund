"""
16_multi_kalshi_fetch.py — Pull current Kalshi market data for every city
configured in config/cities.yaml.

This is the model-free counterpart to 13_live_signal.py.  It does not need
trained models — it just records what Kalshi is currently pricing for each
bucket of each open event.  This means we can populate the multi-city
terminal even before per-city models have finished training.

For cities that DO have trained models, 13_live_signal.py (per-city version)
later overlays p_model and EV.  build_fund_state.py merges the two.

Reads:  config/cities.yaml
Writes: reports/live_markets_<slug>.json   (one per city, per refresh)

Usage:
  python3 code/16_multi_kalshi_fetch.py
  python3 code/16_multi_kalshi_fetch.py --city nyc            # one city
  python3 code/16_multi_kalshi_fetch.py --max-workers 6        # parallel

The Kalshi public market-data API is unauthenticated.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import City, load_cities, get_city  # noqa: E402

API = "https://api.elections.kalshi.com/trade-api/v2"


# Kalshi public market-data API rate-limits aggressively when many concurrent
# clients hit it from one IP.  We saw 429s at >2 parallel workers.  In addition
# to lowering parallelism in main(), we exponential-backoff specifically on 429
# (rather than treating it as a hard error like 404).
def http_get(url: str, retries: int = 5, timeout: int = 15) -> dict:
    last_err = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "ariadne-multi-fetch/1.0",
            })
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # too many requests — back off harder
                time.sleep(1.5 * (a + 1))
                last_err = e
                continue
            if e.code in (404, 400):
                return {"_status": e.code, "events": [], "markets": []}
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(0.4 * (a + 1))
    raise RuntimeError(f"GET {url}: {last_err}")


def fetch_series_events(series_ticker: str) -> list[dict]:
    """Return list of {event, markets} for currently-open events in this series."""
    out: list[dict] = []
    cursor = None
    while True:
        url = f"{API}/events?series_ticker={series_ticker}&status=open&limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        d = http_get(url)
        events = d.get("events") or []
        for e in events:
            mu = f"{API}/markets?event_ticker={e['event_ticker']}&limit=100"
            md = http_get(mu)
            out.append({
                "event": e,
                "markets": md.get("markets") or [],
            })
        cursor = d.get("cursor")
        if not cursor or not events:
            break
    return out


def _strike_sort_key(market: dict):
    """Order buckets along the temperature axis: less → between → greater."""
    st = market.get("strike_type") or ""
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    if st == "less":
        return (0, cap if cap is not None else -1e9)
    if st == "between":
        mid = ((floor or 0) + (cap or 0)) / 2.0
        return (1, mid)
    if st == "greater":
        return (2, floor if floor is not None else 1e9)
    return (3, 0)


def _market_to_bucket(m: dict) -> dict:
    """Extract bid/ask/mid for a Kalshi market, handling both API schemas:

    OLD (integer cents, possibly all-null now):
      m['yes_bid']        # int 0..100  e.g. 22
      m['yes_ask']

    NEW (decimal-dollar strings, populated even when old fields are null):
      m['no_bid_dollars']    # str e.g. "0.7700"
      m['no_ask_dollars']    # str e.g. "0.7800"
      m['last_price_dollars']
      m['previous_yes_bid_dollars'], m['previous_yes_ask_dollars']

    For a binary YES/NO market: yes_bid = 1 − no_ask, yes_ask = 1 − no_bid.
    We try new fields first (current API), fall back to old, fall back to
    last_price as a single-point estimate, then to None.
    """
    def _f(x):
        if x is None: return None
        try: return float(x)
        except (TypeError, ValueError): return None

    # Path 1: new "no_*_dollars" fields → invert to yes side.
    no_bid = _f(m.get("no_bid_dollars"))
    no_ask = _f(m.get("no_ask_dollars"))
    yb_d = (1.0 - no_ask) if no_ask is not None else None
    ya_d = (1.0 - no_bid) if no_bid is not None else None

    # Path 2: legacy integer-cent fields if path 1 was null.
    if yb_d is None:
        yb_raw = m.get("yes_bid")
        if yb_raw is not None:
            yb_d = float(yb_raw) / 100.0
    if ya_d is None:
        ya_raw = m.get("yes_ask")
        if ya_raw is not None:
            ya_d = float(ya_raw) / 100.0

    # Path 3: last_price as a single-point fallback if no quotes either side.
    last = _f(m.get("last_price_dollars"))
    if yb_d is None and ya_d is None and last is not None:
        yb_d = ya_d = last

    mid = None
    if yb_d is not None and ya_d is not None:
        mid = (yb_d + ya_d) / 2.0

    return {
        "ticker":       m.get("ticker"),
        "subtitle":     m.get("yes_sub_title") or m.get("subtitle") or "",
        "strike_type":  m.get("strike_type"),
        "floor_strike": m.get("floor_strike"),
        "cap_strike":   m.get("cap_strike"),
        "yes_bid":      yb_d,
        "yes_ask":      ya_d,
        "yes_mid":      mid,
        "last_price":   last,
        "open_interest": _f(m.get("open_interest_fp")) or m.get("open_interest"),
        "volume":       _f(m.get("volume_24h_fp")) or m.get("volume"),
        "status":       m.get("status"),
    }


def fetch_city(city: City) -> dict:
    """Fetch markets for both HIGH and LOW series of one city."""
    t0 = time.time()
    series_data: dict[str, list] = {}
    errors: list[str] = []

    for kind, ticker in [("HIGH", city.kalshi.high_series), ("LOW", city.kalshi.low_series)]:
        if not ticker:
            continue
        try:
            series_data[kind] = fetch_series_events(ticker)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            series_data[kind] = []

    # Group markets by event into bucket charts ordered along the strike axis.
    bucket_charts = []
    for kind, raw_events in series_data.items():
        for entry in raw_events:
            evt = entry["event"]
            markets = entry["markets"]
            if not markets:
                continue
            day_d = (evt.get("event_ticker") or "").split("-")[-1]
            # Convert YYMMMDD → YYYY-MM-DD if possible
            day_iso = _maybe_parse_kalshi_date(day_d)
            sorted_mkts = sorted(markets, key=_strike_sort_key)
            bucket_charts.append({
                "city_slug":     city.slug,
                "city_name":     city.name,
                "icao":          city.icao,
                "side_label":    kind,
                "series_ticker": (city.kalshi.high_series if kind == "HIGH"
                                  else city.kalshi.low_series),
                "event_ticker":  evt.get("event_ticker"),
                "day_D":         day_iso,
                "title":         f"{city.name} · {kind} · {day_iso or day_d}",
                "buckets":       [_market_to_bucket(m) for m in sorted_mkts],
            })

    bucket_charts.sort(key=lambda c: (c["day_D"] or "", 0 if c["side_label"] == "HIGH" else 1))

    return {
        "city_slug":  city.slug,
        "city_name":  city.name,
        "icao":       city.icao,
        "timezone":   city.timezone,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec":   round(time.time() - t0, 2),
        "errors":     errors,
        "bucket_charts": bucket_charts,
        "n_events":   sum(len(v) for v in series_data.values()),
    }


def _maybe_parse_kalshi_date(s: str) -> str | None:
    """Kalshi event tickers end with e.g. '26MAY03'.  Return ISO yyyy-mm-dd."""
    if not s or len(s) != 7:
        return None
    months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
              "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
    try:
        yy = s[0:2]; mm = months[s[2:5]]; dd = s[5:7]
        # YY is two-digit, in 2026 = "26".  Hard-pin to 2000+.
        return f"20{yy}-{mm}-{dd}"
    except (KeyError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", help="single city slug (default: all)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="parallel HTTP workers across cities (default 2 — Kalshi rate-limits)")
    args = ap.parse_args()

    cities = [get_city(args.city)] if args.city else load_cities()
    print(f"[multi-fetch] fetching {len(cities)} cities, parallel={args.max_workers}", flush=True)

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    summary = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "cities": []}

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(fetch_city, c): c for c in cities}
        for fut in cf.as_completed(futures):
            c = futures[fut]
            try:
                result = fut.result()
                out_path = c.live_signals_path.parent / f"live_markets_{c.slug}.json"
                out_path.write_text(json.dumps(result, indent=2))
                line = (f"  {c.slug:>4}  events={result['n_events']:>2}  "
                        f"buckets={len(result['bucket_charts']):>2}  "
                        f"{result['duration_sec']}s")
                if result["errors"]:
                    line += f"  ERR: {result['errors'][0][:50]}"
                print(line, flush=True)
                summary["cities"].append({
                    "slug": c.slug,
                    "n_events": result["n_events"],
                    "n_charts": len(result["bucket_charts"]),
                    "errors":   result["errors"],
                })
            except Exception as e:
                print(f"  {c.slug:>4}  FAIL: {e}", flush=True)
                summary["cities"].append({"slug": c.slug, "error": str(e)})

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "live_markets_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[multi-fetch] done. wrote per-city JSONs and live_markets_summary.json")


if __name__ == "__main__":
    main()
