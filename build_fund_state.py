"""Build data/fund_state.json from real model + backtest + live data.

Reads:
  reports/backtest_metrics.json
  reports/live_signals.json                       (SFO model-based signals; legacy)
  reports/live_signals_<slug>.json                (per-city model-based signals)
  reports/live_markets_<slug>.json                (per-city Kalshi-only market data)
  reports/forecast.json
  data/trade_log_B_realistic_short.parquet
  data/<slug>_hourly.parquet                      (per-city METAR observations)
  config/cities.yaml                              (via cities_config)

Writes:
  data/fund_state.json   — single aggregated payload for the public terminal,
                           with one entry per city under state['cities'].
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# After the repo restructure the site lives at the repo root, so the
# project root and the site public root are the same directory.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "fund_state.json"

sys.path.insert(0, str(ROOT / "code"))
try:
    from cities_config import load_cities  # noqa: E402
    CITIES = load_cities()
except Exception as e:
    print(f"warn: could not load cities config ({e}); proceeding SFO-only")
    CITIES = []

# Stated AUM (set by founders)
AUM = 600_000

bt = json.loads((ROOT / "reports" / "backtest_metrics.json").read_text())
sig = json.loads((ROOT / "reports" / "live_signals.json").read_text()) if (ROOT / "reports" / "live_signals.json").exists() else {"signals": []}
fc = json.loads((ROOT / "reports" / "forecast.json").read_text()) if (ROOT / "reports" / "forecast.json").exists() else None

log = pd.read_parquet(ROOT / "data" / "trade_log_B_realistic_short.parquet")
log["decision_time"] = pd.to_datetime(log["decision_time"])
log = log.sort_values("decision_time").reset_index(drop=True)

# Backtest summary numbers (from B = ≤12h short-horizon strategy)
init_bk = 1000.0
final_bk = float(log["bankroll_after"].iloc[-1])
backtest_return_pct = (final_bk - init_bk) / init_bk * 100
n_trades = int(len(log))
win_rate = float((log["won"] == 1).mean())

# Days span
days = (log["decision_time"].iloc[-1] - log["decision_time"].iloc[0]).days
years = max(days / 365.25, 0.001)
annualized_return = float(((final_bk / init_bk) ** (1 / years) - 1) * 100) if years > 0 else 0.0

# Sharpe (simplified): mean daily P&L / std daily P&L * sqrt(252)
log["day"] = log["decision_time"].dt.floor("D")
daily_pnl = log.groupby("day")["pnl"].sum()
if daily_pnl.std() > 0:
    daily_return = daily_pnl / init_bk  # rough fractional return
    sharpe = float(daily_return.mean() / daily_return.std() * np.sqrt(252))
else:
    sharpe = 0.0

# Max drawdown
peak = log["bankroll_after"].cummax()
dd = (log["bankroll_after"] - peak) / peak
max_dd = float(dd.min() * 100)

# Recent trades (last 12)
recent_trades = []
for _, r in log.tail(12).iterrows():
    recent_trades.append({
        "time": r["decision_time"].strftime("%Y-%m-%d %H:%M"),
        "ticker": r["ticker"],
        "side": r["side"].upper(),
        "size_usd": round(float(r["cash_paid"]), 2),
        "contracts": int(r["contracts"]),
        "p_model": round(float(r["p_model"]), 3),
        "cost": round(float(r["cost_per_contract"]), 3),
        "won": int(r["won"]),
        "pnl": round(float(r["pnl"]), 2),
    })

# Open positions / actionable signals
open_positions = []
if sig.get("signals"):
    for s in sig["signals"]:
        if s.get("trade_side") is None:
            continue
        open_positions.append({
            "ticker": s["ticker"],
            "side": s["trade_side"].upper(),
            "bucket": s.get("yes_sub_title") or "",
            "day": s.get("day_D"),
            "cost": round(float(s.get("trade_cost") or 0), 3),
            "p_model": round(float(s.get("model_prob_yes") or 0), 3),
            "p_final": round(float(s.get("p_final") or 0), 3),
            "ev_per_c": round(float(s.get("trade_ev_per_c") or 0), 4),
            "kelly_used": round(float(s.get("kelly_used") or 0), 4),
        })

# Bankroll equity curve (downsample to ~50 points for chart)
n = len(log)
step = max(1, n // 50)
equity = []
for i in range(0, n, step):
    r = log.iloc[i]
    equity.append({"t": r["decision_time"].strftime("%Y-%m-%d"), "v": round(float(r["bankroll_after"]), 2)})
# always include the last point
equity.append({"t": log["decision_time"].iloc[-1].strftime("%Y-%m-%d"),
               "v": round(float(log["bankroll_after"].iloc[-1]), 2)})

# Per-horizon skill from backtest
horizon_skill = []
for h in ["1", "3", "6", "12", "24", "48", "72"]:
    if h not in bt["horizon_metrics"]: continue
    m = bt["horizon_metrics"][h]
    horizon_skill.append({
        "horizon_h": int(h),
        "mae": round(float(m["mae"]), 3),
        "skill_persist": round(float(m["skill_vs_persist"]) * 100, 1),
        "skill_clim": round(float(m["skill_vs_clim"]) * 100, 1),
    })

# Forecast headline (next-day high)
forecast_summary = None
if fc:
    f12 = next((x for x in fc["forecast"] if x["horizon"] == 12), None)
    f24 = next((x for x in fc["forecast"] if x["horizon"] == 24), None)
    forecast_summary = {
        "issued": fc.get("issued"),
        "current_temp": fc.get("current_temp_f"),
        "h12_median": f12["median"] if f12 else None,
        "h24_median": f24["median"] if f24 else None,
    }

# Bucket-probability charts: per (series_ticker, day_D) — list of buckets with our model
# probability vs the Kalshi market price. The terminal renders one chart per group.
bucket_charts = []
if sig.get("signals"):
    groups = {}
    for s in sig["signals"]:
        key = (s.get("series_ticker"), s.get("day_D"))
        groups.setdefault(key, []).append(s)

    def _strike_sort(s):
        # order buckets along the temperature axis: less → between → greater
        st = s.get("strike_type")
        floor = s.get("floor_strike")
        cap = s.get("cap_strike")
        if st == "less":
            return (0, cap if cap is not None else -1e9)
        if st == "between":
            mid = ((floor or 0) + (cap or 0)) / 2.0
            return (1, mid)
        if st == "greater":
            return (2, floor if floor is not None else 1e9)
        return (3, 0)

    for (series, day), items in groups.items():
        if not series or not day:
            continue
        items_sorted = sorted(items, key=_strike_sort)
        buckets = []
        for s in items_sorted:
            ask = s.get("yes_ask")
            bid = s.get("yes_bid")
            mid = None
            if ask is not None and bid is not None:
                mid = (float(ask) + float(bid)) / 2.0
            buckets.append({
                "label": s.get("yes_sub_title") or "",
                "p_model": round(float(s.get("model_prob_yes") or 0), 3),
                "p_final": round(float(s.get("p_final") or 0), 3),
                "p_market": round(float(mid), 3) if mid is not None else None,
                "yes_ask": round(float(ask), 3) if ask is not None else None,
                "yes_bid": round(float(bid), 3) if bid is not None else None,
            })
        side_label = items_sorted[0].get("side_label") or ""
        bucket_charts.append({
            "series_ticker": series,
            "day_D": day,
            "side_label": side_label,
            "title": f"{side_label} · {day}",
            "buckets": buckets,
        })
    # sort by day then side (HIGH before LOW for visual consistency)
    bucket_charts.sort(key=lambda c: (c["day_D"], 0 if c["side_label"] == "HIGH" else 1))

# Last 72h of hourly observations (temp + dew) for the terminal line chart
obs_72h = []
try:
    h = pd.read_parquet(ROOT / "data" / "sfo_hourly.parquet")
    h = h.dropna(subset=["temp_f"]).sort_values("hour").tail(72)
    for _, r in h.iterrows():
        obs_72h.append({
            "t": pd.Timestamp(r["hour"]).strftime("%Y-%m-%d %H:%M"),
            "temp_f": round(float(r["temp_f"]), 2),
            "dew_f": round(float(r["dew_f"]), 2) if pd.notna(r["dew_f"]) else None,
        })
except Exception as e:
    print(f"warn: could not build obs_72h ({e})")

# ─────────────────────────────────────────────────────────────────────────
# Multi-city aggregation
#
# For each city in cities.yaml, we look for two artifacts:
#   reports/live_markets_<slug>.json   — Kalshi-only data (always emitted by
#                                         16_multi_kalshi_fetch.py for every
#                                         city with a configured series).
#   reports/live_signals_<slug>.json   — full model-based signals (only emitted
#                                         once a city has trained models).
#
# We merge them into a per-city object the terminal renders as one tab.  The
# `model_status` field is what the UI uses to decide whether to show p_model
# overlays and EV — "trained" if signals are present, "pending" otherwise.
# ─────────────────────────────────────────────────────────────────────────
def _build_city_section(city) -> dict:
    mk_path = ROOT / "reports" / f"live_markets_{city.slug}.json"
    sg_path = ROOT / "reports" / f"live_signals_{city.slug}.json"
    hr_path = ROOT / "data"    / f"{city.slug}_hourly.parquet"

    # SFO predates the multi-city pipeline — fall back to the legacy paths.
    if city.slug == "sfo":
        if not sg_path.exists() and (ROOT / "reports" / "live_signals.json").exists():
            sg_path = ROOT / "reports" / "live_signals.json"
        if not hr_path.exists() and (ROOT / "data" / "sfo_hourly.parquet").exists():
            hr_path = ROOT / "data" / "sfo_hourly.parquet"

    section = {
        "slug":         city.slug,
        "name":         city.name,
        "icao":         city.icao,
        "timezone":     city.timezone,
        "trains_high":  city.trains_high,
        "trains_low":   city.trains_low,
        "model_status": "pending",
        "bucket_charts": [],
        "obs_72h":       [],
        "open_positions": [],
        "fetched_at_utc": None,
    }

    # ── Kalshi market-only data (always present once 16_multi_kalshi_fetch ran)
    if mk_path.exists():
        try:
            mk = json.loads(mk_path.read_text())
            section["fetched_at_utc"] = mk.get("fetched_at_utc")
            for c in mk.get("bucket_charts", []):
                section["bucket_charts"].append({
                    "side_label":    c["side_label"],
                    "series_ticker": c["series_ticker"],
                    "day_D":         c["day_D"],
                    "title":         f"{c['side_label']} · {c['day_D'] or ''}",
                    "buckets": [{
                        "ticker":    b.get("ticker"),
                        "label":     b.get("subtitle", ""),
                        "p_model":   None,
                        "p_final":   None,
                        "p_market":  b.get("yes_mid"),
                        "yes_bid":   b.get("yes_bid"),
                        "yes_ask":   b.get("yes_ask"),
                        "floor_strike": b.get("floor_strike"),
                        "cap_strike":   b.get("cap_strike"),
                        "strike_type":  b.get("strike_type"),
                    } for b in c.get("buckets", [])],
                })
        except Exception as e:
            print(f"warn: {city.slug}: failed to load live_markets ({e})")

    # ── Per-city trained-model signals (only present once city has models)
    # ── Critical: live_signals.json is refreshed by 13_live_signal which only
    #     runs in the SLOW path (every 10 min).  live_markets.json is refreshed
    #     every fast cycle (~90s).  We therefore use live_markets as the source
    #     of truth for current bid/ask/p_market, and OVERLAY model_prob_yes /
    #     p_final from live_signals onto the matching ticker.  This way the
    #     displayed Kalshi prices update every 90s for trained cities, while
    #     the model probabilities update every 10 min (which is fine — they
    #     only change when new METARs arrive anyway, ~1 per hour).
    if sg_path.exists():
        try:
            sg = json.loads(sg_path.read_text())
            section["model_status"] = "trained"
            # Build a ticker → signal map for fast lookup
            sig_by_ticker = {}
            for s in sg.get("signals", []) or []:
                t = s.get("ticker")
                if t:
                    sig_by_ticker[t] = s

            # Walk the existing bucket_charts (built from live_markets above)
            # and overlay model_prob_yes / p_final per matching ticker.  Keep
            # the FRESH bid/ask/p_market from live_markets.
            for chart in section["bucket_charts"]:
                for b in chart["buckets"]:
                    sig = sig_by_ticker.get(b.get("ticker"))
                    if sig:
                        b["p_model"] = round(float(sig.get("model_prob_yes") or 0), 3)
                        b["p_final"] = round(float(sig.get("p_final") or 0), 3)
                        # Keep the FRESH yes_bid/yes_ask/p_market from live_markets
                        # — DO NOT overwrite from sig (which is up to 10 min old)
                        # Only fill these if they were missing in live_markets
                        if b.get("yes_bid") is None and sig.get("yes_bid") is not None:
                            b["yes_bid"] = round(float(sig["yes_bid"]), 3)
                        if b.get("yes_ask") is None and sig.get("yes_ask") is not None:
                            b["yes_ask"] = round(float(sig["yes_ask"]), 3)
                        if b.get("p_market") is None and b.get("yes_bid") is not None and b.get("yes_ask") is not None:
                            b["p_market"] = round((b["yes_bid"] + b["yes_ask"]) / 2.0, 3)

            # Actionable signals → open positions
            for s in sg.get("signals", []) or []:
                if s.get("trade_side") is None:
                    continue
                section["open_positions"].append({
                    "ticker":     s["ticker"],
                    "side":       s["trade_side"].upper(),
                    "bucket":     s.get("yes_sub_title") or "",
                    "day":        s.get("day_D"),
                    "cost":       round(float(s.get("trade_cost") or 0), 3),
                    "p_model":    round(float(s.get("model_prob_yes") or 0), 3),
                    "p_final":    round(float(s.get("p_final") or 0), 3),
                    "ev_per_c":   round(float(s.get("trade_ev_per_c") or 0), 4),
                    "kelly_used": round(float(s.get("kelly_used") or 0), 4),
                })
        except Exception as e:
            print(f"warn: {city.slug}: failed to load live_signals ({e})")

    # ── Per-city last 72h observations (only present once 02_build_dataset ran)
    if hr_path.exists():
        try:
            hh = pd.read_parquet(hr_path)
            hh = hh.dropna(subset=["temp_f"]).sort_values("hour").tail(72)
            for _, r in hh.iterrows():
                section["obs_72h"].append({
                    "t": pd.Timestamp(r["hour"]).strftime("%Y-%m-%d %H:%M"),
                    "temp_f": round(float(r["temp_f"]), 2),
                    "dew_f":  round(float(r["dew_f"]), 2) if pd.notna(r.get("dew_f")) else None,
                })
        except Exception as e:
            print(f"warn: {city.slug}: failed to load hourly ({e})")

    section["n_open"] = len(section["open_positions"])
    section["n_charts"] = len(section["bucket_charts"])
    return section


def _strike_sort(s):
    st = s.get("strike_type"); floor = s.get("floor_strike"); cap = s.get("cap_strike")
    if st == "less":    return (0, cap if cap is not None else -1e9)
    if st == "between": return (1, ((floor or 0) + (cap or 0)) / 2.0)
    if st == "greater": return (2, floor if floor is not None else 1e9)
    return (3, 0)


# ─────────────────────────────────────────────────────────────────────────
# Lock-detector: zero out bucket probabilities that are mathematically
# impossible given today's already-observed temperature extremes, then
# renormalize.
#
# Rationale: our quantile model is statistically humble — even when the
# day's daily extreme is essentially locked (e.g., the morning low was
# 53°F and temp has risen monotonically since), the model still spreads
# probability across nearby buckets.  The meta-calibrator helps but
# doesn't have a "truth-already-observed" feature, so its p_final stays
# diffuse too.  This post-process fixes that with hard math:
#
#   For LOW market, bucket "X to Y" wins iff day_low ∈ [X-0.5, Y+0.5).
#   Since day_low ≤ obs_min_so_far (always, since day_low is min over the
#   whole day), if obs_min_so_far < X-0.5 the bucket cannot win.  Snap to 0.
#
#   For HIGH market, bucket "X to Y" wins iff day_high ∈ [X-0.5, Y+0.5).
#   day_high ≥ obs_max_so_far, so if obs_max_so_far > Y+0.5 the bucket
#   cannot win.  Snap to 0.
#
# We don't snap "winning" buckets to 1 because future cooling (LOW) or
# warming (HIGH) can still produce a more extreme value — we'd need a
# "no further extreme expected" detector for that, which is harder.
# Just removing impossible buckets and renormalizing is safe and correct.
# ─────────────────────────────────────────────────────────────────────────
def _bucket_excluded(side: str, b: dict, obs_min: float, obs_max: float) -> bool:
    """Decide if a bucket cannot win YES given the day's observed extremes
    so far.  Uses Kalshi's strike-naming conventions:
      between  (floor=X, cap=Y): YES if reported temp ∈ {X..Y} → actual in [X-0.5, Y+0.5)
      greater  (floor=X):        YES if reported temp > X     → actual ≥ X+0.5
      less     (cap=Y):          YES if reported temp < Y     → actual < Y-0.5
    """
    floor = b.get("floor_strike")
    cap   = b.get("cap_strike")
    st    = b.get("strike_type")

    if side == "LOW":
        # day_low ≤ obs_min ALWAYS (since obs is partial; future could be lower).
        # We can only safely exclude buckets whose lower bound exceeds obs_min.
        if st == "between" and floor is not None:
            return obs_min < float(floor) - 0.5
        if st == "greater" and floor is not None:
            # YES iff day_low ≥ floor+0.5.  obs_min < floor+0.5 ⇒ day_low ≤ obs_min < floor+0.5.
            return obs_min < float(floor) + 0.5
        # "less" buckets: future cooling could still hit them — never exclude.
        return False

    if side == "HIGH":
        # day_high ≥ obs_max ALWAYS.  We can exclude buckets whose upper bound is
        # already exceeded by the observed maximum.
        if st == "between" and cap is not None:
            return obs_max > float(cap) + 0.5
        if st == "less" and cap is not None:
            # YES iff day_high < cap-0.5.  obs_max ≥ cap-0.5 ⇒ day_high ≥ obs_max ≥ cap-0.5.
            return obs_max >= float(cap) - 0.5
        # "greater" buckets: future warming could still hit them — never exclude.
        return False

    return False


def _apply_lock_detector(chart: dict, obs_72h: list) -> dict:
    """Zero out impossible buckets given today's observations, renormalize."""
    if not chart["buckets"] or not obs_72h or not chart.get("day_D"):
        return chart
    day_D = chart["day_D"]   # YYYY-MM-DD in local-standard time
    side = chart["side_label"]
    # Match obs that fall on the contract's local-standard day.
    todays = [o["temp_f"] for o in obs_72h if o["t"][:10] == day_D]
    if not todays:
        return chart
    obs_min = min(todays)
    obs_max = max(todays)

    excluded_idx = [i for i, b in enumerate(chart["buckets"])
                    if _bucket_excluded(side, b, obs_min, obs_max)]
    if not excluded_idx:
        return chart

    # Renormalize p_model and p_final independently.
    for field in ("p_model", "p_final"):
        kept_total = sum(
            (b.get(field) or 0.0) for i, b in enumerate(chart["buckets"])
            if i not in excluded_idx
        )
        if kept_total <= 1e-9:
            continue
        for i, b in enumerate(chart["buckets"]):
            cur = b.get(field)
            if cur is None:
                continue
            if i in excluded_idx:
                b[field] = 0.001                                  # near-zero, not exactly 0
            else:
                b[field] = round(min(0.999, cur / kept_total), 3) # rescale to ~1.0
    chart["_lock_detector_excluded"] = excluded_idx               # for debug visibility
    return chart


cities_payload = []
for city in CITIES:
    section = _build_city_section(city)
    # Apply lock-detector to each bucket chart using the city's own obs_72h
    for chart in section.get("bucket_charts", []):
        _apply_lock_detector(chart, section.get("obs_72h", []))
    cities_payload.append(section)

# AUM scaling from backtest
# Backtest used $1k → $X; scale to AUM
aum_scale = AUM / init_bk
ytd_pnl = (final_bk - init_bk) * aum_scale
total_pnl_at_aum = ytd_pnl

state = {
    "fund": {
        "name": "Ariadne Labs",
        "aum_usd": AUM,
        "strategy": "Quantitative prediction-market arbitrage",
        "inception": "2026-01",
        "domicile": "Stanford, CA",
    },
    "performance": {
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "backtest_return_pct": round(backtest_return_pct, 1),
        "annualized_return_pct": round(annualized_return, 1),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "ytd_pnl_at_aum_usd": round(ytd_pnl, 2),
        "starting_bankroll_usd": init_bk,
        "current_bankroll_usd": round(final_bk, 2),
        "days_live": days,
    },
    "exposures": {
        "asset_class": "Event contracts (Kalshi)",
        "venue": "Kalshi",
        "instruments": ["KXHIGHTSFO daily HIGH", "KXLOWTSFO daily LOW"],
        "open_positions": open_positions,
        "n_open": len(open_positions),
    },
    "forecast": forecast_summary,
    "horizon_skill": horizon_skill,
    "equity_curve": equity,
    "recent_trades": recent_trades,
    # ── Legacy single-city (SFO) shape — terminal still reads these for
    #     backwards compatibility while the multi-city UI is rolling out.
    "bucket_charts": bucket_charts,
    "obs_72h": obs_72h,
    # ── New multi-city payload — array of per-city objects, see _build_city_section.
    "cities": cities_payload,
    "n_cities": len(cities_payload),
    "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(state, indent=2, default=str))
print(f"wrote {OUT}")
print(f"  AUM=${AUM:,}  Sharpe={sharpe:.2f}  win_rate={win_rate*100:.1f}%  n_trades={n_trades}")
