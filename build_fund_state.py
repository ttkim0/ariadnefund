"""Build data/fund_state.json from real model + backtest + live data.

Reads:
  reports/backtest_metrics.json
  reports/live_signals.json
  reports/forecast.json
  data/trade_log_B_realistic_short.parquet
Writes:
  data/fund_state.json
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

# After the repo restructure the site lives at the repo root, so the
# project root and the site public root are the same directory.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "fund_state.json"

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
    "bucket_charts": bucket_charts,
    "obs_72h": obs_72h,
    "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(state, indent=2, default=str))
print(f"wrote {OUT}")
print(f"  AUM=${AUM:,}  Sharpe={sharpe:.2f}  win_rate={win_rate*100:.1f}%  n_trades={n_trades}")
