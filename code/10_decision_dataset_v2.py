"""
10_decision_dataset_v2.py — Rebuild the decision dataset using DAILY-EXTREME models.

This replaces 08_decision_dataset.py. Key change: instead of taking the hourly
forecast at 15:00 PST and applying a small offset, we use 14 dedicated
daily-extreme quantile models from `09_daily_extreme_train.py` that target
the actual daily max/min on day D.

Output: data/decision_dataset_v2.parquet

For each (market, decision_time t) pair:
  * Determine kind ∈ {high, low} from the series ticker.
  * Determine k = settlement_day_D - day(t) (in PST days), clipped to [0,3].
  * Compute hours_to_settle = midnight_after(D, PST) - t in hours.
  * Predict 7 quantiles using dxmodel_{kind}_q{Q}.joblib with all features
    at t + hours_to_settle as the extra feature.
  * Map quantiles to model_prob_yes for each strike type using
    F(cap+0.5) - F(floor-0.5) (between), 1 - F(floor) (greater), F(cap) (less).
  * Joint with market state (price, bid, ask, volume, OI) at t.
  * Truth (actual_high / actual_low) and yes_outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
EVENTS_PATH = ROOT / "data" / "kalshi_events.parquet"
MARKETS_PATH = ROOT / "data" / "kalshi_markets.parquet"
CANDLES_PATH = ROOT / "data" / "kalshi_candles.parquet"
HOURLY_PATH = ROOT / "data" / "sfo_hourly.parquet"
FEATURES_PATH = ROOT / "data" / "sfo_features.parquet"
DX_META = ROOT / "reports" / "daily_extreme_metrics.json"
HOURLY_META = ROOT / "reports" / "train_metrics.json"
MODEL_DIR = ROOT / "models"

OUT_PATH = ROOT / "data" / "decision_dataset_v2.parquet"
SUMMARY_PATH = ROOT / "reports" / "decision_dataset_v2_summary.md"

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def utc_to_pst(ts_utc) -> pd.Timestamp:
    t = pd.Timestamp(ts_utc)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t - pd.Timedelta(hours=8)


def to_f32(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for c in cols:
        s = d[c]
        if pd.api.types.is_extension_array_dtype(s) or s.dtype == "float64":
            out[c] = s.astype("float32")
        else:
            out[c] = s
    return out


def isotonic_monotone(qpred: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(qpred, axis=1)


def cdf_at_value(qpred_row: np.ndarray, qs: np.ndarray, x: float) -> float:
    return float(np.interp(x, qpred_row, qs, left=0.0, right=1.0))


def main():
    print("[v2] loading inputs ...", flush=True)
    events = pd.read_parquet(EVENTS_PATH)
    markets = pd.read_parquet(MARKETS_PATH)
    candles = pd.read_parquet(CANDLES_PATH)
    hourly = pd.read_parquet(HOURLY_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    dx_meta = json.loads(DX_META.read_text())
    fcols = dx_meta["feature_cols"]   # includes "hours_to_settle"
    hourly_fcols = json.loads(HOURLY_META.read_text())["feature_cols"]
    print(f"  events {len(events):,}, markets {len(markets):,}, candles {len(candles):,}", flush=True)
    print(f"  dx feature_cols: {len(fcols)} (last={fcols[-1]})", flush=True)

    # daily extremes truth
    h = hourly[["hour", "temp_f"]].dropna(subset=["temp_f"]).copy()
    h["day"] = h["hour"].dt.floor("D")
    daily = h.groupby("day")["temp_f"].agg(actual_high="max", actual_low="min").reset_index()

    # Join events
    events_idx = events.set_index("event_ticker")[["strike_date", "title"]]
    m = markets.merge(events_idx, left_on="event_ticker", right_index=True, how="left")
    m["side"] = np.where(m["_series_ticker"] == "KXLOWTSFO", "LOW", "HIGH")
    strike_date_pst = m["strike_date"].apply(utc_to_pst)
    m["day_D"] = (strike_date_pst - pd.Timedelta(days=1)).dt.floor("D")
    settled = m[m["status"].eq("finalized")].copy()
    print(f"[v2] finalized markets: {len(settled):,}", flush=True)

    # Candle x market join
    cand = candles.merge(settled[["ticker", "event_ticker", "_series_ticker",
                                  "floor_strike", "cap_strike", "strike_type",
                                  "result", "open_time", "close_time", "side", "day_D"]],
                         on="ticker", how="inner")
    cand["decision_time"] = cand["end_time"].apply(utc_to_pst).dt.floor("h")
    # k = (day_D - day(decision_time)) in days
    cand["day_t"] = cand["decision_time"].dt.floor("D")
    cand["k"] = ((cand["day_D"] - cand["day_t"]).dt.total_seconds() / 86400).round().astype("Int64")
    # hours_to_settle: midnight after day D in PST minus decision_time
    settle_ts = cand["day_D"] + pd.Timedelta(days=1)
    cand["hours_to_settle"] = ((settle_ts - cand["decision_time"]).dt.total_seconds() / 3600.0)
    cand["hours_to_close"] = (cand["close_time"] - cand["end_time"]).dt.total_seconds() / 3600

    # Drop bad rows
    cand = cand.dropna(subset=["k", "hours_to_settle"]).copy()
    cand = cand[cand["k"].between(0, 3)].copy()
    cand["k"] = cand["k"].astype(int)
    print(f"[v2] (candle×market) rows: {len(cand):,}", flush=True)

    # Build feature matrix per row: pull features at decision_time and add
    # hours_to_settle at the end.
    feat_idx = features.set_index("hour")
    # Reindex by decision_time
    dec_index = pd.Index(cand["decision_time"].values)
    feat_aligned = feat_idx.reindex(dec_index)
    feat_aligned = feat_aligned.reset_index(drop=True)
    feat_aligned["hours_to_settle"] = cand["hours_to_settle"].values

    valid_mask = feat_aligned[hourly_fcols[0]].notna().values  # rough proxy for "decision_time exists"
    n_valid = int(valid_mask.sum())
    print(f"[v2] decision_time matched in features: {n_valid}/{len(cand)}", flush=True)

    # Predict quantiles per kind
    qpred_high = np.full((len(cand), len(QUANTILES)), np.nan)
    qpred_low = np.full((len(cand), len(QUANTILES)), np.nan)

    if n_valid > 0:
        X_all = to_f32(feat_aligned, fcols).values
        X_valid = X_all[valid_mask]
        # Predict both kinds
        for kind, target in [("high", qpred_high), ("low", qpred_low)]:
            for j, q in enumerate(QUANTILES):
                mpath = MODEL_DIR / f"dxmodel_{kind}_q{int(q*100):02d}.joblib"
                m = joblib.load(mpath)
                yhat = m.predict(X_valid)
                target_idx = np.where(valid_mask)[0]
                target[target_idx, j] = yhat
            print(f"[v2] predicted {kind} quantiles", flush=True)

    qpred_high = isotonic_monotone(qpred_high)
    qpred_low = isotonic_monotone(qpred_low)
    qs_arr = np.array(QUANTILES)

    # Pick the right kind per row
    is_high = (cand["side"].values == "HIGH")
    qpred = np.where(is_high[:, None], qpred_high, qpred_low)

    for j, q in enumerate(QUANTILES):
        cand[f"q{int(q*100):02d}"] = qpred[:, j]

    # Compute model_prob_yes
    floor = cand["floor_strike"].astype("float64").values
    cap = cand["cap_strike"].astype("float64").values
    stype = cand["strike_type"].values
    model_prob = np.full(len(cand), np.nan)
    for i in range(len(cand)):
        if np.isnan(qpred[i, 0]):
            continue
        if stype[i] == "greater":
            # YES if temp > floor (strictly). NWS rounds to integer °F.
            # P(temp > floor) = 1 - F(floor + 0.5)
            model_prob[i] = 1.0 - cdf_at_value(qpred[i], qs_arr, floor[i] + 0.5)
        elif stype[i] == "less":
            # YES if temp < cap (strictly). P(temp < cap) = F(cap - 0.5)
            model_prob[i] = cdf_at_value(qpred[i], qs_arr, cap[i] - 0.5)
        elif stype[i] == "between":
            model_prob[i] = (cdf_at_value(qpred[i], qs_arr, cap[i] + 0.5)
                             - cdf_at_value(qpred[i], qs_arr, floor[i] - 0.5))
            model_prob[i] = max(0.0, min(1.0, model_prob[i]))
    cand["model_prob_yes"] = model_prob

    # Market state
    cand["market_yes_close"] = cand["price_close"]
    cand["market_yes_bid"] = cand["yes_bid_close"]
    cand["market_yes_ask"] = cand["yes_ask_close"]
    cand["spread"] = (cand["yes_ask_close"] - cand["yes_bid_close"])
    cand["edge"] = cand["model_prob_yes"] - cand["market_yes_close"]

    # Truth + outcome
    cand = cand.merge(daily, left_on="day_D", right_on="day", how="left").drop(columns=["day"])
    cand["actual_value"] = np.where(cand["side"] == "HIGH", cand["actual_high"], cand["actual_low"])

    # Derive yes_outcome from truth
    derived = np.zeros(len(cand), dtype=int) - 1
    for i in range(len(cand)):
        v = cand["actual_value"].iloc[i]
        if pd.isna(v):
            continue
        st = cand["strike_type"].iloc[i]
        if st == "greater":
            derived[i] = int(v > cand["floor_strike"].iloc[i])
        elif st == "less":
            derived[i] = int(v < cand["cap_strike"].iloc[i])
        elif st == "between":
            derived[i] = int((v >= cand["floor_strike"].iloc[i]) and (v <= cand["cap_strike"].iloc[i]))
    cand["yes_outcome_derived"] = derived

    keep = [
        "ticker", "event_ticker", "_series_ticker", "side", "day_D", "k",
        "strike_type", "floor_strike", "cap_strike",
        "decision_time", "hours_to_settle", "hours_to_close",
        "price_close", "price_open", "price_mean",
        "yes_bid_close", "yes_ask_close", "spread",
        "open_interest", "volume",
        "q05","q10","q25","q50","q75","q90","q95",
        "model_prob_yes", "market_yes_close",
        "market_yes_bid", "market_yes_ask", "edge",
        "actual_value", "actual_high", "actual_low",
        "result", "yes_outcome_derived",
    ]
    out = cand[keep].rename(columns={"_series_ticker": "series_ticker"}).copy()
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n[v2] wrote {OUT_PATH} ({len(out):,} rows)", flush=True)

    # Quality checks
    elig = out.dropna(subset=["model_prob_yes", "market_yes_close"])
    elig = elig[elig["yes_outcome_derived"].isin([0, 1])]
    print(f"[v2] eligible rows: {len(elig):,}", flush=True)

    eps = 1e-6
    p_m = np.clip(elig["model_prob_yes"].values, eps, 1-eps)
    p_k = np.clip(elig["market_yes_close"].values, eps, 1-eps)
    y = elig["yes_outcome_derived"].values.astype(float)
    ll_m = float(-np.mean(y * np.log(p_m) + (1-y) * np.log(1-p_m)))
    ll_k = float(-np.mean(y * np.log(p_k) + (1-y) * np.log(1-p_k)))
    brier_m = float(np.mean((p_m - y) ** 2))
    brier_k = float(np.mean((p_k - y) ** 2))
    print(f"\n=== ALL ROWS ===")
    print(f"  log-loss   model={ll_m:.4f}   market={ll_k:.4f}   skill={1 - ll_m/ll_k:+.3f}")
    print(f"  Brier      model={brier_m:.4f} market={brier_k:.4f} skill={1 - brier_m/brier_k:+.3f}")

    print("\n=== BY STRIKE TYPE ===")
    for st in ["greater", "less", "between"]:
        sub = elig[elig["strike_type"] == st]
        if len(sub) < 10:
            continue
        p_m = np.clip(sub["model_prob_yes"].values, eps, 1-eps)
        p_k = np.clip(sub["market_yes_close"].values, eps, 1-eps)
        y = sub["yes_outcome_derived"].values.astype(float)
        ll_m = float(-np.mean(y * np.log(p_m) + (1-y) * np.log(1-p_m)))
        ll_k = float(-np.mean(y * np.log(p_k) + (1-y) * np.log(1-p_k)))
        print(f"  {st:8s}  n={len(sub):>5,}  yes_rate={y.mean():.3f}  "
              f"model_p_mean={sub['model_prob_yes'].mean():.3f}  "
              f"market_p_mean={sub['market_yes_close'].mean():.3f}  "
              f"LL: model={ll_m:.4f} market={ll_k:.4f}")

    print("\n=== BY HOURS-TO-SETTLE ===")
    bins = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 96)]
    for lo, hi in bins:
        sub = elig[(elig["hours_to_settle"] >= lo) & (elig["hours_to_settle"] < hi)]
        if len(sub) < 10:
            continue
        p_m = np.clip(sub["model_prob_yes"].values, eps, 1-eps)
        p_k = np.clip(sub["market_yes_close"].values, eps, 1-eps)
        y = sub["yes_outcome_derived"].values.astype(float)
        ll_m = float(-np.mean(y * np.log(p_m) + (1-y) * np.log(1-p_m)))
        ll_k = float(-np.mean(y * np.log(p_k) + (1-y) * np.log(1-p_k)))
        print(f"  [{lo:>2}h–{hi:>2}h)  n={len(sub):>5,}  LL: model={ll_m:.4f} market={ll_k:.4f}  "
              f"skill={1 - ll_m/ll_k:+.3f}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("# Decision Dataset v2 Summary\n\n"
                            f"- Rows: **{len(out):,}**\n"
                            f"- Eligible: **{len(elig):,}**\n"
                            f"- Overall LL — model {ll_m:.4f} vs market {ll_k:.4f}\n")


if __name__ == "__main__":
    main()
