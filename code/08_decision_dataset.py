"""
08_decision_dataset.py — Build the joint decision-time dataset.

For every (market_ticker, decision_time_t) pair from the Kalshi candle history,
produce ONE row containing:
  * Market metadata (event, settlement day, strike side, floor/cap, hours-to-close)
  * Market state at t (yes_mid, bid, ask, spread, volume, open_interest, momentum)
  * Truth (actual daily high or low on day D, derived from NOAA data)
  * Our model's quantile forecast for day-D's daily extreme, mapped from the
    49-quantile hourly forecast via a calibrated "afternoon-peak → daily-high"
    correction (or "early-morning → daily-low").
  * Derived: model_prob_yes for the market's strike + edge vs market mid.

Output: data/decision_dataset.parquet  (one row per opportunity)

Inputs:
  data/kalshi_events.parquet
  data/kalshi_markets.parquet
  data/kalshi_candles.parquet
  data/sfo_hourly.parquet
  data/sfo_features.parquet
  reports/train_metrics.json
  models/qmodel_h{H}_q{Q}.joblib   (49 models)

Conventions:
  * All timestamps in our pipeline are PST (UTC-8). Kalshi candle end_period_ts
    is Unix seconds (UTC) — we convert to PST consistently.
  * "Daily high for day D" uses the PST calendar day of D as the integration
    window (max of hourly temp_f over D 00:00 PST to D 23:59 PST). NWS uses
    local clock time which during DST shifts by 1h; the impact on day-max is
    negligible because the peak is well inside the day window.
"""

from __future__ import annotations

import json
import time
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
TRAIN_META_PATH = ROOT / "reports" / "train_metrics.json"
MODEL_DIR = ROOT / "models"

OUT_PATH = ROOT / "data" / "decision_dataset.parquet"
SUMMARY_PATH = ROOT / "reports" / "decision_dataset_summary.md"

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Target hour-of-day in PST that we use as the proxy for the daily extreme.
# 15:00 PST (= 4pm PDT in summer, 3pm PST in winter) is roughly the SFO daily
# peak. 05:00 PST (= 6am PDT in summer, 5am in winter) is roughly the daily low.
TARGET_HOUR_HIGH = 15
TARGET_HOUR_LOW = 5


# ---------- helpers ----------

def utc_to_pst(ts_utc) -> pd.Timestamp:
    t = pd.Timestamp(ts_utc)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t - pd.Timedelta(hours=8)


def snap_horizon(h: float) -> int | None:
    """Snap a hours-ahead value to the nearest available model horizon."""
    if pd.isna(h) or h < 0:
        return None
    options = np.array(HORIZONS)
    idx = int(np.argmin(np.abs(options - h)))
    return int(options[idx])


def isotonic_monotone(qpred: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(qpred, axis=1)


def cdf_at_value(qpred_row: np.ndarray, qs: np.ndarray, x: float) -> float:
    """Linear interpolation of CDF at value x. qpred_row is 7 quantile preds in
    ascending q order; qs is the corresponding q values. Returns F(x) ∈ [0,1]."""
    return float(np.interp(x, qpred_row, qs, left=0.0, right=1.0))


def shift_quantiles(qpred: np.ndarray, mean_shift: float, std_inflate: float) -> np.ndarray:
    """Shift each quantile by mean_shift and inflate standard deviation around
    the median by `std_inflate` (>= 1). For std_inflate=1.0 just adds a constant.
    For std_inflate>1, broadens the spread symmetrically around q=0.5."""
    qpred = qpred.copy()
    median = qpred[:, QUANTILES.index(0.50)]
    if std_inflate != 1.0:
        for j, q in enumerate(QUANTILES):
            qpred[:, j] = median + (qpred[:, j] - median) * std_inflate
    return qpred + mean_shift


# ---------- daily extremes (truth) ----------

def compute_daily_extremes(hourly: pd.DataFrame) -> pd.DataFrame:
    """For each PST calendar day, compute max and min of hourly temp_f."""
    df = hourly[["hour", "temp_f"]].dropna(subset=["temp_f"]).copy()
    df["day"] = df["hour"].dt.floor("D")
    g = df.groupby("day")["temp_f"]
    out = pd.DataFrame({
        "day": g.size().index,
        "actual_high": g.max().values,
        "actual_low":  g.min().values,
        "n_obs":       g.size().values,
    })
    return out


# ---------- model prediction batching ----------

def to_f32_matrix(features: pd.DataFrame, fcols: list[str]) -> np.ndarray:
    out = pd.DataFrame(index=features.index)
    for c in fcols:
        s = features[c]
        if pd.api.types.is_extension_array_dtype(s) or s.dtype == "float64":
            out[c] = s.astype("float32")
        else:
            out[c] = s
    return out.values


def predict_quantiles(features: pd.DataFrame, fcols: list[str],
                      decision_hours: pd.Series, horizons: pd.Series) -> np.ndarray:
    """Return (n, 7) array of quantile predictions, one row per (decision_time,
    horizon) pair. We group by horizon to batch model loading & prediction."""
    n = len(decision_hours)
    # Map decision_time → row index in features
    feat_index = features.set_index("hour").index
    feat_loc = pd.Index(feat_index).get_indexer(decision_hours)
    valid_mask = feat_loc >= 0
    print(f"[decision] {valid_mask.sum()}/{n} decision times found in features", flush=True)

    out = np.full((n, len(QUANTILES)), np.nan)
    if valid_mask.sum() == 0:
        return out

    feat_values = to_f32_matrix(features, fcols)
    rows_X = feat_values[feat_loc[valid_mask]]   # (n_valid, n_features)

    for h in HORIZONS:
        sub = (horizons.values == h) & valid_mask
        if not sub.any():
            continue
        # Map sub-mask onto rows_X (which is in valid_mask order)
        sub_in_valid = sub[valid_mask]
        X = rows_X[sub_in_valid]
        for j, q in enumerate(QUANTILES):
            mpath = MODEL_DIR / f"qmodel_h{h}_q{int(q*100):02d}.joblib"
            m = joblib.load(mpath)
            yhat = m.predict(X)
            # write back into out at the correct rows
            target_idx = np.where(sub)[0]
            out[target_idx, j] = yhat
        print(f"  horizon h={h}: {sub.sum()} predictions", flush=True)

    return isotonic_monotone(out)


# ---------- daily-extreme correction (afternoon-peak vs daily-max) ----------

def fit_extreme_correction(hourly: pd.DataFrame, target_hour: int,
                           kind: str, fit_until: pd.Timestamp) -> tuple[float, float]:
    """For each calendar day, compute the value of temp_f at `target_hour` PST
    and the daily extreme (max or min). Return (mean, std) of (extreme - target)
    on data up to fit_until.
    These are used to shift our forecast quantiles from "temp at target_hour" to
    "daily extreme"."""
    df = hourly[["hour", "temp_f"]].dropna(subset=["temp_f"]).copy()
    df = df[df["hour"] <= fit_until].copy()
    df["day"] = df["hour"].dt.floor("D")
    df["hod"] = df["hour"].dt.hour

    # Pivot: per day, the target-hour value
    target_df = df[df["hod"] == target_hour].set_index("day")["temp_f"].rename("target_val")
    # Daily extreme
    if kind == "high":
        ext = df.groupby("day")["temp_f"].max().rename("extreme")
    else:
        ext = df.groupby("day")["temp_f"].min().rename("extreme")

    j = pd.concat([target_df, ext], axis=1).dropna()
    delta = (j["extreme"] - j["target_val"]).values
    mean = float(np.mean(delta))
    std = float(np.std(delta))
    print(f"[decision] correction kind={kind} target_hour={target_hour}: "
          f"mean={mean:+.2f}°F  std={std:.2f}°F  (n={len(delta):,} days, fit ≤ {fit_until.date()})",
          flush=True)
    return mean, std


# ---------- main ----------

def main():
    print("[decision] loading inputs...", flush=True)
    events = pd.read_parquet(EVENTS_PATH)
    markets = pd.read_parquet(MARKETS_PATH)
    candles = pd.read_parquet(CANDLES_PATH)
    hourly = pd.read_parquet(HOURLY_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    meta = json.loads(TRAIN_META_PATH.read_text())
    fcols = meta["feature_cols"]

    print(f"  events {len(events):,}, markets {len(markets):,}, candles {len(candles):,}", flush=True)

    # Daily extremes (truth) on PST calendar days
    daily_ext = compute_daily_extremes(hourly)
    print(f"[decision] daily extremes computed for {len(daily_ext):,} days", flush=True)

    # Fit "afternoon → daily-max" and "early-morning → daily-min" corrections
    # using training-window data only.
    fit_until = pd.Timestamp("2019-12-31 23:00:00")
    high_shift, _ = fit_extreme_correction(hourly, TARGET_HOUR_HIGH, "high", fit_until)
    low_shift, _  = fit_extreme_correction(hourly, TARGET_HOUR_LOW,  "low",  fit_until)
    # std_inflate: for now we keep 1.0 (just shift). Calibration has shown that
    # the daily-extreme distribution is similar in width to the single-hour
    # distribution in SF (the diurnal cycle is tight). std_inflate=1.05
    # adds a small amount of broadening.
    std_inflate = 1.05

    # Join markets with events to get strike_date
    events_idx = events.set_index("event_ticker")[["strike_date", "title"]]
    m = markets.merge(events_idx, left_on="event_ticker", right_index=True, how="left")
    print(f"[decision] markets joined to events: {len(m):,}", flush=True)

    # We use HIGH series for high-temp markets, LOW for low-temp.
    m["side"] = np.where(m["_series_ticker"] == "KXLOWTSFO", "LOW", "HIGH")

    # The "settlement day D" in PST: the event title says e.g. "May 2, 2026" and
    # strike_date is the close ts (e.g., May 3 08:00 UTC = May 3 00:00 PST).
    # The day-D in PST is strike_date - 1 day.
    strike_date_pst = m["strike_date"].apply(utc_to_pst)
    m["day_D"] = (strike_date_pst - pd.Timedelta(days=1)).dt.floor("D")

    # Keep only finalized markets with a real outcome for backtest table
    settled = m[m["status"].eq("finalized")].copy()
    print(f"[decision] finalized markets: {len(settled):,}", flush=True)

    # Build (decision_time, market) pairs from candles
    cand = candles.merge(settled[["ticker", "event_ticker", "_series_ticker",
                                  "floor_strike", "cap_strike", "strike_type",
                                  "result", "open_time", "close_time", "side", "day_D"]],
                         on="ticker", how="inner")
    print(f"[decision] candle×market pairs: {len(cand):,}", flush=True)

    # Decision time in PST = end_time(UTC) - 8h
    cand["decision_time"] = cand["end_time"].apply(utc_to_pst).dt.floor("h")
    # Hours from decision_time to settlement-target hour
    target_hod = np.where(cand["side"].values == "HIGH", TARGET_HOUR_HIGH, TARGET_HOUR_LOW)
    target_time = cand["day_D"] + pd.to_timedelta(target_hod, unit="h")
    cand["target_time"] = target_time
    cand["raw_horizon_hours"] = (cand["target_time"] - cand["decision_time"]).dt.total_seconds() / 3600
    cand["snapped_horizon"] = cand["raw_horizon_hours"].apply(snap_horizon).astype("Int64")
    # Also keep close-time-relative (hours-to-close) for the trading backtest
    cand["hours_to_close"] = (cand["close_time"] - cand["end_time"]).dt.total_seconds() / 3600

    # Drop rows where we can't predict (no horizon / decision in future)
    cand = cand.dropna(subset=["snapped_horizon"]).copy()
    cand["snapped_horizon"] = cand["snapped_horizon"].astype(int)
    print(f"[decision] rows with valid horizon: {len(cand):,}", flush=True)

    # Predict quantiles per row
    print("[decision] predicting quantiles ...", flush=True)
    qpred = predict_quantiles(features, fcols, cand["decision_time"], cand["snapped_horizon"])

    # Apply daily-extreme correction (HIGH vs LOW) per row
    is_high = (cand["side"].values == "HIGH")
    # Inflate std (multiplicative around median)
    median = qpred[:, QUANTILES.index(0.50)]
    qpred_inflated = qpred.copy()
    for j in range(len(QUANTILES)):
        qpred_inflated[:, j] = median + (qpred[:, j] - median) * std_inflate
    # Apply per-row shift
    shift = np.where(is_high, high_shift, low_shift)
    qpred_corrected = qpred_inflated + shift[:, None]

    for j, q in enumerate(QUANTILES):
        cand[f"q{int(q*100):02d}"] = qpred_corrected[:, j]

    # Compute model_prob_yes for each row's strike
    floor = cand["floor_strike"].astype("float64").values
    cap = cand["cap_strike"].astype("float64").values
    stype = cand["strike_type"].values
    model_prob = np.full(len(cand), np.nan)
    qs_arr = np.array(QUANTILES)
    for i in range(len(cand)):
        if np.isnan(qpred_corrected[i, 0]):
            continue
        if stype[i] == "greater":
            # YES if temp > floor
            model_prob[i] = 1.0 - cdf_at_value(qpred_corrected[i], qs_arr, floor[i])
        elif stype[i] == "less":
            # YES if temp < cap
            model_prob[i] = cdf_at_value(qpred_corrected[i], qs_arr, cap[i])
        elif stype[i] == "between":
            # YES if floor <= temp <= cap. Daily-high temps are integer-valued
            # in NWS reports, so we interpret strictly:
            # floor=61, cap=62 means YES iff (60 < temp <= 62) within ±0.5 rounding.
            # Use F(cap+0.5) - F(floor-0.5).
            model_prob[i] = (cdf_at_value(qpred_corrected[i], qs_arr, cap[i] + 0.5)
                             - cdf_at_value(qpred_corrected[i], qs_arr, floor[i] - 0.5))
            model_prob[i] = max(0.0, min(1.0, model_prob[i]))
    cand["model_prob_yes"] = model_prob

    # Market price at decision time = candle close price (mid).
    # Use price_close as the "settled" price for that hour. Compute spread, etc.
    cand["market_yes_close"] = cand["price_close"]   # already in dollars (0..1)
    cand["market_yes_bid"] = cand["yes_bid_close"]
    cand["market_yes_ask"] = cand["yes_ask_close"]
    cand["spread"] = (cand["yes_ask_close"] - cand["yes_bid_close"])
    cand["edge"] = cand["model_prob_yes"] - cand["market_yes_close"]

    # Add the eventual truth (actual daily extreme) and result
    cand = cand.merge(daily_ext[["day", "actual_high", "actual_low"]],
                      left_on="day_D", right_on="day", how="left").drop(columns=["day"])
    cand["actual_value"] = np.where(cand["side"]=="HIGH", cand["actual_high"], cand["actual_low"])

    # YES outcome from market metadata (already in `result`); also derive from truth as a check
    cand["yes_outcome"] = (cand["result"] == "yes").astype("Int8")
    derived_yes = np.zeros(len(cand), dtype=int)
    for i in range(len(cand)):
        v = cand["actual_value"].iloc[i]
        if pd.isna(v):
            derived_yes[i] = -1   # unknown
            continue
        st = cand["strike_type"].iloc[i]
        if st == "greater":
            derived_yes[i] = int(v > cand["floor_strike"].iloc[i])
        elif st == "less":
            derived_yes[i] = int(v < cand["cap_strike"].iloc[i])
        elif st == "between":
            derived_yes[i] = int((v >= cand["floor_strike"].iloc[i]) and (v <= cand["cap_strike"].iloc[i]))
    cand["yes_outcome_derived"] = derived_yes

    # Trim columns we don't need
    keep = [
        "ticker", "event_ticker", "_series_ticker", "side", "day_D",
        "strike_type", "floor_strike", "cap_strike",
        "decision_time", "target_time", "raw_horizon_hours", "snapped_horizon",
        "hours_to_close",
        "price_close", "price_open", "price_mean",
        "yes_bid_close", "yes_ask_close", "spread",
        "open_interest", "volume",
        "q05","q10","q25","q50","q75","q90","q95",
        "model_prob_yes", "market_yes_close",
        "market_yes_bid", "market_yes_ask", "edge",
        "actual_value", "actual_high", "actual_low",
        "result", "yes_outcome", "yes_outcome_derived",
    ]
    out = cand[keep].copy()
    out = out.rename(columns={"_series_ticker": "series_ticker"})
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n[decision] wrote {OUT_PATH} ({len(out):,} rows, {out.shape[1]} cols)", flush=True)

    # Validation: rows with valid model_prob and outcome
    eligible = out.dropna(subset=["model_prob_yes", "market_yes_close"])
    eligible = eligible[eligible["yes_outcome_derived"].isin([0, 1])]
    print(f"[decision] eligible for backtest: {len(eligible):,}", flush=True)

    # Quick log-loss comparison: model vs market
    eps = 1e-6
    p_m = np.clip(eligible["model_prob_yes"].values, eps, 1-eps)
    p_k = np.clip(eligible["market_yes_close"].values, eps, 1-eps)
    y = eligible["yes_outcome_derived"].values.astype(float)
    ll_m = float(-np.mean(y * np.log(p_m) + (1-y) * np.log(1-p_m)))
    ll_k = float(-np.mean(y * np.log(p_k) + (1-y) * np.log(1-p_k)))
    brier_m = float(np.mean((p_m - y) ** 2))
    brier_k = float(np.mean((p_k - y) ** 2))
    print(f"\n=== ALL ROWS (no time-of-decision filter) ===")
    print(f"  log-loss   model={ll_m:.4f}   market={ll_k:.4f}   skill={1 - ll_m/ll_k:+.3f}")
    print(f"  Brier      model={brier_m:.4f} market={brier_k:.4f} skill={1 - brier_m/brier_k:+.3f}")

    # Per snapped_horizon bucket
    print("\n=== BY HORIZON ===")
    for h in HORIZONS:
        sub = eligible[eligible["snapped_horizon"] == h]
        if len(sub) < 10:
            continue
        p_m = np.clip(sub["model_prob_yes"].values, eps, 1-eps)
        p_k = np.clip(sub["market_yes_close"].values, eps, 1-eps)
        y = sub["yes_outcome_derived"].values.astype(float)
        ll_m = float(-np.mean(y * np.log(p_m) + (1-y) * np.log(1-p_m)))
        ll_k = float(-np.mean(y * np.log(p_k) + (1-y) * np.log(1-p_k)))
        print(f"  h={h:>2}h  n={len(sub):>5,}  model_LL={ll_m:.4f}  market_LL={ll_k:.4f}  skill={1 - ll_m/ll_k:+.3f}")

    # Quick markdown summary
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Decision Dataset Summary\n",
             f"- Rows: **{len(out):,}**",
             f"- Eligible for backtest: **{len(eligible):,}**",
             f"- Horizons used: {HORIZONS}",
             f"- High-extreme correction: target_hour={TARGET_HOUR_HIGH}, mean_shift={high_shift:+.2f}°F",
             f"- Low-extreme correction: target_hour={TARGET_HOUR_LOW}, mean_shift={low_shift:+.2f}°F",
             f"- std_inflate factor: {std_inflate}",
             "",
             "## Log-loss skill (negative if market beats us)",
             f"- Overall: model {ll_m:.4f} vs market {ll_k:.4f}",
             ""]
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[decision] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
