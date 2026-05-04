"""
09_daily_extreme_train.py — Train quantile models for SFO daily HIGH and LOW.

For Kalshi-style daily-temperature contracts, what matters is the daily extreme
(max or min) of a calendar day, NOT the temperature at a specific hour. Our
hourly forecasting models from `04_train.py` are tuned for hourly temperature;
mapping them onto daily-extreme contracts via a single fixed offset (e.g.,
"add 2.6°F to predicted 15:00 temp") loses information and is biased toward
the off-peak hour.

This script trains a separate set of quantile models targeting actual daily
extremes:

  Target_HIGH(t, D) = max of temp_f over PST calendar day D
  Target_LOW(t,  D) = min of temp_f over PST calendar day D

For each training row at hour t, and each settlement-day offset k ∈ {0,1,2,3}
(today through 3 days ahead), we create a training pair with:
  * Existing 250 features at t (lags, rolling, climatology, marine-layer, …)
  * Extra feature `hours_to_settlement` = (midnight after day D) − t in hours
  * Target = the actual daily extreme of day D

We train 7 quantiles × 2 kinds = 14 models with HistGradientBoostingRegressor.
NaN handling is native; horizons are encoded as a single feature.

Outputs:
  models/dxmodel_{kind}_q{Q}.joblib   for kind ∈ {high, low}, Q ∈ {05,10,...,95}
  reports/daily_extreme_metrics.json
  reports/daily_extreme_summary.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/Users/terrykim/Documents/SF Weather")
HOURLY_PATH = ROOT / "data" / "sfo_hourly.parquet"
FEATURES_PATH = ROOT / "data" / "sfo_features.parquet"
TRAIN_META_PATH = ROOT / "reports" / "train_metrics.json"

MODEL_DIR = ROOT / "models"
METRICS_OUT = ROOT / "reports" / "daily_extreme_metrics.json"
SUMMARY_OUT = ROOT / "reports" / "daily_extreme_summary.md"

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
KIND_DAY_OFFSETS = [0, 1, 2, 3]  # day(t)+k as the target day D

TRAIN_END = pd.Timestamp("2019-12-31 23:00:00")
VAL_END = pd.Timestamp("2022-12-31 23:00:00")  # test starts 2023-01-01

HGB_PARAMS = dict(
    loss="quantile",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=63,
    min_samples_leaf=120,    # bigger leaves than hourly model — fewer training rows
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)


def to_f32(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for c in cols:
        s = d[c]
        if pd.api.types.is_extension_array_dtype(s) or s.dtype == "float64":
            out[c] = s.astype("float32")
        else:
            out[c] = s
    return out


def pinball(y, yhat, q):
    e = y - yhat
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def coverage(y, yhat, q):
    return float(np.mean(y <= yhat))


def main():
    print("[dx] loading inputs ...", flush=True)
    hourly = pd.read_parquet(HOURLY_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    meta = json.loads(TRAIN_META_PATH.read_text())
    fcols = meta["feature_cols"]
    print(f"[dx] hourly {len(hourly):,}, features {features.shape}", flush=True)

    # 1. Compute daily extremes (PST calendar day) on the hourly grid.
    h = hourly[["hour", "temp_f"]].dropna(subset=["temp_f"]).copy()
    h["day"] = h["hour"].dt.floor("D")
    daily = h.groupby("day")["temp_f"].agg(daily_high="max", daily_low="min", n_obs="size").reset_index()
    print(f"[dx] daily extremes: {len(daily):,} days", flush=True)

    # 2. For each (t, k) pair, build (features_at_t, target_day, hours_to_settle, target_value)
    #    target_day = day(t) + k.  hours_to_settle = (target_day + 1d 00:00 PST) - t in hours.
    print(f"[dx] building (t, k) training pairs for k ∈ {KIND_DAY_OFFSETS} ...", flush=True)

    feat = features.copy()
    feat["day"] = feat["hour"].dt.floor("D")
    feat = feat.merge(daily, on="day", how="left")  # adds today's high/low (k=0)
    feat = feat.rename(columns={"daily_high": "target_high_k0",
                                "daily_low":  "target_low_k0",
                                "n_obs":      "_today_n_obs"})

    # Also bring in target_high/low for k=1,2,3 by shifting `daily` by k days
    for k in [1, 2, 3]:
        d_shift = daily.copy()
        d_shift["day"] = d_shift["day"] - pd.Timedelta(days=k)  # value at "day" represents target on day+k
        d_shift = d_shift[["day", "daily_high", "daily_low"]].rename(columns={
            "daily_high": f"target_high_k{k}",
            "daily_low":  f"target_low_k{k}",
        })
        feat = feat.merge(d_shift, on="day", how="left")
    feat = feat.drop(columns=["_today_n_obs"])

    # Build a long table: one row per (hour, kind, k)
    long_rows = []
    for kind in ["high", "low"]:
        for k in KIND_DAY_OFFSETS:
            target_col = f"target_{kind}_k{k}"
            sub = feat[["hour"] + fcols + [target_col]].copy()
            sub = sub.rename(columns={target_col: "y"})
            sub["k"] = k
            sub["kind"] = kind
            # hours_to_settle = midnight AFTER target day D in PST minus t
            target_day = sub["hour"].dt.floor("D") + pd.Timedelta(days=k)
            settle_ts = target_day + pd.Timedelta(days=1)  # next midnight PST
            sub["hours_to_settle"] = (settle_ts - sub["hour"]).dt.total_seconds() / 3600.0
            long_rows.append(sub)

    long = pd.concat(long_rows, ignore_index=True)
    print(f"[dx] long table rows: {len(long):,}", flush=True)
    # Drop rows with NaN target
    long = long.dropna(subset=["y"]).reset_index(drop=True)
    # Drop early warmup rows (need lag features)
    earliest = long["hour"].min() + pd.Timedelta(hours=336 + 24)
    long = long[long["hour"] >= earliest].copy()
    print(f"[dx] after warmup drop: {len(long):,}", flush=True)

    # Train per kind
    train_metrics = {}
    feature_cols_used = fcols + ["hours_to_settle"]

    for kind in ["high", "low"]:
        print(f"\n[dx] === kind={kind.upper()} ===", flush=True)
        sub = long[long["kind"] == kind].copy()
        train = sub[sub["hour"] <= TRAIN_END].copy()
        val = sub[(sub["hour"] > TRAIN_END) & (sub["hour"] <= VAL_END)].copy()
        print(f"  train rows {len(train):,}, val rows {len(val):,}", flush=True)

        X_train = to_f32(train, feature_cols_used).values
        y_train = train["y"].values.astype("float32")
        X_val = to_f32(val, feature_cols_used).values
        y_val = val["y"].values.astype("float32")

        kmetrics = {"train_rows": len(train), "val_rows": len(val)}
        for q in QUANTILES:
            t0 = time.time()
            params = dict(HGB_PARAMS); params["quantile"] = q
            m = HistGradientBoostingRegressor(**params)
            m.fit(X_train, y_train)
            yp_train = m.predict(X_train)
            yp_val = m.predict(X_val)
            pl_t = pinball(y_train, yp_train, q)
            pl_v = pinball(y_val, yp_val, q)
            cov_v = coverage(y_val, yp_val, q)
            elapsed = time.time() - t0
            print(f"    q={q:.2f}  iters={m.n_iter_:3d}  pinball train={pl_t:.4f} val={pl_v:.4f}  "
                  f"coverage={cov_v:.3f}  {elapsed:.1f}s", flush=True)
            mpath = MODEL_DIR / f"dxmodel_{kind}_q{int(q*100):02d}.joblib"
            joblib.dump(m, mpath)
            kmetrics[f"q{int(q*100):02d}"] = {
                "iters": int(m.n_iter_),
                "pinball_train": pl_t,
                "pinball_val": pl_v,
                "coverage_val": cov_v,
                "fit_seconds": elapsed,
                "model_path": str(mpath.relative_to(ROOT)),
            }
        train_metrics[kind] = kmetrics

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps({
        "quantiles": QUANTILES,
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
        "feature_cols": feature_cols_used,
        "metrics": train_metrics,
    }, indent=2))
    print(f"\n[dx] wrote {METRICS_OUT}", flush=True)

    # Summary
    lines = ["# Daily Extreme Quantile Training Summary\n",
             f"- Train ends: {TRAIN_END}, Val ends: {VAL_END}",
             f"- Quantiles: {QUANTILES}",
             ""]
    for kind in ["high", "low"]:
        lines.append(f"## {kind.upper()}\n")
        lines.append("| q | iters | pinball train | pinball val | coverage |")
        lines.append("|---:|---:|---:|---:|---:|")
        for q in QUANTILES:
            r = train_metrics[kind][f"q{int(q*100):02d}"]
            lines.append(f"| {q:.2f} | {r['iters']} | {r['pinball_train']:.4f} | "
                         f"{r['pinball_val']:.4f} | {r['coverage_val']:.3f} |")
        lines.append("")
    SUMMARY_OUT.write_text("\n".join(lines))
    print(f"[dx] wrote {SUMMARY_OUT}", flush=True)


if __name__ == "__main__":
    main()
