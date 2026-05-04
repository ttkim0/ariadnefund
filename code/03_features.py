"""
03_features.py — Feature engineering for SFO temperature forecasting.

Reads:
  data/sfo_hourly.parquet
  data/sfo_daily.parquet

Writes:
  data/sfo_features.parquet      # one row per UTC hour t, with features known at t
  data/sfo_targets.parquet       # one row per UTC hour t, with temp_f at t+h
                                   for forecast horizons h ∈ {1,3,6,12,24,48,72} hours
  data/sfo_climatology.parquet   # hour-of-year climatology table (training-only)
  reports/feature_summary.md

Key design choices:
  * The "issuance time" of a forecast is `t`. All features at row t use only data
    observed strictly at or before t. There is NO leakage from the future.
  * The target columns (temp_f_h{H}) are simply temp_f shifted by -H hours.
    These will be predicted from the features at row t.
  * Climatology is fit only on data BEFORE the validation cutoff (so test data
    cannot contaminate the seasonal baseline). Two cutoffs are used:
      - clim_train_end:  last day used to fit climatology (default 2019-12-31)
      - The same climatology is then used to derive features for ALL rows.
  * Dew point is filled where missing using Magnus formula from temp_f + rh
    (this is a deterministic physical relation, not synthetic data).
  * Wind direction is split into onshore/offshore components specific to SFO
    geography (Pacific is to the W/NW, ~270° = pure onshore).
  * Marine-layer features: dew-point depression, low-vis flag, overcast flag.
  * Daily (prior-day) features come from SOD records — never the day's own SOD.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
HOURLY_PATH = ROOT / "data" / "sfo_hourly.parquet"
DAILY_PATH = ROOT / "data" / "sfo_daily.parquet"
FEAT_OUT = ROOT / "data" / "sfo_features.parquet"
TARG_OUT = ROOT / "data" / "sfo_targets.parquet"
CLIM_OUT = ROOT / "data" / "sfo_climatology.parquet"
SUMMARY_PATH = ROOT / "reports" / "feature_summary.md"

# Forecast horizons (hours ahead) — what we want to predict.
HORIZONS = [1, 3, 6, 12, 24, 48, 72]

# Climatology training window end (inclusive). Only data at or before this date
# is used to fit the seasonal baseline. This avoids test-set contamination.
CLIM_TRAIN_END = pd.Timestamp("2019-12-31 23:00:00")

# Lag offsets in hours (history available at time t).
LAG_HOURS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 48, 72, 168, 336]
# Rolling windows in hours.
ROLL_WINDOWS = [3, 6, 12, 24, 72, 168]


# ---------- Magnus formula for dew point fill ----------

def dew_point_from_t_rh(temp_f: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus-Tetens approximation. Inputs in F and %; output in F.
    Valid for temps -45..60 C. RH must be > 0; we clip to [1, 100]."""
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    rh_c = rh.clip(lower=1.0, upper=100.0) / 100.0
    a, b = 17.625, 243.04
    alpha = (a * t_c) / (b + t_c) + np.log(rh_c)
    dew_c = (b * alpha) / (a - alpha)
    return dew_c * 9.0 / 5.0 + 32.0


# ---------- Climatology ----------

def fit_climatology(df: pd.DataFrame, end_ts: pd.Timestamp) -> pd.DataFrame:
    """For each (month, day, hour) compute mean / std of temp_f using rows
    strictly at or before end_ts. Smooth with a rolling ±15-day, all-hours-of-day
    window (so each (mo, day, hr) cell is averaged across nearby calendar days)."""
    train = df.loc[df["hour"] <= end_ts, ["hour", "temp_f"]].dropna(subset=["temp_f"]).copy()
    train["mo"] = train["hour"].dt.month
    train["dy"] = train["hour"].dt.day
    train["hr"] = train["hour"].dt.hour

    # Pivot to month/day/hour cell mean & std
    grp = train.groupby(["mo", "dy", "hr"])["temp_f"]
    cell_mean = grp.mean().rename("clim_mean")
    cell_std = grp.std().rename("clim_std")
    cell_n = grp.size().rename("clim_n")

    # Build a 366*24 hour-of-year index (use 366 to include Feb 29).
    # We'll smooth across 31 surrounding days (centered, ±15) for each fixed hour-of-day.
    out_rows = []
    by_hour = {h: pd.DataFrame() for h in range(24)}
    cell_df = pd.concat([cell_mean, cell_std, cell_n], axis=1).reset_index()

    # Build a complete (mo, dy) grid for smoothing
    valid_md = []
    for mo in range(1, 13):
        for dy in range(1, 32):
            try:
                pd.Timestamp(year=2020, month=mo, day=dy)  # 2020 leap year, all valid
                valid_md.append((mo, dy))
            except ValueError:
                pass
    md_index = pd.MultiIndex.from_tuples(valid_md, names=["mo", "dy"])

    for hr in range(24):
        sub = cell_df[cell_df["hr"] == hr].set_index(["mo", "dy"]).reindex(md_index)
        # Sort by day-of-year for circular smoothing
        sub = sub.sort_index()
        sub["doy"] = [pd.Timestamp(2020, mo, dy).day_of_year for mo, dy in sub.index]
        sub = sub.sort_values("doy").reset_index()
        # Circular padding ±15 days
        pad = 15
        head = sub.tail(pad).copy()
        tail = sub.head(pad).copy()
        head["doy"] -= 366
        tail["doy"] += 366
        full = pd.concat([head, sub, tail], ignore_index=True)
        # Centered rolling 31-day mean (in days, but we have 1 row per day at this hour)
        smoothed_mean = full["clim_mean"].rolling(window=31, center=True, min_periods=10).mean()
        smoothed_std = full["clim_std"].rolling(window=31, center=True, min_periods=10).mean()
        sub["clim_mean_s"] = smoothed_mean.iloc[pad:pad + len(sub)].values
        sub["clim_std_s"] = smoothed_std.iloc[pad:pad + len(sub)].values
        sub["hr"] = hr
        by_hour[hr] = sub

    clim = pd.concat(by_hour.values(), ignore_index=True)
    clim = clim[["mo", "dy", "hr", "clim_mean_s", "clim_std_s", "clim_mean", "clim_std", "clim_n"]]
    clim = clim.rename(columns={"clim_mean_s": "clim_mean_smooth",
                                "clim_std_s":  "clim_std_smooth"})
    print(f"[features] climatology fit on {len(train):,} rows up to {end_ts}; "
          f"{len(clim)} (mo,dy,hr) cells", flush=True)
    return clim


def apply_climatology(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mo"] = df["hour"].dt.month
    df["dy"] = df["hour"].dt.day
    df["hr"] = df["hour"].dt.hour
    df = df.merge(clim[["mo", "dy", "hr", "clim_mean_smooth", "clim_std_smooth"]],
                  on=["mo", "dy", "hr"], how="left")
    df = df.drop(columns=["mo", "dy", "hr"])
    return df


# ---------- Lag and rolling features ----------

def add_lag_and_rolling(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    df = df.sort_values("hour").reset_index(drop=True)
    for col in base_cols:
        for h in LAG_HOURS:
            df[f"{col}_lag{h}"] = df[col].shift(h).astype("float32")
        for w in ROLL_WINDOWS:
            df[f"{col}_rmean{w}"] = df[col].rolling(window=w, min_periods=max(2, w // 4)).mean().astype("float32")
            if col == "temp_f":
                df[f"{col}_rstd{w}"] = df[col].rolling(window=w, min_periods=max(2, w // 4)).std().astype("float32")
                df[f"{col}_rmin{w}"] = df[col].rolling(window=w, min_periods=max(2, w // 4)).min().astype("float32")
                df[f"{col}_rmax{w}"] = df[col].rolling(window=w, min_periods=max(2, w // 4)).max().astype("float32")
    return df


# ---------- Time-of-year cyclic features ----------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h = df["hour"]
    hour_of_day = h.dt.hour
    doy = h.dt.dayofyear
    minute_of_year = (doy - 1) * 24 * 60 + hour_of_day * 60
    minutes_in_year = 366 * 24 * 60  # use 366 — within ~0.3% always
    moy_frac = minute_of_year / minutes_in_year

    df["sin_hod"] = np.sin(2 * np.pi * hour_of_day / 24).astype("float32")
    df["cos_hod"] = np.cos(2 * np.pi * hour_of_day / 24).astype("float32")
    df["sin_doy"] = np.sin(2 * np.pi * moy_frac).astype("float32")
    df["cos_doy"] = np.cos(2 * np.pi * moy_frac).astype("float32")
    df["sin_doy2"] = np.sin(4 * np.pi * moy_frac).astype("float32")
    df["cos_doy2"] = np.cos(4 * np.pi * moy_frac).astype("float32")
    df["dow"] = h.dt.dayofweek.astype("int8")
    df["month"] = h.dt.month.astype("int8")
    df["year"] = h.dt.year.astype("int16")
    df["hod"] = hour_of_day.astype("int8")
    return df


# ---------- SFO marine-layer & wind decomposition ----------

def add_marine_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dew-point depression: small values mean saturated air → fog/clouds.
    df["dewdep_f"] = (df["temp_f"] - df["dew_f"]).astype("float32")

    # Wind decomposition. SFO is at ~37.6N, -122.4W; the Pacific is to the WNW,
    # land/East Bay to the E. Pure onshore wind ≈ 270° (W). Offshore ≈ 90° (E).
    wd_rad = np.deg2rad(df["wind_dir"])
    ws = df["wind_speed"].fillna(0.0)
    df["u_wind"] = (-ws * np.sin(wd_rad)).astype("float32")  # east component
    df["v_wind"] = (-ws * np.cos(wd_rad)).astype("float32")  # north component
    # Onshore (W-NW wind, common during sea breeze): wind FROM 220-340°.
    df["onshore_flag"] = ((df["wind_dir"] >= 220) & (df["wind_dir"] <= 340)).astype("Int8")
    # Offshore (warm winds from inland): FROM 30-130°.
    df["offshore_flag"] = ((df["wind_dir"] >= 30) & (df["wind_dir"] <= 130)).astype("Int8")

    # Fog / marine-layer proxies.
    df["fog_proxy"] = (
        ((df["vis_mi"] < 3) & df["vis_mi"].notna())
        | ((df["rh"] >= 95) & df["rh"].notna())
    ).astype("Int8")
    df["low_vis"] = ((df["vis_mi"] < 5) & df["vis_mi"].notna()).astype("Int8")

    # Pressure tendency (3h) is encoded as a small float (-3..3) — keep raw.
    # Also derive 6h pressure change from rolling.
    df["slp_change_6h"] = (df["slp_inhg"] - df["slp_inhg"].shift(6)).astype("float32")
    df["slp_change_24h"] = (df["slp_inhg"] - df["slp_inhg"].shift(24)).astype("float32")

    # Temperature dynamics
    df["temp_change_1h"] = (df["temp_f"] - df["temp_f"].shift(1)).astype("float32")
    df["temp_change_3h"] = (df["temp_f"] - df["temp_f"].shift(3)).astype("float32")
    df["temp_change_6h"] = (df["temp_f"] - df["temp_f"].shift(6)).astype("float32")
    df["temp_change_24h"] = (df["temp_f"] - df["temp_f"].shift(24)).astype("float32")

    return df


# ---------- Daily (prior-day) context features ----------

def add_daily_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Join yesterday's, day-before's, and 7-day lagged daily summaries.
    Critical: a row at hour t on day D may NOT use day-D's daily summary
    (which is computed from the full day's hourly data — leakage)."""
    df = df.copy()
    df["day"] = df["hour"].dt.floor("D")
    daily = daily[["day", "daily_avg_f", "daily_max_f", "daily_min_f", "daily_precip_in",
                   "sunrise_min", "sunset_min"]].copy()

    # Build lagged-day frames
    for k in [1, 2, 7]:
        d_lag = daily.copy()
        d_lag["day"] = d_lag["day"] + pd.Timedelta(days=k)
        d_lag = d_lag.rename(columns={
            "daily_avg_f":     f"avg_d{k}",
            "daily_max_f":     f"max_d{k}",
            "daily_min_f":     f"min_d{k}",
            "daily_precip_in": f"precip_d{k}",
        })
        # Sunrise/sunset only meaningful for k=1 (yesterday's are essentially today's)
        if k != 1:
            d_lag = d_lag.drop(columns=["sunrise_min", "sunset_min"])
        df = df.merge(d_lag, on="day", how="left")
    df = df.drop(columns=["day"])
    return df


# ---------- Targets ----------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["hour"]].copy()
    for h in HORIZONS:
        out[f"temp_f_h{h}"] = df["temp_f"].shift(-h).astype("float32")
    return out


# ---------- Main ----------

def main():
    print(f"[features] reading {HOURLY_PATH}", flush=True)
    df = pd.read_parquet(HOURLY_PATH)
    print(f"[features] hourly rows: {len(df):,}", flush=True)
    daily = pd.read_parquet(DAILY_PATH)
    print(f"[features] daily rows: {len(daily):,}", flush=True)

    df = df.sort_values("hour").reset_index(drop=True)

    # Fill dew_f from Magnus when missing but temp+rh present.
    fill_mask = df["dew_f"].isna() & df["temp_f"].notna() & df["rh"].notna()
    n_fill = int(fill_mask.sum())
    if n_fill:
        df.loc[fill_mask, "dew_f"] = dew_point_from_t_rh(
            df.loc[fill_mask, "temp_f"], df.loc[fill_mask, "rh"]
        )
        print(f"[features] filled {n_fill:,} dew_f via Magnus", flush=True)

    # Climatology fit + apply
    clim = fit_climatology(df, CLIM_TRAIN_END)
    CLIM_OUT.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(CLIM_OUT, index=False)

    df = apply_climatology(df, clim)
    print("[features] climatology applied", flush=True)

    # Calendar features
    df = add_calendar_features(df)

    # Marine-layer + wind features
    df = add_marine_features(df)

    # Daily-history features (prior days only)
    df = add_daily_features(df, daily)
    print("[features] marine + daily features added", flush=True)

    # Lag and rolling features for the most informative columns
    base_cols = [c for c in ["temp_f", "dew_f", "rh", "slp_inhg", "wind_speed",
                             "vis_mi", "dewdep_f", "u_wind", "v_wind"]
                 if c in df.columns]
    df = add_lag_and_rolling(df, base_cols)
    print("[features] lags + rolling added", flush=True)

    # Down-cast numeric to float32 to save memory.
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    # Build targets BEFORE we drop temp_f from feature df.
    targets = build_targets(df)

    # Drop columns we don't want as model inputs.
    drop_cols = ["temp_source", "isd_rt", "lcd_rt", "isd_qc",
                 "p_tendency", "wind_gust"]  # gust >91% missing; tendency is categorical
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Save
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEAT_OUT, index=False)
    targets.to_parquet(TARG_OUT, index=False)
    print(f"[features] wrote features ({df.shape}) → {FEAT_OUT}", flush=True)
    print(f"[features] wrote targets ({targets.shape}) → {TARG_OUT}", flush=True)

    # Summary
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature Summary\n"]
    lines.append(f"- Rows: **{len(df):,}** ({df['hour'].min()} → {df['hour'].max()})")
    lines.append(f"- Features: **{df.shape[1] - 1}** (excluding `hour`)")
    lines.append(f"- Targets: **{len(HORIZONS)}** horizons {HORIZONS}")
    lines.append(f"- Climatology fit window ends: {CLIM_TRAIN_END}")
    lines.append("")
    lines.append("**Feature columns:**\n")
    for c in df.columns:
        if c == "hour":
            continue
        nn = int(df[c].notna().sum())
        lines.append(f"- `{c}` ({df[c].dtype}): {nn:,} non-null ({nn/len(df)*100:.1f}%)")
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[features] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
