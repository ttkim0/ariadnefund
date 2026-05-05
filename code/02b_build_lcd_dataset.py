"""
02b_build_lcd_dataset.py — Multi-city, LCD-only canonical dataset builder.

This is the slim cousin of 02_build_dataset.py.  Where the SFO pipeline merges
NOAA LCD with NOAA ISD (the global UTC archive) for richer coverage, we don't
have ISD downloads for the 19 new cities — and the LCD alone is plenty for
the model's purposes (hourly METAR-derived obs go back to 1970).

Inputs:  data/lcd_raw/{slug}.csv         (per-city LCD CSV — symlinked)
         config/cities.yaml              (timezone, name)
Outputs: data/{slug}_hourly.parquet      (one row per local-standard-time hour)
         data/{slug}_daily.parquet       (one row per calendar day, from SOD)

Timezone discipline (the SFO mistake we are not repeating):
  * NOAA LCD v2 reports timestamps in the station's LOCAL STANDARD TIME
    (no DST adjustment, ever).  We store hours in that time base, with the
    city's IANA timezone written into a `timezone` column attribute and
    metadata so downstream code never has to guess.
  * Kalshi settlement uses the local CIVIL day (which observes DST), so
    when downstream code maps the hourly grid to a Kalshi event_ticker
    day_D, it must convert from local-standard → local-civil first.
    Phoenix (no DST) is identical; everywhere else has a 1-hour offset
    half the year.

Usage:
  python3 code/02b_build_lcd_dataset.py --city nyc
  python3 code/02b_build_lcd_dataset.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import City, get_city, load_cities  # noqa: E402

# Hourly-style report types in priority order — same as the SFO pipeline.
RT_PRIORITY = {
    "FM-15": 0, "SAO": 1, "FM-12": 2, "FM-16": 3,
    "SAOSP": 4, "SYSA": 5, "SY-SA": 5, "AUTO": 6, "NSRDB": 7,
}
HOURLY_REPORT_TYPES = set(RT_PRIORITY.keys())

# Universal clip bounds — wider than SFO's because we cover the continental
# US.  Phoenix in summer hits 115°F; Minneapolis in winter hits −30°F.
CLIP_BOUNDS = {
    "temp_f":     (-30.0, 125.0),
    "dew_f":      (-40.0, 90.0),
    "wetbulb_f":  (-30.0, 100.0),
    "rh":         (0.0, 100.0),
    "slp_inhg":   (28.0, 31.5),
    "vis_mi":     (0.0, 30.0),
    "wind_dir":   (0.0, 360.0),
    "wind_speed": (0.0, 100.0),
    "wind_gust":  (0.0, 130.0),
    "p_change":   (-2.5, 2.5),
    "precip_in":  (0.0, 10.0),
}

LCD_HOURLY_COLS = [
    "HourlyDryBulbTemperature", "HourlyDewPointTemperature",
    "HourlyWetBulbTemperature", "HourlyRelativeHumidity",
    "HourlySeaLevelPressure", "HourlyPressureChange",
    "HourlyPressureTendency", "HourlySkyConditions", "HourlyVisibility",
    "HourlyWindDirection", "HourlyWindSpeed", "HourlyWindGustSpeed",
    "HourlyPrecipitation",
]
LCD_RENAME = {
    "HourlyDryBulbTemperature":   "temp_f",
    "HourlyDewPointTemperature":  "dew_f",
    "HourlyWetBulbTemperature":   "wetbulb_f",
    "HourlyRelativeHumidity":     "rh",
    "HourlySeaLevelPressure":     "slp_inhg",
    "HourlyPressureChange":       "p_change",
    "HourlyPressureTendency":     "p_tendency",
    "HourlySkyConditions":        "sky_raw",
    "HourlyVisibility":           "vis_mi",
    "HourlyWindDirection":        "wind_dir",
    "HourlyWindSpeed":            "wind_speed",
    "HourlyWindGustSpeed":        "wind_gust",
    "HourlyPrecipitation":        "precip_in",
}

_TRAILING_FLAG = re.compile(r"[^\d\-.]+$")


def to_num(s: pd.Series, *, trace_value: Optional[float] = None) -> pd.Series:
    out = s.astype("string").str.replace(_TRAILING_FLAG, "", regex=True)
    out = out.replace({"": np.nan, "M": np.nan, "*": np.nan})
    if trace_value is not None:
        out = out.replace({"T": str(trace_value)})
    else:
        out = out.replace({"T": np.nan})
    return pd.to_numeric(out, errors="coerce")


def derive_cloud_max(sky: pd.Series) -> pd.Series:
    """LCD HourlySkyConditions is e.g. 'BKN:7 100' — the integer after each
    layer is the cover in oktas (0-8).  Take the max layer cover as the
    summary cloud-cover feature, like the SFO pipeline does."""
    s = sky.fillna("").astype(str)
    out = pd.Series(np.nan, index=s.index, dtype="Float64")
    pat = re.compile(r":\s*(\d+)")
    for i, txt in enumerate(s):
        if not txt:
            continue
        nums = [int(m.group(1)) for m in pat.finditer(txt)]
        nums = [n for n in nums if 0 <= n <= 10]
        if nums:
            out.iloc[i] = float(max(nums))
    return out


def build_for_city(city: City) -> dict:
    t0 = time.time()
    if not city.lcd_raw.exists():
        print(f"[{city.slug}] SKIP — LCD not found at {city.lcd_raw}")
        return {"slug": city.slug, "skipped": True}

    print(f"[{city.slug}] reading {city.lcd_raw}", flush=True)
    keep_cols = (
        ["DATE", "REPORT_TYPE"] + LCD_HOURLY_COLS
        + ["DailyAverageDryBulbTemperature", "DailyMaximumDryBulbTemperature",
           "DailyMinimumDryBulbTemperature", "DailyPrecipitation",
           "Sunrise", "Sunset"]
    )
    str_cols = {c: "string" for c in keep_cols if c != "DATE"}
    df = pd.read_csv(
        city.lcd_raw, usecols=keep_cols, dtype=str_cols, parse_dates=["DATE"],
    )
    df["REPORT_TYPE"] = df["REPORT_TYPE"].str.strip()
    n_raw = len(df)

    # ── Daily extract from SOD rows (the source of truth for next-day forecasting)
    sod = df[df["REPORT_TYPE"].eq("SOD")].copy()
    sod["day"] = sod["DATE"].dt.floor("D")
    daily = pd.DataFrame({
        "day":             sod["day"],
        "daily_avg_f":     to_num(sod["DailyAverageDryBulbTemperature"]),
        "daily_max_f":     to_num(sod["DailyMaximumDryBulbTemperature"]),
        "daily_min_f":     to_num(sod["DailyMinimumDryBulbTemperature"]),
        "daily_precip_in": to_num(sod["DailyPrecipitation"], trace_value=0.001),
        "sunrise_raw":     sod["Sunrise"],
        "sunset_raw":      sod["Sunset"],
    })
    daily = daily.dropna(subset=["day"]).drop_duplicates("day", keep="last")
    daily = daily.sort_values("day").reset_index(drop=True)

    def hhmm_to_min(s):
        n = pd.to_numeric(s, errors="coerce")
        return ((n // 100) * 60 + (n % 100)).astype("Float64")
    daily["sunrise_min"] = hhmm_to_min(daily["sunrise_raw"])
    daily["sunset_min"]  = hhmm_to_min(daily["sunset_raw"])
    daily = daily.drop(columns=["sunrise_raw", "sunset_raw"])

    # ── Hourly extract from non-summary report types
    h = df.loc[df["REPORT_TYPE"].isin(HOURLY_REPORT_TYPES)].copy().rename(columns=LCD_RENAME)
    for col in ["temp_f", "dew_f", "wetbulb_f", "rh", "slp_inhg", "p_change",
                "vis_mi", "wind_dir", "wind_speed", "wind_gust"]:
        h[col] = to_num(h[col])
    h["precip_in"] = to_num(h["precip_in"], trace_value=0.001)
    h.loc[h["wind_dir"].eq(999), "wind_dir"] = np.nan
    h["sky_raw"] = h["sky_raw"].astype("string")
    h["cloud_max"] = derive_cloud_max(h["sky_raw"])
    h["sky_overcast"] = (h["cloud_max"] >= 8).astype("Int8")
    h["sky_obscured"] = h["sky_raw"].str.contains("VV", na=False).astype("Int8")

    h["hour"] = h["DATE"].dt.floor("h")
    h["rt_priority"] = h["REPORT_TYPE"].map(RT_PRIORITY).fillna(99).astype(int)
    qual_cols = ["temp_f", "dew_f", "rh", "slp_inhg", "vis_mi", "wind_dir", "wind_speed"]
    h["_qual"] = h[qual_cols].notna().sum(axis=1)
    h = h.sort_values(["hour", "rt_priority", "_qual"], ascending=[True, True, False])
    h = h.drop_duplicates("hour", keep="first").reset_index(drop=True)
    h = h.rename(columns={"REPORT_TYPE": "lcd_rt"})

    # ── Outlier clipping
    n_clip = 0
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col not in h.columns:
            continue
        bad = h[col].notna() & ((h[col] < lo) | (h[col] > hi))
        n_clip += int(bad.sum())
        h.loc[bad, col] = np.nan

    # ── Output schema — keep the same column set as sfo_hourly.parquet so
    #     the existing 03_features.py / model code works unchanged.
    cols_keep = [
        "hour", "cloud_max", "dew_f", "lcd_rt", "p_change",
        "p_tendency", "precip_in", "rh", "sky_obscured", "sky_overcast",
        "slp_inhg", "temp_f", "vis_mi", "wetbulb_f", "wind_dir", "wind_gust",
        "wind_speed",
    ]
    # Add empty isd columns for schema compatibility with SFO consumer code.
    h["isd_qc"] = pd.Series([pd.NA] * len(h), dtype="string")
    h["isd_rt"] = pd.Series([pd.NA] * len(h), dtype="string")
    h["temp_source"] = "lcd"
    cols_keep += ["isd_qc", "isd_rt", "temp_source"]
    out_h = h[cols_keep].sort_values("hour").reset_index(drop=True)

    # Attach timezone metadata as a parquet user-key (and an in-memory col).
    # Downstream code can read it and avoid guessing.
    out_h["timezone"] = city.timezone

    out_h.to_parquet(city.hourly_path, index=False)
    daily.to_parquet(city.daily_path, index=False)

    dt = round(time.time() - t0, 1)
    print(f"[{city.slug}] hourly={len(out_h):,}  daily={len(daily):,}  "
          f"clipped={n_clip:,}  raw={n_raw:,}  ({dt}s)", flush=True)

    return {
        "slug":       city.slug,
        "n_hourly":   int(len(out_h)),
        "n_daily":    int(len(daily)),
        "n_clipped":  int(n_clip),
        "duration_s": dt,
        "min_hour":   str(out_h["hour"].min()) if len(out_h) else None,
        "max_hour":   str(out_h["hour"].max()) if len(out_h) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", help="single city slug")
    ap.add_argument("--all", action="store_true", help="all cities except sfo")
    ap.add_argument("--include-sfo", action="store_true",
                    help="when --all is set, also rebuild SFO from LCD-only "
                         "(default: skip SFO, since 02_build_dataset.py owns it)")
    args = ap.parse_args()

    if args.city:
        targets = [get_city(args.city)]
    elif args.all:
        targets = [c for c in load_cities() if c.slug != "sfo" or args.include_sfo]
    else:
        ap.error("must pass --city <slug> or --all")
        return

    results = []
    for c in targets:
        try:
            results.append(build_for_city(c))
        except Exception as e:
            print(f"[{c.slug}] FAILED: {e}", flush=True)
            results.append({"slug": c.slug, "error": str(e)})

    print("\n[summary]")
    for r in results:
        if r.get("error"):
            print(f"  {r['slug']:>4}  FAILED: {r['error']}")
        elif r.get("skipped"):
            print(f"  {r['slug']:>4}  skipped (no LCD)")
        else:
            print(f"  {r['slug']:>4}  hourly={r['n_hourly']:>7,}  "
                  f"daily={r['n_daily']:>5,}  range={r['min_hour']} → {r['max_hour']}")


if __name__ == "__main__":
    main()
