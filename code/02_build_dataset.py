"""
02_build_dataset.py — Build a canonical hourly SFO weather dataset.

Inputs:
  * raw global hourly.csv  (ISD)
  * raw LCD datas.csv      (Local Climatological Data v2)

Output:
  * data/sfo_hourly.parquet  — one row per UTC hour from 1970-01-01 to the latest
                                  available timestamp, with cleaned, source-tagged
                                  features. NO synthetic data.
  * data/sfo_daily.parquet   — one row per calendar day, with SOD-derived
                                  daily max / min / avg / precip.
  * reports/build_summary.md — what was kept, dropped, clipped.

Design:
  1. Parse ISD TMP -> Fahrenheit. Keep only QC flags {1, 5, A} (passed checks).
     Drop QC=9 (erroneous/missing).
  2. Parse LCD numeric columns. Strip trailing 's' (suspect), '*' (estimated),
     'V' (variable). Treat 'M' as missing, 'T' (trace precip) as 0.001.
  3. Filter LCD to hourly-style REPORT_TYPEs (FM-15 preferred, then SAO, FM-16,
     FM-12, SAOSP, SYSA, SY-SA). Exclude SOD/SOM/SY-MT/SMARS for hourly bins.
  4. Per UTC-hour bin, choose the single best observation:
        priority: FM-15 > SAO > FM-12 > FM-16 > SAOSP > SYSA/SY-SA > AUTO > NSRDB
     If multiple at same priority, take the one with the most non-null fields
     (richest record).
  5. Clip outliers per-feature using historical SFO bounds.
  6. ISD fills gaps where LCD has no temperature.
  7. Build sfo_daily separately from SOD rows (these are the authoritative
     daily summaries, used as features for next-day forecasting).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
ISD_PATH = ROOT / "global hourly.csv"
LCD_PATH = ROOT / "LCD datas.csv"
OUT_HOURLY = ROOT / "data" / "sfo_hourly.parquet"
OUT_DAILY = ROOT / "data" / "sfo_daily.parquet"
SUMMARY_PATH = ROOT / "reports" / "build_summary.md"

# Report-type priority for picking the canonical hourly observation.
# Lower number = higher priority.
RT_PRIORITY = {
    "FM-15": 0,    # METAR — the standard hourly aviation observation
    "SAO":   1,    # legacy surface aviation observation
    "FM-12": 2,    # synoptic SYNOP from manned stations (often has best wind/pressure)
    "FM-16": 3,    # SPECI — special METARs (off-cycle but valid)
    "SAOSP": 4,
    "SYSA":  5,
    "SY-SA": 5,
    "AUTO":  6,
    "NSRDB": 7,
}
HOURLY_REPORT_TYPES = set(RT_PRIORITY.keys())

# Fahrenheit clip bounds based on physical plausibility for SFO and surrounds.
CLIP_BOUNDS = {
    "temp_f":     (10.0, 115.0),
    "dew_f":      (-20.0, 80.0),
    "wetbulb_f":  (0.0, 90.0),
    "rh":         (0.0, 100.0),
    "slp_inhg":   (28.5, 31.0),
    "vis_mi":     (0.0, 30.0),
    "wind_dir":   (0.0, 360.0),     # 999 already nulled
    "wind_speed": (0.0, 80.0),
    "wind_gust":  (0.0, 110.0),
    "p_change":   (-2.0, 2.0),
    "precip_in":  (0.0, 6.0),
}


# ---------- Parsing helpers ----------

_TRAILING_FLAG = re.compile(r"[^\d\-.]+$")  # strip 's', '*', 'V', 'A', etc.

def to_num(s: pd.Series, *, trace_value: float | None = None) -> pd.Series:
    """Coerce LCD-style messy strings to numeric. 'M' -> NaN. 'T' -> trace_value."""
    out = s.astype("string").str.replace(_TRAILING_FLAG, "", regex=True)
    out = out.replace({"": np.nan, "M": np.nan, "*": np.nan})
    if trace_value is not None:
        out = out.replace({"T": str(trace_value)})
    else:
        out = out.replace({"T": np.nan})
    return pd.to_numeric(out, errors="coerce")


def parse_isd_tmp(value: str) -> tuple[float, str | None]:
    if not isinstance(value, str) or "," not in value:
        return (np.nan, None)
    raw, qc = value.split(",", 1)
    try:
        n = int(raw)
    except ValueError:
        return (np.nan, qc)
    if n == 9999:
        return (np.nan, qc)
    return (n / 10.0, qc)


# ---------- ISD ingestion ----------

def load_isd() -> pd.DataFrame:
    """Load ISD. Critical: NOAA ISD timestamps are UTC, but NOAA LCD v2 uses
    Local Standard Time (PST = UTC-8 for SFO). To merge consistently we
    convert ISD UTC → PST by subtracting 8 hours. The whole dataset then
    lives on a PST time axis (no DST shifts ever, by design)."""
    print(f"[build] reading ISD {ISD_PATH}", flush=True)
    df = pd.read_csv(
        ISD_PATH,
        dtype={
            "STATION": "string", "REPORT_TYPE": "string",
            "SOURCE": "string", "QUALITY_CONTROL": "string",
            "TMP": "string",
        },
        usecols=["DATE", "REPORT_TYPE", "QUALITY_CONTROL", "TMP"],
        parse_dates=["DATE"],
    )
    df["REPORT_TYPE"] = df["REPORT_TYPE"].str.strip()
    # UTC → PST (no DST). LCD is already in PST so this aligns the two sources.
    df["DATE"] = df["DATE"] - pd.Timedelta(hours=8)

    parsed = df["TMP"].map(parse_isd_tmp)
    df["temp_c"] = parsed.map(lambda t: t[0])
    df["temp_qc"] = parsed.map(lambda t: t[1])

    # Keep only QC flags 1 (passed first level), 5 (passed all), A (auto-pass).
    # Drop 9 (failed/missing), 6, 2, 7, P (problematic — rare anyway).
    keep_qc = df["temp_qc"].isin({"1", "5", "A"})
    before, after = len(df), int(keep_qc.sum())
    df = df.loc[keep_qc & df["temp_c"].notna()].copy()
    print(f"[build] ISD QC keep: {after}/{before} ({after/before*100:.1f}%)", flush=True)

    df["temp_f"] = df["temp_c"] * 9.0 / 5.0 + 32.0
    df["hour"] = df["DATE"].dt.floor("h")
    df["rt_priority"] = df["REPORT_TYPE"].map(RT_PRIORITY).fillna(99).astype(int)

    # one obs per hour: lowest priority number wins; ties broken by highest temp_f
    # density (just keep first after sort — they are very close anyway)
    df = df.sort_values(["hour", "rt_priority", "DATE"])
    df = df.drop_duplicates("hour", keep="first")
    print(f"[build] ISD unique hours after dedup: {len(df):,}", flush=True)
    return df[["hour", "temp_f", "REPORT_TYPE", "temp_qc"]].rename(
        columns={"REPORT_TYPE": "isd_rt", "temp_qc": "isd_qc"}
    )


# ---------- LCD ingestion ----------

LCD_HOURLY_COLS = [
    "HourlyDryBulbTemperature",
    "HourlyDewPointTemperature",
    "HourlyWetBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlySeaLevelPressure",
    "HourlyPressureChange",
    "HourlyPressureTendency",
    "HourlySkyConditions",
    "HourlyVisibility",
    "HourlyWindDirection",
    "HourlyWindSpeed",
    "HourlyWindGustSpeed",
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


def load_lcd_hourly_and_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[build] reading LCD {LCD_PATH}", flush=True)

    keep_cols = (
        ["DATE", "REPORT_TYPE"]
        + LCD_HOURLY_COLS
        + ["DailyAverageDryBulbTemperature", "DailyMaximumDryBulbTemperature",
           "DailyMinimumDryBulbTemperature", "DailyPrecipitation",
           "Sunrise", "Sunset"]
    )
    str_cols = {c: "string" for c in keep_cols if c not in ("DATE",)}
    df = pd.read_csv(
        LCD_PATH,
        usecols=keep_cols,
        dtype=str_cols,
        parse_dates=["DATE"],
    )
    df["REPORT_TYPE"] = df["REPORT_TYPE"].str.strip()
    print(f"[build] LCD raw rows: {len(df):,}", flush=True)

    # ===== Daily extract from SOD rows =====
    sod = df[df["REPORT_TYPE"].eq("SOD")].copy()
    sod["day"] = sod["DATE"].dt.floor("D")
    daily = pd.DataFrame({
        "day":            sod["day"],
        "daily_avg_f":    to_num(sod["DailyAverageDryBulbTemperature"]),
        "daily_max_f":    to_num(sod["DailyMaximumDryBulbTemperature"]),
        "daily_min_f":    to_num(sod["DailyMinimumDryBulbTemperature"]),
        "daily_precip_in": to_num(sod["DailyPrecipitation"], trace_value=0.001),
        "sunrise_raw":    sod["Sunrise"],
        "sunset_raw":     sod["Sunset"],
    })
    daily = daily.dropna(subset=["day"]).drop_duplicates("day", keep="last")
    daily = daily.sort_values("day").reset_index(drop=True)

    # Parse Sunrise/Sunset which are HHMM ints in local time
    def hhmm_to_minutes(s):
        n = pd.to_numeric(s, errors="coerce")
        return ((n // 100) * 60 + (n % 100)).astype("Float64")
    daily["sunrise_min"] = hhmm_to_minutes(daily["sunrise_raw"])
    daily["sunset_min"] = hhmm_to_minutes(daily["sunset_raw"])
    daily = daily.drop(columns=["sunrise_raw", "sunset_raw"])
    print(f"[build] LCD daily rows (SOD): {len(daily):,}", flush=True)

    # ===== Hourly extract from non-summary report types =====
    hourly_mask = df["REPORT_TYPE"].isin(HOURLY_REPORT_TYPES)
    h = df.loc[hourly_mask].copy()
    print(f"[build] LCD hourly rows after RT filter: {len(h):,}", flush=True)

    h = h.rename(columns=LCD_RENAME)
    # Coerce numerics. Precip uses 'T'=trace=0.001 inches.
    for col in ["temp_f", "dew_f", "wetbulb_f", "rh", "slp_inhg", "p_change",
                "vis_mi", "wind_dir", "wind_speed", "wind_gust"]:
        h[col] = to_num(h[col])
    h["precip_in"] = to_num(h["precip_in"], trace_value=0.001)

    # 999 wind direction means "variable" — null it out.
    h.loc[h["wind_dir"].eq(999), "wind_dir"] = np.nan

    # Sky conditions: keep raw string, derive cloud cover later. Null obvious bad values.
    h["sky_raw"] = h["sky_raw"].astype("string")

    h["hour"] = h["DATE"].dt.floor("h")
    h["rt_priority"] = h["REPORT_TYPE"].map(RT_PRIORITY).fillna(99).astype(int)

    # Quality count for tiebreak: number of non-null observed fields.
    quality_cols = ["temp_f", "dew_f", "rh", "slp_inhg", "vis_mi",
                    "wind_dir", "wind_speed"]
    h["_qual"] = h[quality_cols].notna().sum(axis=1)

    # Pick best obs per hour: lower rt_priority first, then higher _qual.
    h = h.sort_values(
        ["hour", "rt_priority", "_qual"],
        ascending=[True, True, False],
    )
    h_dedup = h.drop_duplicates("hour", keep="first").reset_index(drop=True)
    print(f"[build] LCD hourly unique hours: {len(h_dedup):,}", flush=True)
    h_dedup = h_dedup.rename(columns={"REPORT_TYPE": "lcd_rt"})

    cols_keep = ["hour", "lcd_rt"] + list(LCD_RENAME.values())
    return h_dedup[cols_keep], daily


# ---------- Outlier clipping ----------

def clip_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {}
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col not in df.columns:
            continue
        before = df[col].notna().sum()
        bad = (df[col].notna()) & ((df[col] < lo) | (df[col] > hi))
        n_bad = int(bad.sum())
        df.loc[bad, col] = np.nan
        stats[col] = {"before_non_null": int(before), "nulled": n_bad,
                      "after_non_null": int(df[col].notna().sum()),
                      "lo": lo, "hi": hi}
    return df, stats


# ---------- Cloud cover encoding ----------

# Sky conditions look like "FEW:02 5 SCT:04 25 BKN:07 80" — concatenated layers.
# We extract the maximum cloud-cover code (CLR=0, FEW=2, SCT=4, BKN=7, OVC=9)
# as a numeric "ceiling category" 0..9, plus an indicator if any layer is BKN/OVC.
SKY_CODES = {
    "CLR": 0, "SKC": 0, "FEW": 2, "SCT": 4, "BKN": 7, "OVC": 9,
    "VV":  9,  # vertical visibility = obscured sky (fog)
    "X":   0,  # clear (rare encoding)
}

_SKY_LAYER = re.compile(r"\b(CLR|SKC|FEW|SCT|BKN|OVC|VV|X)\b")

def encode_sky(s: str | float) -> tuple[float, int, int]:
    """Return (max_code 0-9, has_overcast, has_obscured)."""
    if not isinstance(s, str):
        return (np.nan, 0, 0)
    layers = _SKY_LAYER.findall(s.upper())
    if not layers:
        return (np.nan, 0, 0)
    codes = [SKY_CODES[l] for l in layers]
    has_ovc = int(any(l == "OVC" for l in layers))
    has_obs = int(any(l == "VV" for l in layers))
    return (float(max(codes)), has_ovc, has_obs)


# ---------- Main build ----------

def main():
    isd = load_isd()
    lcd_hourly, lcd_daily = load_lcd_hourly_and_daily()

    # Outer-merge on hour
    merged = lcd_hourly.merge(isd, on="hour", how="outer", suffixes=("_lcd", "_isd"))

    # temp_f provenance: LCD primary; ISD fills gaps.
    src = pd.Series("none", index=merged.index, dtype="string")
    src[merged["temp_f_lcd"].notna()] = "lcd"
    isd_only = merged["temp_f_lcd"].isna() & merged["temp_f_isd"].notna()
    src[isd_only] = "isd"
    merged["temp_f"] = merged["temp_f_lcd"].combine_first(merged["temp_f_isd"])
    merged["temp_source"] = src
    merged = merged.drop(columns=["temp_f_lcd", "temp_f_isd"])

    # Sky encoding
    sky_enc = merged["sky_raw"].map(encode_sky)
    merged["cloud_max"] = sky_enc.map(lambda t: t[0])
    merged["sky_overcast"] = sky_enc.map(lambda t: t[1]).astype("Int8")
    merged["sky_obscured"] = sky_enc.map(lambda t: t[2]).astype("Int8")
    merged = merged.drop(columns=["sky_raw"])

    # Outlier clipping
    merged, clip_stats = clip_outliers(merged)

    # Reindex to a strict hourly grid so missing hours are explicit
    grid = pd.date_range(merged["hour"].min(), merged["hour"].max(), freq="h", name="hour")
    merged = merged.set_index("hour").reindex(grid).reset_index()
    print(f"[build] hourly grid rows: {len(merged):,}", flush=True)

    # Sanity: number of non-null per important column
    for c in ["temp_f", "dew_f", "rh", "slp_inhg", "wind_speed", "vis_mi"]:
        nn = int(merged[c].notna().sum())
        print(f"[build]   {c}: {nn:,} non-null ({nn/len(merged)*100:.2f}%)")

    # Save hourly
    OUT_HOURLY.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_HOURLY, index=False)
    print(f"[build] wrote {OUT_HOURLY}", flush=True)

    # Save daily
    lcd_daily.to_parquet(OUT_DAILY, index=False)
    print(f"[build] wrote {OUT_DAILY}", flush=True)

    # Summary report
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Dataset Build Summary\n"]
    lines.append(f"- Hourly grid: **{len(merged):,}** rows  "
                 f"({merged['hour'].min()} → {merged['hour'].max()})")
    lines.append(f"- Daily SOD: **{len(lcd_daily):,}** rows")
    lines.append("")
    lines.append("**temp_f provenance:**")
    src_counts = merged["temp_source"].value_counts(dropna=False).to_dict()
    for k, v in src_counts.items():
        lines.append(f"- {k}: {v:,} ({v/len(merged)*100:.2f}%)")
    lines.append("")
    lines.append("**Outlier clipping:**\n")
    lines.append("| Field | Range | Nulled | After-non-null |")
    lines.append("|---|---|---:|---:|")
    for col, st in clip_stats.items():
        lines.append(f"| {col} | [{st['lo']}, {st['hi']}] | {st['nulled']:,} | {st['after_non_null']:,} |")
    lines.append("")
    lines.append("**Hourly column non-null counts (after merge+clip):**\n")
    lines.append("| Column | Non-null | % |")
    lines.append("|---|---:|---:|")
    for c in merged.columns:
        if c == "hour":
            continue
        nn = int(merged[c].notna().sum())
        lines.append(f"| {c} | {nn:,} | {nn/len(merged)*100:.2f}% |")
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[build] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
