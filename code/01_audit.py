"""
01_audit.py — Deep data audit of the two SFO weather CSVs.

Goals:
  * Confirm schema, dtypes, time coverage, station identity for both files.
  * Quantify missingness per column.
  * Inspect REPORT_TYPE breakdown for LCD (multiple report kinds per hour).
  * Inspect TMP parsing for ISD (format like "+0139,5" -> 13.9 C, QC flag 5).
  * Detect duplicate timestamps and clock-skew issues.
  * Detect impossible / out-of-range values.
  * Write a single Markdown report at reports/audit.md.

This script does NOT modify or write any cleaned data. It only reads the raw
files and produces a report. Cleaning happens in 02_build_dataset.py.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
ISD_PATH = ROOT / "global hourly.csv"
LCD_PATH = ROOT / "LCD datas.csv"
REPORT_PATH = ROOT / "reports" / "audit.md"


def fmt_int(n) -> str:
    return f"{int(n):,}"


def parse_isd_tmp(value: str):
    """Parse ISD TMP field. Format: '+0139,5' -> (13.9 C, qc='5'). Missing = +9999."""
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


def audit_isd() -> dict:
    print(f"[audit] reading ISD: {ISD_PATH}", flush=True)
    df = pd.read_csv(
        ISD_PATH,
        dtype={
            "STATION": "string",
            "NAME": "string",
            "SOURCE": "string",
            "REPORT_TYPE": "string",
            "CALL_SIGN": "string",
            "QUALITY_CONTROL": "string",
            "TMP": "string",
        },
        parse_dates=["DATE"],
    )
    print(f"[audit] ISD rows: {len(df):,}", flush=True)

    parsed = df["TMP"].map(parse_isd_tmp)
    df["TMP_C"] = parsed.map(lambda t: t[0])
    df["TMP_QC"] = parsed.map(lambda t: t[1])
    df["TMP_F"] = df["TMP_C"] * 9.0 / 5.0 + 32.0

    out = {}
    out["rows"] = int(len(df))
    out["date_min"] = str(df["DATE"].min())
    out["date_max"] = str(df["DATE"].max())
    out["stations"] = df["STATION"].value_counts().to_dict()
    out["report_types"] = df["REPORT_TYPE"].str.strip().value_counts().to_dict()
    out["sources"] = df["SOURCE"].value_counts(dropna=False).to_dict()
    out["call_signs"] = df["CALL_SIGN"].str.strip().value_counts().to_dict()
    out["qc_flags"] = df["TMP_QC"].value_counts(dropna=False).to_dict()

    out["tmp_missing"] = int(df["TMP_C"].isna().sum())
    out["tmp_present"] = int(df["TMP_C"].notna().sum())
    out["tmp_c_min"] = float(df["TMP_C"].min())
    out["tmp_c_max"] = float(df["TMP_C"].max())
    out["tmp_c_mean"] = float(df["TMP_C"].mean())
    out["tmp_f_min"] = float(df["TMP_F"].min())
    out["tmp_f_max"] = float(df["TMP_F"].max())
    out["tmp_f_mean"] = float(df["TMP_F"].mean())

    # impossible values: SF historical record range is ~25F to ~106F
    out["tmp_f_below_15"] = int((df["TMP_F"] < 15).sum())
    out["tmp_f_above_115"] = int((df["TMP_F"] > 115).sum())

    # duplicates by timestamp (multiple reports per hour are normal in ISD)
    dup_per_hour = (
        df.groupby(df["DATE"].dt.floor("h"))
        .size()
        .reset_index(name="n")
    )
    out["hours_total"] = int(len(dup_per_hour))
    out["hours_with_multi_obs"] = int((dup_per_hour["n"] > 1).sum())
    out["max_obs_per_hour"] = int(dup_per_hour["n"].max())
    out["mean_obs_per_hour"] = float(dup_per_hour["n"].mean())

    # gap analysis: hours with NO observation
    full_index = pd.date_range(df["DATE"].dt.floor("h").min(), df["DATE"].dt.floor("h").max(), freq="h")
    present_hours = set(df["DATE"].dt.floor("h").unique())
    missing_hours = [h for h in full_index if h not in present_hours]
    out["expected_hours"] = int(len(full_index))
    out["missing_hours"] = int(len(missing_hours))
    out["coverage_pct"] = 100.0 * (1 - len(missing_hours) / len(full_index))

    # Year-by-year coverage
    by_year = (
        pd.Series(1, index=df["DATE"].dt.floor("h").drop_duplicates())
        .resample("YE")
        .sum()
    )
    out["yearly_unique_hours"] = {str(idx.year): int(v) for idx, v in by_year.items()}

    return out


def audit_lcd() -> dict:
    print(f"[audit] reading LCD: {LCD_PATH}", flush=True)

    # Read first to discover columns
    head = pd.read_csv(LCD_PATH, nrows=1)
    cols = list(head.columns)
    print(f"[audit] LCD columns: {cols}", flush=True)

    # All hourly/daily fields are messy strings: trailing 's' (suspect), '*' (estimated),
    # 'V' (variable), 'M' (missing), 'T' (trace). Read as string then coerce.
    str_cols = [c for c in cols if c.startswith(("Hourly", "Daily", "Sunrise", "Sunset"))]
    dtype_map = {c: "string" for c in str_cols}
    dtype_map.update({
        "STATION": "string",
        "NAME": "string",
        "REPORT_TYPE": "string",
        "SOURCE": "string",
    })

    df = pd.read_csv(LCD_PATH, dtype=dtype_map, parse_dates=["DATE"])
    print(f"[audit] LCD rows: {len(df):,}", flush=True)

    out = {}
    out["rows"] = int(len(df))
    out["columns"] = cols
    out["date_min"] = str(df["DATE"].min())
    out["date_max"] = str(df["DATE"].max())
    out["stations"] = df["STATION"].value_counts().to_dict()
    out["report_types"] = df["REPORT_TYPE"].str.strip().value_counts().to_dict()
    out["sources"] = df["SOURCE"].value_counts(dropna=False).head(10).to_dict()

    # Per-column raw missingness
    out["missing_raw"] = {c: int(df[c].isna().sum()) for c in cols}

    # Hourly columns: parse numeric and inspect
    hourly_cols = [c for c in cols if c.startswith("Hourly")]
    daily_cols = [c for c in cols if c.startswith("Daily")]

    def to_num(s: pd.Series) -> pd.Series:
        # Strip trailing letters/symbols like 's', '*', 'V', 'T'. Keep digits, '-', '.'.
        cleaned = s.str.replace(r"[^\d\-.]+$", "", regex=True)
        cleaned = cleaned.replace({"": np.nan, "M": np.nan, "T": "0.001", "*": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")

    out["hourly_stats"] = {}
    for c in hourly_cols:
        if c == "HourlySkyConditions" or c == "HourlyPressureTendency":
            # categorical-ish; report unique value count
            uniq = df[c].dropna().nunique()
            sample = df[c].dropna().value_counts().head(8).to_dict()
            out["hourly_stats"][c] = {"type": "categorical", "unique": int(uniq), "top": sample}
        else:
            n = to_num(df[c])
            out["hourly_stats"][c] = {
                "type": "numeric",
                "non_null": int(n.notna().sum()),
                "null": int(n.isna().sum()),
                "min": float(n.min()) if n.notna().any() else None,
                "max": float(n.max()) if n.notna().any() else None,
                "mean": float(n.mean()) if n.notna().any() else None,
                "p1": float(n.quantile(0.01)) if n.notna().any() else None,
                "p99": float(n.quantile(0.99)) if n.notna().any() else None,
            }

    out["daily_stats"] = {}
    for c in daily_cols:
        n = to_num(df[c])
        out["daily_stats"][c] = {
            "non_null": int(n.notna().sum()),
            "null": int(n.isna().sum()),
            "min": float(n.min()) if n.notna().any() else None,
            "max": float(n.max()) if n.notna().any() else None,
            "mean": float(n.mean()) if n.notna().any() else None,
        }

    # Report-type breakdown by year for understanding what kinds of rows exist
    year = df["DATE"].dt.year
    rt_year = pd.crosstab(year, df["REPORT_TYPE"].str.strip())
    out["report_type_by_year_head"] = rt_year.head().to_dict()
    out["report_type_by_year_tail"] = rt_year.tail().to_dict()

    # Hourly DryBulb: how many rows have it?
    dry = to_num(df["HourlyDryBulbTemperature"])
    out["dry_bulb_present"] = int(dry.notna().sum())
    out["dry_bulb_missing"] = int(dry.isna().sum())

    # Coverage of hourly dry bulb by year
    df["_dry"] = dry
    yearly = (
        df.dropna(subset=["_dry"])
        .assign(hour=lambda d: d["DATE"].dt.floor("h"))
        .drop_duplicates("hour")
        .groupby(df["DATE"].dt.year)
        .size()
    )
    out["dry_bulb_unique_hours_by_year"] = {str(int(k)): int(v) for k, v in yearly.items()}

    # Daily values are emitted only on SOD rows (one per day). Confirm:
    sod_mask = df["REPORT_TYPE"].str.strip().eq("SOD")
    daily_max = to_num(df["DailyMaximumDryBulbTemperature"])
    out["daily_max_present_total"] = int(daily_max.notna().sum())
    out["daily_max_present_on_sod"] = int(daily_max[sod_mask].notna().sum())
    out["daily_max_present_off_sod"] = int(daily_max[~sod_mask].notna().sum())

    # Hours with dry bulb: distinct hourly bins
    dry_df = df.dropna(subset=["_dry"]).copy()
    dry_df["hour_bin"] = dry_df["DATE"].dt.floor("h")
    obs_per_hour = dry_df.groupby("hour_bin").size()
    out["hours_with_dry_bulb"] = int(obs_per_hour.shape[0])
    out["dry_bulb_obs_per_hour_max"] = int(obs_per_hour.max())
    out["dry_bulb_obs_per_hour_mean"] = float(obs_per_hour.mean())

    # Coverage gap: expected hours vs hours with at least one dry bulb obs
    full_index = pd.date_range(dry_df["hour_bin"].min(), dry_df["hour_bin"].max(), freq="h")
    out["dry_bulb_expected_hours"] = int(len(full_index))
    out["dry_bulb_coverage_pct"] = 100.0 * out["hours_with_dry_bulb"] / len(full_index)

    return out


def write_report(isd: dict, lcd: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    add = lines.append

    add("# SFO Weather Data Audit\n")
    add(f"Generated by `code/01_audit.py`. Working dir: `{ROOT}`\n")

    add("\n## ISD / Global Hourly\n")
    add(f"- Path: `{ISD_PATH.name}`")
    add(f"- Rows: **{fmt_int(isd['rows'])}**")
    add(f"- Date range: **{isd['date_min']}** → **{isd['date_max']}**")
    add(f"- Expected hours in range: {fmt_int(isd['expected_hours'])}")
    add(f"- Hours with at least one obs: {fmt_int(isd['hours_total'])}")
    add(f"- Missing hours (no obs at all): {fmt_int(isd['missing_hours'])}")
    add(f"- Coverage: **{isd['coverage_pct']:.2f}%**")
    add(f"- Hours with multi-obs: {fmt_int(isd['hours_with_multi_obs'])}  "
        f"(max {isd['max_obs_per_hour']}/hr, mean {isd['mean_obs_per_hour']:.2f})")
    add("")
    add("**Stations seen:**")
    for s, n in isd["stations"].items():
        add(f"- `{s}`: {fmt_int(n)}")
    add("")
    add("**Report types:**")
    for s, n in list(isd["report_types"].items())[:15]:
        add(f"- `{s}`: {fmt_int(n)}")
    add("")
    add("**TMP field:**")
    add(f"- Parsed non-null: {fmt_int(isd['tmp_present'])}  /  null: {fmt_int(isd['tmp_missing'])}")
    add(f"- Range (C): {isd['tmp_c_min']:.2f} to {isd['tmp_c_max']:.2f}, mean {isd['tmp_c_mean']:.2f}")
    add(f"- Range (F): {isd['tmp_f_min']:.2f} to {isd['tmp_f_max']:.2f}, mean {isd['tmp_f_mean']:.2f}")
    add(f"- Below 15F: {isd['tmp_f_below_15']},  above 115F: {isd['tmp_f_above_115']} (suspicious)")
    add("")
    add("**QC flag distribution:**")
    for s, n in isd["qc_flags"].items():
        add(f"- `{s}`: {fmt_int(n)}")
    add("")
    add("**Yearly unique-hours coverage (sampled):**")
    yrs = sorted(isd["yearly_unique_hours"].keys())
    for y in yrs[:3] + ["..."] + yrs[-3:]:
        if y == "...":
            add("- ...")
        else:
            add(f"- {y}: {fmt_int(isd['yearly_unique_hours'][y])} unique hours")

    add("\n## LCD / Local Climatological Data v2\n")
    add(f"- Path: `{LCD_PATH.name}`")
    add(f"- Rows: **{fmt_int(lcd['rows'])}**")
    add(f"- Date range: **{lcd['date_min']}** → **{lcd['date_max']}**")
    add(f"- Columns ({len(lcd['columns'])}): {', '.join(lcd['columns'])}")
    add("")
    add("**Stations seen:**")
    for s, n in lcd["stations"].items():
        add(f"- `{s}`: {fmt_int(n)}")
    add("")
    add("**Report types:**")
    for s, n in list(lcd["report_types"].items())[:15]:
        add(f"- `{s}`: {fmt_int(n)}")
    add("")
    add(f"- HourlyDryBulbTemperature non-null: {fmt_int(lcd['dry_bulb_present'])} ({lcd['dry_bulb_present']/lcd['rows']*100:.1f}%)")
    add(f"- Hours with at least one dry-bulb obs: {fmt_int(lcd['hours_with_dry_bulb'])}")
    add(f"- Expected hours in dry-bulb range: {fmt_int(lcd['dry_bulb_expected_hours'])}")
    add(f"- Dry bulb coverage: **{lcd['dry_bulb_coverage_pct']:.2f}%**")
    add(f"- Daily max present total: {fmt_int(lcd['daily_max_present_total'])}, "
        f"on SOD: {fmt_int(lcd['daily_max_present_on_sod'])}, off SOD: {fmt_int(lcd['daily_max_present_off_sod'])}")
    add("")
    add("**Hourly column stats:**\n")
    add("| Column | Type | Non-null | Min | Mean | Max | p1 | p99 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for c, st in lcd["hourly_stats"].items():
        if st.get("type") == "categorical":
            add(f"| {c} | categorical | {fmt_int(st['unique'])} unique | | | | | |")
        else:
            mn = "—" if st["min"] is None else f"{st['min']:.2f}"
            mx = "—" if st["max"] is None else f"{st['max']:.2f}"
            mean = "—" if st["mean"] is None else f"{st['mean']:.2f}"
            p1 = "—" if st["p1"] is None else f"{st['p1']:.2f}"
            p99 = "—" if st["p99"] is None else f"{st['p99']:.2f}"
            add(f"| {c} | num | {fmt_int(st['non_null'])} | {mn} | {mean} | {mx} | {p1} | {p99} |")
    add("")
    add("**Daily column stats:**\n")
    add("| Column | Non-null | Min | Mean | Max |")
    add("|---|---:|---:|---:|---:|")
    for c, st in lcd["daily_stats"].items():
        mn = "—" if st["min"] is None else f"{st['min']:.2f}"
        mx = "—" if st["max"] is None else f"{st['max']:.2f}"
        mean = "—" if st["mean"] is None else f"{st['mean']:.2f}"
        add(f"| {c} | {fmt_int(st['non_null'])} | {mn} | {mean} | {mx} |")
    add("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"[audit] wrote {REPORT_PATH}", flush=True)


def main():
    isd = audit_isd()
    lcd = audit_lcd()
    write_report(isd, lcd)
    print("\n=== AUDIT SUMMARY ===")
    print(f"ISD rows={isd['rows']:,}  hours={isd['hours_total']:,}  "
          f"coverage={isd['coverage_pct']:.2f}%  "
          f"TMP F range=[{isd['tmp_f_min']:.1f}, {isd['tmp_f_max']:.1f}]")
    print(f"LCD rows={lcd['rows']:,}  dry-bulb hours={lcd['hours_with_dry_bulb']:,}  "
          f"coverage={lcd['dry_bulb_coverage_pct']:.2f}%")


if __name__ == "__main__":
    main()
