"""
14_refresh_noaa.py — Pull the latest KSFO METARs from the NWS Aviation Weather
API and merge them into the canonical hourly grid, then rebuild features so
the live-signal pipeline uses up-to-date observations.

This closes the freshness gap between the static LCD/ISD CSVs (which lag by
days/weeks) and live trading. NO authentication required.

Source:
  https://aviationweather.gov/api/data/metar?ids=KSFO&format=json&hours=N

Reads:
  data/sfo_hourly.parquet  (existing canonical grid)

Writes:
  data/sfo_hourly.parquet  (updated with recent rows)
  data/sfo_features.parquet (regenerated for the last ~30 days)

Usage:
  python3 code/14_refresh_noaa.py [--hours 72]

Notes:
  * METAR temp / dew are in °C. We convert to °F (rounded to 0.1 for parity
    with LCD which is integer °F).
  * METAR pressure (`altim`) is millibars; we convert to inHg.
  * Wind speed is in knots; we convert to mph.
  * Visibility "10+" → 10 mi; fractional "1 1/2" → 1.5 mi.
  * RH is recomputed from temp+dew via Magnus.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
HOURLY_PATH = ROOT / "data" / "sfo_hourly.parquet"
DAILY_PATH = ROOT / "data" / "sfo_daily.parquet"

API = "https://aviationweather.gov/api/data/metar"


def fetch_metars(hours: int = 72) -> list[dict]:
    url = f"{API}?ids=KSFO&format=json&hours={hours}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "sfo-weather-research/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_visibility(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s.endswith("+"):
        s = s[:-1]
    # Fractional like "1 1/2" or "1/2"
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def magnus_rh(temp_c, dew_c):
    if temp_c is None or dew_c is None:
        return None
    a, b = 17.625, 243.04
    es = np.exp(a * temp_c / (b + temp_c))
    ed = np.exp(a * dew_c / (b + dew_c))
    return float(100.0 * ed / es)


def parse_metars_to_hourly(metars: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metars:
        ts = m.get("obsTime")
        if ts is None:
            continue
        # obsTime is Unix seconds. Convert to PST.
        utc = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        pst = utc - pd.Timedelta(hours=8)
        temp_c = m.get("temp")
        dew_c = m.get("dewp")
        rh = magnus_rh(temp_c, dew_c) if temp_c is not None and dew_c is not None else None
        vis_mi = parse_visibility(m.get("visib"))
        slp_mb = m.get("altim")  # use altimeter (typical METAR field) as slp proxy
        slp_inhg = (float(slp_mb) / 33.8639) if slp_mb is not None else None
        # also try the slp field if present
        if m.get("slp") is not None:
            slp_inhg = float(m["slp"]) / 33.8639
        wspd_kt = m.get("wspd")
        wspd_mph = float(wspd_kt) * 1.15078 if wspd_kt is not None else None
        wgst_kt = m.get("wgst")
        wgst_mph = float(wgst_kt) * 1.15078 if wgst_kt is not None else None
        wdir = m.get("wdir")
        if isinstance(wdir, str) and wdir.upper() == "VRB":
            wdir = None

        # Cloud info
        clouds = m.get("clouds") or []
        max_cover = None
        sky_overcast = 0
        sky_obscured = 0
        cover_codes = {"CLR": 0, "SKC": 0, "FEW": 2, "SCT": 4, "BKN": 7, "OVC": 9, "VV": 9}
        for layer in clouds:
            c = (layer.get("cover") or "").upper()
            v = cover_codes.get(c)
            if v is not None and (max_cover is None or v > max_cover):
                max_cover = v
            if c == "OVC": sky_overcast = 1
            if c == "VV":  sky_obscured = 1

        rows.append({
            "hour": pst.floor("h"),
            "raw_obs_time_pst": pst,
            "temp_f": temp_c * 9.0 / 5.0 + 32.0 if temp_c is not None else None,
            "dew_f":  dew_c * 9.0 / 5.0 + 32.0 if dew_c is not None else None,
            "rh":     rh,
            "slp_inhg": slp_inhg,
            "wind_dir": float(wdir) if wdir is not None else None,
            "wind_speed": wspd_mph,
            "wind_gust": wgst_mph,
            "vis_mi": vis_mi,
            "cloud_max": float(max_cover) if max_cover is not None else None,
            "sky_overcast": pd.Series([sky_overcast], dtype="Int8")[0],
            "sky_obscured": pd.Series([sky_obscured], dtype="Int8")[0],
            "temp_source": "metar_live",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Per hour bin, keep the obs closest to HH:53 (METAR convention)
    # Simpler: keep the latest within each hour bin
    df = df.sort_values(["hour", "raw_obs_time_pst"])
    df = df.drop_duplicates("hour", keep="last").drop(columns=["raw_obs_time_pst"])
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=72, help="hours of METAR history to fetch (max 168)")
    args = ap.parse_args()
    hours = min(168, args.hours)

    print(f"[refresh] fetching {hours}h of KSFO METARs ...", flush=True)
    raw = fetch_metars(hours)
    print(f"[refresh] received {len(raw)} METAR records", flush=True)

    new = parse_metars_to_hourly(raw)
    if new.empty:
        print("[refresh] nothing parsed; aborting", flush=True)
        return
    print(f"[refresh] parsed {len(new)} hourly bins ({new['hour'].min()} → {new['hour'].max()} PST)",
          flush=True)

    # Load existing
    if HOURLY_PATH.exists():
        existing = pd.read_parquet(HOURLY_PATH)
    else:
        raise SystemExit(f"No existing {HOURLY_PATH}; run 02_build_dataset.py first.")

    # Upsert: replace rows where new.hour overlaps existing.hour, keep new values where
    # temp_f was missing in existing.
    merged = existing.set_index("hour").combine_first(new.set_index("hour"))
    # Where new has a temp_f and existing's was NaN OR new is more recent METAR, prefer new.
    # Specifically: for hour bins in new, we override with new for the columns we just measured.
    new_idx = new.set_index("hour").index
    for col in ["temp_f", "dew_f", "rh", "slp_inhg", "wind_dir", "wind_speed", "wind_gust",
                "vis_mi", "cloud_max", "sky_overcast", "sky_obscured"]:
        if col in new.columns:
            merged.loc[new_idx, col] = new.set_index("hour")[col]
    merged.loc[new_idx, "temp_source"] = "metar_live"

    # Extend hour grid if METARs go beyond existing range
    full_idx = pd.date_range(merged.index.min(), max(merged.index.max(), new_idx.max()),
                             freq="h", name="hour")
    merged = merged.reindex(full_idx)
    merged = merged.reset_index()
    print(f"[refresh] hourly grid now: {len(merged):,} rows ({merged['hour'].min()} → {merged['hour'].max()})",
          flush=True)
    merged.to_parquet(HOURLY_PATH, index=False)
    print(f"[refresh] wrote {HOURLY_PATH}", flush=True)

    # Trigger feature regeneration
    print("[refresh] regenerating features (calls 03_features.py) ...", flush=True)
    import subprocess
    r = subprocess.run(["python3", str(ROOT / "code" / "03_features.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print("[refresh] feature regeneration FAILED:", r.stderr[-500:])
        raise SystemExit(1)
    # Print final lines
    for line in r.stdout.splitlines()[-5:]:
        print("  " + line)
    print("[refresh] features regenerated. Now run:")
    print("    python3 code/13_live_signal.py")


if __name__ == "__main__":
    main()
