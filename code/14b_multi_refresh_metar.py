"""
14b_multi_refresh_metar.py — Pull the latest METARs for every configured city
from the NWS Aviation Weather API and merge them into each city's hourly grid.

This is the multi-city counterpart to 14_refresh_noaa.py.  Same API, same
parsing logic, but loops over cities.yaml.  Per-city timezone discipline:
  * NWS API returns UTC `obsTime`.
  * We subtract the city's LOCAL STANDARD TIME UTC offset (not the IANA
    rule with DST!) so the resulting `hour` column matches the LCD parquet
    that 02b built (LCD is always local-standard).

Reads:  data/{slug}_hourly.parquet            (existing canonical grid)
        config/cities.yaml                     (ICAO + IANA timezone)
Writes: data/{slug}_hourly.parquet             (updated with recent rows)

Usage:
  python3 code/14b_multi_refresh_metar.py             # all cities
  python3 code/14b_multi_refresh_metar.py --hours 168 # last week
  python3 code/14b_multi_refresh_metar.py --city nyc

Cities whose station does not produce JSON-format METARs (rare — e.g. KNYC
is a special non-airport identifier) gracefully fall back: the row is just
not added, no crash.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import warnings

import numpy as np
import pandas as pd

# pandas warns about all-NA column concatenation behavior changing — suppress
# since our schemas are stable and the warning fires on every city.
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import City, get_city, load_cities  # noqa: E402

API = "https://aviationweather.gov/api/data/metar"

# Local-standard-time UTC offset by IANA timezone, hours.  This is the offset
# the LCD parquet uses (no DST shift).  We only need a small set since cities.yaml
# uses a known set of zones.
LOCAL_STD_OFFSET_H = {
    "America/Los_Angeles": 8,   # PST = UTC-8
    "America/Denver":      7,   # MST = UTC-7
    "America/Phoenix":     7,   # MST year-round
    "America/Chicago":     6,   # CST = UTC-6
    "America/New_York":    5,   # EST = UTC-5
}


def http_get_json(url: str, timeout: int = 30) -> list:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "ariadne-multi-metar/1.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def parse_visibility(v) -> float | None:
    if v is None: return None
    s = str(v).strip()
    if s.endswith("+"): s = s[:-1]
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
    if m: return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m: return float(m.group(1)) / float(m.group(2))
    try: return float(s)
    except ValueError: return None


def magnus_rh(temp_c, dew_c):
    if temp_c is None or dew_c is None: return None
    a, b = 17.625, 243.04
    es = np.exp(a * temp_c / (b + temp_c))
    ed = np.exp(a * dew_c / (b + dew_c))
    return float(100.0 * ed / es)


def parse_metars(metars: list[dict], local_std_offset_h: int) -> pd.DataFrame:
    rows = []
    for m in metars:
        ts = m.get("obsTime")
        if ts is None: continue
        utc = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        local_std = utc - pd.Timedelta(hours=local_std_offset_h)

        temp_c = m.get("temp"); dew_c = m.get("dewp")
        temp_f = (float(temp_c) * 9 / 5 + 32) if temp_c is not None else None
        dew_f  = (float(dew_c)  * 9 / 5 + 32) if dew_c  is not None else None
        rh = magnus_rh(temp_c, dew_c) if (temp_c is not None and dew_c is not None) else None
        vis_mi = parse_visibility(m.get("visib"))
        slp_mb = m.get("altim")
        slp_inhg = (float(slp_mb) / 33.8639) if slp_mb is not None else None
        if m.get("slp") is not None:
            slp_inhg = float(m["slp"]) / 33.8639
        wspd = m.get("wspd"); wgst = m.get("wgst"); wdir = m.get("wdir")
        wspd_mph = (float(wspd) * 1.15078) if wspd is not None else None
        wgst_mph = (float(wgst) * 1.15078) if wgst is not None else None
        if isinstance(wdir, str) and wdir.upper() == "VRB":
            wdir = None

        # Cloud cover
        clouds = m.get("clouds") or []
        cover_codes = {"CLR": 0, "SKC": 0, "FEW": 2, "SCT": 4, "BKN": 7, "OVC": 9, "VV": 9}
        max_cover = None; overcast = 0; obscured = 0
        for layer in clouds:
            cov = (layer.get("cover") or "").upper()
            n = cover_codes.get(cov)
            if n is not None:
                if max_cover is None or n > max_cover: max_cover = n
                if cov == "OVC": overcast = 1
                if cov == "VV":  obscured = 1

        rows.append({
            "hour":         pd.Timestamp(local_std).floor("h"),
            "temp_f":       temp_f,
            "dew_f":        dew_f,
            "rh":           rh,
            "wetbulb_f":    None,
            "slp_inhg":     slp_inhg,
            "p_change":     None,
            "p_tendency":   None,
            "vis_mi":       vis_mi,
            "wind_dir":     float(wdir) if wdir is not None else None,
            "wind_speed":   wspd_mph,
            "wind_gust":    wgst_mph,
            "precip_in":    None,
            "cloud_max":    max_cover,
            "sky_overcast": overcast,
            "sky_obscured": obscured,
            "lcd_rt":       "FM-15",   # METAR
            "isd_qc":       pd.NA,
            "isd_rt":       pd.NA,
            "temp_source":  "metar",
        })
    return pd.DataFrame(rows)


def refresh_one(city: City, hours: int) -> dict:
    if city.timezone not in LOCAL_STD_OFFSET_H:
        return {"slug": city.slug, "error": f"unsupported timezone {city.timezone}"}
    offset_h = LOCAL_STD_OFFSET_H[city.timezone]

    if not city.hourly_path.exists():
        return {"slug": city.slug, "error": f"hourly parquet missing — run 02b_build_lcd_dataset first"}

    t0 = time.time()
    url = f"{API}?ids={city.icao}&format=json&hours={hours}"
    try:
        metars = http_get_json(url)
    except urllib.error.HTTPError as e:
        return {"slug": city.slug, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"slug": city.slug, "error": str(e)}

    if not metars:
        return {"slug": city.slug, "error": "no METARs returned (station may not be in NWS feed)"}

    new_rows = parse_metars(metars, offset_h)
    if new_rows.empty:
        return {"slug": city.slug, "error": "parsed empty"}

    # Pick best obs per hour from the new METAR set
    new_rows = new_rows.sort_values("hour").drop_duplicates("hour", keep="last")
    n_new = len(new_rows)

    # Merge with existing parquet
    existing = pd.read_parquet(city.hourly_path)
    # Keep new where existing is missing or older; existing values otherwise
    new_rows["timezone"] = city.timezone
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.sort_values("hour").drop_duplicates("hour", keep="last")
    merged = merged.reset_index(drop=True)

    n_added = len(merged) - len(existing)

    merged.to_parquet(city.hourly_path, index=False)
    return {
        "slug":      city.slug,
        "n_metars":  int(len(metars)),
        "n_new_hours": n_new,
        "n_added":   int(n_added),
        "max_hour":  str(merged["hour"].max()),
        "duration_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", help="single city slug")
    ap.add_argument("--hours", type=int, default=168,
                    help="how many hours of METARs to pull (default 168 = 1 week)")
    args = ap.parse_args()

    if args.city:
        cities = [get_city(args.city)]
    else:
        cities = load_cities()

    print(f"[multi-metar] refreshing {len(cities)} cities, last {args.hours}h")
    results = []
    for c in cities:
        r = refresh_one(c, args.hours)
        results.append(r)
        if "error" in r:
            print(f"  {r['slug']:>4}  ERR: {r['error']}", flush=True)
        else:
            print(f"  {r['slug']:>4}  +{r['n_added']:>3} hours  latest={r['max_hour']}  ({r['duration_s']}s)", flush=True)

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"[multi-metar] done: {n_ok}/{len(cities)} ok")


if __name__ == "__main__":
    main()
