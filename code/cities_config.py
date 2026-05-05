"""Single source of truth for the multi-city config.

Every other script imports `load_cities()` and `get_city(slug)` from here.
The YAML file (config/cities.yaml) is the only place city facts are
written down — slug, ICAO, timezone, Kalshi suffix, raw LCD path.

We also do a few light validations on load so a typo in cities.yaml
fails immediately instead of producing a confusingly empty model later.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Avoid PyYAML dependency by parsing the cities.yaml directly with a tiny
# loader.  cities.yaml is a deliberately flat structure so this works.
# (PyYAML is fine to add later but keeping the dep surface small for now.)
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "cities.yaml"


@dataclass(frozen=True)
class KalshiSeries:
    high_series: Optional[str]
    low_series: Optional[str]


@dataclass(frozen=True)
class City:
    slug: str
    name: str
    icao: str
    nws_office: str
    zip: str
    timezone: str
    lcd_raw: Path
    kalshi: KalshiSeries
    climatology: dict  # month-key -> [low, high]

    # Per-city derived paths.  Centralizing here so no other code spells
    # them out — change the layout once, the whole pipeline follows.
    @property
    def hourly_path(self) -> Path:
        return REPO_ROOT / "data" / f"{self.slug}_hourly.parquet"

    @property
    def daily_path(self) -> Path:
        return REPO_ROOT / "data" / f"{self.slug}_daily.parquet"

    @property
    def features_path(self) -> Path:
        return REPO_ROOT / "data" / f"{self.slug}_features.parquet"

    @property
    def targets_path(self) -> Path:
        return REPO_ROOT / "data" / f"{self.slug}_targets.parquet"

    @property
    def climatology_path(self) -> Path:
        return REPO_ROOT / "data" / f"{self.slug}_climatology.parquet"

    @property
    def models_dir(self) -> Path:
        return REPO_ROOT / "models" / self.slug

    @property
    def live_signals_path(self) -> Path:
        return REPO_ROOT / "reports" / f"live_signals_{self.slug}.json"

    @property
    def forecast_path(self) -> Path:
        return REPO_ROOT / "reports" / f"forecast_{self.slug}.json"

    @property
    def trains_high(self) -> bool:
        return self.kalshi.high_series is not None

    @property
    def trains_low(self) -> bool:
        return self.kalshi.low_series is not None

    def kalshi_series(self) -> list[str]:
        out = []
        if self.kalshi.high_series:
            out.append(self.kalshi.high_series)
        if self.kalshi.low_series:
            out.append(self.kalshi.low_series)
        return out


def _parse_yaml_simple(text: str) -> dict:
    """Tiny YAML subset parser — only what cities.yaml uses.

    Supports: top-level keys, nested mappings, lists of mappings, list scalars,
    inline lists like [1, 2], comments after #, basic quoting.

    If PyYAML is available we use it; otherwise this fallback handles the
    flat structure used by cities.yaml.
    """
    if HAVE_YAML:
        return yaml.safe_load(text)

    # Manual parse for the specific shape of cities.yaml.
    # We assume 2-space indentation and very limited YAML features.
    lines = text.splitlines()
    out: dict = {}
    cities: list[dict] = []
    cur_city: Optional[dict] = None
    cur_dict_key: Optional[str] = None
    i = 0

    def strip_comment(s: str) -> str:
        idx = s.find("#")
        return s[:idx].rstrip() if idx >= 0 else s.rstrip()

    def parse_value(v: str):
        v = v.strip()
        if v == "" or v.lower() == "null":
            return None
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [parse_value(p) for p in parts]
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            return v[1:-1]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    while i < len(lines):
        raw = lines[i]
        line = strip_comment(raw)
        if not line.strip():
            i += 1
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        # Top-level "cities:" trigger
        if indent == 0 and stripped == "cities:":
            i += 1
            continue

        # Start of a city entry: "  - slug: foo"
        if indent == 2 and stripped.startswith("- "):
            cur_city = {}
            cities.append(cur_city)
            stripped = stripped[2:]
            indent = 4
            cur_dict_key = None
            # fall through to handle the key:value on this line
        # Nested dict header: "    kalshi:" (with no value on same line)
        if indent == 4 and stripped.endswith(":") and ":" in stripped[:-1] is False:
            cur_dict_key = stripped[:-1]
            assert cur_city is not None
            cur_city[cur_dict_key] = {}
            i += 1
            continue
        # 6-space-indent line under a nested dict
        if indent == 6 and cur_dict_key:
            assert cur_city is not None
            k, _, v = stripped.partition(":")
            cur_city[cur_dict_key][k.strip()] = parse_value(v)
            i += 1
            continue
        # 4-space-indent simple key:value
        if indent == 4:
            assert cur_city is not None
            cur_dict_key = None
            k, _, v = stripped.partition(":")
            v = v.strip()
            if v == "":
                # Next lines are nested
                cur_dict_key = k.strip()
                cur_city[cur_dict_key] = {}
            else:
                cur_city[k.strip()] = parse_value(v)
            i += 1
            continue
        i += 1

    out["cities"] = cities
    return out


_CACHED: Optional[list[City]] = None


def load_cities() -> list[City]:
    """Load and validate cities.yaml.  Cached after first call."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"cities config not found: {CONFIG_PATH}")

    text = CONFIG_PATH.read_text()
    data = _parse_yaml_simple(text)
    raw_cities = data.get("cities") or []
    out: list[City] = []
    seen_slugs: set[str] = set()
    seen_icao: set[str] = set()

    for raw in raw_cities:
        slug = raw["slug"]
        if slug in seen_slugs:
            raise ValueError(f"duplicate city slug: {slug}")
        seen_slugs.add(slug)
        if raw["icao"] in seen_icao:
            raise ValueError(f"duplicate ICAO {raw['icao']} (cities {slug} and existing)")
        seen_icao.add(raw["icao"])

        kal = raw.get("kalshi") or {}
        kalshi = KalshiSeries(
            high_series=kal.get("high_series"),
            low_series=kal.get("low_series"),
        )
        if not kalshi.high_series and not kalshi.low_series:
            raise ValueError(f"city {slug} has no Kalshi markets configured")

        city = City(
            slug=slug,
            name=raw["name"],
            icao=raw["icao"],
            nws_office=raw.get("nws_office", ""),
            zip=str(raw.get("zip", "")),
            timezone=raw["timezone"],
            lcd_raw=REPO_ROOT / raw["lcd_raw"],
            kalshi=kalshi,
            climatology=raw.get("climatology", {}),
        )
        out.append(city)

    _CACHED = out
    return out


def get_city(slug: str) -> City:
    """Look up a city by slug.  Raises KeyError if not found."""
    for c in load_cities():
        if c.slug == slug:
            return c
    raise KeyError(f"city slug not found: {slug}")


def all_slugs() -> list[str]:
    return [c.slug for c in load_cities()]


def kalshi_series_to_city() -> dict[str, City]:
    """Reverse map: Kalshi series ticker → City."""
    out: dict[str, City] = {}
    for c in load_cities():
        if c.kalshi.high_series:
            out[c.kalshi.high_series] = c
        if c.kalshi.low_series:
            out[c.kalshi.low_series] = c
    return out


if __name__ == "__main__":
    # Quick sanity dump for `python3 code/cities_config.py`
    cities = load_cities()
    print(f"loaded {len(cities)} cities from {CONFIG_PATH}")
    for c in cities:
        h = c.kalshi.high_series or "—"
        l = c.kalshi.low_series or "—"
        print(f"  {c.slug:>4}  {c.icao}  {c.timezone:<22}  HIGH={h:<14}  LOW={l}")
