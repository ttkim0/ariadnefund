"""
15_dashboard.py — Generate a static HTML dashboard for SFO weather forecasts
and Kalshi trading signals.

Reads:
  data/sfo_hourly.parquet
  data/sfo_features.parquet
  reports/forecast.json
  reports/live_signals.json
  reports/daily_extreme_metrics.json
  models/dxmodel_{kind}_q{Q}.joblib  (for daily HIGH/LOW per day)

Writes:
  reports/dashboard.html  (open in any browser; no server needed)

Charts:
  * Now-bar: current temp/dew/wind, today's high/low so far
  * Daily HIGH forecast — next 4 days, with explicit dates and 80% interval
  * Daily LOW forecast — next 4 days, with explicit dates and 80% interval
  * Per-day bucket probabilities for upcoming Kalshi markets (us vs market)
  * Recent observations — last 72h
  * Actionable trades table
  * Hourly forecast samples (clearly labeled as discrete sample points, not a
    continuous prediction; explanatory note included)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
HOURLY_PATH = ROOT / "data" / "sfo_hourly.parquet"
FEAT_PATH = ROOT / "data" / "sfo_features.parquet"
DX_META = ROOT / "reports" / "daily_extreme_metrics.json"
FORECAST_JSON = ROOT / "reports" / "forecast.json"
SIGNALS_JSON = ROOT / "reports" / "live_signals.json"
MODEL_DIR = ROOT / "models"
DASH_OUT = ROOT / "reports" / "dashboard.html"

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def to_f32_row(row: pd.Series, cols: list[str], extra: dict[str, float]) -> np.ndarray:
    arr = np.empty(len(cols), dtype="float32")
    for i, c in enumerate(cols):
        if c in extra:
            arr[i] = float(extra[c])
            continue
        v = row[c] if c in row.index else np.nan
        arr[i] = np.nan if pd.isna(v) else float(v)
    return arr.reshape(1, -1)


def isotonic(qpred: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(qpred, axis=1)


def predict_daily_extreme(kind: str, feat_row: pd.Series, fcols: list[str],
                          hours_to_settle: float) -> list[float]:
    qpred = np.empty((1, len(QUANTILES)))
    x = to_f32_row(feat_row, fcols, {"hours_to_settle": hours_to_settle})
    for j, q in enumerate(QUANTILES):
        m = joblib.load(MODEL_DIR / f"dxmodel_{kind}_q{int(q*100):02d}.joblib")
        qpred[0, j] = float(m.predict(x)[0])
    qpred = isotonic(qpred)
    return qpred[0].tolist()


def main():
    print("[dash] reading inputs ...", flush=True)
    hourly = pd.read_parquet(HOURLY_PATH)
    features = pd.read_parquet(FEAT_PATH)
    dx_meta = json.loads(DX_META.read_text())
    fcols = dx_meta["feature_cols"]

    forecast = json.loads(FORECAST_JSON.read_text()) if FORECAST_JSON.exists() else None
    signals = json.loads(SIGNALS_JSON.read_text()) if SIGNALS_JSON.exists() else None

    latest = features[features["temp_f"].notna()].iloc[-1]
    issued = pd.Timestamp(latest["hour"])
    print(f"[dash] issuance time: {issued} PST", flush=True)

    # Today's observations so far
    today_pst = issued.floor("D")
    today_obs = hourly[(hourly["hour"] >= today_pst) & (hourly["hour"] <= issued)].dropna(subset=["temp_f"])
    today_high_so_far = float(today_obs["temp_f"].max()) if not today_obs.empty else None
    today_low_so_far = float(today_obs["temp_f"].min()) if not today_obs.empty else None

    # Last 72h observations
    obs = hourly.dropna(subset=["temp_f"]).tail(72).copy()
    obs_chart = {
        "x": [str(t) for t in obs["hour"]],
        "y": [float(v) for v in obs["temp_f"]],
        "dew": [float(v) if pd.notna(v) else None for v in obs["dew_f"]],
    }

    # Hourly forecast samples
    fan = None
    if forecast and "forecast" in forecast:
        fan = []
        for h in forecast["forecast"]:
            fan.append({
                "valid_time": h["valid_time"],
                "horizon": h["horizon"],
                "median": h["median"],
                "q10": h["q10"], "q90": h["q90"],
            })

    # Next 4 calendar days HIGH and LOW from daily-extreme model
    daily_pts = []
    for k in range(0, 4):
        target_day = today_pst + pd.Timedelta(days=k)
        settle_ts = target_day + pd.Timedelta(days=1)
        hours_to_settle = (settle_ts - issued).total_seconds() / 3600.0
        if hours_to_settle <= 0:
            continue
        qhigh = predict_daily_extreme("high", latest, fcols, hours_to_settle)
        qlow = predict_daily_extreme("low",  latest, fcols, hours_to_settle)
        daily_pts.append({
            "day": str(target_day.date()),
            "hours_to_settle": round(hours_to_settle, 1),
            "high_q10": qhigh[1], "high_q50": qhigh[3], "high_q90": qhigh[5],
            "low_q10":  qlow[1],  "low_q50":  qlow[3],  "low_q90":  qlow[5],
        })

    # Per-day bucket charts (HIGH and LOW), grouped by day
    bucket_charts = []
    if signals and signals.get("signals"):
        for series, kind in [("KXHIGHTSFO", "HIGH"), ("KXLOWTSFO", "LOW")]:
            days_seen = {}
            for s in signals["signals"]:
                if s.get("series_ticker") != series:
                    continue
                day = s.get("day_D")
                days_seen.setdefault(day, []).append(s)
            for day, sigs in sorted(days_seen.items()):
                buckets = []
                for s in sigs:
                    p_market = None
                    if s.get("yes_bid") is not None and s.get("yes_ask") is not None:
                        p_market = 0.5 * (s["yes_bid"] + s["yes_ask"])
                    stype = s.get("strike_type")
                    if stype == "less":
                        sort_v = -1000
                    elif stype == "greater":
                        sort_v = 1000
                    else:
                        sort_v = (s.get("floor_strike") or 0) + (s.get("cap_strike") or 0)
                    buckets.append({
                        "label": s.get("yes_sub_title") or "",
                        "sort": sort_v,
                        "p_model": s.get("model_prob_yes"),
                        "p_market": p_market,
                        "ticker": s.get("ticker"),
                    })
                buckets.sort(key=lambda b: b["sort"])
                bucket_charts.append({"kind": kind, "day_D": day, "buckets": buckets})

    actionable = []
    if signals and signals.get("signals"):
        for s in signals["signals"]:
            if s.get("trade_side") is None:
                continue
            actionable.append(s)

    # ---------- Joint outcomes for best HIGH+LOW pair per day ----------
    joint_outcomes = []
    if actionable:
        by_day = {}
        for s in actionable:
            by_day.setdefault(s["day_D"], {"HIGH": None, "LOW": None})
            kind = "LOW" if s["series_ticker"] == "KXLOWTSFO" else "HIGH"
            current = by_day[s["day_D"]][kind]
            if current is None or (s.get("trade_ev_per_c") or -1) > (current.get("trade_ev_per_c") or -1):
                by_day[s["day_D"]][kind] = s
        for day in sorted(by_day.keys()):
            pair = by_day[day]
            if not pair["HIGH"] or not pair["LOW"]:
                continue
            ph = pair["HIGH"].get("p_final") or pair["HIGH"].get("model_prob_yes") or 0
            pl = pair["LOW"].get("p_final")  or pair["LOW"].get("model_prob_yes")  or 0
            # Probability we WIN each trade (not the bucket settling YES — depends on side)
            # If trade_side == 'yes', we win if YES → use ph as is for HIGH, pl as is for LOW
            # If trade_side == 'no',  we win if NO  → use 1-ph for HIGH, 1-pl for LOW
            wh = (1.0 - ph) if pair["HIGH"]["trade_side"] == "no" else ph
            wl = (1.0 - pl) if pair["LOW"]["trade_side"]  == "no" else pl
            joint_outcomes.append({
                "day": day,
                "high_ticker": pair["HIGH"]["ticker"],
                "low_ticker":  pair["LOW"]["ticker"],
                "high_side":   pair["HIGH"]["trade_side"],
                "low_side":    pair["LOW"]["trade_side"],
                "high_bucket": pair["HIGH"]["yes_sub_title"],
                "low_bucket":  pair["LOW"]["yes_sub_title"],
                "p_high_win":  round(wh, 3),
                "p_low_win":   round(wl, 3),
                "both_win":    round(wh * wl, 3),
                "high_only":   round(wh * (1 - wl), 3),
                "low_only":    round((1 - wh) * wl, 3),
                "both_lose":   round((1 - wh) * (1 - wl), 3),
                "high_cost":   pair["HIGH"]["trade_cost"],
                "low_cost":    pair["LOW"]["trade_cost"],
                "high_ev":     pair["HIGH"]["trade_ev_per_c"],
                "low_ev":      pair["LOW"]["trade_ev_per_c"],
            })

    # The actual time this dashboard was generated (for "is the auto-refresh working?" diagnosis)
    generated_at = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=8)  # PST
    generated_at_pdt = generated_at + pd.Timedelta(hours=1)  # naive PDT during DST
    data_blob = {
        "issued": str(issued),
        "issued_label": issued.strftime("%a %b %-d, %Y at %-I:%M %p PST"),
        "today_pst": str(today_pst.date()),
        "today_label": today_pst.strftime("%a %b %-d, %Y"),
        "generated_at_pst": generated_at.strftime("%Y-%m-%d %H:%M:%S PST"),
        "generated_at_pdt": generated_at_pdt.strftime("%Y-%m-%d %H:%M:%S PDT"),
        "current_temp": float(latest["temp_f"]),
        "current_dew": float(latest["dew_f"]) if pd.notna(latest["dew_f"]) else None,
        "current_wind_speed": float(latest["wind_speed"]) if pd.notna(latest["wind_speed"]) else None,
        "current_wind_dir": float(latest["wind_dir"]) if pd.notna(latest["wind_dir"]) else None,
        "today_high_so_far": today_high_so_far,
        "today_low_so_far": today_low_so_far,
        "obs": obs_chart,
        "fan": fan,
        "daily": daily_pts,
        "bucket_charts": bucket_charts,
        "actionable": actionable,
        "joint_outcomes": joint_outcomes,
    }

    html = HTML_TEMPLATE.replace(
        "__DATA_BLOB__",
        json.dumps(data_blob, indent=2, default=str)
    )
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(html)
    print(f"[dash] wrote {DASH_OUT}", flush=True)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<title>SFO Weather Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #fff; color: #111; font: 14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; margin: 0; padding: 18px; max-width: 1280px; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  h2 { margin: 22px 0 8px; font-size: 16px; }
  h3 { margin: 14px 0 6px; font-size: 14px; font-weight: 600; }
  .small { color: #666; font-size: 12px; }
  .panel { border: 1px solid #ccc; padding: 12px; margin-bottom: 12px; }
  .nowbar { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 12px; }
  .stat { border: 1px solid #ccc; padding: 8px 12px; min-width: 130px; }
  .stat .v { font-size: 20px; font-weight: 700; }
  .stat .l { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #ddd; }
  th { background: #f3f3f3; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .yes { color: #1a6e1a; font-weight: 600; }
  .no  { color: #a82929; font-weight: 600; }
  .row2 { display: grid; gap: 12px; grid-template-columns: 1fr 1fr; }
  .explain { background: #fffbe6; border: 1px solid #e6d99a; padding: 10px 12px; font-size: 12px; margin-bottom: 12px; }
  @media (max-width: 760px) { .row2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>SFO Weather + Kalshi Dashboard</h1>
<div class="small" id="header-sub"></div>
<div class="small" id="page-loaded"></div>
<div class="small" id="freshness"></div>

<div class="nowbar" id="nowbar"></div>

<div class="row2">
  <div>
    <h2 id="ddh_title">Daily HIGH forecast — next 4 days</h2>
    <div class="panel"><div id="daily_high_chart" style="height:300px;"></div></div>
  </div>
  <div>
    <h2 id="ddl_title">Daily LOW forecast — next 4 days</h2>
    <div class="panel"><div id="daily_low_chart" style="height:300px;"></div></div>
  </div>
</div>

<h2>Joint outcomes — if you take BOTH the best HIGH and best LOW trade for each day</h2>
<div class="explain">
  <b>This is the math you asked about.</b> Probabilities <i>multiply</i> for independent events (you can't add them).
  The table below shows, for each day where we have an actionable HIGH bet AND an actionable LOW bet,
  the probability that <b>both win</b>, <b>only HIGH wins</b>, <b>only LOW wins</b>, or <b>both lose</b>.
  These come from your actual model probabilities for the trades, not a generic 65%.
</div>
<div id="joint_panel"></div>

<h2>Bucket probabilities for upcoming Kalshi markets</h2>
<div class="explain">
  <b>How to read this:</b> Each bar shows the probability the daily HIGH (or LOW) for that calendar day
  lands in that 2°F bucket. Blue = our model. Yellow = Kalshi market mid price (= market's implied probability).
  Numeric % is shown above each bar. When our blue is taller, we think it's underpriced; when shorter, overpriced.
</div>
<div id="bucket_charts_container"></div>

<h2>Recent SFO observations — last 72h</h2>
<div class="panel"><div id="obs_chart" style="height:280px;"></div></div>

<h2>Actionable Kalshi trades (after fees and spread crossing)</h2>
<div id="signals_panel"></div>

<h2>Hourly forecast samples — diagnostic only</h2>
<div class="explain">
  <b>Why this looks lower than the daily HIGH chart above:</b> The hourly model is sampled at exactly
  7 horizons from issuance time: <code>+1h, +3h, +6h, +12h, +24h, +48h, +72h</code>.
  None of those land at SF's typical daily peak hour (12-13 PST), so the points below systematically miss
  the warmest part of each day. <b>For Kalshi daily-HIGH/LOW contracts, use the daily-extreme charts
  above</b> (which are produced by a different model targeting day-max/day-min directly). This panel is
  shown only to display short-horizon dynamics, not to predict daily peaks.
</div>
<div class="panel"><div id="fan_chart" style="height:280px;"></div></div>

<div class="small" style="margin-top:18px">
  Built from NOAA LCD + ISD + live KSFO METARs. All times PST (UTC−8). Buckets settle on NWS Climatological Report for SFO.
</div>

<script id="data" type="application/json">__DATA_BLOB__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const fmtDay = ymd => {
  const [y, m, d] = ymd.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  return dt.toLocaleDateString(undefined, {weekday:'short', month:'short', day:'numeric'});
};
document.getElementById('page-loaded').textContent = 'page loaded ' + new Date().toLocaleString();
document.getElementById('header-sub').textContent =
  'forecasts issued at ' + DATA.issued_label + '  ·  today is ' + DATA.today_label;

// Compute and display data freshness so you can tell if the auto-refresh ran
const generatedAt = new Date(DATA.generated_at_pdt.replace(' PDT', ''));
const ageSeconds = (new Date() - generatedAt) / 1000;
const ageStr = ageSeconds < 120 ? Math.round(ageSeconds) + ' sec ago'
             : ageSeconds < 7200 ? Math.round(ageSeconds/60) + ' min ago'
             : Math.round(ageSeconds/3600) + ' hours ago';
const freshColor = ageSeconds < 3900 ? '#1a6e1a' : '#a82929';
document.getElementById('freshness').innerHTML =
  '<span style="color:' + freshColor + '; font-weight: 600;">dashboard last regenerated: ' +
  DATA.generated_at_pdt + ' (' + ageStr + ')</span>' +
  ' — auto-refresh runs hourly via launchd';

const nowbar = document.getElementById('nowbar');
function stat(label, value) {
  const d = document.createElement('div');
  d.className = 'stat';
  d.innerHTML = '<div class="v">' + value + '</div><div class="l">' + label + '</div>';
  nowbar.appendChild(d);
}
stat('current temp', DATA.current_temp.toFixed(0) + '°F');
if (DATA.current_dew !== null) stat('dew point', DATA.current_dew.toFixed(0) + '°F');
if (DATA.current_wind_speed !== null) {
  let s = DATA.current_wind_speed.toFixed(0) + ' mph';
  if (DATA.current_wind_dir !== null) s += ' from ' + Math.round(DATA.current_wind_dir) + '°';
  stat('wind', s);
}
if (DATA.today_high_so_far !== null) stat("today's high so far", DATA.today_high_so_far.toFixed(0) + '°F');
if (DATA.today_low_so_far !== null)  stat("today's low so far",  DATA.today_low_so_far.toFixed(0)  + '°F');
if (DATA.actionable) stat('actionable trades', DATA.actionable.length);

// ---------- Daily HIGH / LOW bar charts with explicit dates ----------
function dailyChart(divId, kind) {
  if (!DATA.daily) return;
  const xLabels = DATA.daily.map(d => fmtDay(d.day) + '<br>' + d.day);
  const med = DATA.daily.map(d => d[kind + '_q50']);
  const q10 = DATA.daily.map(d => d[kind + '_q10']);
  const q90 = DATA.daily.map(d => d[kind + '_q90']);
  const lo = med.map((m, i) => m - q10[i]);
  const hi = med.map((m, i) => q90[i] - m);
  const color = kind === 'high' ? '#1a6e1a' : '#1f4ea8';
  const traces = [{
    x: xLabels, y: med, type: 'bar',
    marker: {color: color, opacity: 0.65, line: {color: color, width: 1.2}},
    text: med.map((m, i) =>
      m.toFixed(1) + '°F<br><span style="font-size:10px;color:#444">[' +
      q10[i].toFixed(0) + '–' + q90[i].toFixed(0) + ']</span>'),
    textposition: 'outside',
    error_y: {type: 'data', symmetric: false, array: hi, arrayminus: lo,
              color: color, thickness: 1.5, width: 8},
  }];
  Plotly.newPlot(divId, traces, {
    margin: {l: 50, r: 20, t: 10, b: 60}, font: {color: '#111'},
    xaxis: {gridcolor: '#eee'}, yaxis: {gridcolor: '#eee', title: '°F'},
    showlegend: false,
  }, {displayModeBar: false, responsive: true});
}
dailyChart('daily_high_chart', 'high');
dailyChart('daily_low_chart',  'low');

// ---------- Joint outcomes table ----------
const jp = document.getElementById('joint_panel');
if (DATA.joint_outcomes && DATA.joint_outcomes.length > 0) {
  const rows = DATA.joint_outcomes.map(j => {
    const dateLabel = fmtDay(j.day) + ' (' + j.day + ')';
    const pct = v => (v*100).toFixed(1) + '%';
    return '<tr>' +
      '<td>' + dateLabel + '</td>' +
      '<td><code>' + j.high_ticker + '</code> ' + j.high_side.toUpperCase() + ' @ $' + j.high_cost.toFixed(3) + '</td>' +
      '<td><code>' + j.low_ticker  + '</code> ' + j.low_side.toUpperCase()  + ' @ $' + j.low_cost.toFixed(3)  + '</td>' +
      '<td class="num">' + pct(j.p_high_win) + '</td>' +
      '<td class="num">' + pct(j.p_low_win)  + '</td>' +
      '<td class="num yes">' + pct(j.both_win) + '</td>' +
      '<td class="num">' + pct(j.high_only)  + '</td>' +
      '<td class="num">' + pct(j.low_only)   + '</td>' +
      '<td class="num no">'  + pct(j.both_lose) + '</td>' +
    '</tr>';
  }).join('');
  jp.innerHTML = '<div class="panel"><table>' +
    '<thead><tr><th>day</th><th>HIGH trade</th><th>LOW trade</th>' +
    '<th class="num">P(HIGH wins)</th><th class="num">P(LOW wins)</th>' +
    '<th class="num">both win</th><th class="num">only HIGH</th>' +
    '<th class="num">only LOW</th><th class="num">both lose</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table>' +
    '<div class="small" style="margin-top:8px;">Probabilities multiply (independent). ' +
    '"Both win" = max profit. "Both lose" = max loss. The middle two are ~breakeven days.</div>' +
  '</div>';
} else {
  jp.innerHTML = '<div class="panel" style="color:#666;">' +
    'Need both an actionable HIGH trade and an actionable LOW trade for the same day. ' +
    'No paired opportunities right now.</div>';
}

// ---------- Per-day bucket charts (Kalshi) ----------
const bcc = document.getElementById('bucket_charts_container');
if (DATA.bucket_charts && DATA.bucket_charts.length > 0) {
  DATA.bucket_charts.forEach((bc, idx) => {
    const wrap = document.createElement('div');
    const dateLabel = fmtDay(bc.day_D) + ' (' + bc.day_D + ')';
    wrap.innerHTML = '<h3>' + bc.kind + ' for ' + dateLabel + '</h3>' +
                     '<div class="panel"><div id="bc_' + idx + '" style="height:300px;"></div></div>';
    bcc.appendChild(wrap);
    const labels = bc.buckets.map(b => b.label);
    const pmodel  = bc.buckets.map(b => (b.p_model  != null ? b.p_model  * 100 : 0));
    const pmarket = bc.buckets.map(b => (b.p_market != null ? b.p_market * 100 : 0));
    const traces = [
      {x: labels, y: pmodel, type:'bar', marker:{color:'#3a8dff'}, name:'our model',
       text: pmodel.map(v => v.toFixed(0) + '%'), textposition:'outside'},
      {x: labels, y: pmarket, type:'bar', marker:{color:'#d4a017'}, name:'Kalshi market',
       text: pmarket.map(v => v.toFixed(0) + '%'), textposition:'outside'},
    ];
    Plotly.newPlot('bc_' + idx, traces, {
      barmode: 'group', margin: {l: 50, r: 20, t: 10, b: 60}, font: {color: '#111'},
      xaxis: {gridcolor: '#eee'}, yaxis: {gridcolor: '#eee', title: 'P(YES) %', range: [0, 110]},
      legend: {x: 0.02, y: 1.10, orientation:'h'},
    }, {displayModeBar: false, responsive: true});
  });
} else {
  bcc.innerHTML = '<div class="panel" style="color:#666;">No upcoming Kalshi markets.</div>';
}

// ---------- Recent observations line chart ----------
if (DATA.obs) {
  const traces = [
    {x: DATA.obs.x, y: DATA.obs.y, mode: 'lines+markers', line: {color: '#1f4ea8'},
     marker: {size: 4}, name: 'observed temp'},
  ];
  if (DATA.obs.dew && DATA.obs.dew.some(v => v !== null)) {
    traces.push({x: DATA.obs.x, y: DATA.obs.dew, mode: 'lines',
                 line: {color: '#1a6e1a', dash: 'dot'}, name: 'dew point'});
  }
  Plotly.newPlot('obs_chart', traces, {
    margin: {l: 50, r: 20, t: 10, b: 40}, font: {color: '#111'},
    xaxis: {gridcolor: '#eee'}, yaxis: {gridcolor: '#eee', title: '°F'},
    legend: {x: 0.02, y: 1.10, orientation:'h'},
  }, {displayModeBar: false, responsive: true});
}

// ---------- Hourly forecast samples (kept for context, clearly labeled) ----------
if (DATA.fan) {
  const x = DATA.fan.map(p => p.valid_time);
  const med = DATA.fan.map(p => p.median);
  const q10 = DATA.fan.map(p => p.q10);
  const q90 = DATA.fan.map(p => p.q90);
  const traces = [
    {x: x.concat(x.slice().reverse()), y: q10.concat(q90.slice().reverse()),
     fill: 'toself', fillcolor: 'rgba(31,78,168,0.15)', line: {color: 'rgba(0,0,0,0)'},
     hoverinfo: 'skip', showlegend: false},
    {x: x, y: med, mode: 'lines+markers', line: {color: '#1f4ea8', width: 2},
     marker: {size: 8}, name: 'median (each dot = one model horizon)',
     text: DATA.fan.map(p => '+' + p.horizon + 'h: ' + p.median.toFixed(1) + '°F'),
     hovertemplate: '%{text} (valid %{x})<extra></extra>'},
  ];
  Plotly.newPlot('fan_chart', traces, {
    margin: {l: 50, r: 20, t: 10, b: 40}, font: {color: '#111'},
    xaxis: {gridcolor: '#eee', title: 'Valid time (PST) — only 7 sample points'},
    yaxis: {gridcolor: '#eee', title: '°F'},
    showlegend: false,
  }, {displayModeBar: false, responsive: true});
}

// ---------- Actionable signals table ----------
const sp = document.getElementById('signals_panel');
if (DATA.actionable && DATA.actionable.length > 0) {
  const rows = DATA.actionable.map(s => {
    const sideClass = s.trade_side === 'yes' ? 'yes' : 'no';
    const dateLabel = fmtDay(s.day_D) + ' (' + s.day_D + ')';
    return '<tr>' +
      '<td>' + dateLabel + '</td>' +
      '<td><code>' + s.ticker + '</code></td>' +
      '<td>' + (s.yes_sub_title || '') + '</td>' +
      '<td class="' + sideClass + '">' + (s.trade_side || '').toUpperCase() + '</td>' +
      '<td class="num">$' + (s.trade_cost || 0).toFixed(3) + '</td>' +
      '<td class="num">' + (s.model_prob_yes != null ? (s.model_prob_yes * 100).toFixed(1) + '%' : '—') + '</td>' +
      '<td class="num">' + (s.p_final != null ? (s.p_final * 100).toFixed(1) + '%' : '—') + '</td>' +
      '<td class="num yes">+$' + (s.trade_ev_per_c || 0).toFixed(3) + '</td>' +
      '<td class="num">' + ((s.kelly_used || 0) * 100).toFixed(2) + '%</td>' +
    '</tr>';
  }).join('');
  sp.innerHTML = '<div class="panel"><table>' +
    '<thead><tr><th>day</th><th>ticker</th><th>bucket</th><th>side</th>' +
    '<th class="num">cost</th><th class="num">p_model</th><th class="num">p_final</th>' +
    '<th class="num">EV/contract</th><th class="num">Kelly</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>';
} else {
  sp.innerHTML = '<div class="panel" style="color:#666;">No actionable trades right now.</div>';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
