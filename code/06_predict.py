"""
06_predict.py — Live forecasting for SFO temperature.

Reads:
  data/sfo_features.parquet         (the latest issuance time = last row with feats)
  models/qmodel_h{H}_q{Q}.joblib    (49 models)
  reports/train_metrics.json        (feature-column ordering)

Writes:
  reports/forecast.json             machine-readable forecast
  reports/forecast.md               human-readable forecast (markdown table per horizon)

Usage:
  python3 code/06_predict.py [--issued YYYY-MM-DDTHH:00]
                             [--bucket-width 5]    # °F bin width for bucket prob
                             [--bucket-min 30 --bucket-max 105]

Default issuance is the latest hour with full feature data. Forecast horizons
are the same {1,3,6,12,24,48,72} the models were trained on.

Output for each horizon:
  - median (point forecast)
  - 50% interval (q25..q75) and 80% interval (q10..q90)
  - bucket probabilities (5°F bins by default)
  - most-likely bucket
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
FEAT_PATH = ROOT / "data" / "sfo_features.parquet"
MODEL_DIR = ROOT / "models"
TRAIN_META_PATH = ROOT / "reports" / "train_metrics.json"
FORECAST_JSON = ROOT / "reports" / "forecast.json"
FORECAST_MD = ROOT / "reports" / "forecast.md"

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def to_f32_row(row: pd.Series, cols: list[str]) -> np.ndarray:
    arr = np.empty(len(cols), dtype="float32")
    for i, c in enumerate(cols):
        v = row[c]
        if pd.isna(v):
            arr[i] = np.nan
        else:
            arr[i] = float(v)
    return arr.reshape(1, -1)


def isotonic_monotone(quantile_preds: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(quantile_preds, axis=1)


def cdf_at_edges(qpred: np.ndarray, qs: list[float], edges: list[float]) -> np.ndarray:
    qpred = isotonic_monotone(qpred)
    out = np.empty((qpred.shape[0], len(edges)), dtype=np.float64)
    for i in range(qpred.shape[0]):
        out[i] = np.interp(edges, qpred[i], qs, left=0.0, right=1.0)
    return out


def bucket_probs(cdf: np.ndarray) -> np.ndarray:
    diffs = np.diff(cdf, axis=1)
    diffs = np.clip(diffs, 0.0, 1.0)
    diffs[:, 0] += cdf[:, 0]
    diffs[:, -1] += 1.0 - cdf[:, -1]
    s = diffs.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return diffs / s


def render_md(issued: pd.Timestamp, results: list[dict], edges: list[int]) -> str:
    lines = [f"# SFO Temperature Forecast",
             f"Issued at **{issued}** PST  "
             f"(= {issued + pd.Timedelta(hours=8)} UTC).\n",
             "All times below are PST (UTC-8). "
             "During DST add 1h to convert to PDT.\n"]
    for r in results:
        valid_pst = pd.Timestamp(r['valid_time'])
        valid_utc = valid_pst + pd.Timedelta(hours=8)
        lines.append(f"## +{r['horizon']}h  (valid at {valid_pst} PST = {valid_utc} UTC)")
        lines.append(f"")
        lines.append(f"- **Median:** {r['median']:.1f}°F")
        lines.append(f"- **50% interval:** {r['q25']:.1f}°F – {r['q75']:.1f}°F")
        lines.append(f"- **80% interval:** {r['q10']:.1f}°F – {r['q90']:.1f}°F")
        lines.append(f"- **90% interval:** {r['q05']:.1f}°F – {r['q95']:.1f}°F")
        lines.append(f"- **Most likely bucket:** {r['top_bucket']} (P={r['top_prob']*100:.1f}%)")
        lines.append("")
        lines.append("| Bucket | Probability |")
        lines.append("|---|---:|")
        # Show only buckets with >0.5% mass
        for label, p in zip(r['bucket_labels'], r['bucket_probs']):
            if p < 0.005:
                continue
            bar = "▎" * max(1, round(p * 40))  # 40-char visualization at p=1
            lines.append(f"| {label} | {p*100:5.1f}%  {bar} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issued", default=None,
                    help="Issuance timestamp (UTC). Default: latest available hour.")
    ap.add_argument("--bucket-width", type=int, default=5)
    ap.add_argument("--bucket-min", type=int, default=30)
    ap.add_argument("--bucket-max", type=int, default=105)
    args = ap.parse_args()

    edges = list(range(args.bucket_min, args.bucket_max + 1, args.bucket_width))
    labels = [f"[{a},{b})°F" for a, b in zip(edges[:-1], edges[1:])]

    features = pd.read_parquet(FEAT_PATH)
    meta = json.loads(TRAIN_META_PATH.read_text())
    fcols = meta["feature_cols"]

    if args.issued is None:
        # Latest hour with non-null essentials.
        valid = features[features["temp_f"].notna()]
        issued = valid["hour"].max()
    else:
        issued = pd.Timestamp(args.issued)

    row = features.loc[features["hour"].eq(issued)]
    if row.empty:
        raise SystemExit(f"No feature row for {issued}")
    row = row.iloc[0]
    print(f"[predict] issuance time: {issued}")
    print(f"[predict] current observed temp_f: {row['temp_f']:.1f}°F")

    Xrow = to_f32_row(row, fcols)

    results = []
    for h in HORIZONS:
        qpred = np.empty((1, len(QUANTILES)), dtype=np.float64)
        for j, q in enumerate(QUANTILES):
            mpath = MODEL_DIR / f"qmodel_h{h}_q{int(q*100):02d}.joblib"
            m = joblib.load(mpath)
            qpred[0, j] = float(m.predict(Xrow)[0])
        qpred = isotonic_monotone(qpred)
        median = float(qpred[0, QUANTILES.index(0.50)])
        q05 = float(qpred[0, QUANTILES.index(0.05)])
        q10 = float(qpred[0, QUANTILES.index(0.10)])
        q25 = float(qpred[0, QUANTILES.index(0.25)])
        q75 = float(qpred[0, QUANTILES.index(0.75)])
        q90 = float(qpred[0, QUANTILES.index(0.90)])
        q95 = float(qpred[0, QUANTILES.index(0.95)])

        cdf_e = cdf_at_edges(qpred, QUANTILES, edges)
        probs = bucket_probs(cdf_e)[0]
        top_idx = int(np.argmax(probs))
        results.append({
            "horizon": h,
            "valid_time": str(issued + pd.Timedelta(hours=h)),
            "median": median,
            "q05": q05, "q10": q10, "q25": q25, "q75": q75, "q90": q90, "q95": q95,
            "bucket_edges": edges,
            "bucket_labels": labels,
            "bucket_probs": probs.tolist(),
            "top_bucket": labels[top_idx],
            "top_prob": float(probs[top_idx]),
        })
        print(f"[predict] h={h:>2}h  median={median:5.1f}°F  "
              f"80%CI=[{q10:.1f},{q90:.1f}]  top={labels[top_idx]} ({probs[top_idx]*100:.1f}%)")

    FORECAST_JSON.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_JSON.write_text(json.dumps({
        "issued": str(issued),
        "current_temp_f": float(row["temp_f"]),
        "forecast": results,
    }, indent=2))
    FORECAST_MD.write_text(render_md(issued, results, edges))
    print(f"[predict] wrote {FORECAST_JSON} and {FORECAST_MD}")


if __name__ == "__main__":
    main()
