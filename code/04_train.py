"""
04_train.py — Train multi-horizon quantile regression models for SFO temperature.

Inputs:
  data/sfo_features.parquet
  data/sfo_targets.parquet

Outputs:
  models/qmodel_h{H}_q{Q}.joblib   — one HistGradientBoostingRegressor per (horizon, quantile)
  reports/train_metrics.json       — train/val pinball loss & calibration per model
  reports/train_summary.md         — markdown summary

Design:
  * Train on 1970-01-01 → 2019-12-31, validate on 2020-01-01 → 2022-12-31.
  * Test set (2023+) is NOT touched here — backtest script handles it.
  * For each horizon h ∈ {1,3,6,12,24,48,72} and quantile q ∈ {.05,.1,.25,.5,.75,.9,.95},
    fit HistGradientBoostingRegressor(loss='quantile', quantile=q) using native NaN support.
  * Use early stopping based on the validation pinball loss for that quantile.
  * Persist each fitted model individually so backtest can load only what's needed.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/Users/terrykim/Documents/SF Weather")
FEAT_PATH = ROOT / "data" / "sfo_features.parquet"
TARG_PATH = ROOT / "data" / "sfo_targets.parquet"
MODEL_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "train_metrics.json"
SUMMARY_PATH = ROOT / "reports" / "train_summary.md"

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

TRAIN_END = pd.Timestamp("2019-12-31 23:00:00")
VAL_END = pd.Timestamp("2022-12-31 23:00:00")  # test starts 2023-01-01

# Hyperparameters. Tuned to be robust across all (horizon, quantile) pairs.
HGB_PARAMS = dict(
    loss="quantile",
    learning_rate=0.05,
    max_iter=500,
    max_depth=None,        # let leaves control depth
    max_leaf_nodes=63,
    min_samples_leaf=80,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,   # uses train-tail as internal early-stopping val
    n_iter_no_change=20,
    random_state=42,
)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Fraction of y_true at or below y_pred (quantile coverage). Should ≈ q for a calibrated model."""
    return float(np.mean(y_true <= y_pred))


def prepare_xy(features: pd.DataFrame, targets: pd.DataFrame, h: int):
    target_col = f"temp_f_h{h}"
    df = features.copy()
    df[target_col] = targets[target_col].values

    # Drop rows where the target is missing (last `h` hours)
    df = df.dropna(subset=[target_col]).copy()

    # Drop rows in early period before the longest lag is available (warmup)
    earliest = df["hour"].min() + pd.Timedelta(hours=336 + 24)  # max LAG_HOURS + safety
    df = df[df["hour"] >= earliest].copy()

    train = df[df["hour"] <= TRAIN_END]
    val = df[(df["hour"] > TRAIN_END) & (df["hour"] <= VAL_END)]

    feature_cols = [c for c in df.columns if c not in {"hour", target_col}]

    # Convert nullable dtypes (Int8, Int64, Float64) to float32 with NaN.
    def to_float32(d):
        out = pd.DataFrame(index=d.index)
        for c in feature_cols:
            s = d[c]
            if pd.api.types.is_extension_array_dtype(s):
                out[c] = s.astype("float32")
            elif s.dtype == "float64":
                out[c] = s.astype("float32")
            else:
                out[c] = s
        return out

    X_train = to_float32(train).values
    y_train = train[target_col].values.astype("float32")
    X_val = to_float32(val).values
    y_val = val[target_col].values.astype("float32")

    return X_train, y_train, X_val, y_val, feature_cols


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[train] reading features {FEAT_PATH}", flush=True)
    features = pd.read_parquet(FEAT_PATH)
    print(f"[train] reading targets {TARG_PATH}", flush=True)
    targets = pd.read_parquet(TARG_PATH)
    assert (features["hour"].values == targets["hour"].values).all()

    print(f"[train] features shape: {features.shape}", flush=True)
    print(f"[train] training horizons {HORIZONS}, quantiles {QUANTILES}", flush=True)

    metrics = {}
    feature_cols_used: list[str] | None = None

    total_start = time.time()
    for h in HORIZONS:
        print(f"\n[train] === horizon h={h}h ===", flush=True)
        X_train, y_train, X_val, y_val, feat_cols = prepare_xy(features, targets, h)
        if feature_cols_used is None:
            feature_cols_used = feat_cols
        print(f"[train]   train rows {len(y_train):,}, val rows {len(y_val):,}, "
              f"features {len(feat_cols)}", flush=True)

        h_metrics = {"train_rows": len(y_train), "val_rows": len(y_val)}

        for q in QUANTILES:
            t0 = time.time()
            params = dict(HGB_PARAMS)
            params["quantile"] = q
            model = HistGradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            pl_train = pinball_loss(y_train, y_pred_train, q)
            pl_val = pinball_loss(y_val, y_pred_val, q)
            cov_val = coverage(y_val, y_pred_val, q)
            elapsed = time.time() - t0
            print(f"[train]   q={q:.2f} | iters={model.n_iter_:3d} | "
                  f"pinball train={pl_train:.4f} val={pl_val:.4f} | "
                  f"coverage val={cov_val:.3f} (target {q:.2f}) | {elapsed:.1f}s",
                  flush=True)
            mpath = MODEL_DIR / f"qmodel_h{h}_q{int(q*100):02d}.joblib"
            joblib.dump(model, mpath)
            h_metrics[f"q{int(q*100):02d}"] = {
                "iters": int(model.n_iter_),
                "pinball_train": pl_train,
                "pinball_val": pl_val,
                "coverage_val": cov_val,
                "fit_seconds": elapsed,
                "model_path": str(mpath.relative_to(ROOT)),
            }

        metrics[f"h{h}"] = h_metrics

    total_elapsed = time.time() - total_start
    print(f"\n[train] total time: {total_elapsed/60:.1f} min", flush=True)

    # Save metrics + feature column order (needed for inference).
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "horizons": HORIZONS,
        "quantiles": QUANTILES,
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
        "feature_cols": feature_cols_used,
        "hgb_params": {k: v for k, v in HGB_PARAMS.items() if k != "quantile"},
        "metrics": metrics,
        "total_minutes": total_elapsed / 60,
    }, indent=2))
    print(f"[train] wrote {METRICS_PATH}", flush=True)

    # Markdown summary
    lines = ["# Training Summary\n",
             f"- Train end: **{TRAIN_END}**, Validation end: **{VAL_END}**",
             f"- Horizons: {HORIZONS}",
             f"- Quantiles: {QUANTILES}",
             f"- Total wall time: **{total_elapsed/60:.1f} min**",
             ""]
    lines.append("## Validation pinball loss by horizon × quantile (lower is better)\n")
    header = "| horizon | " + " | ".join(f"q={q:.2f}" for q in QUANTILES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(QUANTILES)) + "|"
    lines.append(header); lines.append(sep)
    for h in HORIZONS:
        row = [f"h={h}h"]
        for q in QUANTILES:
            v = metrics[f"h{h}"][f"q{int(q*100):02d}"]["pinball_val"]
            row.append(f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Validation coverage by horizon × quantile (target = q)\n")
    lines.append(header); lines.append(sep)
    for h in HORIZONS:
        row = [f"h={h}h"]
        for q in QUANTILES:
            v = metrics[f"h{h}"][f"q{int(q*100):02d}"]["coverage_val"]
            row.append(f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[train] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
