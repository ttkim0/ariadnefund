"""
05_backtest.py — Walk-forward evaluation on the held-out test set.

Reads:
  data/sfo_features.parquet
  data/sfo_targets.parquet
  models/qmodel_h{H}_q{Q}.joblib
  reports/train_metrics.json   (for feature column ordering)

Writes:
  reports/backtest_metrics.json
  reports/backtest_summary.md
  data/test_predictions.parquet  (per-row predictions for all horizons & quantiles)

Test window: 2023-01-01 → end of data (2026-04-22).

Metrics reported per horizon:
  * MAE / RMSE of the median (q=0.50) prediction
  * Pinball loss per quantile, plus mean pinball (CRPS-like proxy)
  * Coverage at each quantile (fraction of obs <= predicted q)
  * 80% interval (q10 to q90) coverage and average width
  * Skill scores vs persistence and vs climatology
  * MAE & RMSE for two baselines for direct comparison

Bucket evaluation:
  * 5°F bins from 30 to 105 (16 buckets)
  * For each row, derive bucket probabilities by linear interpolation of the
    empirical CDF defined by the 7 predicted quantiles, after isotonic
    monotonization to remove quantile crossing.
  * Compute log-loss & Brier of the bucket-prob forecast vs realized bucket.
  * Compare to climatology bucket prob (P from clim mean & std assuming Normal).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/terrykim/Documents/SF Weather")
FEAT_PATH = ROOT / "data" / "sfo_features.parquet"
TARG_PATH = ROOT / "data" / "sfo_targets.parquet"
MODEL_DIR = ROOT / "models"
TRAIN_META_PATH = ROOT / "reports" / "train_metrics.json"

PRED_OUT = ROOT / "data" / "test_predictions.parquet"
METRICS_OUT = ROOT / "reports" / "backtest_metrics.json"
SUMMARY_OUT = ROOT / "reports" / "backtest_summary.md"

TEST_START = pd.Timestamp("2023-01-01 00:00:00")

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Bucket edges in °F. Edges define inclusive-left, exclusive-right bins.
# 30,35,...,100,105 → 15 bins.
BUCKET_EDGES = list(range(30, 110, 5))  # [30,35,40,...,105]
BUCKET_LABELS = [f"[{a},{b})" for a, b in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])]


def to_f32(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = pd.DataFrame(index=d.index)
    for c in cols:
        s = d[c]
        if pd.api.types.is_extension_array_dtype(s) or s.dtype == "float64":
            out[c] = s.astype("float32")
        else:
            out[c] = s
    return out.values


def pinball(y, yhat, q):
    e = y - yhat
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def isotonic_monotone(quantile_preds: np.ndarray) -> np.ndarray:
    """Enforce that predicted quantiles are non-decreasing across q.
    quantile_preds: (n_rows, n_quantiles) sorted by ascending q.
    Returns same shape with each row monotonized (cumulative max)."""
    return np.maximum.accumulate(quantile_preds, axis=1)


def cdf_from_quantile_preds(quantile_preds: np.ndarray,
                            quantiles: list[float],
                            edges: list[int]) -> np.ndarray:
    """For each row, given quantile preds at given quantiles, linearly interpolate
    the CDF F(x) at each edge x. We extrapolate flat at the tails:
      F(x) = 0 for x <= q_0; F(x) = 1 for x >= q_max.
    Returns array of shape (n_rows, len(edges)) with CDF values at each edge.
    """
    qp = isotonic_monotone(quantile_preds)
    n = qp.shape[0]
    qarr = np.array(quantiles)
    edges = np.array(edges, dtype=float)
    # Output array
    out = np.empty((n, len(edges)), dtype=np.float64)
    for i in range(n):
        xs = qp[i]   # ascending temperature values
        ys = qarr    # corresponding CDF values
        # np.interp handles outside-range with constant extrapolation if we
        # explicitly set left=0 and right=1, but values must be increasing.
        # If quantile values have duplicates (saturated), np.interp still works.
        # If two qp values are exactly equal, np.interp picks the lower y, which is fine.
        out[i] = np.interp(edges, xs, ys, left=0.0, right=1.0)
    return out


def bucket_probs_from_cdf(cdf_at_edges: np.ndarray) -> np.ndarray:
    """Given F at edges [e_0, e_1, ..., e_K], the prob mass in [e_k, e_{k+1}) is
    F(e_{k+1}) - F(e_k). Returns shape (n_rows, K)."""
    diffs = np.diff(cdf_at_edges, axis=1)
    diffs = np.clip(diffs, 0.0, 1.0)  # safety
    # Mass below first edge or above last edge gets folded into nearest end bucket
    below = cdf_at_edges[:, [0]]
    above = 1.0 - cdf_at_edges[:, [-1]]
    diffs[:, 0] += below[:, 0]
    diffs[:, -1] += above[:, 0]
    # Renormalize for numerical safety
    s = diffs.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return diffs / s


def bucket_index(values: np.ndarray, edges) -> np.ndarray:
    """For each value, return bucket index in [0, K-1] where K = len(edges)-1.
    Values <= edges[0] go to bucket 0; values >= edges[-1] go to bucket K-1."""
    idx = np.searchsorted(edges, values, side="right") - 1
    K = len(edges) - 1
    idx = np.clip(idx, 0, K - 1)
    return idx


def normal_bucket_probs(mean: np.ndarray, std: np.ndarray, edges) -> np.ndarray:
    """Climatology baseline: assume temp at horizon ~ Normal(mean, std), where
    mean = clim_mean_smooth and std = clim_std_smooth at the FORECAST VALID
    time (t + h)."""
    from scipy.stats import norm
    K = len(edges) - 1
    out = np.empty((len(mean), K), dtype=np.float64)
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        out[:, k] = norm.cdf(hi, loc=mean, scale=std) - norm.cdf(lo, loc=mean, scale=std)
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return out / s


def load_models(horizons, quantiles):
    models = {}
    for h in horizons:
        for q in quantiles:
            mpath = MODEL_DIR / f"qmodel_h{h}_q{int(q*100):02d}.joblib"
            models[(h, q)] = joblib.load(mpath)
    print(f"[backtest] loaded {len(models)} models", flush=True)
    return models


def main():
    print("[backtest] loading features/targets...", flush=True)
    features = pd.read_parquet(FEAT_PATH)
    targets = pd.read_parquet(TARG_PATH)

    meta = json.loads(TRAIN_META_PATH.read_text())
    feature_cols = meta["feature_cols"]
    print(f"[backtest] {len(feature_cols)} features", flush=True)

    test_mask = features["hour"] >= TEST_START
    feat_test = features.loc[test_mask].reset_index(drop=True)
    targ_test = targets.loc[test_mask].reset_index(drop=True)
    print(f"[backtest] test rows: {len(feat_test):,} ({feat_test['hour'].min()} → {feat_test['hour'].max()})", flush=True)

    X_test = to_f32(feat_test, feature_cols)
    print(f"[backtest] X_test shape: {X_test.shape}", flush=True)

    models = load_models(HORIZONS, QUANTILES)

    # Pre-compute climatology mean/std at the FORECAST VALID time (t + h).
    # We do this by shifting clim_mean_smooth / clim_std_smooth by -h hours.
    feat_full = features  # use the entire frame for shifting
    clim_at_target = {}
    for h in HORIZONS:
        cm = features["clim_mean_smooth"].shift(-h).values
        cs = features["clim_std_smooth"].shift(-h).values
        clim_at_target[h] = (cm[test_mask.values], cs[test_mask.values])

    # Predictions
    pred_records = {"hour": feat_test["hour"].values}
    horizon_metrics = {}
    bucket_metrics = {}

    for h in HORIZONS:
        target = targ_test[f"temp_f_h{h}"].values
        valid = ~np.isnan(target)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        # Predict each quantile
        Qpred = np.empty((len(X_test), len(QUANTILES)), dtype=np.float64)
        for j, q in enumerate(QUANTILES):
            yhat = models[(h, q)].predict(X_test)
            Qpred[:, j] = yhat
            pred_records[f"h{h}_q{int(q*100):02d}"] = yhat.astype("float32")

        Qpred = isotonic_monotone(Qpred)
        median = Qpred[:, QUANTILES.index(0.50)]
        lo80 = Qpred[:, QUANTILES.index(0.10)]
        hi80 = Qpred[:, QUANTILES.index(0.90)]

        y = target[valid]
        m = median[valid]
        l = lo80[valid]
        u = hi80[valid]

        mae = float(np.mean(np.abs(y - m)))
        rmse = float(np.sqrt(np.mean((y - m) ** 2)))
        bias = float(np.mean(m - y))
        cov80 = float(np.mean((y >= l) & (y <= u)))
        width80 = float(np.mean(u - l))

        # Pinball losses & coverage
        pl = []
        cov = []
        for j, q in enumerate(QUANTILES):
            yp = Qpred[valid, j]
            pl.append(pinball(y, yp, q))
            cov.append(float(np.mean(y <= yp)))
        mean_pinball = float(np.mean(pl))

        # Baselines
        # Persistence: forecast at t+h is temp_f at t
        persist = feat_test["temp_f"].values[valid]
        mae_persist = float(np.mean(np.abs(y - persist)))
        rmse_persist = float(np.sqrt(np.mean((y - persist) ** 2)))

        # Climatology at valid time
        cm_t, cs_t = clim_at_target[h]
        cm_t_v = cm_t[valid]
        cs_t_v = cs_t[valid]
        mae_clim = float(np.mean(np.abs(y - cm_t_v)))
        rmse_clim = float(np.sqrt(np.mean((y - cm_t_v) ** 2)))

        skill_persist = 1 - mae / mae_persist
        skill_clim = 1 - mae / mae_clim

        horizon_metrics[h] = {
            "n": n_valid,
            "mae": mae, "rmse": rmse, "bias": bias,
            "cov80": cov80, "width80": width80,
            "mae_persist": mae_persist, "rmse_persist": rmse_persist,
            "mae_clim": mae_clim, "rmse_clim": rmse_clim,
            "skill_vs_persist": skill_persist,
            "skill_vs_clim": skill_clim,
            "mean_pinball": mean_pinball,
            "pinball_by_q": dict(zip([f"q{int(q*100):02d}" for q in QUANTILES], pl)),
            "coverage_by_q": dict(zip([f"q{int(q*100):02d}" for q in QUANTILES], cov)),
        }

        # ---- Bucket forecasting ----
        cdf_e = cdf_from_quantile_preds(Qpred, QUANTILES, BUCKET_EDGES)
        probs_model = bucket_probs_from_cdf(cdf_e)[valid]
        # Climatology bucket probs (Normal at valid time)
        try:
            from scipy.stats import norm
            probs_clim = normal_bucket_probs(cm_t_v, cs_t_v, BUCKET_EDGES)
        except ImportError:
            probs_clim = None

        # Realized bucket
        true_bucket = bucket_index(y, BUCKET_EDGES)

        eps = 1e-9
        log_loss_model = float(-np.mean(np.log(np.clip(probs_model[np.arange(len(true_bucket)), true_bucket], eps, 1.0))))
        # Brier = sum_k (p_k - 1{k==true})^2 averaged over rows
        K = len(BUCKET_EDGES) - 1
        onehot = np.zeros_like(probs_model)
        onehot[np.arange(len(true_bucket)), true_bucket] = 1.0
        brier_model = float(np.mean(np.sum((probs_model - onehot) ** 2, axis=1)))

        bm = {
            "log_loss": log_loss_model, "brier": brier_model,
            "n_buckets": K,
        }
        if probs_clim is not None:
            log_loss_clim = float(-np.mean(np.log(np.clip(probs_clim[np.arange(len(true_bucket)), true_bucket], eps, 1.0))))
            brier_clim = float(np.mean(np.sum((probs_clim - onehot) ** 2, axis=1)))
            bm["log_loss_clim"] = log_loss_clim
            bm["brier_clim"] = brier_clim
            bm["log_loss_skill_vs_clim"] = 1 - log_loss_model / log_loss_clim
            bm["brier_skill_vs_clim"] = 1 - brier_model / brier_clim
        bucket_metrics[h] = bm

        print(f"[backtest] h={h}h | MAE={mae:.3f}  RMSE={rmse:.3f}  bias={bias:+.2f}  "
              f"cov80={cov80:.3f}  width80={width80:.2f} | "
              f"vs_persist={skill_persist:+.3f}  vs_clim={skill_clim:+.3f} | "
              f"bucket_LL={log_loss_model:.3f}", flush=True)

    # Save predictions
    pred_df = pd.DataFrame(pred_records)
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PRED_OUT, index=False)
    print(f"[backtest] wrote {PRED_OUT}", flush=True)

    # Save metrics
    out = {
        "test_start": str(TEST_START),
        "horizons": HORIZONS,
        "quantiles": QUANTILES,
        "bucket_edges": BUCKET_EDGES,
        "horizon_metrics": horizon_metrics,
        "bucket_metrics": bucket_metrics,
    }
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    print(f"[backtest] wrote {METRICS_OUT}", flush=True)

    # Markdown summary
    lines = ["# Backtest Summary\n",
             f"Test window: **{TEST_START}** → end of data\n"]
    lines.append("## Point forecast accuracy (median = q0.50)\n")
    lines.append("| horizon | n | MAE | RMSE | Bias | Persist MAE | Clim MAE | Skill vs Persist | Skill vs Clim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        m = horizon_metrics[h]
        lines.append(f"| {h}h | {m['n']:,} | **{m['mae']:.3f}** | {m['rmse']:.3f} | "
                     f"{m['bias']:+.2f} | {m['mae_persist']:.3f} | {m['mae_clim']:.3f} | "
                     f"**{m['skill_vs_persist']:+.3f}** | **{m['skill_vs_clim']:+.3f}** |")
    lines.append("")
    lines.append("## 80% prediction-interval calibration & width\n")
    lines.append("| horizon | Coverage | Target | Avg width (°F) |")
    lines.append("|---|---:|---:|---:|")
    for h in HORIZONS:
        m = horizon_metrics[h]
        lines.append(f"| {h}h | {m['cov80']:.3f} | 0.800 | {m['width80']:.2f} |")
    lines.append("")
    lines.append("## Per-quantile coverage (target = q)\n")
    header = "| horizon | " + " | ".join(f"q={q:.2f}" for q in QUANTILES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(QUANTILES)) + "|"
    lines.append(header); lines.append(sep)
    for h in HORIZONS:
        c = horizon_metrics[h]["coverage_by_q"]
        row = [f"{h}h"] + [f"{c[f'q{int(q*100):02d}']:.3f}" for q in QUANTILES]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"## Bucket forecasting (5°F bins, edges = {BUCKET_EDGES})\n")
    lines.append("| horizon | Log loss (model) | Log loss (clim) | Skill | Brier (model) | Brier (clim) | Skill |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        b = bucket_metrics[h]
        lines.append(f"| {h}h | **{b['log_loss']:.3f}** | "
                     f"{b.get('log_loss_clim','—'):.3f} | {b.get('log_loss_skill_vs_clim',float('nan')):+.3f} | "
                     f"**{b['brier']:.4f}** | {b.get('brier_clim','—'):.4f} | "
                     f"{b.get('brier_skill_vs_clim',float('nan')):+.3f} |")
    SUMMARY_OUT.write_text("\n".join(lines))
    print(f"[backtest] wrote {SUMMARY_OUT}", flush=True)


if __name__ == "__main__":
    main()
