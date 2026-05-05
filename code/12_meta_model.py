"""
12_meta_model.py — Meta-calibration model that blends our forecast with the market.

Reads:
  data/decision_dataset_v2.parquet

Writes:
  models/meta_calibrator.joblib
  data/decision_dataset_v2_meta.parquet  (adds `meta_prob_yes` column)
  reports/meta_model_summary.md

The meta-calibrator is a small logistic-regression ensemble over:
  * logit(model_prob_yes)
  * logit(market_yes_close)
  * hours_to_settle
  * spread
  * log(open_interest+1), log(volume+1)
  * abs(model_prob - market_prob)   (disagreement magnitude)
  * strike_type one-hot

Target: yes_outcome_derived

Goal: learn a calibrated blend that's better than either alone, especially in
regions where one signal is more informative than the other.

Eval: log-loss & Brier of meta vs market vs model on a chronological hold-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import get_city  # noqa: E402

ROOT = REPO_ROOT

# Chronological split: by default the last 25% of the eligible window is held
# out for testing.  Per-city, we derive the split from the actual data range
# rather than hardcoding a date — different cities have different Kalshi
# launch dates.


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    f = pd.DataFrame()
    f["logit_model"] = logit(df["model_prob_yes"].values)
    f["logit_market"] = logit(df["market_yes_close"].values)
    f["disagreement"] = np.abs(df["model_prob_yes"].values - df["market_yes_close"].values)
    f["hours_to_settle"] = df["hours_to_settle"].fillna(48).values
    f["spread"] = df["spread"].fillna(0.10).values
    f["log_oi"] = np.log1p(df["open_interest"].fillna(0).values)
    f["log_vol"] = np.log1p(df["volume"].fillna(0).values)
    # one-hot strike type
    f["st_greater"] = (df["strike_type"] == "greater").astype(int).values
    f["st_less"] = (df["strike_type"] == "less").astype(int).values
    f["st_between"] = (df["strike_type"] == "between").astype(int).values
    cols = list(f.columns)
    return f.values.astype("float32"), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sfo")
    args = ap.parse_args()
    city = get_city(args.city)

    if city.slug == "sfo":
        DECISION_PATH = ROOT / "data" / "decision_dataset_v2.parquet"
        META_OUT      = ROOT / "models" / "meta_calibrator.joblib"
        DATASET_OUT   = ROOT / "data" / "decision_dataset_v2_meta.parquet"
        SUMMARY_OUT   = ROOT / "reports" / "meta_model_summary.md"
    else:
        DECISION_PATH = ROOT / "data" / f"decision_dataset_v2_{city.slug}.parquet"
        META_OUT      = city.models_dir / "meta_calibrator.joblib"
        DATASET_OUT   = ROOT / "data" / f"decision_dataset_v2_meta_{city.slug}.parquet"
        SUMMARY_OUT   = ROOT / "reports" / f"meta_model_summary_{city.slug}.md"

    if not DECISION_PATH.exists():
        print(f"[meta:{city.slug}] no decision dataset — run 10_decision_dataset_v2 first; skipping")
        return

    print(f"[meta:{city.slug}] reading {DECISION_PATH}", flush=True)
    df = pd.read_parquet(DECISION_PATH)
    elig = df.dropna(subset=["model_prob_yes", "market_yes_close", "spread"])
    elig = elig[elig["yes_outcome_derived"].isin([0, 1])].copy()
    elig = elig.sort_values("decision_time").reset_index(drop=True)
    print(f"[meta:{city.slug}] eligible rows: {len(elig):,}", flush=True)

    if len(elig) < 200:
        print(f"[meta:{city.slug}] insufficient eligible rows ({len(elig)}); skipping")
        return

    # Per-city split: hold out last 25% of rows chronologically.
    cutoff_idx = int(len(elig) * 0.75)
    META_TRAIN_END = elig["decision_time"].iloc[cutoff_idx]
    train = elig.iloc[:cutoff_idx]
    test  = elig.iloc[cutoff_idx:]
    print(f"[meta:{city.slug}] train rows {len(train):,}, test rows {len(test):,} "
          f"(cutoff {META_TRAIN_END})", flush=True)
    if len(train) < 100 or len(test) < 50:
        print(f"[meta:{city.slug}] insufficient data; skipping meta-model fit", flush=True)
        return

    X_train, fnames = build_features(train)
    y_train = train["yes_outcome_derived"].astype(int).values
    X_test, _ = build_features(test)
    y_test = test["yes_outcome_derived"].astype(int).values

    # Logistic with mild regularization
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    # Optional isotonic post-calibration on training (avoid look-ahead)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    p_train_iso = iso.transform(p_train)
    p_test_iso = iso.transform(p_test)

    eps = 1e-6
    def ll(y, p):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    def brier(y, p):
        return float(np.mean((p - y) ** 2))

    # Compare vs market and our model alone on the test set
    p_market = test["market_yes_close"].values
    p_model = test["model_prob_yes"].values
    print(f"\n=== TEST SET (n={len(test)}) ===")
    print(f"  log-loss   model={ll(y_test, p_model):.4f}  market={ll(y_test, p_market):.4f}  "
          f"meta_raw={ll(y_test, p_test):.4f}  meta_iso={ll(y_test, p_test_iso):.4f}")
    print(f"  Brier      model={brier(y_test, p_model):.4f}  market={brier(y_test, p_market):.4f}  "
          f"meta_raw={brier(y_test, p_test):.4f}  meta_iso={brier(y_test, p_test_iso):.4f}")

    # Print learned coefficients for interpretation
    coef = pipe.named_steps["logreg"].coef_[0]
    intercept = pipe.named_steps["logreg"].intercept_[0]
    print("\n=== Logistic-regression coefficients (standardized features) ===")
    for n, c in zip(fnames, coef):
        print(f"  {n:>16s}: {c:+.3f}")
    print(f"  intercept: {intercept:+.3f}")

    # Save artifacts
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "city": city.slug,
        "pipeline": pipe,
        "isotonic": iso,
        "feature_names": fnames,
    }, META_OUT)
    print(f"\n[meta:{city.slug}] wrote {META_OUT}", flush=True)

    # Add meta_prob_yes to the full eligible df and save
    X_all, _ = build_features(elig)
    p_all = pipe.predict_proba(X_all)[:, 1]
    p_all_iso = iso.transform(p_all)
    elig = elig.copy()
    elig["meta_prob_raw"] = p_all
    elig["meta_prob_yes"] = p_all_iso
    elig.to_parquet(DATASET_OUT, index=False)
    print(f"[meta] wrote {DATASET_OUT} ({len(elig):,} rows)", flush=True)

    # Markdown summary
    lines = ["# Meta-Calibration Summary\n",
             f"- Train rows: {len(train):,}  Test rows: {len(test):,}",
             f"- Train end: {META_TRAIN_END}",
             "",
             "## Test-set log-loss\n",
             f"| Model | LL | Brier |",
             f"|---|---:|---:|",
             f"| Our forecast (model_prob) | {ll(y_test, p_model):.4f} | {brier(y_test, p_model):.4f} |",
             f"| Market (yes_close)        | {ll(y_test, p_market):.4f} | {brier(y_test, p_market):.4f} |",
             f"| Meta (logreg)             | {ll(y_test, p_test):.4f} | {brier(y_test, p_test):.4f} |",
             f"| Meta (logreg + isotonic)  | {ll(y_test, p_test_iso):.4f} | {brier(y_test, p_test_iso):.4f} |",
             ""]
    SUMMARY_OUT.write_text("\n".join(lines))
    print(f"[meta] wrote {SUMMARY_OUT}", flush=True)


if __name__ == "__main__":
    main()
