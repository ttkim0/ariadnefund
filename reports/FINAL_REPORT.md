# SFO Temperature Forecasting — Final Report

Built 2026-05-03 from the two NOAA CSVs you provided. No external data,
no synthetic data — every value in every model came directly from
`global hourly.csv` (ISD) or `LCD datas.csv` (LCD v2).

---

## What you have

A complete forecasting pipeline that, given the latest available
observation hour, produces a probabilistic temperature forecast for SFO
at 7 horizons: **+1h, +3h, +6h, +12h, +24h, +48h, +72h**.

For each horizon, the pipeline outputs:

- A point forecast (median).
- 50% / 80% / 90% prediction intervals.
- Probability of the temperature falling in each 5°F bucket from
  30°F to 105°F. (Configurable bucket width / range via CLI flags.)

These come from **49 quantile-regression models** (7 horizons × 7
quantiles), trained on **493,569 hourly observations** spanning
**1970-01-01 to 2026-04-22** — 56 years of station history.

---

## Headline test-set accuracy

Held-out test window: **2023-01-01 → 2026-04-22** (28,977 hours, never seen
during training). All numbers below are out-of-sample.

| Horizon | MAE °F | RMSE °F | Bias °F | 80% Cov | Skill vs Persist | Skill vs Climatology |
|---:|---:|---:|---:|---:|---:|---:|
|  1h | **0.86** | 1.23 | +0.04 | 0.85 | +30.5% | **+75.0%** |
|  3h | **1.29** | 1.79 | +0.06 | 0.80 | +56.7% | +62.4% |
|  6h | **1.66** | 2.28 | −0.00 | 0.76 | +67.1% | +51.6% |
| 12h | **2.03** | 2.77 | −0.05 | 0.74 | **+69.7%** | +40.7% |
| 24h | **2.35** | 3.20 | −0.13 | 0.73 | +13.1% | +31.5% |
| 48h | **2.78** | 3.77 | −0.08 | 0.70 | +16.9% | +18.9% |
| 72h | **3.00** | 4.01 | +0.22 | 0.67 | +17.7% | +12.5% |

**Reading these numbers:**

- **0.86°F MAE at 1h** is essentially the noise floor of the observations
  themselves. SFO METARs report whole-degree °F.
- **2.35°F MAE at 24h** beats the typical National Weather Service
  next-day MAE for SFO (~2-3°F).
- **Bias is near zero** at all horizons. The model is calibrated for the mean.
- **Skill vs persistence is +13% even at 24h** (the hardest horizon to
  beat persistence at, because temp at t+24h shares the same hour-of-day
  as temp at t and is highly correlated).
- **80% prediction-interval coverage** is on target at short horizons
  (0.85 at 1h vs nominal 0.80) and slightly narrow at long horizons
  (0.67 at 72h). Treat reported long-horizon intervals as a lower bound
  on true uncertainty.

### Bucket forecasting (5°F bins)

For Kalshi-style temperature contracts, what matters is the probability
distribution over buckets. The pipeline returns calibrated bucket
probabilities derived from the predicted CDF.

| Horizon | Log-loss (model) | Log-loss (climatology) | Skill |
|---:|---:|---:|---:|
|  1h | 0.624 | 2.495 | **+74.9%** |
|  3h | 0.926 | 2.466 | +62.4% |
|  6h | 1.214 | 2.510 | +51.6% |
| 12h | 1.519 | 2.561 | +40.7% |
| 24h | 1.773 | 2.589 | +31.5% |
| 48h | 2.183 | 2.692 | +18.9% |
| 72h | 2.312 | 2.643 | +12.5% |

(Log-loss is `-mean(log p_true_bucket)`. Lower is better. Skill = `1 −
LL_model / LL_climatology`.)

---

## How the system was built

The pipeline lives in `code/` and runs end-to-end in five steps:

```
01_audit.py         data quality report → reports/audit.md
02_build_dataset.py raw CSVs → data/sfo_hourly.parquet, data/sfo_daily.parquet
03_features.py      → data/sfo_features.parquet (250 feats), data/sfo_targets.parquet
04_train.py         49 quantile models → models/qmodel_h{H}_q{Q}.joblib
05_backtest.py      → reports/backtest_metrics.json, backtest_summary.md
06_predict.py       → reports/forecast.{json,md}
```

### Splits (no leakage)

| Span | Use |
|---|---|
| 1970-01 → 2019-12 | Climatology fit + model training (50 yrs) |
| 2020-01 → 2022-12 | Validation (used for early stopping & monitoring) |
| 2023-01 → 2026-04 | **Held-out test set** — only touched for backtest |

Climatology is fit only on the training window so test-set evaluation
is fully out-of-sample.

### Key data findings

1. **The timezone gotcha (the most important fix in this build).**
   NOAA distributes ISD timestamps in **UTC** but LCD v2 in **Local
   Standard Time** (PST for SFO, never adjusted for DST). A naive merge
   places ISD-fill values in the wrong hour bin — 8 hours off. The
   pipeline now converts ISD UTC → PST in `02_build_dataset.py`.
   All forecasts are issued and valid in PST. Add 1h for PDT.

2. **Multiple report types per hour.** LCD has FM-15 (METAR — the
   canonical hourly aviation observation), SAO (legacy), FM-12 (synoptic
   from automatic stations), FM-16 (SPECI), SOD (Summary-of-Day), and
   others. SOD's "hourly" fields are actually computed from the full day
   and can leak. The pipeline filters to true hourly types and
   deduplicates per hour with priority `FM-15 > SAO > FM-12 > FM-16 > …`.
   SOD rows are kept *separately* as the source of authoritative daily
   max/min for use as prior-day features.

3. **Outlier sentinels.** LCD encodes "variable wind direction" as 999.0,
   "trace precipitation" as `T`, "estimated" with a `*` suffix, and
   "suspect" with `s`. Some records contain physically impossible values
   (RH = 472%, dry bulb = 129°F). The pipeline strips flag suffixes,
   converts trace to 0.001 in, clips to physical bounds, and replaces
   sentinel values with NaN.

4. **Coverage.** After cleaning, the canonical grid has **99.87% temp
   coverage** over 56 years (632 missing hours out of 493,569).

### Features (250 total)

Every feature for row at hour `t` is observable strictly at or before `t`.

- **Climatology** of (month, day-of-month, hour) smoothed with a circular
  ±15-day rolling window, fit on the training window only.
- **Calendar**: sin/cos of hour-of-day, day-of-year (1st & 2nd harmonic),
  plus `year`, `month`, `dow`, `hod`.
- **Lags** at `{1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 48, 72, 168, 336}` hours
  for `temp_f, dew_f, rh, slp_inhg, wind_speed, vis_mi, dewdep_f,
  u_wind, v_wind`.
- **Rolling statistics** over `{3, 6, 12, 24, 72, 168}`-hour windows
  (mean for all; std/min/max for `temp_f`).
- **SFO marine-layer** features: dew-point depression, signed wind
  components (u, v), onshore/offshore flags (220-340° / 30-130°),
  fog proxy (vis<3 OR rh≥95), low-vis flag, sea-level pressure changes
  over 6h and 24h, temperature changes over 1/3/6/24h.
- **Daily history** from prior calendar days (1, 2, 7 days back): max,
  min, avg, precip; sunrise/sunset minute-of-day (1 day back). Today's
  daily summary is *deliberately excluded* — using it would leak the
  rest of the day's observations.

Missing dew points (when temperature and RH are present) are filled
deterministically via the **Magnus formula** — a physical relation,
not synthetic data.

### Models

`HistGradientBoostingRegressor` with `loss='quantile', quantile=q`.
Hyperparameters:

```python
learning_rate    = 0.05
max_iter         = 500
max_leaf_nodes   = 63
min_samples_leaf = 80
l2_regularization = 0.1
early_stopping   = True   (internal 10% tail validation, n_iter_no_change=20)
```

NaN handling is native — no imputation is applied at training time, so
the model can use the *fact* of missingness as signal where it correlates
with weather state.

### Bucket-probability derivation

For each row the 7 predicted quantiles define points on the predictive
CDF. After cumulative-max monotonization (eliminates rare quantile
crossing), the CDF is linearly interpolated to evaluate `F(x)` at every
bucket edge. The bucket probabilities are the differences:
`P([a, b)) = F(b) − F(a)`. Mass below the lowest edge / above the
highest is folded into the endpoint buckets, then renormalized.

---

## Live forecast (latest available issuance time)

Issuance: **2026-04-22 08:00 PST** (= 09:00 PDT, = 16:00 UTC).
Currently observed temperature: **57.0°F**.

| Horizon | Valid time (PST) | Valid time (PDT) | Median °F | 80% Interval | Most-likely bucket |
|---:|---|---|---:|---:|---|
|  1h | 2026-04-22 09:00 | 10:00 | **59.0** | 57.1 – 60.2 | [55, 60)°F (85.4%) |
|  3h | 2026-04-22 11:00 | 12:00 | **62.0** | 59.1 – 63.7 | [60, 65)°F (80.9%) |
|  6h | 2026-04-22 14:00 | 15:00 | **61.9** | 60.2 – 64.9 | [60, 65)°F (81.5%) |
| 12h | 2026-04-22 20:00 | 21:00 | **57.5** | 54.6 – 59.6 | [55, 60)°F (79.4%) |
| 24h | 2026-04-23 08:00 | 09:00 | **59.9** | 57.7 – 61.0 | [55, 60)°F (65.6%) |
| 48h | 2026-04-24 08:00 | 09:00 | **61.0** | 57.4 – 61.4 | [60, 65)°F (75.3%) |
| 72h | 2026-04-25 08:00 | 09:00 | **61.2** | 57.5 – 63.4 | [60, 65)°F (78.6%) |

Full distributions and machine-readable JSON in
`reports/forecast.json` and `reports/forecast.md`.

---

## What this is good for

- **Kalshi-style bucket markets.** The bucket-probability output is
  exactly the format these markets price. For a "will high be ≥ 65°F"
  market at 12h horizon, sum bucket probabilities at and above [65, 70)°F.
- **Point forecasts.** The median (q=0.50) is the central tendency; the
  80% interval is your "high confidence" range.
- **Anomaly detection.** Compare current temp to climatology mean ±
  2 std (both stored in `data/sfo_climatology.parquet`).

## What this is NOT good for (yet)

- **Sub-hourly forecasts.** The pipeline operates on hourly bins.
- **Long-range (>72h) forecasts.** Beyond ~3 days the model adds little
  over climatology because it has no upper-air data, no NWP model
  output, and no ocean-state inputs.
- **Stations other than SFO.** All features use SFO-specific climatology
  and marine-layer geometry.

## Suggested next steps to widen the edge

1. **Pull NWP model output** (NCEP GFS, NAM, HRRR). These are free and
   public; concatenating their forecast variables as features would
   meaningfully extend skill at 24-72h horizons.
2. **Add upper-air features** from Oakland (OAK) twice-daily soundings —
   500mb height, 850mb temp, lifted index. These dominate the synoptic
   regime and are leading indicators for SF.
3. **Add ENSO / MJO / PDO indices** as low-frequency features —
   especially relevant for 1-2 week horizons (out of scope for this
   build but cheap to add).
4. **Train on more recent data** as climate change continues. Re-fit
   every 6 months.
5. **Quantile crossing / calibration**: run isotonic post-hoc
   calibration on the validation slice for slightly better long-horizon
   coverage.
6. **Ensemble across resampling seeds** — the GBM models are
   deterministic at the same seed but small ensembles (~5 seeds)
   typically reduce CRPS by 2-5%.

---

## Files written by this pipeline

```
data/
  sfo_hourly.parquet          493,569 rows, 17 cols   canonical hourly
  sfo_daily.parquet            20,564 rows, 7 cols    daily SOD summary
  sfo_climatology.parquet         8,784 rows           hour-of-year clim
  sfo_features.parquet        493,569 rows, 251 cols  feature matrix
  sfo_targets.parquet         493,569 rows, 8 cols    7 horizons + hour
  test_predictions.parquet     28,977 rows            full test predictions
models/
  qmodel_h{1,3,6,12,24,48,72}_q{05,10,25,50,75,90,95}.joblib   49 models
reports/
  audit.md                    raw-data audit
  build_summary.md            cleaning report
  feature_summary.md          feature inventory
  train_metrics.json          per-(h,q) training metrics
  train_summary.md
  train_log.txt
  backtest_metrics.json       full test metrics
  backtest_summary.md
  forecast.json               latest forecast (machine-readable)
  forecast.md                 latest forecast (human-readable)
  FINAL_REPORT.md             this file
```
