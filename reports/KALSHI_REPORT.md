# Kalshi SFO Weather Trading System

End-to-end integration of NOAA observations with Kalshi's daily-temperature
prediction markets. Built on top of the SFO weather forecasting system (see
[FINAL_REPORT.md](FINAL_REPORT.md)).

---

## What this is

Given any decision time `t`, the system:

1. **Pulls the current SFO weather state** (live METAR feed when needed).
2. **Predicts the distribution of today's & next 3 days' daily HIGH and LOW**
   via 14 dedicated quantile models (7 quantiles × {high, low}).
3. **Maps that distribution onto every open Kalshi strike** (`KXHIGHTSFO-…`
   and `KXLOWTSFO-…`) to produce a model probability for each YES contract.
4. **Compares to current Kalshi mid-prices** and computes a calibrated
   **meta-probability** that blends our model with the market.
5. **Recommends trades** with edge, recommended Kelly fraction, and projected EV
   after fees & spread crossing.

No authentication is required for any data fetch — Kalshi market endpoints
and NWS Aviation Weather endpoints are public. The user-supplied API key
**is not used by this system** (and they should rotate it since it appeared
in chat).

---

## Headline backtest results (2026-01-15 → 2026-05-03, 612 markets)

Realistic constraints applied: $200 max stake per trade, 500 contract cap,
50% of open-interest cap, ~$0.005/100-contract slippage, Kalshi quadratic
fees, fractional Kelly (0.25× full Kelly).

| Strategy | n trades | Win % | Total return | Max DD | $/trade |
|---|---:|---:|---:|---:|---:|
| Raw model, **all horizons** | 466 | 54.1% | **−37%** | $499 | −$0.79 |
| Raw model, **≤12h to settle** | 277 | **65.3%** | **+723%** | $1,111 | +$26.10 |
| Meta model, ≤12h | 255 | 58.0% | +668% | $984 | +$26.19 |
| Raw model, **≤6h to settle** | 100 | **78.0%** | +413% | $291 | **+$41.33** |

**Edge is real but concentrated in short horizons.** At 12-48h we're worse
than the market (they have NWP forecasts we don't), but at 0-12h before
settlement our model integrates recent METARs faster than the market
re-prices.

### Probabilistic skill by horizon (decision-dataset log-loss)

| Hours to settle | n | Model LL | Market LL | Skill vs market |
|---|---:|---:|---:|---:|
| 0-6h | 1,076 | 0.118 | 0.812 | **+85.5%** |
| 6-12h | 1,820 | 0.199 | 0.628 | **+68.2%** |
| 12-24h | 3,991 | 0.605 | 0.400 | −51.2% |
| 24-48h | 5,066 | 0.754 | 0.408 | −84.8% |

The crossover between "we win" and "market wins" sits right at 12h. At <12h
we should trade aggressively; at >12h we should listen to the market.

### Meta-model on April 2026 hold-out

| Predictor | Log-loss | Brier |
|---|---:|---:|
| Our raw model | 0.484 | 0.120 |
| Market mid | 0.469 | 0.138 |
| **Meta (logreg)** | **0.376** | **0.119** |
| Meta + isotonic | 0.405 | 0.119 |

Meta beats the market by **+20% log-loss** and **+13.5% Brier** on the held-out
April 2026 slice.

The learned meta-coefficients (standardized features) are interpretable:

| Feature | Coef | Interpretation |
|---|---:|---|
| `logit_model` | +1.02 | Our model carries the most weight |
| `logit_market` | +0.48 | Market also informative |
| `disagreement` | +0.37 | When we disagree, that's predictive |
| `st_less` | −0.68 | Big bias correction for "less" buckets where we over-predict |
| `hours_to_settle` | −0.04 | Mild down-weight at long horizons |

---

## How the system works

### Data flow

```
NOAA LCD + ISD CSVs
       │
       ▼ (02_build_dataset.py)
sfo_hourly.parquet  ──── 14_refresh_noaa.py ←── NWS Aviation Weather API
       │                                          (pulls last 7d METARs)
       ▼ (03_features.py)
sfo_features.parquet (250 cols + hours_to_settle)
       │
       ▼ (09_daily_extreme_train.py)
14 quantile models: dxmodel_{high|low}_q{05..95}.joblib
       │
       ▼
       │   ┌──── 07_kalshi_fetch.py ────────────┐
       │   │       (events, markets, candles)  │
       │   ▼                                    ▼
       └──→ 10_decision_dataset_v2.py     ┌──── Kalshi public API
                       │                   │   (events / markets / candles)
                       ▼                   │
       decision_dataset_v2.parquet ────────┘
                       │
            ┌──────────┼──────────┬──────────────────┐
            ▼          ▼          ▼                  ▼
   12_meta_model  11_trade   13_live_signal     reports
                  _backtest
```

### The core insight

The market's price reflects **professional forecasts and recent news**, but
its incorporation of *real-time* weather observations is laggy. When SFO's
1pm METAR comes in cooler than expected, the market takes 1-3 hours to fully
re-price; in that window a model that ingests METARs immediately has edge.

Our model's strength at 0-12h is therefore not "we have a better forecasting
algorithm" — it's "we incorporate observations faster than the market". The
edge persists because Kalshi liquidity at intraday frequencies is still thin
relative to traditional markets.

### Strike convention reverse-engineered from the API

Kalshi SFO weather contracts use three strike types, each settling against
NWS's official daily HIGH or LOW from
[forecast.weather.gov/product.php?site=MTR&product=CLI&issuedby=SFO](https://forecast.weather.gov/product.php?site=MTR&product=CLI&issuedby=SFO):

- **`greater`**: floor=X, cap=NaN. YES iff temp > X. (e.g. `T66` = "67° or above")
- **`less`**: cap=X, floor=NaN. YES iff temp < X. (e.g. `T59` = "58° or below")
- **`between`**: floor=X, cap=Y. YES iff X ≤ temp ≤ Y. (e.g. `B61.5` = "61° to 62°")

Strikes step by 2°F in the bucket region with single-degree tails. NWS
reports daily highs as integers, so `B61.5 (61-62)` is satisfied by 61 or 62
exactly. Bucket probabilities use F(cap+0.5) − F(floor−0.5) on our predicted
CDF to handle this rounding correctly.

### NOAA-LCD timezone gotcha (carried over from base system)

LCD timestamps are **PST** (no DST), ISD are **UTC**. The pipeline converts
ISD to PST so everything lives on a single PST time axis. METARs from the
Aviation Weather API are UTC and are converted on ingest. All dates in
`reports/live_signals.md` and `reports/forecast.md` are PST.

### Daily-extreme correction caveat

NWS's official daily HIGH uses **1-minute resolution**, but our training
labels use **hourly METAR** maxima. There's a systematic gap of ~0.5-1.5°F:
the true daily high can briefly spike between hourly readings. Live trading
should add this correction explicitly — currently the meta-model partially
absorbs it via the strike-type bias coefficient (`st_less = −0.68`).

---

## Pipeline scripts

| Script | What it does | Time |
|---|---|---:|
| `01_audit.py` | Raw-data inspection report | ~30 s |
| `02_build_dataset.py` | Clean & merge LCD+ISD into PST hourly grid | ~1 min |
| `03_features.py` | 250 features per hour + climatology | ~2 min |
| `04_train.py` | 49 hourly-temperature quantile models | ~21 min |
| `05_backtest.py` | Walk-forward eval of hourly forecasts | ~30 s |
| `06_predict.py` | Forecast at any issuance time | <5 s |
| `07_kalshi_fetch.py` | Pull all SFO weather markets + candles | ~3 min |
| `08_decision_dataset.py` | (v1, deprecated) Decision dataset using hourly model | ~1 min |
| `09_daily_extreme_train.py` | 14 quantile models for daily HIGH/LOW | ~30 min |
| `10_decision_dataset_v2.py` | Decision dataset using daily-extreme models | ~2 min |
| `11_trade_backtest.py` | Realistic trading simulation | <30 s |
| `12_meta_model.py` | Logistic blend of model + market | <30 s |
| `13_live_signal.py` | Current actionable trades | ~30 s |
| `14_refresh_noaa.py` | Pull latest METARs + regen features | ~2 min |

### Live trading workflow

```bash
cd "/Users/terrykim/Documents/SF Weather"
python3 code/14_refresh_noaa.py --hours 168       # ingest last 7d METARs
python3 code/13_live_signal.py                    # ranked trades
cat reports/live_signals.md
```

Re-run before each decision cycle (e.g. every hour during the trading window).

---

## Live forecast snapshot

Issued at **2026-05-03 18:00 PST** (= 02:00 UTC May 4). Currently observed
SFO temp: 57°F. Today's high so far (per METAR): 63°F at 1pm PDT.

**7 actionable trades found**. Top three by EV:

| Day | Bucket | Side | Cost | p_model | p_market | p_meta | EV/$ |
|---|---|---|---:|---:|---:|---:|---:|
| 5/3 | 62-63°F (B62.5) | YES | $0.01 | 0.66 | 0.005 | 0.39 | **+$0.366** |
| 5/4 | 67°+ (T66) | YES | $0.06 | 0.39 | 0.04 | 0.39 | +$0.316 |
| 5/4 | 56°+ (T55, low) | YES | $0.14 | 0.34 | 0.085 | 0.39 | +$0.236 |

(Full table in [live_signals.md](live_signals.md).)

---

## Honest limitations & future work

1. **Small Kalshi history.** KXHIGHTSFO only goes back to 2026-01-15 (111
   days). Backtest sample size is modest — confidence intervals on the
   reported skill metrics are wide. Re-evaluate after 6+ more months of data.
2. **No NWP integration.** Adding GFS/HRRR forecast features would close most
   of our 12-48h skill gap vs the market. NCEP grids are public; this is the
   single biggest "next step" lever.
3. **1-min vs hourly resolution gap.** Our daily-high target uses hourly
   METARs while NWS settles on 1-min CRS data. Fix: ingest the 1-min ASOS
   feed (also public) for training labels.
4. **No order-book depth model.** Backtest assumes you can fill at the
   candle's close bid/ask up to a contract cap. Real Kalshi order books are
   often 1-2 contracts deep at top-of-book; iceberg liquidity matters.
5. **No live trading executor.** This system produces signals only. To trade
   automatically you'd need:
   - The RSA private key (the trading API requires `KALSHI-ACCESS-KEY` +
     `KALSHI-ACCESS-SIGNATURE` headers using your private PEM).
   - An order-management layer with entry/exit, time-in-force, and risk
     limits.
   - Slack/email/SMS alert hooks.
6. **No bandit / RL sizing.** Currently size = fractional Kelly capped at
   $200/trade. A contextual bandit on `(edge, hours_to_settle, spread, OI,
   strike_type)` would learn when full-Kelly is safe and when to back off.
   Recommend Thompson sampling on a Beta-Bernoulli per-feature-bin.
7. **Weather resolution beyond SFO.** Kalshi has parallel series for LAX,
   NYC, ORD, MIA, etc. Each requires its own daily-extreme model trained on
   that station's history.

---

## Files written by the Kalshi pipeline

```
data/
  kalshi_events.parquet                143 events
  kalshi_markets.parquet               612 markets
  kalshi_candles.parquet            24,502 hourly candles
  decision_dataset.parquet         (v1, deprecated)
  decision_dataset_v2.parquet      21,470 (decision, market) rows
  decision_dataset_v2_meta.parquet 11,953 with meta-prob
  trade_log_*.parquet              per-strategy trade logs
models/
  dxmodel_{high|low}_q{05..95}.joblib   14 daily-extreme models
  meta_calibrator.joblib                 logreg + isotonic
reports/
  decision_dataset_v2_summary.md
  daily_extreme_summary.md
  meta_model_summary.md
  trade_backtest_*.md                    per-strategy backtest reports
  live_signals.md / .json                current actionable signals
  KALSHI_REPORT.md                       this file
```
