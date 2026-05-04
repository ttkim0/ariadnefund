# Using the SFO forecasting system

## Generate a forecast for the latest hour

```bash
cd "/Users/terrykim/Documents/SF Weather"
python3 code/06_predict.py
```

This reads `data/sfo_features.parquet`, picks the most recent hour with a
non-null observed temperature, runs all 49 models, and writes both
`reports/forecast.json` and `reports/forecast.md`.

## Forecast for a specific issuance time

```bash
python3 code/06_predict.py --issued "2026-04-22T08:00"
```

The timestamp is **PST** (this whole system runs on a PST time axis;
see README for the timezone gotcha). Add 8 hours to get UTC.

## Custom buckets

Default is 5°F bins from 30°F to 105°F. Override:

```bash
python3 code/06_predict.py --bucket-width 2 --bucket-min 40 --bucket-max 90
```

For a Kalshi-style "will the high temperature be at or above 65°F" market,
look at the cumulative bucket probability. For example,
`P(temp >= 65°F)` at the 12h horizon is the sum of all bucket probabilities
in `[65,70)°F` and above.

## Re-running the pipeline

If the source CSVs change (e.g. you re-download with newer data):

```bash
python3 code/02_build_dataset.py    # ~1 min
python3 code/03_features.py         # ~2 min
python3 code/04_train.py            # ~21 min (49 models)
python3 code/05_backtest.py         # ~30 sec
python3 code/06_predict.py          # ~5 sec
```

## Interpreting the output

A typical forecast row looks like:

```
+24h  median=59.7°F  80%CI=[56.5,60.7]  top=[55,60)°F (61.0%)
```

- **median** is the point forecast — the central tendency. Use this if
  you want a single number.
- **80%CI** is the 80% prediction interval `[q10, q90]`. The actual
  temperature is expected to fall in this range about 80% of the time.
  Backtest empirical coverage was 0.69-0.86 across horizons (slightly narrow
  at long horizons — true uncertainty is a touch larger than the model thinks).
- **top** is the most-likely 5°F bucket and its probability. Use this for
  bucket-style markets.

For market trading, the most useful column in `forecast.json` is
`bucket_probs` — an array of probabilities aligned with `bucket_labels`.
This is the full conditional distribution.

## Known limits

- **Climate-change drift**: training data ends 2019. Test backtest showed
  small (+0.17°F) positive bias at 72h, consistent with mild warming.
  Re-run training every ~6 months to incorporate fresh data.
- **80% interval slightly narrow at long horizons**: empirical coverage
  drops to 0.69 at 72h vs 0.80 nominal. For high-stakes trading at long
  horizons, treat reported intervals as a lower bound on true uncertainty.
- **No external data**: this system uses ONLY the two NOAA station files.
  It has no access to NWS forecasts, NCEP/GFS model output, satellite
  imagery, or upper-air soundings. Combining with those would improve
  long-horizon skill substantially but adds operational complexity.
