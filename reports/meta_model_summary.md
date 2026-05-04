# Meta-Calibration Summary

- Train rows: 5,769  Test rows: 6,184
- Train end: 2026-03-31 23:00:00

## Test-set log-loss

| Model | LL | Brier |
|---|---:|---:|
| Our forecast (model_prob) | 0.4838 | 0.1204 |
| Market (yes_close)        | 0.4688 | 0.1375 |
| Meta (logreg)             | 0.3761 | 0.1193 |
| Meta (logreg + isotonic)  | 0.4051 | 0.1185 |
