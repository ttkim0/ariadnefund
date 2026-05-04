# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 277
- **n_wins**: 181
- **win_rate**: 0.6534
- **total_pnl**: 203310.0234
- **total_fee**: 35585.6266
- **final_bankroll**: 204310.0234
- **return_pct**: 20331.0023
- **mean_kelly_used**: 0.0454
- **mean_size_dollars**: 2938.7847
- **max_drawdown_dollars**: 82759.4122
- **trades_yes**: 97
- **trades_no**: 180
- **avg_pnl_per_trade**: 733.9712
- **prob_col**: model_prob_yes
- **max_hours_to_settle**: 12.0000
- **label**: B_model_short

## Trade-side breakdown

- YES: n=97, win_rate=0.340, P&L=$+16921.74
- NO: n=180, win_rate=0.822, P&L=$+186388.28

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B65.5 | no | 0.691 | 0.250 | 20672 | 0 | -5439.32 | 257673.89 |
| 2026-04-20 12:00:00 | KXLOWTSFO-26APR20-B54.5 | no | 0.759 | 0.160 | 38688 | 0 | -6576.96 | 251096.93 |
| 2026-04-20 14:00:00 | KXLOWTSFO-26APR20-B52.5 | yes | 0.227 | 0.090 | 104785 | 0 | -10478.50 | 240618.43 |
| 2026-04-20 14:00:00 | KXHIGHTSFO-26APR20-B67.5 | yes | 0.261 | 0.140 | 60328 | 0 | -9049.20 | 231569.23 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-T65 | no | 0.000 | 0.970 | 11936 | 1 | +238.72 | 231807.95 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B62.5 | no | 0.493 | 0.190 | 61002 | 0 | -12247.55 | 219560.39 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B50.5 | yes | 0.131 | 0.060 | 68991 | 0 | -4829.37 | 214731.02 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B48.5 | no | 0.000 | 0.940 | 11421 | 1 | +571.05 | 215302.07 |
| 2026-04-21 13:00:00 | KXLOWTSFO-26APR21-B52.5 | no | 0.664 | 0.140 | 76893 | 0 | -11533.95 | 203768.12 |
| 2026-04-21 15:00:00 | KXHIGHTSFO-26APR21-B64.5 | no | 0.000 | 0.940 | 10838 | 1 | +541.90 | 204310.02 |