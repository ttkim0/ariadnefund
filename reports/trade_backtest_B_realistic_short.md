# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 277
- **n_wins**: 181
- **win_rate**: 0.6534
- **total_pnl**: 7231.0285
- **total_fee**: 876.9490
- **final_bankroll**: 8231.0285
- **return_pct**: 723.1028
- **mean_kelly_used**: 0.0454
- **mean_size_dollars**: 109.6102
- **max_drawdown_dollars**: 1110.7913
- **trades_yes**: 97
- **trades_no**: 180
- **avg_pnl_per_trade**: 26.1048
- **prob_col**: model_prob_yes
- **max_hours_to_settle**: 12.0000
- **label**: B_realistic_short

## Trade-side breakdown

- YES: n=97, win_rate=0.340, P&L=$+2833.55
- NO: n=180, win_rate=0.822, P&L=$+4397.48

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B65.5 | no | 0.691 | 0.250 | 500 | 0 | -132.60 | 8496.57 |
| 2026-04-20 12:00:00 | KXLOWTSFO-26APR20-B54.5 | no | 0.759 | 0.160 | 185 | 0 | -31.66 | 8464.91 |
| 2026-04-20 14:00:00 | KXLOWTSFO-26APR20-B52.5 | yes | 0.227 | 0.090 | 198 | 0 | -20.04 | 8444.86 |
| 2026-04-20 14:00:00 | KXHIGHTSFO-26APR20-B67.5 | yes | 0.261 | 0.140 | 500 | 0 | -76.00 | 8368.86 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-T65 | no | 0.000 | 0.970 | 206 | 1 | +3.85 | 8372.72 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B62.5 | no | 0.493 | 0.190 | 500 | 0 | -101.43 | 8271.29 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B50.5 | yes | 0.131 | 0.060 | 165 | 0 | -11.71 | 8259.57 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B48.5 | no | 0.000 | 0.940 | 29 | 1 | +1.45 | 8261.02 |
| 2026-04-21 13:00:00 | KXLOWTSFO-26APR21-B52.5 | no | 0.664 | 0.140 | 266 | 0 | -40.31 | 8220.71 |
| 2026-04-21 15:00:00 | KXHIGHTSFO-26APR21-B64.5 | no | 0.000 | 0.940 | 212 | 1 | +10.32 | 8231.03 |