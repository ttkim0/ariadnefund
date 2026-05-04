# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 255
- **n_wins**: 148
- **win_rate**: 0.5804
- **total_pnl**: 6678.7916
- **total_fee**: 804.9384
- **final_bankroll**: 7678.7916
- **return_pct**: 667.8792
- **mean_kelly_used**: 0.0410
- **mean_size_dollars**: 93.3658
- **max_drawdown_dollars**: 983.6887
- **trades_yes**: 96
- **trades_no**: 159
- **avg_pnl_per_trade**: 26.1913
- **prob_col**: meta_prob_yes
- **max_hours_to_settle**: 12.0000
- **label**: C_realistic_meta_short

## Trade-side breakdown

- YES: n=96, win_rate=0.292, P&L=$+3189.18
- NO: n=159, win_rate=0.755, P&L=$+3489.61

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B65.5 | no | 0.512 | 0.250 | 500 | 0 | -132.60 | 8068.51 |
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B69.5 | yes | 0.073 | 0.030 | 500 | 0 | -21.00 | 8047.51 |
| 2026-04-20 12:00:00 | KXLOWTSFO-26APR20-B54.5 | no | 0.512 | 0.160 | 185 | 0 | -31.66 | 8015.84 |
| 2026-04-20 14:00:00 | KXLOWTSFO-26APR20-B52.5 | yes | 0.219 | 0.090 | 198 | 0 | -20.04 | 7995.80 |
| 2026-04-20 14:00:00 | KXHIGHTSFO-26APR20-B67.5 | yes | 0.189 | 0.140 | 500 | 0 | -76.00 | 7919.80 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B52.5 | no | 0.624 | 0.110 | 254 | 0 | -30.86 | 7888.93 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B50.5 | yes | 0.189 | 0.060 | 165 | 0 | -11.71 | 7877.22 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B64.5 | yes | 0.185 | 0.130 | 500 | 0 | -71.00 | 7806.22 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B62.5 | no | 0.512 | 0.190 | 500 | 0 | -101.43 | 7704.79 |
| 2026-04-21 13:00:00 | KXHIGHTSFO-26APR21-T65 | yes | 0.073 | 0.040 | 500 | 0 | -26.00 | 7678.79 |