# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 100
- **n_wins**: 78
- **win_rate**: 0.7800
- **total_pnl**: 4132.9777
- **total_fee**: 230.8223
- **final_bankroll**: 5132.9777
- **return_pct**: 413.2978
- **mean_kelly_used**: 0.0468
- **mean_size_dollars**: 88.0920
- **max_drawdown_dollars**: 291.1368
- **trades_yes**: 27
- **trades_no**: 73
- **avg_pnl_per_trade**: 41.3298
- **prob_col**: model_prob_yes
- **max_hours_to_settle**: 6.0000
- **label**: D_realistic_v_short

## Trade-side breakdown

- YES: n=27, win_rate=0.556, P&L=$+1386.84
- NO: n=73, win_rate=0.863, P&L=$+2746.14

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-16 18:00:00 | KXLOWTSFO-26APR16-B47.5 | no | 0.000 | 0.960 | 208 | 1 | +5.97 | 4243.08 |
| 2026-04-16 18:00:00 | KXLOWTSFO-26APR16-B49.5 | no | 0.212 | 0.150 | 500 | 1 | +419.00 | 4662.08 |
| 2026-04-17 18:00:00 | KXLOWTSFO-26APR17-B50.5 | no | 0.000 | 0.970 | 125 | 1 | +2.44 | 4664.52 |
| 2026-04-17 20:00:00 | KXLOWTSFO-26APR17-B48.5 | no | 0.000 | 0.970 | 149 | 1 | +2.86 | 4667.37 |
| 2026-04-17 23:00:00 | KXLOWTSFO-26APR17-B52.5 | yes | 0.847 | 0.730 | 102 | 1 | +26.13 | 4693.50 |
| 2026-04-17 23:00:00 | KXLOWTSFO-26APR17-T53 | no | 0.000 | 0.800 | 249 | 1 | +46.65 | 4740.16 |
| 2026-04-19 18:00:00 | KXLOWTSFO-26APR19-B51.5 | no | 0.280 | 0.060 | 500 | 1 | +464.00 | 5204.16 |
| 2026-04-20 18:00:00 | KXLOWTSFO-26APR20-B52.5 | yes | 0.327 | 0.090 | 158 | 0 | -15.94 | 5188.21 |
| 2026-04-20 18:00:00 | KXLOWTSFO-26APR20-B54.5 | no | 0.673 | 0.170 | 216 | 0 | -39.17 | 5149.04 |
| 2026-04-21 18:00:00 | KXLOWTSFO-26APR21-B50.5 | yes | 0.080 | 0.050 | 261 | 0 | -16.06 | 5132.98 |