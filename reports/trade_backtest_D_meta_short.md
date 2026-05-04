# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 255
- **n_wins**: 148
- **win_rate**: 0.5804
- **total_pnl**: 186091.7035
- **total_fee**: 23491.9665
- **final_bankroll**: 187091.7035
- **return_pct**: 18609.1703
- **mean_kelly_used**: 0.0410
- **mean_size_dollars**: 1622.8758
- **max_drawdown_dollars**: 80700.0484
- **trades_yes**: 96
- **trades_no**: 159
- **avg_pnl_per_trade**: 729.7714
- **prob_col**: meta_prob_yes
- **max_hours_to_settle**: 12.0000
- **label**: D_meta_short

## Trade-side breakdown

- YES: n=96, win_rate=0.292, P&L=$+32097.55
- NO: n=159, win_rate=0.755, P&L=$+153994.16

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B65.5 | no | 0.512 | 0.250 | 53558 | 0 | -14092.45 | 253699.30 |
| 2026-04-20 12:00:00 | KXHIGHTSFO-26APR20-B69.5 | yes | 0.073 | 0.030 | 93084 | 0 | -3723.36 | 249975.94 |
| 2026-04-20 12:00:00 | KXLOWTSFO-26APR20-B54.5 | no | 0.512 | 0.160 | 78117 | 0 | -13279.89 | 236696.05 |
| 2026-04-20 14:00:00 | KXLOWTSFO-26APR20-B52.5 | yes | 0.219 | 0.090 | 93519 | 0 | -9351.90 | 227344.15 |
| 2026-04-20 14:00:00 | KXHIGHTSFO-26APR20-B67.5 | yes | 0.189 | 0.140 | 23361 | 0 | -3504.15 | 223840.00 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B52.5 | no | 0.624 | 0.110 | 101745 | 0 | -12209.40 | 211630.60 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B50.5 | yes | 0.189 | 0.060 | 121470 | 0 | -8502.90 | 203127.70 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B64.5 | yes | 0.185 | 0.130 | 24597 | 0 | -3443.58 | 199684.12 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR21-B62.5 | no | 0.512 | 0.190 | 52548 | 0 | -10550.22 | 189133.90 |
| 2026-04-21 13:00:00 | KXHIGHTSFO-26APR21-T65 | yes | 0.073 | 0.040 | 40844 | 0 | -2042.20 | 187091.70 |