# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 466
- **n_wins**: 252
- **win_rate**: 0.5408
- **total_pnl**: -367.2514
- **total_fee**: 173.9289
- **final_bankroll**: 632.7486
- **return_pct**: -36.7251
- **mean_kelly_used**: 0.0425
- **mean_size_dollars**: 13.3333
- **max_drawdown_dollars**: 499.2679
- **trades_yes**: 180
- **trades_no**: 286
- **avg_pnl_per_trade**: -0.7881
- **prob_col**: model_prob_yes
- **max_hours_to_settle**: None
- **label**: A_realistic_all

## Trade-side breakdown

- YES: n=180, win_rate=0.122, P&L=$-152.13
- NO: n=286, win_rate=0.804, P&L=$-215.12

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-21 07:00:00 | KXLOWTSFO-26APR22-B48.5 | no | 0.000 | 0.860 | 2 | 1 | +0.26 | 634.98 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-B64.5 | no | 0.271 | 0.680 | 2 | 1 | +0.61 | 635.59 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-B66.5 | no | 0.104 | 0.830 | 38 | 1 | +6.08 | 641.67 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-T60 | yes | 0.101 | 0.060 | 68 | 1 | +63.24 | 704.91 |
| 2026-04-21 08:00:00 | KXHIGHTSFO-26APR22-B62.5 | yes | 0.356 | 0.280 | 66 | 0 | -19.41 | 685.50 |
| 2026-04-21 08:00:00 | KXLOWTSFO-26APR22-T53 | yes | 0.241 | 0.190 | 13 | 0 | -2.61 | 682.89 |
| 2026-04-21 10:00:00 | KXLOWTSFO-26APR22-B50.5 | no | 0.318 | 0.610 | 51 | 0 | -31.96 | 650.93 |
| 2026-04-21 10:00:00 | KXHIGHTSFO-26APR22-T67 | no | 0.000 | 0.940 | 14 | 1 | +0.70 | 651.63 |
| 2026-04-21 11:00:00 | KXLOWTSFO-26APR22-B52.5 | yes | 0.464 | 0.390 | 50 | 0 | -20.33 | 631.30 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B48.5 | no | 0.000 | 0.940 | 29 | 1 | +1.45 | 632.75 |