# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 466
- **n_wins**: 252
- **win_rate**: 0.5408
- **total_pnl**: -912.7731
- **total_fee**: 644.7731
- **final_bankroll**: 87.2269
- **return_pct**: -91.2773
- **mean_kelly_used**: 0.0424
- **mean_size_dollars**: 25.3348
- **max_drawdown_dollars**: 1758.9797
- **trades_yes**: 182
- **trades_no**: 284
- **avg_pnl_per_trade**: -1.9587
- **prob_col**: model_prob_yes
- **max_hours_to_settle**: None
- **label**: A_model_all

## Trade-side breakdown

- YES: n=182, win_rate=0.126, P&L=$-867.89
- NO: n=284, win_rate=0.806, P&L=$-44.88

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-21 07:00:00 | KXLOWTSFO-26APR22-B48.5 | no | 0.000 | 0.860 | 4 | 1 | +0.52 | 84.88 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-B64.5 | no | 0.271 | 0.680 | 4 | 1 | +1.22 | 86.10 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-B66.5 | no | 0.104 | 0.830 | 5 | 1 | +0.80 | 86.90 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-T60 | yes | 0.101 | 0.060 | 15 | 1 | +13.95 | 100.85 |
| 2026-04-21 07:00:00 | KXLOWTSFO-26APR22-T53 | yes | 0.302 | 0.170 | 23 | 0 | -4.14 | 96.71 |
| 2026-04-21 08:00:00 | KXHIGHTSFO-26APR22-B62.5 | yes | 0.356 | 0.280 | 9 | 0 | -2.65 | 94.06 |
| 2026-04-21 10:00:00 | KXLOWTSFO-26APR22-B50.5 | no | 0.318 | 0.610 | 7 | 0 | -4.39 | 89.67 |
| 2026-04-21 10:00:00 | KXHIGHTSFO-26APR22-T67 | no | 0.000 | 0.940 | 4 | 1 | +0.20 | 89.87 |
| 2026-04-21 11:00:00 | KXLOWTSFO-26APR22-B52.5 | yes | 0.464 | 0.390 | 7 | 0 | -2.85 | 87.03 |
| 2026-04-21 12:00:00 | KXLOWTSFO-26APR21-B48.5 | no | 0.000 | 0.940 | 4 | 1 | +0.20 | 87.23 |