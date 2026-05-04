# Trade Backtest

- Decision window: 2026-01-15 → present
- Initial bankroll: $1,000
- Strategy: edge ≥ $0.02/contract, Kelly × 0.25, cap 5.0% per trade

## Summary

- **n_trades**: 447
- **n_wins**: 206
- **win_rate**: 0.4609
- **total_pnl**: -592.3181
- **total_fee**: 1441.8881
- **final_bankroll**: 407.6819
- **return_pct**: -59.2318
- **mean_kelly_used**: 0.0354
- **mean_size_dollars**: 56.3723
- **max_drawdown_dollars**: 3621.7815
- **trades_yes**: 239
- **trades_no**: 208
- **avg_pnl_per_trade**: -1.3251
- **prob_col**: meta_prob_yes
- **max_hours_to_settle**: None
- **label**: C_meta_all

## Trade-side breakdown

- YES: n=239, win_rate=0.159, P&L=$-1038.72
- NO: n=208, win_rate=0.808, P&L=$+446.40

## Last 10 trades

| time | ticker | side | p_model | cost | contracts | won | pnl | bk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04-21 07:00:00 | KXLOWTSFO-26APR22-T53 | yes | 0.338 | 0.170 | 146 | 0 | -26.28 | 470.30 |
| 2026-04-21 07:00:00 | KXLOWTSFO-26APR22-B46.5 | no | 0.003 | 0.960 | 24 | 1 | +0.72 | 471.02 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-B60.5 | yes | 0.189 | 0.100 | 117 | 0 | -12.87 | 458.15 |
| 2026-04-21 07:00:00 | KXHIGHTSFO-26APR22-T60 | no | 0.000 | 0.950 | 24 | 0 | -23.04 | 435.11 |
| 2026-04-21 08:00:00 | KXHIGHTSFO-26APR22-T67 | yes | 0.189 | 0.140 | 44 | 0 | -6.60 | 428.51 |
| 2026-04-21 09:00:00 | KXLOWTSFO-26APR22-B52.5 | no | 0.288 | 0.620 | 34 | 1 | +12.36 | 440.87 |
| 2026-04-21 09:00:00 | KXLOWTSFO-26APR22-B50.5 | no | 0.288 | 0.610 | 36 | 0 | -22.56 | 418.31 |
| 2026-04-21 12:00:00 | KXHIGHTSFO-26APR22-B66.5 | yes | 0.189 | 0.150 | 32 | 0 | -5.12 | 413.19 |
| 2026-04-21 13:00:00 | KXHIGHTSFO-26APR22-B64.5 | no | 0.288 | 0.670 | 19 | 1 | +5.98 | 419.17 |
| 2026-04-21 16:00:00 | KXHIGHTSFO-26APR22-B62.5 | yes | 0.338 | 0.260 | 42 | 0 | -11.49 | 407.68 |