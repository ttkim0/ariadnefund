"""
11_trade_backtest.py — Realistic trading backtest using the v2 decision dataset.

Reads:
  data/decision_dataset_v2.parquet

Writes:
  data/trade_log.parquet           one row per simulated trade
  reports/trade_backtest.json
  reports/trade_backtest.md

Strategy: edge-threshold + fractional Kelly + spread crossing + Kalshi fees.

For each row in the decision dataset (one (market, decision_time) pair) within
the test window:
  * Compute YES-side opportunity:
      buy at yes_ask_close (cross the spread)
      EV per $1 staked = model_prob_yes * (1 / yes_ask_close - 1) - (1 - model_prob_yes)
      Better cast: pay $yes_ask_close per contract, get $1 if YES else $0
      pure EV per contract = model_prob_yes - yes_ask_close
  * Compute NO-side opportunity:
      buy NO at (1 - yes_bid_close); pay that, get $1 if NO else $0
      pure EV per contract = (1 - model_prob_yes) - (1 - yes_bid_close)
                            = yes_bid_close - model_prob_yes
  * Apply Kalshi quadratic fee: ~0.07 * P * (1-P), capped at $0.01 per contract.
  * Threshold: only trade if post-fee EV > MIN_EV (default $0.02 per contract).
  * Sizing: fractional Kelly. For YES, p_yes=model_prob_yes, b_yes=(1-ask)/ask.
    Kelly fraction f = (p*b - (1-p)) / b. Cap at MAX_FRACTION (default 5%
    of bankroll, or 25% of unconstrained Kelly, whichever is smaller).
  * Each trade: position = fraction * current_bankroll, contracts = position / cost
  * P&L: settle at $1 if outcome matches side, $0 otherwise; pay fee on entry only.

Anti-leakage:
  * Only one trade per market: take the FIRST (earliest decision_time) row that
    crosses the threshold. Avoid double-counting an opportunity that persists.
  * Time-walked: trades are processed in chronological order; bankroll updates
    after each settlement.

Realism notes:
  * Kalshi market hours: SFO weather markets trade 24/7 on Kalshi but liquidity
    varies. We trust candle close prices as executable mid; bid/ask close as
    executable spread.
  * Slippage beyond bid/ask not modeled. Order fills are assumed instant at
    the close price of the candle hour.
  * Position size limits: real Kalshi has per-contract position limits ($25k
    or so). For simulation we cap at a fraction of bankroll.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from cities_config import get_city  # noqa: E402

ROOT = REPO_ROOT
DEFAULT_DECISION_PATH = ROOT / "data" / "decision_dataset_v2.parquet"
META_DECISION_PATH = ROOT / "data" / "decision_dataset_v2_meta.parquet"

# === Strategy parameters (overridable via CLI) ===
MIN_EV_PER_CONTRACT = 0.02       # require >= $0.02 EV/contract after fees
MAX_FRACTION = 0.05              # cap per-trade size at 5% of current bankroll
KELLY_MULTIPLIER = 0.25          # use 25% of full Kelly
INITIAL_BANKROLL = 1000.0        # starting bankroll, $
MIN_PRICE = 0.01                 # don't trade at 1¢ or 99¢ (illiquid extremes)
MAX_PRICE = 0.99
TEST_START = pd.Timestamp("2026-01-15 00:00:00")
ENABLE_DECISION_FILTER = True    # take the first crossing per market only

# === Realism caps (override the Kelly cap if smaller) ===
# Real Kalshi liquidity: weather markets typically have OI of 100-2000 contracts,
# hourly volume of 50-500. A naive market order beyond top-of-book gets bad fills.
# We cap stakes both by absolute dollars and by a fraction of the candle's
# observed open_interest at decision time. Slippage models a 0.5¢ extra cost
# per 100 contracts beyond the first 100.
MAX_STAKE_PER_TRADE_DOLLARS = 200.0   # absolute $ cap per trade
MAX_CONTRACTS_PER_TRADE = 500          # absolute contract cap
MAX_OI_FRACTION = 0.5                  # don't take more than 50% of open interest
SLIPPAGE_PER_100_CONTRACTS = 0.005    # extra $/contract beyond 100 contracts


def kalshi_quadratic_fee(price: float, contracts: int) -> float:
    """Kalshi standard quadratic fee: 0.07 * P * (1-P) per contract, $0.01 floor.
    Source: kalshi.com/docs (subject to change). KXHIGHTSFO has fee_type=quadratic
    and fee_multiplier=1. We apply on the entry only (not on settlement)."""
    if contracts <= 0:
        return 0.0
    per_contract = max(0.01, 0.07 * price * (1.0 - price))
    return per_contract * contracts


def kelly_fraction(p_true: float, price: float) -> float:
    """Full Kelly for buying a binary contract at `price` with true win prob p_true.
    EV per dollar staked = p_true / price - 1. Loss probability = 1 - p_true.
    Kelly f* = (p*b - q) / b  where b = (1-price)/price, q = 1 - p.
    Returns float in [0, 1]. Negative or 0 → no position."""
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - p_true
    f = (p_true * b - q) / b
    return max(0.0, f)


def simulate(df: pd.DataFrame, prob_col: str = "model_prob_yes",
             max_hours_to_settle: float = None) -> tuple[pd.DataFrame, dict]:
    """Process trades chronologically; return trade log and summary."""
    df = df.sort_values("decision_time").reset_index(drop=True)
    if max_hours_to_settle is not None:
        df = df[df["hours_to_settle"] <= max_hours_to_settle].copy()
    seen_markets = set()

    bankroll = INITIAL_BANKROLL
    rows = []

    for i, r in df.iterrows():
        if pd.isna(r[prob_col]) or pd.isna(r["yes_ask_close"]) or pd.isna(r["yes_bid_close"]):
            continue
        if r["yes_outcome_derived"] not in (0, 1):
            continue
        ticker = r["ticker"]
        if ENABLE_DECISION_FILTER and ticker in seen_markets:
            continue

        ask = float(r["yes_ask_close"])
        bid = float(r["yes_bid_close"])
        if ask <= MIN_PRICE or ask >= MAX_PRICE or bid <= MIN_PRICE or bid >= MAX_PRICE:
            continue
        p = float(r[prob_col])

        # YES side
        yes_cost = ask
        yes_ev_per_c = p * 1.0 - yes_cost   # EV per contract before fee
        # NO side
        no_cost = 1.0 - bid
        no_ev_per_c = (1.0 - p) * 1.0 - no_cost

        # Estimate fees per contract (we don't yet know contract count, but quadratic
        # fee per contract depends only on price)
        yes_fee_per_c = max(0.01, 0.07 * yes_cost * (1.0 - yes_cost))
        no_fee_per_c = max(0.01, 0.07 * no_cost * (1.0 - no_cost))
        yes_post = yes_ev_per_c - yes_fee_per_c
        no_post = no_ev_per_c - no_fee_per_c

        # Pick the better side
        side = None
        if yes_post >= MIN_EV_PER_CONTRACT and yes_post >= no_post:
            side = "yes"
        elif no_post >= MIN_EV_PER_CONTRACT:
            side = "no"
        if side is None:
            continue

        if side == "yes":
            cost = yes_cost
            p_win = p
            outcome_pays_one = (r["yes_outcome_derived"] == 1)
        else:
            cost = no_cost
            p_win = 1.0 - p
            outcome_pays_one = (r["yes_outcome_derived"] == 0)

        # Kelly fraction with multiplier and cap
        f_kelly = kelly_fraction(p_win, cost)
        f = min(f_kelly * KELLY_MULTIPLIER, MAX_FRACTION)
        if f <= 0:
            continue
        capital_to_stake = min(bankroll * f, MAX_STAKE_PER_TRADE_DOLLARS)
        # Number of whole contracts at top-of-book cost
        contracts = int(capital_to_stake // cost)
        if contracts <= 0:
            continue
        # Apply contract cap and OI fraction cap
        contracts = min(contracts, MAX_CONTRACTS_PER_TRADE)
        oi = r.get("open_interest", 0) or 0
        if oi and oi > 0:
            contracts = min(contracts, int(oi * MAX_OI_FRACTION))
        if contracts <= 0:
            continue
        # Slippage: average extra cost per contract beyond first 100
        if contracts > 100:
            extra = (contracts - 100) * SLIPPAGE_PER_100_CONTRACTS / 2.0  # average over the size
            effective_cost = cost + extra * 1.0 / max(1, contracts)
        else:
            effective_cost = cost
        cash_paid = contracts * effective_cost
        fee = kalshi_quadratic_fee(effective_cost, contracts)
        # Settlement: each contract pays $1 if outcome matches the side, else $0
        payout = contracts * (1.0 if outcome_pays_one else 0.0)
        pnl = payout - cash_paid - fee
        bankroll += pnl

        rows.append({
            "decision_time": r["decision_time"],
            "ticker": ticker,
            "side": side,
            "p_model": p,
            "yes_close": r["market_yes_close"],
            "yes_ask": ask,
            "yes_bid": bid,
            "spread": r["spread"],
            "hours_to_close": r["hours_to_close"],
            "hours_to_settle": r["hours_to_settle"],
            "kelly_full": f_kelly,
            "kelly_used": f,
            "contracts": contracts,
            "cost_per_contract": cost,
            "cash_paid": cash_paid,
            "fee": fee,
            "outcome_yes": r["yes_outcome_derived"],
            "won": int(outcome_pays_one),
            "payout": payout,
            "pnl": pnl,
            "bankroll_after": bankroll,
        })
        if ENABLE_DECISION_FILTER:
            seen_markets.add(ticker)

    log = pd.DataFrame(rows)
    summary = {}
    if len(log) > 0:
        wins = log["won"] == 1
        summary = {
            "n_trades": int(len(log)),
            "n_wins": int(wins.sum()),
            "win_rate": float(wins.mean()),
            "total_pnl": float(log["pnl"].sum()),
            "total_fee": float(log["fee"].sum()),
            "final_bankroll": float(bankroll),
            "return_pct": float((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100),
            "mean_kelly_used": float(log["kelly_used"].mean()),
            "mean_size_dollars": float(log["cash_paid"].mean()),
            "max_drawdown_dollars": _max_drawdown(log["bankroll_after"]),
            "trades_yes": int((log["side"] == "yes").sum()),
            "trades_no": int((log["side"] == "no").sum()),
            "avg_pnl_per_trade": float(log["pnl"].mean()),
        }
    return log, summary


def _max_drawdown(bk: pd.Series) -> float:
    if bk.empty: return 0.0
    peak = bk.cummax()
    dd = peak - bk
    return float(dd.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sfo", help="city slug")
    ap.add_argument("--prob-col", default="model_prob_yes",
                    help="probability column to use ('model_prob_yes' or 'meta_prob_yes')")
    ap.add_argument("--decision-path", default=None,
                    help="decision dataset path (default: per-city auto-derived)")
    ap.add_argument("--max-hours-to-settle", type=float, default=None,
                    help="only trade decisions with hours_to_settle <= this")
    ap.add_argument("--label", default="default",
                    help="label suffix for output files")
    args = ap.parse_args()

    city = get_city(args.city)
    if args.decision_path:
        DECISION_PATH = Path(args.decision_path)
    elif city.slug == "sfo":
        DECISION_PATH = DEFAULT_DECISION_PATH
    else:
        DECISION_PATH = ROOT / "data" / f"decision_dataset_v2_{city.slug}.parquet"
    label = args.label
    suffix = "" if city.slug == "sfo" else f"_{city.slug}"
    LOG_OUT  = ROOT / "data"    / f"trade_log_{label}{suffix}.parquet"
    JSON_OUT = ROOT / "reports" / f"trade_backtest_{label}{suffix}.json"
    MD_OUT   = ROOT / "reports" / f"trade_backtest_{label}{suffix}.md"

    print(f"[backtest] reading {DECISION_PATH}", flush=True)
    df = pd.read_parquet(DECISION_PATH)
    print(f"[backtest] rows: {len(df):,}", flush=True)
    print(f"[backtest] prob_col={args.prob_col}  max_hours_to_settle={args.max_hours_to_settle}",
          flush=True)

    # Restrict to test window
    df = df[df["decision_time"] >= TEST_START].copy()
    print(f"[backtest] after test start filter: {len(df):,}", flush=True)

    log, summary = simulate(df, prob_col=args.prob_col,
                             max_hours_to_settle=args.max_hours_to_settle)
    summary["prob_col"] = args.prob_col
    summary["max_hours_to_settle"] = args.max_hours_to_settle
    summary["label"] = label
    print(f"\n=== TRADING BACKTEST SUMMARY ({label}) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(LOG_OUT, index=False)
    JSON_OUT.write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# Trade Backtest\n"]
    lines.append(f"- Decision window: {TEST_START.date()} → present")
    lines.append(f"- Initial bankroll: ${INITIAL_BANKROLL:,.0f}")
    lines.append(f"- Strategy: edge ≥ ${MIN_EV_PER_CONTRACT:.2f}/contract, "
                 f"Kelly × {KELLY_MULTIPLIER}, cap {MAX_FRACTION*100:.1f}% per trade")
    lines.append("")
    if summary:
        lines.append("## Summary\n")
        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
    if not log.empty:
        lines.append("\n## Trade-side breakdown\n")
        for side in ["yes", "no"]:
            sub = log[log["side"] == side]
            if sub.empty: continue
            wr = sub["won"].mean()
            pnl = sub["pnl"].sum()
            lines.append(f"- {side.upper()}: n={len(sub)}, win_rate={wr:.3f}, P&L=${pnl:+.2f}")
        lines.append("\n## Last 10 trades\n")
        lines.append("| time | ticker | side | p_model | cost | contracts | won | pnl | bk |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in log.tail(10).iterrows():
            lines.append(f"| {r['decision_time']} | {r['ticker']} | {r['side']} | "
                         f"{r['p_model']:.3f} | {r['cost_per_contract']:.3f} | "
                         f"{int(r['contracts'])} | {int(r['won'])} | "
                         f"{r['pnl']:+.2f} | {r['bankroll_after']:.2f} |")
    MD_OUT.write_text("\n".join(lines))
    print(f"\n[backtest] wrote {LOG_OUT}, {JSON_OUT}, {MD_OUT}", flush=True)


if __name__ == "__main__":
    main()
