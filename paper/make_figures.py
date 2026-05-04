"""make_figures.py — Generate all figures used in the paper from real data.
Carefully redone to fix: line-gap artefacts, label overlaps, calibration
small-multiples, and add more diagnostic figures."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path("/Users/terrykim/Documents/SF Weather")
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.family": "serif",
    "mathtext.fontset": "stix",
})


# --- helper: split a series into contiguous non-null segments and plot
def plot_with_gaps(ax, x, y, color, lw, alpha, label=None, max_gap_hours=2):
    """Plot only contiguous segments, breaking the line on gaps."""
    arr = np.asarray(y, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return
    # group contiguous indices where consecutive timestamps are within max_gap
    x = pd.to_datetime(x)
    seg_start = []
    seg_end = []
    in_seg = False
    for i, v in enumerate(valid):
        if v and not in_seg:
            seg_start.append(i); in_seg = True
        elif not v and in_seg:
            seg_end.append(i); in_seg = False
    if in_seg:
        seg_end.append(len(valid))
    first = True
    for s, e in zip(seg_start, seg_end):
        if e - s < 1: continue
        # further split if there's a time-gap > max_gap inside the segment
        sub_x = x[s:e]; sub_y = arr[s:e]
        if len(sub_x) <= 1:
            ax.plot(sub_x, sub_y, "o", color=color, alpha=alpha, ms=2)
            continue
        diffs = pd.Series(sub_x).diff().dt.total_seconds() / 3600
        breaks = np.where(diffs > max_gap_hours)[0]
        chunk_starts = [0] + list(breaks)
        chunk_ends = list(breaks) + [len(sub_x)]
        for cs, ce in zip(chunk_starts, chunk_ends):
            if ce - cs < 1: continue
            ax.plot(sub_x[cs:ce], sub_y[cs:ce], color=color, lw=lw, alpha=alpha,
                    label=label if first else None)
            if first: first = False


# ============================================================
# Figure 1: SFO daily peak / trough hour distribution by month
# ============================================================
print("[fig] 1: peak/trough hour distribution by month")
hourly = pd.read_parquet(ROOT / "data" / "sfo_hourly.parquet")[["hour", "temp_f"]].dropna()
hourly["day"] = hourly["hour"].dt.floor("D")
hourly["hod"] = hourly["hour"].dt.hour
hourly["month"] = hourly["hour"].dt.month
recent = hourly[hourly["day"] >= "2020-01-01"].copy()

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for ax, kind in zip(axes, ["high", "low"]):
    means = []
    p25s = []
    p75s = []
    for m in range(1, 13):
        sub = recent[recent["month"] == m]
        if kind == "high":
            ext = sub.loc[sub.groupby("day")["temp_f"].idxmax()]
        else:
            ext = sub.loc[sub.groupby("day")["temp_f"].idxmin()]
        means.append(ext["hod"].mean())
        p25s.append(ext["hod"].quantile(0.25))
        p75s.append(ext["hod"].quantile(0.75))
    color = "#1a6e1a" if kind == "high" else "#1f4ea8"
    bars = ax.bar(months, means, color=color, alpha=0.78, zorder=2)
    err_lo = [max(0, m - p) for m, p in zip(means, p25s)]
    err_hi = [max(0, p - m) for m, p in zip(means, p75s)]
    ax.errorbar(months, means, yerr=[err_lo, err_hi], fmt="none",
                color="black", capsize=3, lw=0.8, zorder=3)
    ax.set_title(f"Mean hour of day SFO daily {'HIGH' if kind=='high' else 'LOW'} occurs (PST)\n"
                 f"with interquartile range")
    ax.set_ylabel("hour of day (PST)")
    ax.set_ylim(0, 18)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_diurnal_timing.pdf")
plt.close()


# ============================================================
# Figure 2: Pinball loss heatmap
# ============================================================
print("[fig] 2: pinball loss heatmap")
metrics = json.loads((ROOT / "reports" / "train_metrics.json").read_text())
horizons = metrics["horizons"]
quantiles = metrics["quantiles"]
M = np.zeros((len(horizons), len(quantiles)))
for i, h in enumerate(horizons):
    for j, q in enumerate(quantiles):
        M[i, j] = metrics["metrics"][f"h{h}"][f"q{int(q*100):02d}"]["pinball_val"]

fig, ax = plt.subplots(figsize=(6.5, 3.4))
im = ax.imshow(M, aspect="auto", cmap="viridis_r")
ax.set_xticks(range(len(quantiles)))
ax.set_xticklabels([f"{q:.2f}" for q in quantiles])
ax.set_yticks(range(len(horizons)))
ax.set_yticklabels([f"+{h}h" for h in horizons])
ax.set_xlabel(r"Predicted quantile $\tau$")
ax.set_ylabel("Forecast horizon")
ax.set_title(r"Validation pinball loss $\mathrm{PL}_\tau$ (lower is better, units $^\circ$F)")
for i in range(len(horizons)):
    for j in range(len(quantiles)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                color="white" if M[i,j] > 0.7 else "black", fontsize=7)
cb = plt.colorbar(im, ax=ax)
cb.set_label(r"pinball loss ($^\circ$F)")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_pinball_heatmap.pdf")
plt.close()


# ============================================================
# Figure 3: Test-set MAE vs baselines + skill
# ============================================================
print("[fig] 3: backtest skill")
bt = json.loads((ROOT / "reports" / "backtest_metrics.json").read_text())
hm = bt["horizon_metrics"]
hs = sorted(hm.keys(), key=int)
mae = [hm[h]["mae"] for h in hs]
mae_p = [hm[h]["mae_persist"] for h in hs]
mae_c = [hm[h]["mae_clim"] for h in hs]
skill_p = [hm[h]["skill_vs_persist"] for h in hs]
skill_c = [hm[h]["skill_vs_clim"] for h in hs]
hs_int = [int(h) for h in hs]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
axes[0].plot(hs_int, mae, "o-", color="#2c5aa0", label="our model", lw=2, ms=6)
axes[0].plot(hs_int, mae_p, "s--", color="#a82929", label="persistence", lw=1.5, ms=5)
axes[0].plot(hs_int, mae_c, "^--", color="#1a6e1a", label="climatology", lw=1.5, ms=5)
axes[0].set_xlabel("forecast horizon (hours)")
axes[0].set_ylabel(r"MAE ($^\circ$F)")
axes[0].set_title("Test-set MAE vs baselines")
axes[0].legend(frameon=False)
axes[0].set_xticks(hs_int)
for h, m in zip(hs_int, mae):
    axes[0].annotate(f"{m:.2f}", (h, m), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7,
                     color="#2c5aa0")

axes[1].plot(hs_int, [s*100 for s in skill_p], "s-", color="#a82929", label="vs persistence", lw=2, ms=5)
axes[1].plot(hs_int, [s*100 for s in skill_c], "^-", color="#1a6e1a", label="vs climatology", lw=2, ms=5)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_xlabel("forecast horizon (hours)")
axes[1].set_ylabel(r"skill (\%)")
axes[1].set_title("Skill scores (positive = beat baseline)")
axes[1].legend(frameon=False)
axes[1].set_xticks(hs_int)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_backtest_mae.pdf")
plt.close()


# ============================================================
# Figure 4: Decision-time skill (improved label positioning)
# ============================================================
print("[fig] 4: decision-time skill")
d = pd.read_parquet(ROOT / "data" / "decision_dataset_v2.parquet")
elig = d.dropna(subset=["model_prob_yes", "market_yes_close"])
elig = elig[elig["yes_outcome_derived"].isin([0, 1])].copy()

bins = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 96)]
labels, ll_m_l, ll_k_l, ns = [], [], [], []
for lo, hi in bins:
    sub = elig[(elig["hours_to_settle"] >= lo) & (elig["hours_to_settle"] < hi)]
    if len(sub) < 10: continue
    eps = 1e-6
    pm = np.clip(sub["model_prob_yes"].values, eps, 1-eps)
    pk = np.clip(sub["market_yes_close"].values, eps, 1-eps)
    y = sub["yes_outcome_derived"].values.astype(float)
    ll_m = float(-np.mean(y*np.log(pm)+(1-y)*np.log(1-pm)))
    ll_k = float(-np.mean(y*np.log(pk)+(1-y)*np.log(1-pk)))
    labels.append(f"[{lo}-{hi}h)")
    ll_m_l.append(ll_m)
    ll_k_l.append(ll_k)
    ns.append(len(sub))

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.0, 3.6))
w = 0.36
b1 = ax.bar(x - w/2, ll_m_l, w, color="#2c5aa0", label="our model", alpha=0.88)
b2 = ax.bar(x + w/2, ll_k_l, w, color="#d4a017", label="Kalshi market mid", alpha=0.88)
# Place value labels INSIDE each bar so they don't overlap
for bar, v in zip(b1, ll_m_l):
    ax.text(bar.get_x() + bar.get_width()/2, v - 0.04, f"{v:.3f}",
            ha="center", va="top", color="white", fontsize=7, fontweight="bold")
for bar, v in zip(b2, ll_k_l):
    ax.text(bar.get_x() + bar.get_width()/2, v - 0.04, f"{v:.3f}",
            ha="center", va="top", color="white", fontsize=7, fontweight="bold")
# n labels above the taller bar in each group, offset cleanly
for i, n in enumerate(ns):
    top = max(ll_m_l[i], ll_k_l[i])
    ax.text(x[i], top + 0.04, f"$n={n:,}$", ha="center", fontsize=7, color="#444")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Test-set log-loss (lower is better)")
ax.set_xlabel("Decision time relative to settlement")
ax.set_title("Probabilistic skill: model vs Kalshi market by horizon to settlement")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, max(max(ll_m_l), max(ll_k_l)) * 1.18)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_decision_skill.pdf")
plt.close()


# ============================================================
# Figure 5: Equity curves (4 strategy variants)
# ============================================================
print("[fig] 5: equity curves")
fig, ax = plt.subplots(figsize=(7.0, 3.6))
strategies = [
    ("A_realistic_all", "all horizons (raw model)", "#888888", 1.2),
    ("B_realistic_short", r"$\leq$12h, raw model", "#2c5aa0", 1.8),
    ("C_realistic_meta_short", r"$\leq$12h, meta-blend", "#1a6e1a", 1.4),
    ("D_realistic_v_short", r"$\leq$6h, raw model", "#d4a017", 1.4),
]
for label, name, color, lw in strategies:
    fpath = ROOT / "data" / f"trade_log_{label}.parquet"
    if not fpath.exists(): continue
    log = pd.read_parquet(fpath).sort_values("decision_time").reset_index(drop=True)
    if log.empty: continue
    ax.plot(log["decision_time"], log["bankroll_after"], color=color, lw=lw, label=name)
ax.axhline(1000, color="black", lw=0.6, alpha=0.6, ls=":")
ax.text(log["decision_time"].iloc[0], 1100, "initial bankroll = \\$1{,}000",
        fontsize=7, color="#444")
ax.set_xlabel("trade time")
ax.set_ylabel("bankroll (\\$)")
ax.set_title("Bankroll trajectory by strategy variant (initial \\$1{,}000)")
ax.legend(frameon=False, loc="upper left")
ax.set_yscale("symlog", linthresh=2000)
ax.yaxis.set_major_locator(MaxNLocator(7))
plt.tight_layout()
plt.savefig(FIGDIR / "fig_equity_curves.pdf")
plt.close()


# ============================================================
# Figure 6: Quantile coverage --- now as small multiples
# ============================================================
print("[fig] 6: quantile coverage (small multiples)")
fig, axes = plt.subplots(2, 4, figsize=(8.5, 4.4), sharex=True, sharey=True)
axes = axes.flatten()
for ax, h in zip(axes[:7], horizons):
    cov = []
    for q in quantiles:
        cov.append(metrics["metrics"][f"h{h}"][f"q{int(q*100):02d}"]["coverage_val"])
    ax.plot(quantiles, cov, "o-", color="#2c5aa0", lw=1.5, ms=5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.7)
    ax.set_title(f"horizon h={h}h")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    # annotate gap from diagonal at each point
    for q, c in zip(quantiles, cov):
        ax.text(q, c + 0.04, f"{c:.2f}", ha="center", fontsize=6, color="#444")
# Hide the unused 8th panel
axes[7].axis("off")
fig.text(0.5, 0.0, r"nominal quantile $\tau$", ha="center")
fig.text(0.0, 0.5, "empirical coverage on validation",
         va="center", rotation="vertical")
fig.suptitle("Quantile calibration of the hourly forecast (one panel per horizon)")
plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])
plt.savefig(FIGDIR / "fig_quantile_calibration.pdf")
plt.close()


# ============================================================
# Figure 7: Bucket forecast example with non-overlapping labels
# ============================================================
print("[fig] 7: example bucket forecast")
sig = json.loads((ROOT / "reports" / "live_signals.json").read_text())
tomorrow_sigs = [s for s in sig.get("signals", []) if s.get("series_ticker") == "KXHIGHTSFO"]
if tomorrow_sigs:
    days = sorted({s["day_D"] for s in tomorrow_sigs})
    target_day = days[1] if len(days) >= 2 else days[0]
    sigs = [s for s in tomorrow_sigs if s["day_D"] == target_day]
    sigs.sort(key=lambda s: -1000 if s["strike_type"] == "less"
              else 1000 if s["strike_type"] == "greater"
              else (s["floor_strike"] or 0) + (s["cap_strike"] or 0))
    labels = [s["yes_sub_title"] for s in sigs]
    pmod = [(s["model_prob_yes"] or 0) * 100 for s in sigs]
    pmk = [((s["yes_bid"] or 0) + (s["yes_ask"] or 0)) / 2 * 100
            if s.get("yes_bid") is not None else 0 for s in sigs]

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w/2, pmod, w, color="#2c5aa0", label="our model", alpha=0.88)
    b2 = ax.bar(x + w/2, pmk, w, color="#d4a017", label="Kalshi market mid", alpha=0.88)
    ymax = max(max(pmod), max(pmk)) * 1.25
    for bar, v in zip(b1, pmod):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}\\%",
                ha="center", fontsize=7, color="#2c5aa0")
    for bar, v in zip(b2, pmk):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}\\%",
                ha="center", fontsize=7, color="#a07000")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"P(YES) (\%)")
    ax.set_title(f"SFO daily HIGH bucket probabilities --- settlement day {target_day}")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_example_buckets.pdf")
    plt.close()


# ============================================================
# Figure 8: System architecture
# ============================================================
print("[fig] 8: system architecture")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, label, color="#cfd8dc"):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1.0)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1, color="black"))

box(0.2, 5.0, 1.8, 0.7, "NOAA LCD\n1970-present", "#fce8b2")
box(0.2, 4.0, 1.8, 0.7, "NOAA ISD\n1973-present", "#fce8b2")
box(0.2, 3.0, 1.8, 0.7, "NWS METAR\nlive feed", "#fce8b2")
box(0.2, 2.0, 1.8, 0.7, "Kalshi public\nmarket API", "#fce8b2")

box(2.5, 4.5, 1.8, 0.7, "Canonical hourly grid\n(PST, 56 yrs)", "#cfe5ff")
box(2.5, 3.5, 1.8, 0.7, "Daily extreme labels\n(SOD)", "#cfe5ff")
box(2.5, 2.5, 1.8, 0.7, "Decision-time\nmarket join", "#cfe5ff")

box(5.0, 5.0, 1.8, 0.7, "49 hourly\nquantile models", "#c8e6c9")
box(5.0, 4.0, 1.8, 0.7, "14 daily-extreme\nquantile models", "#c8e6c9")
box(5.0, 3.0, 1.8, 0.7, "Meta-calibrator\n(logreg + isotonic)", "#c8e6c9")

box(7.5, 5.0, 2.2, 0.7, "Calibrated CDF per horizon", "#ffccbc")
box(7.5, 4.0, 2.2, 0.7, r"Strike probability $p_\theta$", "#ffccbc")
box(7.5, 3.0, 2.2, 0.7, "Trading signal & EV", "#ffccbc")
box(7.5, 2.0, 2.2, 0.7, "Live HTML dashboard", "#ffccbc")

arrow(2.0, 5.3, 2.5, 4.9); arrow(2.0, 4.3, 2.5, 4.8)
arrow(2.0, 3.3, 2.5, 4.7); arrow(2.0, 2.3, 2.5, 2.8)
arrow(4.3, 4.9, 5.0, 5.3); arrow(4.3, 3.9, 5.0, 4.3); arrow(4.3, 2.9, 5.0, 3.3)
arrow(6.8, 5.3, 7.5, 5.3); arrow(6.8, 4.3, 7.5, 4.3); arrow(6.8, 3.3, 7.5, 3.3)
arrow(6.5, 3.0, 7.5, 2.3)

ax.set_title("End-to-end system architecture: from NOAA observations to Kalshi trading signal",
             fontsize=10)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_architecture.pdf")
plt.close()


# ============================================================
# Figure 9: Reliability diagram (cleaner)
# ============================================================
print("[fig] 9: reliability diagram")
d_full = pd.read_parquet(ROOT / "data" / "decision_dataset_v2_meta.parquet")
elig_full = d_full.dropna(subset=["model_prob_yes", "market_yes_close"])
elig_full = elig_full[elig_full["yes_outcome_derived"].isin([0, 1])].copy()

bin_edges = np.linspace(0, 1, 11)
def reliability(p, y, edges):
    centers, actuals, counts = [], [], []
    for i in range(len(edges) - 1):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() < 20: continue
        centers.append(p[mask].mean())
        actuals.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(actuals), counts

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, sub_def in zip(axes, [
        ("All eligible decisions", elig_full),
        ("Restricted to $\\leq$12h to settlement", elig_full[elig_full["hours_to_settle"] <= 12])]):
    title, sub = sub_def
    ys = sub["yes_outcome_derived"].values.astype(float)
    for label, col, c in [("our model", "model_prob_yes", "#2c5aa0"),
                          ("Kalshi market mid", "market_yes_close", "#d4a017"),
                          ("meta-blend", "meta_prob_yes", "#1a6e1a")]:
        cs, acts, cnts = reliability(sub[col].values, ys, bin_edges)
        ax.plot(cs, acts, "o-", color=c, label=label, lw=1.6, ms=5, alpha=0.92)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.7, label="perfect calibration")
    ax.set_xlabel("predicted P(YES)")
    ax.set_ylabel("observed YES rate")
    ax.set_title(f"{title}  ($n={len(sub):,}$)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_reliability.pdf")
plt.close()


# ============================================================
# Figure 10: Per-month breakdown
# ============================================================
print("[fig] 10: per-month breakdown")
log = pd.read_parquet(ROOT / "data" / "trade_log_B_realistic_short.parquet").copy()
log["decision_time"] = pd.to_datetime(log["decision_time"])
log["month"] = log["decision_time"].dt.to_period("M").astype(str)
monthly = log.groupby("month").agg(
    trades=("ticker", "count"),
    wins=("won", "sum"),
    pnl=("pnl", "sum"),
).reset_index()
monthly["win_rate"] = monthly["wins"] / monthly["trades"]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
colors_ = ["#1a6e1a" if v >= 0 else "#a82929" for v in monthly["pnl"]]
b = axes[0].bar(monthly["month"], monthly["pnl"], color=colors_, alpha=0.88)
ymax_p = monthly["pnl"].max(); ymin_p = monthly["pnl"].min()
pad = (ymax_p - ymin_p) * 0.12
for bar, n, p in zip(b, monthly["trades"], monthly["pnl"]):
    offset = pad if p >= 0 else -pad
    axes[0].text(bar.get_x() + bar.get_width()/2, p + offset,
                 f"$n={n}$", ha="center", fontsize=7, color="#444")
axes[0].axhline(0, color="black", lw=0.6)
axes[0].set_ylabel("net P\\&L (\\$)")
axes[0].set_title("Strategy B: P\\&L by month ($\\leq$12h, raw model)")
axes[0].set_ylim(ymin_p - pad*2, ymax_p + pad*3)

b2 = axes[1].bar(monthly["month"], monthly["win_rate"]*100, color="#2c5aa0", alpha=0.88)
axes[1].axhline(50, color="black", lw=0.5, ls="--")
axes[1].set_ylabel("win rate (\\%)")
axes[1].set_title("Strategy B: win rate by month")
axes[1].set_ylim(0, 100)
for bar, w_ in zip(b2, monthly["win_rate"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, w_*100 + 2,
                 f"{w_*100:.0f}\\%", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_monthly_breakdown.pdf")
plt.close()


# ============================================================
# Figure 11: Price evolution (FIXED: handle gaps cleanly + larger plot)
# ============================================================
print("[fig] 11: price evolution (clean)")
mk = pd.read_parquet(ROOT / "data" / "kalshi_markets.parquet")
ck = pd.read_parquet(ROOT / "data" / "kalshi_candles.parquet")

chosen_event = "KXHIGHTSFO-26APR15"
event_mks = mk[mk["event_ticker"] == chosen_event].sort_values("floor_strike", na_position="first")

# Build a regular hourly time index spanning all candles for this event
all_times = ck[ck["ticker"].isin(event_mks["ticker"])]["end_time"].dropna()
if not all_times.empty:
    t_min = all_times.min()
    t_max = all_times.max()
    # Use a more distinct color palette
    palette = ["#7b3294","#c2a5cf","#a6dba0","#1a6e1a","#fdb863","#e66101"]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    for (_, r), color in zip(event_mks.iterrows(), palette):
        sub = ck[ck["ticker"] == r["ticker"]].sort_values("end_time")
        if sub.empty: continue
        won = (r["result"] == "yes")
        lw = 2.0 if won else 0.9
        alpha = 1.0 if won else 0.6
        # Plot with explicit gap detection (>2h between consecutive candles -> break line)
        plot_with_gaps(ax, sub["end_time"].values, sub["price_close"].values,
                       color=color, lw=lw, alpha=alpha,
                       label=f"{r['yes_sub_title']} ({'WON' if won else 'lost'})",
                       max_gap_hours=2)
    ax.set_xlabel("trading time (UTC)")
    ax.set_ylabel("YES close price (\\$)")
    ax.set_title(f"Kalshi market price evolution: {chosen_event}\n(6 mutually-exclusive bucket strikes)")
    # Manually rebuild legend so each bucket appears exactly once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax.set_ylim(-0.02, 1.05)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_price_evolution.pdf")
    plt.close()


# ============================================================
# Figure 12: Kelly fraction surface and trade-EV histogram
# ============================================================
print("[fig] 12: trade-level distributions")
log_b = pd.read_parquet(ROOT / "data" / "trade_log_B_realistic_short.parquet").copy()
log_b["decision_time"] = pd.to_datetime(log_b["decision_time"])

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

# (a) histogram of per-trade P&L
ax = axes[0]
ax.hist(log_b["pnl"], bins=40, color="#2c5aa0", alpha=0.85, edgecolor="white")
ax.axvline(log_b["pnl"].mean(), color="#a82929", lw=1.5, ls="--",
           label=f"mean = \\${log_b['pnl'].mean():.1f}")
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("per-trade P\\&L (\\$)")
ax.set_ylabel("count")
ax.set_title(f"Per-trade P\\&L distribution (strategy B, $n={len(log_b):,}$)")
ax.legend(frameon=False)

# (b) scatter: cost per contract vs. won/lost
ax = axes[1]
won = log_b[log_b["won"] == 1]
lost = log_b[log_b["won"] == 0]
ax.scatter(lost["cost_per_contract"], lost["pnl"], color="#a82929",
           alpha=0.5, s=12, label=f"lost ($n={len(lost):,}$)")
ax.scatter(won["cost_per_contract"], won["pnl"], color="#1a6e1a",
           alpha=0.6, s=12, label=f"won ($n={len(won):,}$)")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("cost per contract (\\$)")
ax.set_ylabel("trade P\\&L (\\$)")
ax.set_title("Trade outcomes by entry cost")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_trade_distributions.pdf")
plt.close()


print("\nAll figures regenerated:")
for f in sorted(FIGDIR.glob("*.pdf")):
    print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
