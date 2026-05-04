# Ariadne Labs

A quantitative AI hedge fund for prediction markets, run by Terry Kim,
Andrew Liang, and Krit Phaisamran (Stanford CS, Math minor). This repo
contains the full research, code, paper, and public-facing site.

> **Live site:** Deploy this repo to Vercel — see [Vercel deployment](#vercel-deployment).
>
> **Live terminal locally:** `cd site && ./serve.sh` → open `http://127.0.0.1:8765/`.

---

## Repo layout

```
ariadnefund/
├── site/                       ← public site (deployed to Vercel)
│   ├── index.html              landing
│   ├── research.html           papers list
│   ├── login.html              investor login (open auth in this build)
│   ├── terminal.html           Bloomberg-style investor dashboard
│   ├── css/, js/, img/         assets
│   ├── data/fund_state.json    live fund state read by terminal
│   ├── research/               public PDF of paper 01
│   ├── build_fund_state.py     regenerates fund_state.json from project data
│   └── serve.sh                local-dev http server
│
├── code/                       ← research pipeline (Python)
│   ├── 01_audit.py             data quality report
│   ├── 02_build_dataset.py     LCD + ISD → canonical hourly
│   ├── 03_features.py          250-feature engineering
│   ├── 04_train.py             49 hourly quantile models (~21 min)
│   ├── 05_backtest.py          held-out test on 2023-2026
│   ├── 06_predict.py           live forecast for an issuance hour
│   ├── 07_kalshi_fetch.py      pull Kalshi public market data
│   ├── 08_decision_dataset.py  (v1, deprecated) decision-time dataset
│   ├── 09_daily_extreme_train.py   14 daily-extreme quantile models
│   ├── 10_decision_dataset_v2.py   joint (market × decision-time) data
│   ├── 11_trade_backtest.py    realistic trading backtest
│   ├── 12_meta_model.py        logreg+isotonic meta-calibrator
│   ├── 13_live_signal.py       live actionable trade signals
│   ├── 14_refresh_noaa.py      pull live KSFO METARs and rebuild features
│   ├── 15_dashboard.py         render the strategy-level HTML dashboard
│   ├── refresh_chain.sh        full refresh pipeline (called by launchd)
│   └── com.terrykim.sfo-weather.plist   macOS launchd job def
│
├── data/                       ← parquet outputs (small ones tracked, large ones gitignored)
│   ├── sfo_hourly.parquet               7 MB    canonical hourly grid
│   ├── sfo_daily.parquet                280 KB  daily SOD summary
│   ├── sfo_climatology.parquet          330 KB  hour-of-year clim
│   ├── kalshi_events.parquet            (small)
│   ├── kalshi_markets.parquet           (small)
│   ├── trade_log_*.parquet              backtest trade logs
│   ├── sfo_features.parquet             [GITIGNORED — 136 MB; regen via 03_features.py]
│   ├── sfo_targets.parquet              [GITIGNORED — regen via 03_features.py]
│   ├── kalshi_candles.parquet           [GITIGNORED — regen via 07_kalshi_fetch.py]
│   └── decision_dataset_v2*.parquet     [GITIGNORED — regen via 10_decision_dataset_v2.py]
│
├── models/                     ← 63 trained quantile models (~240 MB, all <100MB each)
│   ├── qmodel_h{H}_q{Q}.joblib          49 hourly models (7 horizons × 7 quantiles)
│   ├── dxmodel_{kind}_q{Q}.joblib       14 daily-extreme models (HIGH/LOW × 7 quantiles)
│   └── meta_calibrator.joblib           tiny logreg+isotonic blender
│
├── paper/                      ← LaTeX source for the research paper
│   ├── main.tex                33-page paper, 65 refs, 12 figures
│   ├── main.pdf                compiled output
│   ├── arxiv.sty
│   ├── references.bib
│   ├── make_figures.py         regenerates figures from the data
│   └── figures/                12 PDF figures
│
├── reports/                    ← project artifacts (audit, summaries, forecasts)
│   ├── audit.md / build_summary.md / feature_summary.md
│   ├── train_metrics.json / train_summary.md
│   ├── backtest_metrics.json / backtest_summary.md
│   ├── decision_dataset_v2_summary.md
│   ├── daily_extreme_metrics.json / daily_extreme_summary.md
│   ├── meta_model_summary.md
│   ├── trade_backtest_*.{json,md}    multiple strategy variants
│   ├── forecast.json / forecast.md   latest live forecast
│   ├── live_signals.json / live_signals.md   actionable trades
│   ├── dashboard.html                strategy-level HTML dashboard
│   ├── KALSHI_REPORT.md / FINAL_REPORT.md
│   └── train_log.txt
│
├── README.md (this file)
├── USAGE.md
├── AUTOMATION.md
├── .gitignore
└── vercel.json                 ← config for Vercel static deploy
```

---

## Vercel deployment

This repo is configured to deploy as a static site on Vercel. The
[`vercel.json`](vercel.json) at the repo root rewrites all incoming
requests to the `site/` directory.

**To deploy:**

1. Push this repo to GitHub (see [Pushing to GitHub](#pushing-to-github)).
2. Go to [vercel.com/new](https://vercel.com/new) → Import Git Repository.
3. Paste the repo URL: `https://github.com/ttkim0/ariadnefund`.
4. Click **Deploy** — leave all settings at default. Vercel will detect
   `vercel.json` and serve `/site/*` as the public site.
5. Vercel gives you a URL like `https://ariadnefund.vercel.app/`.

To attach a custom domain, Vercel project → Settings → Domains.

---

## Local development

### Run the public site locally

```bash
cd site
./serve.sh
# open http://127.0.0.1:8765/
```

The `serve.sh` script:
1. Regenerates `data/fund_state.json` from current backtest + live data.
2. Starts a Python HTTP server on port 8765.

### Re-run the full research pipeline

```bash
# raw NOAA data (gitignored — re-download once if needed)
# put `LCD datas.csv` and `global hourly.csv` in the repo root
# (sources: https://www.ncei.noaa.gov/cdo-web/  and
#          https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

python3 code/01_audit.py            # ~30 s
python3 code/02_build_dataset.py    # ~1 min
python3 code/03_features.py         # ~2 min, regenerates features.parquet
python3 code/04_train.py            # ~21 min, trains 49 hourly models
python3 code/05_backtest.py         # ~30 s
python3 code/07_kalshi_fetch.py     # ~3 min, pulls public Kalshi data
python3 code/09_daily_extreme_train.py   # ~30 min, trains 14 daily models
python3 code/10_decision_dataset_v2.py   # ~2 min
python3 code/11_trade_backtest.py        # < 30 s
python3 code/12_meta_model.py            # < 30 s
python3 code/13_live_signal.py           # ~30 s
python3 code/15_dashboard.py             # < 5 s
```

Or run the full chain end-to-end:

```bash
code/refresh_chain.sh
```

### Auto-refresh via launchd (macOS)

See [`AUTOMATION.md`](AUTOMATION.md) for the launchd setup that runs
`refresh_chain.sh` every 5 minutes, keeping the dashboard fresh.

---

## Pushing to GitHub

This repo is configured to push to
[`https://github.com/ttkim0/ariadnefund`](https://github.com/ttkim0/ariadnefund).

```bash
git init
git add .
git commit -m "initial commit: ariadne labs"
git branch -M main
git remote add origin https://github.com/ttkim0/ariadnefund.git
git push -u origin main
```

If your local git is not authenticated for GitHub, use
[GitHub CLI](https://cli.github.com/):

```bash
gh auth login
gh repo create ttkim0/ariadnefund --public --source=. --push
```

---

## Secrets — what's hidden

This repo contains **no API keys, no passwords, no private credentials**.

- **Kalshi API key:** never written to any file — Kalshi market-data
  endpoints are public and require no authentication for the read-only
  use cases in this repo.
- **`.gitignore`** excludes `.pem`, `.env`, `.key`, `credentials*`,
  `secrets*`, and `~/.kalshi/` patterns by default.
- **The investor login** uses open access in the public build (any
  non-empty username + password). To put non-public information behind
  real auth before deployment, replace `site/js/auth.js` with
  server-side auth at the reverse-proxy or platform level.

---

## License & disclaimer

This repository is distributed for research transparency. It does not
constitute an offer to sell or a solicitation of an offer to buy any
securities. All investments involve risk, including loss of principal.
Past performance and backtest results are not indicative of future
results. Available to qualified purchasers only.

© 2026 Ariadne Labs · Stanford, California
