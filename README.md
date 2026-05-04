# Ariadne Labs

A quantitative AI hedge fund for prediction markets, run by Terry Kim,
Andrew Liang, and Krit Phaisamran (Stanford CS, Math minor). This repo
contains the full research, code, paper, and public-facing site.

> **Live site (Vercel):** push this repo to GitHub, import to Vercel, deploy. Done.
>
> **Local dev:** `./serve.sh` → open `http://127.0.0.1:8765/`.

---

## Repo layout (flat — site files are at the repo root so Vercel auto-detects)

```
ariadnefund/
│   ───── public site (deployed by Vercel) ─────
├── index.html                  landing
├── research.html               papers list
├── login.html                  investor login (open auth in this build)
├── terminal.html               Bloomberg-style investor dashboard
├── css/                        site.css, terminal.css
├── js/                         auth.js
├── img/                        SVG charts (hero-equity, skill-bars)
├── data/fund_state.json        live state read by terminal
├── research/                   public PDF of paper 01
├── build_fund_state.py         regenerates fund_state.json from project data
├── serve.sh                    local-dev http server
├── vercel.json                 deploy config (security headers, clean URLs)
├── .vercelignore               keeps `code/`, `models/`, `paper/`, etc. out of the deploy
│
│   ───── research pipeline (NOT deployed; lives only on GitHub) ─────
├── code/                       14 Python pipeline scripts + launchd plist
│   ├── 01_audit.py             data quality report
│   ├── 02_build_dataset.py     LCD + ISD → canonical hourly grid
│   ├── 03_features.py          250-feature engineering (~2 min)
│   ├── 04_train.py             49 hourly quantile models (~21 min)
│   ├── 05_backtest.py          held-out 2023-2026 test
│   ├── 06_predict.py           live forecast for an issuance hour
│   ├── 07_kalshi_fetch.py      pull public Kalshi market data
│   ├── 09_daily_extreme_train.py   14 daily-extreme quantile models (~30 min)
│   ├── 10_decision_dataset_v2.py
│   ├── 11_trade_backtest.py
│   ├── 12_meta_model.py        logreg + isotonic meta-calibrator
│   ├── 13_live_signal.py
│   ├── 14_refresh_noaa.py      pull live KSFO METARs, rebuild features
│   ├── 15_dashboard.py         strategy-level HTML dashboard
│   ├── refresh_chain.sh        full refresh pipeline (called by launchd)
│   └── com.terrykim.sfo-weather.plist
│
├── data/                       small parquet outputs (large ones gitignored)
│   ├── sfo_hourly.parquet                  7 MB    canonical hourly grid
│   ├── sfo_daily.parquet                  280 KB   daily SOD summary
│   ├── sfo_climatology.parquet            330 KB   hour-of-year clim
│   ├── kalshi_events.parquet              (small)
│   ├── kalshi_markets.parquet             (small)
│   ├── trade_log_*.parquet                backtest trade logs
│   ├── fund_state.json                    live state for terminal
│   └── (sfo_features.parquet, kalshi_candles.parquet, …)   GITIGNORED, regenerable
│
├── models/                     63 trained quantile models (~240 MB total)
│   ├── qmodel_h{H}_q{Q}.joblib            49 hourly models
│   ├── dxmodel_{kind}_q{Q}.joblib         14 daily-extreme models
│   └── meta_calibrator.joblib             tiny logreg + isotonic blender
│
├── paper/                      LaTeX source for the research paper
│   ├── main.tex                33-page paper, 65 refs, 12 figures
│   ├── main.pdf                compiled output
│   ├── references.bib
│   └── figures/                12 PDF figures
│
├── reports/                    audit, train, backtest, decision-skill, etc.
│
├── README.md (this file)
├── USAGE.md
├── AUTOMATION.md
└── .gitignore
```

---

## Vercel deployment — the one-click path

1. **Push this repo to GitHub** (already done at
   [ttkim0/ariadnefund](https://github.com/ttkim0/ariadnefund)).
2. Open **[vercel.com/new](https://vercel.com/new)**.
3. **Import Git Repository** → authorize Vercel for your GitHub if needed → pick `ttkim0/ariadnefund`.
4. **Leave every setting at default** and click **Deploy**.
   - Framework: "Other" (Vercel auto-detects the static `index.html` at the root).
   - Build Command: leave blank.
   - Output Directory: leave blank.
   - The `.vercelignore` ensures `code/`, `models/`, `paper/`, `reports/`, and large parquet files are excluded from the deploy.
5. ~30 seconds later you get a live URL like `https://ariadnefund.vercel.app/`.

---

## Local development

### Run the public site locally

```bash
./serve.sh
# open http://127.0.0.1:8765/
```

The script:
1. Regenerates `data/fund_state.json` from current backtest + live data.
2. Starts a Python HTTP server on port 8765.

### Re-run the research pipeline

```bash
# raw NOAA data (gitignored — re-download once if needed)
# put `LCD datas.csv` and `global hourly.csv` in the repo root
# (https://www.ncei.noaa.gov/cdo-web/  +
#  https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

python3 code/01_audit.py             # ~30 s
python3 code/02_build_dataset.py     # ~1 min
python3 code/03_features.py          # ~2 min
python3 code/04_train.py             # ~21 min, 49 hourly models
python3 code/05_backtest.py          # ~30 s
python3 code/07_kalshi_fetch.py      # ~3 min, pulls public Kalshi data
python3 code/09_daily_extreme_train.py    # ~30 min, 14 daily models
python3 code/10_decision_dataset_v2.py    # ~2 min
python3 code/11_trade_backtest.py         # < 30 s
python3 code/12_meta_model.py             # < 30 s
python3 code/13_live_signal.py            # ~30 s
python3 code/15_dashboard.py              # < 5 s
```

Or end-to-end:

```bash
code/refresh_chain.sh
```

### Auto-refresh via launchd (macOS)

See [`AUTOMATION.md`](AUTOMATION.md) for the setup that runs
`refresh_chain.sh` every 5 minutes.

---

## Secrets — what's hidden

This repo contains **no API keys, no passwords, no private credentials**.

- **Kalshi API key:** never written to any file — Kalshi market-data
  endpoints are public and require no authentication for the read-only
  use cases here.
- **`.gitignore`** excludes `.pem`, `.env`, `.key`, `credentials*`,
  `secrets*`, and `~/.kalshi/` patterns by default.
- **The investor login** uses open access in this build (any non-empty
  username + password). To put non-public information behind real auth,
  replace `js/auth.js` with proper server-side auth at the reverse-proxy
  or platform level.

---

## License & disclaimer

This repository is distributed for research transparency. It does not
constitute an offer to sell or a solicitation of an offer to buy any
securities. All investments involve risk, including loss of principal.
Past performance and backtest results are not indicative of future
results. Available to qualified purchasers only.

© 2026 Ariadne Labs · Stanford, California
