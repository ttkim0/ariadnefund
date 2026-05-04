# Ariadne Labs — site

Public-facing website + private investor terminal for **Ariadne Labs**,
a quantitative AI hedge fund for prediction markets, run by Terry Kim,
Andrew Liang, and Krit Phaisamran (Stanford CS, Math minor).

## Pages

| Path | What it is | Theme |
|---|---|---|
| `/` (`index.html`) | Landing — thesis, fund stats, team | Light, minimalist (Intempus / Aschenbrenner inspired) |
| `/research.html` | Working papers list, PDF download | Light |
| `/login.html` | Investor + cofounder login form | Light |
| `/terminal.html` | Bloomberg-style fund terminal — AUM, P&L, equity curve, open positions, recent trades, horizon skill | **Dark, monospace, Bloomberg-style** |
| `/research/ariadne-paper-01-event-markets.pdf` | Paper 01 (33 pages) | — |

## Run locally

```bash
cd "/Users/terrykim/Documents/SF Weather/site"
./serve.sh
```

The script:

1. Regenerates `data/fund_state.json` from the current backtest + live signal data
   (so the terminal always shows fresh numbers).
2. Starts a local HTTP server on `http://127.0.0.1:8765`.

Open `http://127.0.0.1:8765/` in your browser.

## Default investor credentials

Hardcoded in `js/auth.js`. **Change before any external deployment.**

| Username | Password |
|---|---|
| `tkim` | `ariadne2026` |
| `aliang` | `ariadne2026` |
| `krit` | `ariadne2026` |

To change a password:

```bash
echo -n "your_new_password" | shasum -a 256
# copy the hex into CREDENTIALS in js/auth.js
```

Then **remove the `bootstrapCreds()` lazy-replace function** in `js/auth.js`
(it currently overrides the file's hashes with the seeded `ariadne2026` hash
on first load — that's intentional for the seeded state, but defeats your
custom hashes).

## Security note

`js/auth.js` is **client-side gating only**. Anyone who reads the JS
source can see the hashes and guess. This is acceptable for a
localhost-only or VPN-restricted cofounder tool but is **not real
authentication**. Before deploying to a public URL:

- Put HTTP basic auth at the reverse proxy (nginx/caddy) for `terminal.html`,
  or
- Replace this with server-side auth (Flask/FastAPI session, JWT, etc.), or
- Move the terminal behind a Tailscale / Cloudflare Access tunnel.

## Refreshing fund data

The terminal reads `data/fund_state.json`. Regenerate it any time:

```bash
python3 build_fund_state.py
```

This pulls from:

- `reports/backtest_metrics.json` (test-set MAE, skill scores)
- `reports/live_signals.json` (open actionable trades)
- `reports/forecast.json` (current hourly forecast)
- `data/trade_log_B_realistic_short.parquet` (equity curve, recent trades)

The launchd job documented in `AUTOMATION.md` already keeps these source files
fresh every hour. To also refresh the terminal data automatically,
add a final step to `code/refresh_chain.sh`:

```bash
run_step "fund state for site" "${PY}" site/build_fund_state.py
```

## File map

```
site/
├── README.md                  ← this file
├── serve.sh                   ← run script (regenerates data + starts server)
├── build_fund_state.py        ← generates data/fund_state.json
│
├── index.html                 ← landing
├── research.html              ← papers list
├── login.html                 ← login form
├── terminal.html              ← investor dashboard
│
├── css/
│   ├── site.css               ← light theme (landing, research, login)
│   └── terminal.css           ← dark Bloomberg-terminal theme
├── js/
│   └── auth.js                ← client-side login gating
│
├── data/
│   └── fund_state.json        ← live fund state, read by terminal.html
│
└── research/
    └── ariadne-paper-01-event-markets.pdf
```
