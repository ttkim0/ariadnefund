# Auto-refresh + Dashboard

The system now refreshes itself every hour and renders a single-page HTML
dashboard you can leave open in your browser.

---

## What's running

A `launchd` agent (`com.terrykim.sfo-weather`) is installed at
`~/Library/LaunchAgents/com.terrykim.sfo-weather.plist`. Every 3,600 seconds
(and once at login), it runs:

```
1. code/14_refresh_noaa.py   — pull last 7 d of KSFO METARs from NWS
                                update sfo_hourly.parquet, regen features
2. code/06_predict.py        — hourly temperature forecast (1/3/6/12/24/48/72 h)
3. code/13_live_signal.py    — fetch open Kalshi markets, rank actionable trades
4. code/15_dashboard.py      — regenerate reports/dashboard.html
```

Total wall time per cycle: ~20 seconds. Logs go to
`logs/refresh.log` (rotated to last 2 MB if it grows past 5 MB).

---

## ONE-TIME SETUP YOU MUST DO

macOS protects `~/Documents/` from background processes by default. The
launchd job is registered but blocked until you grant permission. Without
this step, the dashboard will go stale.

**Open System Settings → Privacy & Security → Full Disk Access → click `+`,
then add `/bin/bash`.**

Step-by-step:

1. Apple menu → System Settings → Privacy & Security → **Full Disk Access**
2. Click the **`+`** button (you may need to authenticate with Touch ID / your password)
3. In the file picker, press **⌘⇧G** (or use the Go menu → Go to Folder)
4. Type **`/bin/bash`** and press Return
5. Click **Open**, then make sure the toggle next to `bash` is **ON**

After you do this, the next scheduled run (or the next manual `launchctl kickstart`)
will succeed.

To test immediately without waiting:

```bash
launchctl kickstart -k gui/$(id -u)/com.terrykim.sfo-weather
sleep 25
tail -10 ~/Documents/SF\ Weather/logs/refresh.log
```

You should see a new "refresh_chain.sh done" line dated within the last 30 seconds.

---

## Viewing the dashboard

```bash
open ~/Documents/SF\ Weather/reports/dashboard.html
```

Leave the tab open — it auto-reloads every 5 minutes via a `<meta refresh>`
tag, so it always shows the latest data the launchd job has produced.

The dashboard shows:

- **Current SFO state**: latest temp, dew, wind from the most recent METAR
- **Hourly forecast (next 72h)**: median + 50% / 80% probability fans
- **Daily HIGH / LOW forecasts (today through +3 days)**: bar chart with 80% intervals
- **Tomorrow's HIGH bucket probabilities**: side-by-side us vs Kalshi market
- **Recent observations (last 72h)**: actual temp + dew over time
- **Actionable Kalshi trades**: ranked by EV after fees and spread crossing

---

## Manual operations

```bash
# Manually trigger a refresh
launchctl kickstart -k gui/$(id -u)/com.terrykim.sfo-weather

# Or run the chain directly without launchd (always works from terminal):
~/Documents/SF\ Weather/code/refresh_chain.sh

# Check current schedule / state
launchctl list | grep sfo-weather

# Disable temporarily
launchctl unload ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist

# Re-enable
launchctl load -w ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist

# Uninstall completely
launchctl unload ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist
rm ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist
rm -rf ~/Library/Application\ Support/sfo-weather
```

---

## Files involved

| Path | Role |
|---|---|
| `~/Library/LaunchAgents/com.terrykim.sfo-weather.plist` | launchd job definition (loaded) |
| `~/Library/Application Support/sfo-weather/refresh_chain_wrapper.sh` | tiny wrapper outside `~/Documents/` (launchd-readable) |
| `code/refresh_chain.sh` | the real chain script (logs each step, exit codes, timings) |
| `logs/refresh.log` | append-only log of every run (auto-truncated past 5 MB) |
| `logs/launchd.out.log` / `logs/launchd.err.log` | raw stdout/stderr from launchd |
| `reports/dashboard.html` | the page you actually look at |

---

## Troubleshooting

### "Operation not permitted" in `logs/launchd.err.log`
You haven't done the Full Disk Access step yet. See top of this file.

### Dashboard hasn't updated
Check the most recent line in `logs/refresh.log` — if it's older than an hour,
the job hasn't fired. Try `launchctl kickstart -k gui/$(id -u)/com.terrykim.sfo-weather`
and check again 30 seconds later.

### Want a different refresh interval
Edit `code/com.terrykim.sfo-weather.plist`, change `<key>StartInterval</key><integer>3600</integer>`
(seconds), then:

```bash
cp ~/Documents/SF\ Weather/code/com.terrykim.sfo-weather.plist \
   ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist
launchctl unload ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist
launchctl load -w ~/Library/LaunchAgents/com.terrykim.sfo-weather.plist
```

### Want a notification when new actionable trades appear
The chain currently emits text + JSON only. Adding a `terminal-notifier` or
`osascript` call at the end of `refresh_chain.sh` is straightforward — let me
know and I'll add it.
