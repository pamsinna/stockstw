# Taiwan Stock Auto-Screener

End-to-end automated screening system for ~1,100 Taiwan-listed equities (TWSE, TPEx, Emerging).

Runs daily via GitHub Actions: fetches incremental market data, applies fundamental and technical filters across all stocks, and delivers trade signals to Telegram or Discord.

> **Disclaimer:** This is a personal engineering project. Nothing in this repository constitutes investment advice.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data Sources                                           │
│  FinMind API (price, institutional, revenue, PER)       │
│  TWSE / TPEx official API (stock list)                  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP  (rate-limited: 600 req/hr)
                     ▼
┌─────────────────────────────────────────────────────────┐
│  data/fetcher.py                                        │
│  Retry logic · 402/403 permanent-skip · 429 backoff     │
└────────────────────┬────────────────────────────────────┘
                     │ DataFrame
                     ▼
┌─────────────────────────────────────────────────────────┐
│  data/cache.py  ──  SQLite (data/cache.db)              │
│  Incremental upsert · fetch_log resume cursor           │
│  Tables: daily_price · institutional · monthly_revenue  │
│          financial · per · universe · fetch_log         │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐   ┌────────────────────────────────┐
│  fundamental/    │   │  technical/                    │
│  quality_filter  │   │  indicators.py  signals.py     │
│  EPS · ROE       │   │  MA · KD · MACD · BB · RSI     │
│  Gross margin    │   │  Institutional 60-day flow     │
│  OCF · Revenue   │   │  Market filter (0050 proxy)    │
└──────────┬───────┘   └────────────────┬───────────────┘
           │                            │
           └──────────────┬─────────────┘
                          │ Signal DataFrame
                          ▼
┌─────────────────────────────────────────────────────────┐
│  screener/daily_run.py                                  │
│  Incremental update → fundamental gate → signal scan    │
│  Staleness guard: skip stocks lagging behind 0050 date  │
│  Output: reports/signals_{strategy}_{date}.csv          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  notify/                                                │
│  Telegram (HTML formatting · win-rate monitor)          │
│  Discord  (Markdown · 2000-char chunking)               │
│  Dispatcher routes based on env vars                    │
└─────────────────────────────────────────────────────────┘
```

---

## Strategies

Two live strategies pass out-of-sample validation:

**Strategy 4 — Long-Term Quality Entry**
Fundamental gate (EPS, ROE, gross margin, OCF) + institutional conviction filter (foreign + trust 60-day net buy ≥ 5M shares) + price above MA60 + MACD golden cross within 3 days. Strict market filter: signals suppressed during TAIEX bear phases. TP +30% / SL −10% / max 90 days.

**Strategy 5 — Monthly Revenue Momentum**
Triggered on the first trading day after monthly revenue disclosure (Taiwan statutory deadline: 10th of each month). Requires ≥ 3 consecutive months of YoY revenue growth, positive foreign 20-day flow, RSI < 70. TP +40% / SL −12% / max 120 days.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data wrangling | pandas, numpy |
| Persistence | SQLite (via `sqlite3` + pandas `to_sql`) |
| Market data | FinMind API, TWSE / TPEx official endpoints |
| Scheduling | GitHub Actions (daily cron) |
| Notifications | Telegram Bot API, Discord Webhooks |
| Testing | pytest (43 unit tests) |

---

## Setup

```bash
# Create virtualenv, install dependencies, copy .env template
bash setup_env.sh
source .venv/bin/activate
```

Create a `.env` file (see `.env.example`):

```bash
FINMIND_TOKEN=<your token>        # required — free at finmindtrade.com
TELEGRAM_TOKEN=<bot token>        # optional
TELEGRAM_CHAT_ID=<chat id>        # optional
DISCORD_WEBHOOK_URL=<webhook url> # optional
```

---

## Usage

```bash
# One-time data bootstrap (~1–2 hours due to API rate limits)
python main.py download           # price + institutional data
python main.py download-revenue   # monthly revenue (separate step)

# Daily screening (also runs automatically via GitHub Actions)
python main.py screen

# Backtesting
python main.py backtest           # all strategies, train + test periods
python main.py optimize <index>   # grid search on strategy N

# Benchmark
python -m backtest.benchmark      # strategy vs 0050 ETF (t-test, alpha)

# Tests
pytest
```

---

## GitHub Actions

| Workflow | Trigger | Job |
|---|---|---|
| `daily_screen.yml` | Mon–Fri UTC 21:00 | Incremental update → screen → notify |
| `bootstrap.yml` | Manual (`workflow_dispatch`) | Full historical download (resumable) |
| `ci.yml` | Push / PR | ruff lint + pytest |

The DB is preserved across daily runs via `actions/cache`, keyed by workflow run number with a prefix fallback. On cache miss, the latest bootstrapped DB is downloaded from GitHub Releases.

---

## Key Engineering Details

- **No look-ahead bias.** Signals computed at close T; entries execute at T+1 open with slippage. Monthly revenue signals use T+11 as earliest availability date.
- **Resumable downloads.** `fetch_log` table tracks `last_date` per stock per dataset. All download loops sort by stock ID so interruptions resume from the same position.
- **Rate limiting.** `_finmind()` sleeps 6 s unconditionally after every API call. 402/403 responses write a permanent-skip sentinel; 429 triggers a 60 s backoff.
- **Staleness guard.** Daily screen skips any stock whose latest price date lags behind 0050 (the TAIEX proxy), preventing stale data from generating signals.
- **Cost accounting.** Backtest engine deducts buy fee (0.1425%), sell fee (0.1425%), securities transaction tax (0.3% TWSE/TPEx; 0.15% Emerging), and 0.1% slippage on entry.
