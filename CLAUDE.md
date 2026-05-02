# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

---

## Commands

```bash
# First-time setup
bash setup_env.sh           # creates .venv, installs requirements, copies .env.example

# Activate environment (required each session)
source .venv/bin/activate

# Data bootstrap (one-time, ~1-2 hours due to API rate limits)
python main.py download          # prices + institutional data for all stocks
python main.py download-revenue  # monthly revenue (separate due to rate limiting)

# Daily workflow
python main.py screen            # incremental update + screen + Telegram notify

# Backtesting
python main.py backtest          # run all strategies on cached data
python main.py optimize [N]      # grid-search strategy N (0-indexed)

# Advanced backtest control
python -m backtest.run_backtest --mode full      # download + all strategies
python -m backtest.run_backtest --mode strategy  # strategies only (data already in DB)
python -m backtest.run_backtest --mode optimize --strategy 0
```

## Environment

Secrets go in `.env` (loaded via `python-dotenv` at startup):
- `FINMIND_TOKEN` — FinMind API token (required for all data fetching)
- `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` — Telegram bot notifications

All strategy thresholds, fee rates, and backtest date ranges are in `config.py`. The SQLite database lives at `data/cache.db`.

## Architecture

This is a Taiwan stock auto-screening system with five strategies and a full backtest pipeline.

### Data flow

```
FinMind API / TWSE / TPEX
    ↓ data/fetcher.py          — raw HTTP fetch + rate_limit_sleep (6s between calls)
    ↓ data/cache.py            — SQLite persistence (prices, institutional, financial, revenue)
    ↓ data/universe.py         — build stock universe (TWSE + TPEX + emerging)
```

### Six modules

| Module | Purpose |
|--------|---------|
| `data/` | Fetch + cache all market data in SQLite |
| `technical/` | Compute indicators (`indicators.py`) and emit per-stock signals (`signals.py`) |
| `fundamental/` | Score and filter stocks on EPS, ROE, margins, OCF, revenue growth |
| `backtest/` | Event-driven backtest engine, metrics (Sharpe, drawdown), grid optimizer, ETF benchmark |
| `screener/` | Daily orchestration: incremental update → fundamental filter → signal generation |
| `notify/` | Format and send Telegram messages; append to signal log |

### The five strategies (in `technical/signals.py`)

1. **Short** (`signal_short_vol_breakout`) — 1–5 days: volume surge ≥2.5× MA20 + 20-day high breakout + bullish candle + institutional sync-buy
2. **Swing MA/KD** (`signal_swing_ma_kd_inst`) — 1–4 weeks: MA alignment + KD golden cross (K ≤65) + consecutive institutional buying
3. **Swing Dual** (`signal_swing_dual_inst`) — 1–4 weeks: dual-institutional (foreign + trust) buying confirmation
4. **Long** (`signal_longterm_quality_entry`) — up to 90 days: passes fundamental filter + long-term MA alignment + revenue growth
5. **Revenue Momentum** (`signal_revenue_momentum`) — triggered after monthly revenue release (after 10th of each month)

Every signal is AND-ed with a **market filter**: 0050 ETF as TAIEX proxy, bullish when close > MA60 OR MA20 is turning up (V-recovery allowed). Passing `market_filter=None` disables the filter.

### Key data patterns

- `data/cache.py` functions follow the pattern `save_X(stock_id, df)` / `load_X(stock_id, start=, end=)` / `last_X_date(stock_id)`
- Incremental updates only fetch from `last_price_date(sid)` forward — never re-download full history
- `screener/daily_run.py::run_daily` is the GitHub Actions entry point; it calls `incremental_update` then `screen_today`
- Signal outputs are saved as CSV to `reports/signals_{timeframe}_{date}.csv`
- `backtest/engine.py` uses next-day open prices with slippage for realistic fill simulation

### Backtest structure

`backtest/run_backtest.py::run_all_strategies` iterates the `STRATEGIES` list from `technical/signals.py`, runs `run_portfolio_backtest` for each, and prints metrics via `backtest/metrics.py`. `backtest/optimizer.py::grid_search` wraps this for parameter sweeps.

- **Train period**: 2019-01-01 – 2022-12-31
- **Test period (out-of-sample)**: 2023-01-01 – 2025-12-31
- Entry: T+1 open; exit: first of TP / SL / max-hold (also T+1 open)
- Costs baked in: buy 0.1425% + sell 0.1425% + tax 0.3% (0.15% Emerging) + slippage 0.1%

### Strategy status (out-of-sample results)

| # | Signal function | OOS result | Notes |
|---|----------------|-----------|-------|
| 1 | `signal_short_vol_breakout` | −0.01% | Reference only |
| 2 | `signal_swing_ma_kd_inst` | −0.49% | Reference only |
| 3 | `signal_swing_dual_inst` | −0.31% | Reference only |
| 4 | `signal_longterm_quality_entry` | +1.23% ✅ | Primary; no alpha vs 0050 but higher Sharpe |
| 5 | `signal_revenue_momentum` | — | In development |

Strategy 4's edge comes from a few large winners (+30% TP); median trade is −3.47%. The market filter is critical — performance collapses outside bull periods.
