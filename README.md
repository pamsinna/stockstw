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

## 策略總覽

三個 live strategies，性格不同、互相補位：

| | 策略 | 性格 | 出場 | OOS Sharpe | OOS 年化 | OOS MaxDD |
|---|---|---|---|---|---|---|
| **S4** | 中長線_品質股低接 | 巴菲特型：低 PE 品質股守正 | trail +20%/-15% | **12.92** | +38.97% | **-6.85%** |
| **S5** | 月營收動能 | 林區型：每月情報窗業績轉折 | TP+40% / SL-12% | **14.22** | +50.60% | -8.37% |
| **S6** | 高成長突破 | O'Neil 型：AI 牛市追飆股 | trail +80%/-15% | 12.81 | +52.56% | -10.38% |

OOS = Out-of-sample 驗證期 2023-01-01 ~ 2025-12-31。
回測扣手續費 0.1425% × 2、證交稅 0.3%（興櫃 0.15%）、滑價 0.1%。

---

## 策略邏輯

### 策略四（S4）中長線_品質股低接

**目標股**：基本面紮實、估值合理、法人正在累積、技術面正在整理結束的「便宜好公司」。

**池子過濾**（cross-stock）：
- EPS TTM > 1
- 毛利率 > 15%
- 經營現金流 (OCF) > 0
- ROE ≥ 12%（從 IncomeAfterTaxes / Equity 自算）
- EPS YoY 連續 2 季成長
- 散戶持股 ≤ 50%（TDCC 千張大戶週報，排除套牢盤厚的票）

**進場條件**（per-row）：
- 收盤 > MA60
- MACD 金叉 3 日視窗內
- BB% 30-120%
- RSI < 70
- 外資 + 投信 60 日累計 > 1M 張
- PER < 20 或 NaN
- 大盤 strict 多頭（4 條全到位：close>MA60 & close>MA20 & MA60升 & MA20升）

**出場**：
- 漲到 +20% 啟動 trailing stop（從峰值跌 15% 出場）
- 未達 +20% 時固定停損 -10% / 最長持有 90 天

**OOS（2023-2025）**：

| 交易 | 勝率 | EV/筆 | 年化 | Sharpe | MaxDD |
|------|------|------|------|--------|-------|
| 114 | 64.0% | +16.68% | +38.97% | 12.92 | -6.85% |

出場細項：trailing stop 68 筆 / 停損 39 筆 — 移動停利成主導出場代表「贏家被讓跑」。

---

### 策略五（S5）月營收動能

**目標股**：每月營收公布後第一個交易日的「業績轉折」訊號。

**池子過濾**：同 S4 fundamental（EPS/毛利/OCF/ROE/EPS YoY 成長）

**進場條件**（只在公布日後第一個交易日觸發，每月 1 次）：
- 月營收 YoY > 15%
- YoY 加速 > 5pp（vs 前 3 個月平均）
- 過去 12 個月 ≥ 6 個月正成長
- 2 年 CAGR > 5%（消除基期效應）
- 外資 20 日累計 > 20 日均量的 0.5%
- 站上 MA60 或「近 5 日跌幅 > -8% 且當日紅 K」
- PER < 20 或 NaN
- 大盤 loose 多頭

**出場**：固定停利 +40% / 停損 -12% / 最長 120 天

**OOS（2023-2025）**：

| 交易 | 勝率 | EV/筆 | 年化 | Sharpe | MaxDD |
|------|------|------|------|--------|-------|
| 94 | 58.5% | +11.70% | +50.60% | 14.22 | -8.37% |

---

### 策略六（S6）高成長突破 — regime-conditional

**目標股**：S4 因 PER<20 擋下的高 PE 飆股（鴻勁、奇鋐、世芯-KY 這類）。

**池子過濾**：同 S4 fundamental（但**取消 PER 條件**）

**進場條件**：
- 月營收「3 個月 sum vs 3 個月前 sum」成長 > 10%（用 3M-vs-3M 取代 YoY，IPO 新股 < 12 個月歷史也適用）
- 收盤 = 60 日新高（突破而非金叉，主升段股票很少出現金叉）
- 當日量 > 20 日均量 × 1.5x
- 收盤 > MA60
- 外資 + 投信 60 日累計 > **5M 張**（5 倍 S4 門檻，更高選擇性）
- 大盤 loose 多頭

**出場**：
- 漲到 **+80% 才啟動 trailing**（trail 15%）— 讓贏家跑得夠久
- 未達 +80% 時固定停損 -10% / 最長 90 天

**OOS（2023-2025）**：

| 交易 | 勝率 | EV/筆 | 年化 | Sharpe | MaxDD |
|------|------|------|------|--------|-------|
| 114 | **37.7%** | +15.65% | +52.56% | 12.81 | -10.38% |

⚠️ **regime-conditional 警告**：

| 期間 | 年化 | Sharpe | MaxDD |
|------|------|--------|-------|
| In-Sample 2019-2022（非 AI bull）| +10.7% | 2.6 | -16.9% |
| OOS 2023-2025（AI bull）| **+52.6%** | **12.8** | -10.4% |

S6 OOS 漂亮的數字主要來自 2023-2025 AI 主升段的 regime。**IS 期間 Sharpe 只有 2.6**，遠不如 S4/S5。視為「順勢加強」而非全週期 core。若 AI 行情結束，手動關閉 S6 通知。

**勝率僅 37.7%**：靠少數大贏家拉抬，**連虧 5-6 筆是正常**。建議部位 size 為 S4 的 1/2 ~ 2/3。

---

## 共用 invariant（重要設計約束）

- **No look-ahead bias**：訊號於 T 日收盤後計算，進場一律 T+1 開盤 + 滑價 0.1%。月營收訊號用 T+11 作為「資料可用日」（保守估，避開 10 日當天傍晚才公布的公司）。
- **Strict market filter**（S4 用）：0050 收 > MA60 AND 收 > MA20 AND MA60 升 AND MA20 升，**四條全滿足才視為多頭**。回測顯示「四條全到」比「兩條到」Sharpe 多 0.6、MaxDD 少 2pp。
- **Trailing stop 取代固定 TP**：用 trail 取代 +30% 強制停利，避免砍掉飆股尾巴。S4 用 trigger 20%/trail 15%（保守），S6 用 trigger 80%/trail 15%（激進）。
- **資料新鮮度**：daily_run 以 0050 最後資料日作為「本日交易日」基準，任何股票價格資料落後則跳過，避免用舊資料產生訊號。
- **0050 保護**：mark_fetch_skip() 拒絕標記 0050 為永久跳過，避免一次 403 就讓整個市場過濾失效。

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
