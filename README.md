# stockstw

台股自動選股 + 回測系統。每天早上 8:00（台灣時間）由 GitHub Actions 跑增量更新 → 大盤過濾 → 五個策略訊號 → 推播到 Telegram 與 / 或 Discord。所有資料落地於本地 SQLite (`data/cache.db`)，避免重複打 API。

策略可信度落差很大 — 只有「策略四（中長線品質股低接）」通過樣本外驗證並推為主力。完整脈絡與實際績效見 [`選股邏輯說明.txt`](./選股邏輯說明.txt)，動策略邏輯前先讀。

## Common commands

```bash
# 一鍵建立本地虛擬環境（建立 .venv、安裝依賴、複製 .env.example）
./setup_env.sh
source .venv/bin/activate

# 主入口（main.py 是所有模式的 dispatcher）
python main.py screen           # 每日選股（GitHub Actions 用）
python main.py backtest         # 跑全策略回測（train + test 兩階段）
python main.py download         # 全量下載價格 + 法人籌碼（首次 bootstrap，~1-2 小時）
python main.py download-revenue # 月營收獨立下載（bootstrap phase 2）
python main.py optimize <idx>   # 對 STRATEGIES[idx] 跑 grid search

# 直接呼叫 backtest module（更細控制）
python -m backtest.run_backtest --mode strategy --max-stocks 50
python -m backtest.run_backtest --mode optimize --strategy 0
python -m backtest.benchmark    # 策略四 vs 0050 對照組顯著性測試
```

需要的 secrets（`.env` 或 GitHub Actions secrets）：

- `FINMIND_TOKEN` — 必填，免費註冊：<https://finmindtrade.com/>
- 推播至少設一組（兩組都設則同步推到兩邊）：
  - **Telegram**：`TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID`
  - **Discord**：`DISCORD_WEBHOOK_URL`

測試指令：`pytest`（共 43 個單元測試，~0.2 秒跑完）。策略邏輯驗證另靠 `python main.py backtest` 跑歷史回測比對指標。

## 推播設定

`notify/__init__.py` 是 dispatcher，根據環境變數自動判斷要推到哪些平台。沒設任何一組則 log 一行警告、不會 crash。

### Discord（最簡單，只要一個 URL）

1. 在 Discord 開一個 server（或用既有的）；左側欄底部 **「+」** → 建立伺服器 → 給自己用
2. 進入想接收訊號的頻道 → **齒輪 ⚙️** → **整合 (Integrations)** → **Webhooks** → **新增 Webhook**
3. 點 **複製 Webhook URL**，會是類似這樣：
   ```
   https://discord.com/api/webhooks/<id>/<token>
   ```
4. 貼到 `.env`：
   ```bash
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>
   ```
5. 部署到 GitHub Actions：repo → **Settings → Secrets and variables → Actions** → **New repository secret**，名稱 `DISCORD_WEBHOOK_URL`，值貼上 URL。`daily_screen.yml` 已經 wired 好。

⚠️ Webhook URL 等於密碼，**不要 commit、不要貼到公開地方**。外洩了就回頻道刪掉重建一個，舊的立刻失效。

訊息會自動把 Telegram 的 `<b>` / `<i>` 轉成 Markdown，超過 2000 字會依行切成多則（Discord 單則上限）。

### Telegram

1. 在 Telegram 找 [@BotFather](https://t.me/BotFather) → `/newbot` → 取得 `TELEGRAM_TOKEN`
2. 把 bot 加進你要接收的 group / channel，或直接和 bot 開對話
3. 找出 `chat_id`：傳一句話給 bot 後開 `https://api.telegram.org/bot<TOKEN>/getUpdates`，回應 JSON 裡 `chat.id` 就是
4. `.env`：
   ```bash
   TELEGRAM_TOKEN=<bot token>
   TELEGRAM_CHAT_ID=<chat id>
   ```

## Architecture

### Layered data flow

```
data/fetcher.py    ── HTTP 層，整合 FinMind + TWSE/TPEx 官方 API
   ↓ (DataFrame)
data/cache.py      ── SQLite 落地，所有讀寫經此（init_db, save_*, load_*, last_*_date）
   ↓
data/universe.py   ── 股票池（FinMind TaiwanStockInfo，過濾 4 碼純數字）
   ↓
technical/indicators.py ── 純 pandas 指標（MA / KD / MACD / 布林 / RSI / 量能）
technical/signals.py    ── 五策略訊號函數 + STRATEGIES 清單（回測迴圈用）
fundamental/quality_filter.py ── 基本面評分（EPS/ROE/毛利/OCF）
   ↓
backtest/engine.py      ── 進場 = T+1 開盤；出場 = TP / SL / max_hold；扣手續費 + 證交稅
backtest/run_backtest.py── 策略下載、批次回測、build_market_filter（0050 多頭過濾）
backtest/optimizer.py   ── grid_search + pick_best
backtest/benchmark.py   ── 策略 vs 0050 配對檢定（t-test、alpha）
   ↓
screener/daily_run.py   ── 增量更新 → screen_today → 落 CSV
notify/telegram_bot.py  ── HTML 訊息格式化 + 勝率監控提示（format_signals 共用）
notify/discord_bot.py   ── Webhook 推播（HTML→Markdown、2000 字分段）
notify/__init__.py      ── Dispatcher：依環境變數路由到 Telegram / Discord / 兩者
```

### Critical invariants — don't break these

**No look-ahead bias.** 訊號於 T 日收盤後算出，進場一律 `next_["open"] * (1 + SLIPPAGE)`（次日開盤 + 滑價）。月營收 (`signal_revenue_momentum`) 用次月 11 日作為「資料可用日」（保守估，避開 10 日當天傍晚才公布的公司）。任何指標都不能用未來資料 shift。

**Market filter 優先。** 所有訊號函數都會接 `market_filter: pd.Series | None`，最後一步 `_apply_market_filter` 會 AND 上去。`build_market_filter` 用 0050 收 > MA60 OR (MA20 上揚 + 收 > MA20) 為多頭。空頭日所有訊號被關閉。

**FinMind rate limit = 600 req/hr。** 每股 2-3 次 API call，全量 ~3000 股需在迴圈內 `time.sleep(6)`。`_get` 對 402/403 直接回 `None`（付費或下市，不重試），對 429 sleep 60s 重試。下載迴圈是斷點續跑設計：`last_price_date` / `last_revenue_date` 跳過已抓資料；`download_all` / `download_revenue` 都先 `sorted(stock_id)` 確保中斷後順序一致。

**訊號函數簽名統一。** `(df, inst_df=None, market_filter=None, **extra) -> df with signal_<name>`。回測 / 選股迴圈用 `STRATEGIES` 清單驅動；新增策略時必須補 `signal_col`、`default_tp/sl/hold`，需要月營收則加 `needs_revenue: True`。

**SQLite schema 寫法。** `save_*` 都用自定義 `_insert_or_ignore` method（pandas to_sql 不原生支援 INSERT OR IGNORE），同步維護 `fetch_log` 表記錄每股每資料集的 `last_date`，這是斷點續跑的依據。新增資料表記得擴 `init_db()` DDL 並更新 fetch_log 寫入點。

**手續費口徑要一致。** `engine.Trade.pnl_pct` 自動扣 `FEE_RATE_BUY + FEE_RATE_SELL + tax`（興櫃稅率不同，看 `market` 欄位）。`benchmark.py` 用 `COST_PER_TRADE = 0.0063` 作為 0050 對照組的扣費，動到任一邊都要兩邊同步改。

### GitHub Actions

- `.github/workflows/daily_screen.yml`：每週一至五 UTC 00:00 跑 `python main.py screen`，DB 用 actions/cache 還原避免重抓。
- `.github/workflows/bootstrap.yml`：手動觸發的全量下載；先跑 price+institutional，再跑 revenue（兩段獨立可恢復）；存檔前 `pkill -f "python main.py"; sleep 5` 防 SQLite 寫到一半被 tar 讀到。
