"""
主回測執行腳本：
  python -m backtest.run_backtest --mode full      # 全量下載 + 跑所有策略
  python -m backtest.run_backtest --mode strategy  # 只重跑策略（資料已在 DB）
  python -m backtest.run_backtest --mode optimize  # grid search 參數優化
"""
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

from data.cache import (
    init_db, load_prices, load_institutional,
    save_prices, save_institutional, save_monthly_revenue, last_price_date,
    load_monthly_revenue, last_revenue_date, mark_fetch_skip,
    save_per, last_per_date, load_per,
)
from data.universe import build_universe
from data.fetcher import fetch_price, fetch_institutional, fetch_monthly_revenue, fetch_per
from technical.signals import STRATEGIES
from backtest.engine import run_portfolio_backtest
from backtest.metrics import calc_metrics, print_report
from backtest.optimizer import grid_search, pick_best
from config import (
    BACKTEST_TRAIN_START, BACKTEST_TRAIN_END,
    BACKTEST_TEST_START, BACKTEST_TEST_END,
    DATA_START,
)

TAIEX_PROXY = "0050"  # ETF tracking TAIEX; used as 大盤過濾


def _normalize_and_save_revenue(stock_id: str, raw: pd.DataFrame) -> None:
    """FinMind 月營收欄位正規化後存入 DB。計算 revenue_yoy（若未提供）。"""
    df = raw.copy()
    # FinMind 可能回傳 revenue/revenue_month/monthly_revenue 等不同名稱
    rename = {
        "Revenue": "revenue",
        "revenue_month": "revenue",
        "monthly_revenue": "revenue",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    if "revenue" not in df.columns:
        logger.warning(
            f"{stock_id}: revenue column missing from FinMind response "
            f"(got columns: {list(df.columns)}); skipping save"
        )
        return

    df = df.sort_values("date").reset_index(drop=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    if "revenue_yoy" not in df.columns or df["revenue_yoy"].isna().all():
        df["revenue_yoy"] = df["revenue"].pct_change(12) * 100

    save_monthly_revenue(stock_id, df)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 資料下載 ─────────────────────────────────────────────────────────────────

def _ensure_taiex_proxy(start: str = DATA_START) -> None:
    """確保 0050 資料在 DB（大盤過濾用）"""
    last = last_price_date(TAIEX_PROXY)
    fetch_start = last or start
    price = fetch_price(TAIEX_PROXY, fetch_start)
    if price is None:
        logger.warning("0050 fetch returned 402/403 — using cached data if available")
    elif not price.empty:
        save_prices(TAIEX_PROXY, price)
        logger.info(f"0050 (TAIEX proxy) updated: {len(price)} rows")


def build_market_filter(start: str, end: str, ma_period: int = 60,
                        strict: bool = False) -> pd.Series:
    """
    大盤過濾：
      寬鬆（預設）：收盤 > MA60，OR MA20 開始上揚（V 轉初期允許進場）
      嚴格（strict=True，策略四專用）：收盤 > MA60 AND MA60 本身上升（5日比較）
    """
    df = load_prices(TAIEX_PROXY, start=DATA_START, end=end)
    if len(df) < ma_period:
        logger.warning("0050 data insufficient for market filter — filter disabled")
        return pd.Series(dtype=bool)
    df = df.sort_values("date").reset_index(drop=True)
    df["ma60"] = df["close"].rolling(60).mean()

    above_ma60 = df["close"] > df["ma60"]

    if strict:
        df["ma60_rising"] = df["ma60"] > df["ma60"].shift(5)
        df["market_up"] = above_ma60 & df["ma60_rising"]
    else:
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma20_rising"] = df["ma20"] > df["ma20"].shift(5)
        v_turn_early = df["ma20_rising"] & (df["close"] > df["ma20"])
        df["market_up"] = above_ma60 | v_turn_early

    return df.set_index("date")["market_up"]


def download_all(universe: pd.DataFrame,
                 start: str = "2020-01-01",
                 max_stocks: int | None = None) -> None:
    # 先確保大盤代理資料存在
    _ensure_taiex_proxy(start)

    stocks = sorted(universe["stock_id"].tolist())  # 固定排序，確保斷點續跑順序一致
    if max_stocks:
        stocks = stocks[:max_stocks]

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Downloading price + institutional for {len(stocks)} stocks...")
    skipped = 0
    for sid in tqdm(stocks, desc="Download"):
        last = last_price_date(sid)
        if last and last >= yesterday:
            skipped += 1
            continue
        fetch_start = last or start

        price = fetch_price(sid, fetch_start)  # rate-limited inside _finmind()
        if price is None:
            mark_fetch_skip(sid, "price")    # 402/403: 永久跳過，不再重試
        elif not price.empty:
            save_prices(sid, price)

        inst = fetch_institutional(sid, fetch_start)  # rate-limited inside _finmind()
        if inst is None:
            mark_fetch_skip(sid, "institutional")
        elif not inst.empty:
            save_institutional(sid, inst)

    if skipped:
        logger.info(f"Skipped {skipped} already up-to-date stocks")


def download_revenue(universe: pd.DataFrame,
                     start: str = DATA_START,
                     max_stocks: int | None = None) -> None:
    """月營收獨立下載（bootstrap phase 2，主下載完成後再跑）"""
    stocks = sorted(universe["stock_id"].tolist())  # 固定排序，確保斷點續跑順序一致
    if max_stocks:
        stocks = stocks[:max_stocks]

    stale_before = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
    logger.info(f"Downloading monthly revenue for {len(stocks)} stocks...")
    skipped = 0
    for sid in tqdm(stocks, desc="Revenue"):
        last = last_revenue_date(sid)
        if last and last >= stale_before:
            skipped += 1
            continue
        fetch_start = last or start
        rev = fetch_monthly_revenue(sid, fetch_start)  # rate-limited inside _finmind()
        if rev is not None and not rev.empty:
            _normalize_and_save_revenue(sid, rev)

    if skipped:
        logger.info(f"Skipped {skipped} stocks with recent revenue data")


def download_per(universe: pd.DataFrame,
                 start: str = DATA_START,
                 max_stocks: int | None = None) -> None:
    """每日本益比、股價淨值比、殖利率獨立下載"""
    stocks = sorted(universe["stock_id"].tolist())
    if max_stocks:
        stocks = stocks[:max_stocks]

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Downloading daily PER/PBR for {len(stocks)} stocks...")
    skipped = 0
    for sid in tqdm(stocks, desc="PER"):
        last = last_per_date(sid)
        if last and last >= yesterday:
            skipped += 1
            continue
        fetch_start = last or start
        per = fetch_per(sid, fetch_start)
        if per is None:
            mark_fetch_skip(sid, "per")
        elif not per.empty:
            save_per(sid, per)

    if skipped:
        logger.info(f"Skipped {skipped} already up-to-date stocks")


# ─── 策略回測 ─────────────────────────────────────────────────────────────────

def run_all_strategies(universe: pd.DataFrame,
                       train: bool = True,
                       max_stocks: int | None = None) -> None:
    start = BACKTEST_TRAIN_START if train else BACKTEST_TEST_START
    end   = BACKTEST_TRAIN_END   if train else BACKTEST_TEST_END
    phase = "訓練期" if train else "驗證期（out-of-sample）"

    stocks = universe["stock_id"].tolist()
    if max_stocks:
        stocks = stocks[:max_stocks]

    market_map = dict(zip(universe["stock_id"], universe["market"]))

    # 大盤過濾：寬鬆版（策略一～三、五），嚴格版（策略四專用）
    market_filter = build_market_filter(start, end)
    strict_market_filter = build_market_filter(start, end, strict=True)
    if market_filter.empty:
        logger.warning("Market filter unavailable — running without it")
    else:
        bull_days = int(market_filter.loc[market_filter.index >= pd.Timestamp(start)].sum())
        total_days = int((market_filter.index >= pd.Timestamp(start)).sum())
        logger.info(f"Market filter ready: {bull_days}/{total_days} bull days in period")
        s_days = int(strict_market_filter.loc[strict_market_filter.index >= pd.Timestamp(start)].sum())
        logger.info(f"Strict market filter (S4): {s_days}/{total_days} bull days in period")

    for strategy in STRATEGIES:
        name       = strategy["name"]
        signal_fn  = strategy["signal_fn"]
        signal_col = strategy["signal_col"]
        tp = strategy["default_tp"]
        sl = strategy["default_sl"]
        mh = strategy["default_hold"]

        price_map: dict[str, pd.DataFrame] = {}
        logger.info(f"Preparing signals for [{name}]...")

        needs_rev   = strategy.get("needs_revenue", False)
        needs_per   = strategy.get("needs_per", False)
        use_strict  = strategy.get("strict_market", False)
        active_mf   = strict_market_filter if use_strict else market_filter

        for sid in tqdm(stocks, desc=name, leave=False):
            price = load_prices(sid, start=DATA_START, end=end)
            if len(price) < 60:  # 資料太少跳過
                continue
            inst = load_institutional(sid, start=DATA_START)
            extra: dict = {}
            if needs_rev:
                rev = load_monthly_revenue(sid)
                extra["rev_df"] = rev if not rev.empty else None
            if needs_per:
                per = load_per(sid, start=DATA_START, end=end)
                extra["per_df"] = per if not per.empty else None
            try:
                df = signal_fn(
                    price,
                    inst_df=inst if not inst.empty else None,
                    market_filter=active_mf if not active_mf.empty else None,
                    **extra,
                )
                price_map[sid] = df
            except Exception as e:
                logger.debug(f"{sid} signal error: {e}")

        if not price_map:
            logger.warning(f"No data for strategy {name}")
            continue

        result = run_portfolio_backtest(
            price_map, signal_col, tp, sl, mh,
            start, end, market_map=market_map,
            consec_down_exit=strategy.get("consec_down_exit", False),
        )
        m = calc_metrics(result)
        print_report(name, m, phase=phase)

        # 儲存交易明細
        trades_df = result.to_df()
        if not trades_df.empty:
            phase_tag = "train" if train else "test"
            out = f"reports/{name}_{phase_tag}.csv"
            trades_df.to_csv(out, index=False)
            logger.info(f"  Trades saved → {out}")


# ─── 參數優化 ─────────────────────────────────────────────────────────────────

def optimize(universe: pd.DataFrame, strategy_idx: int = 0,
             max_stocks: int | None = None) -> None:
    strategy = STRATEGIES[strategy_idx]
    name       = strategy["name"]
    signal_fn  = strategy["signal_fn"]
    signal_col = strategy["signal_col"]

    stocks = universe["stock_id"].tolist()
    if max_stocks:
        stocks = stocks[:max_stocks]

    price_map: dict[str, pd.DataFrame] = {}
    market_map = dict(zip(universe["stock_id"], universe["market"]))

    market_filter = build_market_filter(BACKTEST_TRAIN_START, BACKTEST_TRAIN_END)

    logger.info(f"Building signal cache for [{name}]...")
    for sid in tqdm(stocks, desc="Signal prep", leave=False):
        price = load_prices(sid, start=DATA_START)
        if len(price) < 60:
            continue
        inst = load_institutional(sid)
        try:
            df = signal_fn(
                price,
                inst_df=inst if not inst.empty else None,
                market_filter=market_filter if not market_filter.empty else None,
            )
            price_map[sid] = df
        except Exception:
            pass

    if not price_map:
        logger.error("No data available for optimization")
        return

    grid = grid_search(
        price_map, signal_col,
        train_start=BACKTEST_TRAIN_START,
        train_end=BACKTEST_TRAIN_END,
        take_profit_range=[0.06, 0.08, 0.10, 0.12, 0.15],
        stop_loss_range=[0.04, 0.05, 0.06, 0.07, 0.08],
        max_hold_range=[5, 10, 15, 20, 25],
        market_map=market_map,
    )

    if grid.empty:
        logger.error("Grid search returned no results")
        return

    print(f"\n=== [{name}] 訓練期最佳參數 Top 5 ===")
    print(grid[["take_profit","stop_loss","max_hold",
                "win_rate","expectancy_pct","sharpe","max_drawdown_pct"]].head(5).to_string())

    best = pick_best(grid)
    if best:
        logger.info(f"Best params: TP={best['take_profit']}, SL={best['stop_loss']}, Hold={best['max_hold']}")
        grid.to_csv(f"reports/{name}_grid.csv", index=False)


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "strategy", "optimize"],
                        default="strategy")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="限制股票數（測試用）")
    parser.add_argument("--strategy", type=int, default=0,
                        help="optimize 模式選用哪個策略 index")
    args = parser.parse_args()

    init_db()
    universe = build_universe()

    if universe.empty:
        logger.error("Empty universe, check API connection")
        return

    logger.info(f"Universe: {len(universe)} stocks "
                f"({universe['market'].value_counts().to_dict()})")

    if args.mode == "full":
        download_all(universe, max_stocks=args.max_stocks)
        run_all_strategies(universe, train=True, max_stocks=args.max_stocks)
        run_all_strategies(universe, train=False, max_stocks=args.max_stocks)

    elif args.mode == "strategy":
        run_all_strategies(universe, train=True, max_stocks=args.max_stocks)
        run_all_strategies(universe, train=False, max_stocks=args.max_stocks)

    elif args.mode == "optimize":
        optimize(universe, strategy_idx=args.strategy,
                 max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()
