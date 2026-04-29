"""
主回測執行腳本：
  python -m backtest.run_backtest --mode full      # 全量下載 + 跑所有策略
  python -m backtest.run_backtest --mode strategy  # 只重跑策略（資料已在 DB）
  python -m backtest.run_backtest --mode optimize  # grid search 參數優化
"""
import argparse
import logging
import time
import pandas as pd
from tqdm import tqdm

from data.cache import (
    init_db, load_universe, load_prices, load_institutional,
    save_prices, save_institutional, last_price_date,
)
from data.universe import build_universe
from data.fetcher import fetch_price, fetch_institutional
from technical.signals import STRATEGIES
from technical.indicators import add_all, merge_institutional
from backtest.engine import run_portfolio_backtest
from backtest.metrics import calc_metrics, print_report
from backtest.optimizer import grid_search, pick_best
from config import (
    BACKTEST_TRAIN_START, BACKTEST_TRAIN_END,
    BACKTEST_TEST_START, BACKTEST_TEST_END,
)

TAIEX_PROXY = "0050"  # ETF tracking TAIEX; used as 大盤過濾

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 資料下載 ─────────────────────────────────────────────────────────────────

def _ensure_taiex_proxy(start: str = "2018-01-01") -> None:
    """確保 0050 資料在 DB（大盤過濾用）"""
    last = last_price_date(TAIEX_PROXY)
    fetch_start = last or start
    price = fetch_price(TAIEX_PROXY, fetch_start)
    if not price.empty:
        save_prices(TAIEX_PROXY, price)
        logger.info(f"0050 (TAIEX proxy) updated: {len(price)} rows")


def build_market_filter(start: str, end: str, ma_period: int = 60) -> pd.Series:
    """
    大盤過濾（寬鬆版）：
      多頭 = 收盤 > MA60
      OR  收盤雖跌破 MA60，但 MA20 開始上揚（V 轉初期也允許進場）
    避免純 MA60 在 V 轉時錯過最佳買點。
    """
    df = load_prices(TAIEX_PROXY, start="2018-01-01", end=end)
    if len(df) < ma_period:
        logger.warning("0050 data insufficient for market filter — filter disabled")
        return pd.Series(dtype=bool)
    df = df.sort_values("date").reset_index(drop=True)
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma20_rising"] = df["ma20"] > df["ma20"].shift(5)  # MA20 比 5 日前高

    above_ma60   = df["close"] > df["ma60"]
    v_turn_early = df["ma20_rising"] & (df["close"] > df["ma20"])

    df["market_up"] = above_ma60 | v_turn_early
    return df.set_index("date")["market_up"]


def download_all(universe: pd.DataFrame,
                 start: str = "2020-01-01",
                 max_stocks: int | None = None) -> None:
    # 先確保大盤代理資料存在
    _ensure_taiex_proxy(start)
    time.sleep(6)

    stocks = universe["stock_id"].tolist()
    if max_stocks:
        stocks = stocks[:max_stocks]

    logger.info(f"Downloading price + institutional for {len(stocks)} stocks...")
    for i, sid in enumerate(tqdm(stocks, desc="Download")):
        last = last_price_date(sid)
        fetch_start = last or start

        price = fetch_price(sid, fetch_start)
        if not price.empty:
            save_prices(sid, price)
            time.sleep(6)  # 只有成功取得資料才 sleep（403 直接跳過，無需等待）

        inst = fetch_institutional(sid, fetch_start)
        if not inst.empty:
            save_institutional(sid, inst)
            time.sleep(6)


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

    # 大盤過濾：0050 收盤 > MA60 才允許買進
    market_filter = build_market_filter(start, end)
    if market_filter.empty:
        logger.warning("Market filter unavailable — running without it")
    else:
        bull_days = int(market_filter.loc[market_filter.index >= pd.Timestamp(start)].sum())
        total_days = int((market_filter.index >= pd.Timestamp(start)).sum())
        logger.info(f"Market filter ready: {bull_days}/{total_days} bull days in period")

    for strategy in STRATEGIES:
        name       = strategy["name"]
        signal_fn  = strategy["signal_fn"]
        signal_col = strategy["signal_col"]
        tp = strategy["default_tp"]
        sl = strategy["default_sl"]
        mh = strategy["default_hold"]

        price_map: dict[str, pd.DataFrame] = {}
        logger.info(f"Preparing signals for [{name}]...")

        for sid in tqdm(stocks, desc=name, leave=False):
            price = load_prices(sid, start="2018-01-01", end=end)
            if len(price) < 60:  # 資料太少跳過
                continue
            inst = load_institutional(sid, start="2018-01-01")
            try:
                df = signal_fn(
                    price,
                    inst_df=inst if not inst.empty else None,
                    market_filter=market_filter if not market_filter.empty else None,
                )
                price_map[sid] = df
            except Exception as e:
                logger.debug(f"{sid} signal error: {e}")

        if not price_map:
            logger.warning(f"No data for strategy {name}")
            continue

        result = run_portfolio_backtest(
            price_map, signal_col, tp, sl, mh,
            start, end, market_map=market_map
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
        price = load_prices(sid, start="2018-01-01")
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
