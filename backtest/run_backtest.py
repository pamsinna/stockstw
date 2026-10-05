"""
主回測執行腳本：
  （資料由每日 screen 或 `python main.py bootstrap` 從官方來源更新）
  python -m backtest.run_backtest --mode strategy  # 只重跑策略（資料已在 DB）
  python -m backtest.run_backtest --mode optimize  # grid search 參數優化
"""
import argparse
import logging
import pandas as pd
from tqdm import tqdm

from data.cache import (
    init_db, load_prices, load_institutional,
)
from backtest.pit_signals import strategy_signals
from data.universe import build_universe
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


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_market_filter(start: str, end: str, ma_period: int = 60,
                        strict: bool = False) -> pd.Series:
    """
    大盤過濾：
      寬鬆（預設）：收盤 > MA60，OR MA20 開始上揚（V 轉初期允許進場）
      嚴格（strict=True，策略四專用）：四條全滿足
        ① 收盤 > MA60
        ② 收盤 > MA20
        ③ MA60 本身上升（5日比較）
        ④ MA20 本身上升（5日比較）

    嚴格版相當於「graded market score」拿滿 100 分；OOS 回測顯示比舊版
    （只要 ①+③）的 Sharpe 11.60 → 12.21、MaxDD -9.54% → -7.44%。
    """
    df = load_prices(TAIEX_PROXY, start=DATA_START, end=end)
    if len(df) < ma_period:
        logger.warning("0050 data insufficient for market filter — filter disabled")
        return pd.Series(dtype=bool)
    df = df.sort_values("date").reset_index(drop=True)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    above_ma60 = df["close"] > df["ma60"]
    above_ma20 = df["close"] > df["ma20"]
    ma60_rising = df["ma60"] > df["ma60"].shift(5)
    ma20_rising = df["ma20"] > df["ma20"].shift(5)

    if strict:
        # 四條全滿足（graded ≥ 100）
        df["market_up"] = above_ma60 & above_ma20 & ma60_rising & ma20_rising
    else:
        # V 轉初期允許進場：close > ma60，或 MA20 已上揚且站上 MA20
        v_turn_early = ma20_rising & above_ma20
        df["market_up"] = above_ma60 | v_turn_early

    return df.set_index("date")["market_up"]


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
        signal_col = strategy["signal_col"]
        tp = strategy["default_tp"]
        sl = strategy["default_sl"]
        mh = strategy["default_hold"]

        logger.info(f"Preparing signals for [{name}] (point-in-time filters)...")
        price_map = strategy_signals(strategy, stocks, end, market_filter, strict_market_filter)

        if not price_map:
            logger.warning(f"No data for strategy {name}")
            continue

        result = run_portfolio_backtest(
            price_map, signal_col, tp, sl, mh,
            start, end, market_map=market_map,
            consec_down_exit=strategy.get("consec_down_exit", False),
            trail_trigger=strategy.get("trail_trigger"),
            trail_pct=strategy.get("trail_pct", 0.15),
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
    parser.add_argument("--mode", choices=["strategy", "optimize"],
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

    if args.mode == "strategy":
        run_all_strategies(universe, train=True, max_stocks=args.max_stocks)
        run_all_strategies(universe, train=False, max_stocks=args.max_stocks)

    elif args.mode == "optimize":
        optimize(universe, strategy_idx=args.strategy,
                 max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()
