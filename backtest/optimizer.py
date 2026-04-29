"""
參數優化：Grid Search 找最佳停利/停損/持有天數組合
在 train 期跑，用 test 期做 out-of-sample 驗證，避免 overfit
"""
from __future__ import annotations
import itertools
import logging
import pandas as pd
from tqdm import tqdm
from backtest.engine import run_portfolio_backtest
from backtest.metrics import calc_metrics

logger = logging.getLogger(__name__)


def grid_search(
    stock_price_map: dict[str, pd.DataFrame],
    signal_col: str,
    train_start: str,
    train_end: str,
    take_profit_range: list[float],
    stop_loss_range: list[float],
    max_hold_range: list[int],
    min_trades: int = 30,
    market_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    回傳所有參數組合的績效 DataFrame，依期望值排序
    只保留交易次數 >= min_trades 的結果（樣本數太少不可信）
    """
    combos = list(itertools.product(
        take_profit_range, stop_loss_range, max_hold_range
    ))
    rows = []

    for tp, sl, mh in tqdm(combos, desc=f"Grid search {signal_col}"):
        res = run_portfolio_backtest(
            stock_price_map, signal_col,
            tp, sl, mh,
            train_start, train_end,
            market_map=market_map,
        )
        m = calc_metrics(res)
        if "error" in m or m["n_trades"] < min_trades:
            continue
        rows.append({
            "take_profit": tp,
            "stop_loss":   sl,
            "max_hold":    mh,
            **m,
        })

    if not rows:
        logger.warning(f"grid_search: no valid results for {signal_col}")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("expectancy_pct", ascending=False)
    return df.reset_index(drop=True)


def pick_best(grid_df: pd.DataFrame,
              min_win_rate: float = 50.0,
              min_sharpe: float = 0.5) -> dict | None:
    """
    從 grid 結果挑選最佳參數：
    - 勝率 >= min_win_rate
    - Sharpe >= min_sharpe
    - 期望值最高
    """
    if grid_df.empty:
        return None
    filtered = grid_df[
        (grid_df["win_rate"] >= min_win_rate) &
        (grid_df["sharpe"] >= min_sharpe)
    ]
    if filtered.empty:
        logger.warning("No params pass quality filter, using best by expectancy")
        filtered = grid_df
    best = filtered.iloc[0].to_dict()
    return best
