"""
績效指標計算：從 BacktestResult 產生完整報告
"""
import numpy as np
import pandas as pd
from backtest.engine import BacktestResult


def calc_metrics(result: BacktestResult, annual_trading_days: int = 250) -> dict:
    df = result.to_df()
    if df.empty:
        return {"error": "no trades"}

    pnl = df["pnl_pct"].dropna()
    n   = len(pnl)

    win_rate    = (pnl > 0).mean()
    avg_win     = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
    avg_loss    = pnl[pnl < 0].mean() if (pnl < 0).any() else 0
    expectancy  = win_rate * avg_win + (1 - win_rate) * avg_loss
    profit_factor = abs(pnl[pnl > 0].sum() / pnl[pnl < 0].sum()) if (pnl < 0).any() else np.inf

    # 最大連續虧損次數
    streak = _max_loss_streak(pnl)

    # 資金曲線 & 最大回撤
    equity = (1 + pnl).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    # 年化報酬（以實際時間跨度估算）
    if len(df) > 1:
        days_span = (df["exit_date"].max() - df["entry_date"].min()).days
        years = days_span / 365
        cagr = equity.iloc[-1] ** (1 / max(years, 0.1)) - 1 if years > 0 else np.nan
    else:
        cagr = np.nan

    # Sharpe（簡化版，假設無風險利率 1.5%）
    rf_daily = 0.015 / annual_trading_days
    daily_ret = pnl / df["hold_days"].clip(lower=1)
    sharpe = (daily_ret.mean() - rf_daily) / (daily_ret.std() + 1e-9) * np.sqrt(annual_trading_days)

    return {
        "n_trades":       n,
        "win_rate":       round(win_rate * 100, 1),
        "avg_win_pct":    round(avg_win * 100, 2),
        "avg_loss_pct":   round(avg_loss * 100, 2),
        "expectancy_pct": round(expectancy * 100, 2),
        "profit_factor":  round(profit_factor, 2),
        "max_loss_streak":streak,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "cagr_pct":       round(cagr * 100, 2) if not np.isnan(cagr) else "N/A",
        "sharpe":         round(sharpe, 2),
        "exit_reasons":   df["exit_reason"].value_counts().to_dict(),
    }


def _max_loss_streak(pnl: pd.Series) -> int:
    max_s = 0
    cur   = 0
    for v in pnl:
        if v < 0:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def print_report(name: str, metrics: dict, phase: str = "") -> None:
    tag = f"[{phase}] " if phase else ""
    print(f"\n{'='*55}")
    print(f"  {tag}{name}")
    print(f"{'='*55}")
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    print(f"  交易次數：    {metrics['n_trades']}")
    print(f"  勝率：        {metrics['win_rate']}%")
    print(f"  平均獲利：    {metrics['avg_win_pct']}%")
    print(f"  平均虧損：    {metrics['avg_loss_pct']}%")
    print(f"  期望值：      {metrics['expectancy_pct']}%/筆")
    print(f"  獲利因子：    {metrics['profit_factor']}")
    print(f"  最大連敗：    {metrics['max_loss_streak']} 筆")
    print(f"  最大回撤：    {metrics['max_drawdown_pct']}%")
    print(f"  年化報酬：    {metrics['cagr_pct']}%")
    print(f"  Sharpe：      {metrics['sharpe']}")
    reasons = metrics.get("exit_reasons", {})
    print(f"  出場原因：    停利={reasons.get('take_profit',0)}, "
          f"停損={reasons.get('stop_loss',0)}, "
          f"到期={reasons.get('max_hold',0)}")
