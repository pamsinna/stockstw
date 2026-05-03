"""
績效指標計算：從 BacktestResult 產生完整報告
"""
import numpy as np
import pandas as pd
from backtest.engine import BacktestResult


def _daily_portfolio_returns(df: pd.DataFrame) -> pd.Series:
    """
    把並行交易攤到每個交易日，建立真實的等權組合日報酬序列。
    每筆交易的 pnl_pct 均攤到持倉期間每個交易日，
    再對當天所有活躍部位取平均（等權）。
    """
    if df.empty:
        return pd.Series(dtype=float)

    dates = pd.bdate_range(df["entry_date"].min(), df["exit_date"].max())
    n = len(dates)
    date_pos = {d: i for i, d in enumerate(dates)}

    total = np.zeros(n)
    count = np.zeros(n)

    for row in df.itertuples(index=False):
        td = pd.bdate_range(row.entry_date, row.exit_date)
        if len(td) == 0:
            continue
        daily = row.pnl_pct / len(td)
        for d in td:
            if d in date_pos:
                idx = date_pos[d]
                total[idx] += daily
                count[idx] += 1

    active = count > 0
    ret = np.zeros(n)
    ret[active] = total[active] / count[active]
    return pd.Series(ret, index=dates)


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

    # 每日組合報酬（等權並行）→ 正確的 equity curve
    daily = _daily_portfolio_returns(df)
    equity = (1 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    # 年化報酬
    years = (daily.index[-1] - daily.index[0]).days / 365
    cagr = equity.iloc[-1] ** (1 / max(years, 0.1)) - 1 if years > 0 else np.nan

    # Sharpe（只計算有持倉的日子，無風險利率 1.5%）
    rf_daily = 0.015 / annual_trading_days
    active_daily = daily[daily != 0]
    sharpe = ((active_daily.mean() - rf_daily) / (active_daily.std() + 1e-9)
              * np.sqrt(annual_trading_days)) if len(active_daily) > 1 else np.nan

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
