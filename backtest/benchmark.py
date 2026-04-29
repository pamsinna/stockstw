"""
0050 Buy-and-Hold 對照組驗證
目的：檢驗策略四的報酬是 alpha（真選股能力）還是 beta（跟大盤漲而已）

用法：
  python -m backtest.benchmark

需要先跑過 python main.py backtest，讓 reports/中長線_品質股低接_test.csv 存在。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats
from data.cache import load_prices

logger = logging.getLogger(__name__)

STRATEGY_NAME = "中長線_品質股低接"
ETF_ID        = "0050"
COST_PER_TRADE = 0.0063  # 買 + 賣 + 稅 + 滑價（與回測引擎口徑一致）


def load_trades(phase: str = "test") -> pd.DataFrame:
    path = f"reports/{STRATEGY_NAME}_{phase}.csv"
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    if df.empty:
        raise ValueError(f"No trades found in {path}")
    return df


def load_etf_prices() -> pd.DataFrame:
    df = load_prices(ETF_ID, start="2020-01-01")
    if df.empty:
        raise RuntimeError(f"0050 price data not in DB — run download first")
    return df.sort_values("date").reset_index(drop=True)


def run_comparison(trades_df: pd.DataFrame,
                   etf_df: pd.DataFrame) -> dict:
    """
    配對比較：每筆策略交易 vs 同期持有 0050。
    進場用開盤價，出場用收盤價（與策略回測口徑接近）。
    """
    etf = etf_df.set_index("date")
    etf.index = pd.to_datetime(etf.index)

    strategy_returns = []
    benchmark_returns = []
    skipped = 0

    for _, trade in trades_df.iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        exit_ = pd.Timestamp(trade["exit_date"])

        # 找最近的有效交易日（假日可能沒資料）
        entry_row = _nearest_row(etf, entry, "open", direction="forward")
        exit_row  = _nearest_row(etf, exit_,  "close", direction="backward")

        if entry_row is None or exit_row is None or entry_row >= exit_row:
            skipped += 1
            continue

        gross = (exit_row - entry_row) / entry_row
        net   = gross - COST_PER_TRADE

        benchmark_returns.append(net)
        strategy_returns.append(float(trade["pnl_pct"]))

    if skipped:
        logger.warning(f"Skipped {skipped} trades (0050 data missing for that date)")

    strategy_arr  = np.array(strategy_returns)
    benchmark_arr = np.array(benchmark_returns)
    alpha_arr     = strategy_arr - benchmark_arr

    t_stat, p_value = stats.ttest_rel(strategy_arr, benchmark_arr)

    return {
        "n_trades":       len(strategy_arr),
        "strategy_mean":  strategy_arr.mean(),
        "benchmark_mean": benchmark_arr.mean(),
        "alpha_mean":     alpha_arr.mean(),
        "alpha_std":      alpha_arr.std(),
        "alpha_win_rate": (alpha_arr > 0).mean(),
        "strategy_win_rate": (strategy_arr > 0).mean(),
        "benchmark_win_rate": (benchmark_arr > 0).mean(),
        "t_statistic":    t_stat,
        "p_value":        p_value,
        "is_significant": p_value < 0.05,
        "strategy_returns":  strategy_arr,
        "benchmark_returns": benchmark_arr,
    }


def _nearest_row(etf: pd.DataFrame, date: pd.Timestamp,
                 col: str, direction: str) -> float | None:
    """找最近有效交易日的價格，最多漂移 5 天"""
    for delta in range(6):
        d = date + pd.Timedelta(days=delta if direction == "forward" else -delta)
        if d in etf.index:
            return float(etf.loc[d, col])
    return None


def print_report(result: dict) -> None:
    s = result["strategy_mean"] * 100
    b = result["benchmark_mean"] * 100
    a = result["alpha_mean"] * 100
    p = result["p_value"]
    n = result["n_trades"]
    aw = result["alpha_win_rate"] * 100
    sw = result["strategy_win_rate"] * 100
    bw = result["benchmark_win_rate"] * 100
    sig = "顯著 ✓" if p < 0.05 else "不顯著 ✗"

    print(f"""
{'='*60}
  策略四 vs 0050 Buy-and-Hold 對照組驗證
  樣本：{n} 筆交易（驗證期 out-of-sample）
{'='*60}
  策略四平均每筆報酬：  {s:+.2f}%
  0050 對照組（同期）： {b:+.2f}%
  超額報酬（alpha）：   {a:+.2f}%  ← 關鍵數字

  勝率比較
    策略四勝率：       {sw:.1f}%
    0050 同期勝率：    {bw:.1f}%
    跑贏 0050 的比例： {aw:.1f}%

  統計顯著性
    t = {result['t_statistic']:.3f}, p = {p:.4f}（{sig}）

{'='*60}""")

    cost = COST_PER_TRADE * 100
    if a > cost and p < 0.05:
        verdict = f"✅ 有真 alpha（超額報酬 {a:+.2f}% > 成本 {cost:.2f}%，且統計顯著）"
        action  = "策略有獨立選股價值，可繼續執行。"
    elif a > cost and p >= 0.05:
        verdict = f"⚠️  略勝對照組但統計不顯著（p={p:.3f}）"
        action  = (f"目前 {n} 筆樣本不足以確認 alpha，"
                   "建議 paper trade 半年累積更多樣本再判斷。")
    elif abs(a) <= cost:
        verdict = f"❌ 與 0050 差異在成本範圍內（alpha {a:+.2f}%，成本 ±{cost:.2f}%）"
        action  = ("策略沒有超過成本的 alpha，回報主要來自 beta（跟著大盤漲）。\n"
                   "  建議：定期定額 0050 更省力，或重新設計選股因子。")
    else:
        verdict = f"❌ 策略表現劣於 0050（alpha {a:+.2f}%）"
        action  = "直接買 0050 比這個策略更好。"

    print(f"  結論：{verdict}\n  行動：{action}\n")

    # 分段分析（按進場年份）
    _print_yearly_breakdown(result)


def _print_yearly_breakdown(result: dict) -> None:
    pass  # 分段資訊需要 trades_df，由 main() 傳入


def yearly_breakdown(trades_df: pd.DataFrame,
                     etf_df: pd.DataFrame) -> None:
    """按進場年份拆解，看哪幾年是主要貢獻者"""
    etf = etf_df.set_index("date")
    etf.index = pd.to_datetime(etf.index)

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year

    print("  按年份拆解：")
    print(f"  {'年份':<6} {'筆數':>4} {'策略均報酬':>10} {'0050均報酬':>10} {'alpha':>8}")
    print(f"  {'-'*50}")

    for year, grp in trades_df.groupby("year"):
        sub_result = run_comparison(grp, etf_df)
        s = sub_result["strategy_mean"] * 100
        b = sub_result["benchmark_mean"] * 100
        a = sub_result["alpha_mean"] * 100
        flag = "✅" if a > 0 else "❌"
        print(f"  {year:<6} {len(grp):>4}     {s:>+7.2f}%    {b:>+7.2f}%  {a:>+6.2f}% {flag}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    from data.cache import init_db
    init_db()

    try:
        trades_df = load_trades("test")
    except FileNotFoundError:
        print("找不到 reports/中長線_品質股低接_test.csv")
        print("請先執行：python main.py backtest")
        return

    etf_df = load_etf_prices()

    print(f"\n載入 {len(trades_df)} 筆驗證期交易，{len(etf_df)} 天 0050 資料")

    result = run_comparison(trades_df, etf_df)
    print_report(result)
    yearly_breakdown(trades_df, etf_df)


if __name__ == "__main__":
    main()
