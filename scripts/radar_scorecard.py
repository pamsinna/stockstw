"""法人佈局雷達的前瞻成績單：名單出現後 20／60 個交易日的表現 vs 同期 0050 含息。

條件凍結（LAYOUT_RULES_VERSION），不拿歷史資料調參；這份成績單就是唯一的驗證。

  python scripts/radar_scorecard.py

計算方式：
- 同一檔在 20 個交易日內重複上榜只算第一次（避免同一個事件被算很多次）
- 進場 = 上榜隔天開盤；報酬 = 第 20／60 個交易日收盤（還沒到期的列為「進行中」不計入）
- 扣買賣手續費＋證交稅；0050 用同一段的含息報酬
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.portfolio import benchmark_0050  # noqa: E402
from config import FEE_RATE_BUY, FEE_RATE_SELL, TAX_TWSE_OTC  # noqa: E402
from data.cache import load_prices, load_radar_log  # noqa: E402
from technical.signals import LAYOUT_RULES_VERSION  # noqa: E402

HORIZONS = (20, 60)
DEDUP_DAYS = 20
COST = FEE_RATE_BUY + FEE_RATE_SELL + TAX_TWSE_OTC


def episodes(log: pd.DataFrame, trade_days: pd.DatetimeIndex) -> pd.DataFrame:
    """同一檔 DEDUP_DAYS 交易日內重複上榜只留第一次。"""
    pos = {d: i for i, d in enumerate(trade_days)}
    keep, last = [], {}
    for _, r in log.sort_values("date").iterrows():
        i = pos.get(r["date"])
        if i is None:
            continue
        j = last.get(r["stock_id"])
        if j is None or i - j > DEDUP_DAYS:
            keep.append(r)
        last[r["stock_id"]] = i
    return pd.DataFrame(keep)


def scorecard() -> pd.DataFrame:
    log = load_radar_log()
    if log.empty:
        return pd.DataFrame()
    log = log[log["rules_version"] == LAYOUT_RULES_VERSION]
    bench = benchmark_0050(log["date"].min().strftime("%Y-%m-%d"),
                           pd.Timestamp.today().strftime("%Y-%m-%d"))
    days = bench.index
    rows = []
    for _, r in episodes(log, days).iterrows():
        px = load_prices(r["stock_id"], start=r["date"].strftime("%Y-%m-%d")).set_index("date")
        after = px.index[px.index > r["date"]]
        if after.empty:
            continue
        entry_day = after[0]
        entry = px.loc[entry_day, "open"]
        row = {"date": r["date"], "stock_id": r["stock_id"]}
        for h in HORIZONS:
            if len(after) > h - 1:
                exit_day = after[h - 1]
                ret = px.loc[exit_day, "close"] / entry - 1 - COST
                b = bench.loc[entry_day:exit_day]
                row[f"{h}d%"] = ret * 100
                row[f"0050_{h}d%"] = (b.iloc[-1] / bench.shift(1).loc[entry_day] - 1) * 100
            else:
                row[f"{h}d%"] = row[f"0050_{h}d%"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    df = scorecard()
    print(f"法人佈局雷達成績單 — {LAYOUT_RULES_VERSION}")
    if df.empty:
        print("尚無資料（名單從規則凍結後開始累積）")
        return
    print(f"上榜事件 {len(df)} 筆（同檔 {DEDUP_DAYS} 交易日內重複只算一次）")
    for h in HORIZONS:
        x = df.dropna(subset=[f"{h}d%"])
        if x.empty:
            print(f"  {h} 日：尚未有到期的事件")
            continue
        ex = x[f"{h}d%"] - x[f"0050_{h}d%"]
        print(f"  {h} 日：已到期 {len(x)} 筆｜平均 {x[f'{h}d%'].mean():+.2f}%  中位數 {x[f'{h}d%'].median():+.2f}%"
              f"｜同期 0050 {x[f'0050_{h}d%'].mean():+.2f}%｜超額平均 {ex.mean():+.2f}%  贏 0050 比例 {(ex > 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
