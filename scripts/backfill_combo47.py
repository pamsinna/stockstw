"""列出指定區間內的 S4 ∩ S7 高信心進場訊號。

每日針對 universe 中通過 fund + retail 的股票，計算 S4 跟 S7 全序列。
對任一日 D，若當日 S4 或 S7 觸發，且另一邊在過去 COMBO_WINDOW 交易日
內也曾觸發 → 該日該股算「高信心進場」。

用法:
    python scripts/backfill_combo47.py                  # 預設最近 10 天
    python scripts/backfill_combo47.py 2026-05-30 2026-06-08
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from config import DATA_START
from data.cache import (
    init_db, load_prices, load_institutional, load_per,
    load_shareholding_latest,
)
from data.universe import build_universe
from backtest.run_backtest import build_market_filter
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (
    signal_longterm_quality_entry, signal_accumulation_eve, STRATEGIES,
)

COMBO_WINDOW = 20  # 回測甜蜜點（60d 勝率 66.3% > 60d window 的 60.4%）

_S4 = next(s for s in STRATEGIES if s["name"] == "中長線_品質股低接")
_S7 = next(s for s in STRATEGIES if s["name"] == "累積前夕")


def backfill(start: str, end: str):
    init_db()
    uni = build_universe()
    name_map = dict(zip(uni["stock_id"], uni["stock_name"]))
    market_map = dict(zip(uni["stock_id"], uni["market"]))

    fund_df = batch_fundamentals(uni["stock_id"].tolist())
    fund_ok = set(fund_df[fund_df["passes_filter"]]["stock_id"])
    print(f"Universe {len(uni)} | fund pass {len(fund_ok)}")

    retail_ok = None
    s4_retail_max = _S4.get("retail_max_pct")
    if s4_retail_max is not None:
        sh = load_shareholding_latest()
        if not sh.empty:
            retail_ok = set(sh[sh["retail_pct"] <= s4_retail_max]["stock_id"])
            print(f"S4 retail pass: {len(retail_ok)}")

    mf = build_market_filter(start=DATA_START, end=end)
    strict_mf = build_market_filter(start=DATA_START, end=end, strict=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    sids = [s for s in uni["stock_id"].tolist() if s in fund_ok]
    if retail_ok is not None:
        sids = [s for s in sids if s in retail_ok]
    print(f"S4 candidates: {len(sids)}")

    hits = []
    for sid in tqdm(sids, desc="Scan"):
        price = load_prices(sid, start="2020-01-01")
        if len(price) < 200:
            continue
        price = price.sort_values("date").reset_index(drop=True)
        inst = load_institutional(sid, start="2020-01-01")
        inst_arg = inst if not inst.empty else None
        per = load_per(sid, start="2020-01-01")
        per_arg = per if not per.empty else None

        try:
            df4 = signal_longterm_quality_entry(price, inst_arg, per_df=per_arg,
                market_filter=strict_mf, inst_threshold=_S4.get("inst_threshold", 0))
            df7 = signal_accumulation_eve(price, inst_arg, market_filter=mf,
                inst_threshold=_S7.get("inst_threshold", 3_000_000),
                aqs_min=_S7.get("aqs_min", 70.0))
        except Exception:
            continue

        df4 = df4.sort_values("date").reset_index(drop=True)
        df7 = df7.sort_values("date").reset_index(drop=True)

        s4_flag = df4["signal_long"]
        s7_flag = df7["signal_accum"]

        for i in range(len(df4)):
            d = df4.iloc[i]["date"]
            if d < start_ts or d > end_ts:
                continue
            s4_today = bool(s4_flag.iloc[i])
            s7_today = bool(s7_flag.iloc[i])
            if not (s4_today or s7_today):
                continue
            lo = max(0, i - COMBO_WINDOW)
            recent_s4 = bool(s4_flag.iloc[lo:i+1].any())
            recent_s7 = bool(s7_flag.iloc[lo:i+1].any())
            if recent_s4 and recent_s7:
                hits.append({
                    "date": d.date(),
                    "sid": sid,
                    "name": name_map.get(sid, "?"),
                    "market": market_map.get(sid, "TWSE"),
                    "close": float(df4.iloc[i]["close"]),
                    "s4_today": s4_today,
                    "s7_today": s7_today,
                })

    if not hits:
        print("\n沒有訊號")
        return

    df = pd.DataFrame(hits).sort_values(["date", "sid"]).reset_index(drop=True)
    print(f"\n=== 高信心進場 (S4 ∩ S7) {start} ~ {end} — 共 {len(df)} 筆 ===\n")
    for d in sorted(df["date"].unique()):
        day = df[df["date"] == d]
        print(f"─── {d}（{len(day)}）───")
        for _, r in day.iterrows():
            tag = "S4+S7" if r["s4_today"] and r["s7_today"] else ("S4" if r["s4_today"] else "S7")
            print(f"  {r['sid']} {r['name']:<14}  ${r['close']:>7.1f}  今日觸發={tag}")
        print()

    out = f"reports/combo47_{start}_{end}.csv"
    df.to_csv(out, index=False)
    print(f"→ {out}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    backfill(start, end)
