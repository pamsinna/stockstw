"""把 START_DATE ~ END_DATE 之間漏抓的 S4/S5/S6/S7 訊號補出來。

用法：
    python scripts/backfill_signals.py                 # 預設 2026-06-01 ~ 今日
    python scripts/backfill_signals.py 2026-06-01 2026-06-09

注意：使用「現在的 DB 狀態」+「修好的 signal code」算回去，
       不是真的 time-machine（要做完整 time-machine 需要每日 snapshot）。
       這對 S5 yoy bug 補抓 OK，因為 bug 是 code 不是資料。
"""
from __future__ import annotations
import sys
import warnings
import logging
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import DATA_START
from data.cache import (
    init_db, load_prices, load_institutional, load_monthly_revenue, load_per,
    load_shareholding_latest,
)
from data.universe import build_universe
from backtest.run_backtest import build_market_filter
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (
    signal_longterm_quality_entry,
    signal_revenue_momentum,
    signal_growth_breakout,
    signal_accumulation_eve,
    STRATEGIES,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_S4 = next(s for s in STRATEGIES if s["name"] == "中長線_品質股低接")
_S6 = next(s for s in STRATEGIES if s["name"] == "高成長突破")
_S7 = next(s for s in STRATEGIES if s["name"] == "累積前夕")


def backfill(start: str, end: str) -> None:
    init_db()
    print(f"\n=== Signal backfill {start} ~ {end} ===\n")

    uni = build_universe()
    name_map = dict(zip(uni["stock_id"], uni["stock_name"]))
    market_map = dict(zip(uni["stock_id"], uni["market"]))
    print(f"Universe: {len(uni)} stocks")

    # 大盤過濾（loose 給 S5/S6/S7、strict 給 S4）
    mf = build_market_filter(start=DATA_START, end=end)
    strict_mf = build_market_filter(start=DATA_START, end=end, strict=True)

    # 基本面過關
    print("Fundamental filter...")
    fund_df = batch_fundamentals(uni["stock_id"].tolist())
    fund_ok = set(fund_df[fund_df["passes_filter"]]["stock_id"])
    print(f"  passed: {len(fund_ok)}")

    # S4 散戶比例
    retail_ok_s4 = None
    s4_retail_max = _S4.get("retail_max_pct")
    if s4_retail_max is not None:
        sh = load_shareholding_latest()
        if not sh.empty:
            retail_ok_s4 = set(sh[sh["retail_pct"] <= s4_retail_max]["stock_id"])

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # 蒐集：(date, strategy, sid, close, key metrics)
    hits = []

    sids = uni["stock_id"].tolist()
    for sid in tqdm(sids, desc="Backfill"):
        price = load_prices(sid, start="2020-01-01")
        if len(price) < 60:
            continue
        if price["date"].max() < start_ts:
            continue
        inst = load_institutional(sid, start="2020-01-01")
        inst_arg = inst if not inst.empty else None
        rev = load_monthly_revenue(sid)
        rev_arg = rev if not rev.empty else None
        per = load_per(sid, start="2020-01-01")
        per_arg = per if not per.empty else None
        market = market_map.get(sid, "TWSE")

        in_fund = sid in fund_ok
        s4_pass_retail = retail_ok_s4 is None or sid in retail_ok_s4

        try:
            # S4
            if in_fund and s4_pass_retail:
                df = signal_longterm_quality_entry(price, inst_arg, per_df=per_arg,
                    market_filter=strict_mf, inst_threshold=_S4.get("inst_threshold", 0))
                sig = df[df["signal_long"] & (df["date"] >= start_ts) & (df["date"] <= end_ts)]
                for _, r in sig.iterrows():
                    hits.append({"date": r["date"].date(), "strategy": "S4", "sid": sid,
                                 "name": name_map.get(sid, "?"), "market": market,
                                 "close": r["close"]})

            # S5
            df = signal_revenue_momentum(price, inst_arg, rev_arg, per_df=per_arg, market_filter=mf)
            sig = df[df["signal_rev"] & (df["date"] >= start_ts) & (df["date"] <= end_ts)]
            for _, r in sig.iterrows():
                hits.append({"date": r["date"].date(), "strategy": "S5", "sid": sid,
                             "name": name_map.get(sid, "?"), "market": market,
                             "close": r["close"],
                             "yoy": r.get("revenue_yoy")})

            # S6
            if in_fund:
                df = signal_growth_breakout(price, inst_arg, rev_arg,
                    market_filter=mf, inst_threshold=_S6.get("inst_threshold", 0),
                    rev_growth_min=_S6.get("rev_growth_min", 10.0))
                sig = df[df["signal_growth"] & (df["date"] >= start_ts) & (df["date"] <= end_ts)]
                for _, r in sig.iterrows():
                    hits.append({"date": r["date"].date(), "strategy": "S6", "sid": sid,
                                 "name": name_map.get(sid, "?"), "market": market,
                                 "close": r["close"]})

            # S7
            if in_fund:
                df = signal_accumulation_eve(price, inst_arg,
                    market_filter=mf, inst_threshold=_S7.get("inst_threshold", 3_000_000),
                    aqs_min=_S7.get("aqs_min", 70.0))
                sig = df[df["signal_accum"] & (df["date"] >= start_ts) & (df["date"] <= end_ts)]
                for _, r in sig.iterrows():
                    hits.append({"date": r["date"].date(), "strategy": "S7", "sid": sid,
                                 "name": name_map.get(sid, "?"), "market": market,
                                 "close": r["close"]})
        except Exception as e:
            logger.debug(f"{sid}: {type(e).__name__}: {e}")

    if not hits:
        print("\n沒有任何訊號")
        return

    df = pd.DataFrame(hits).sort_values(["date", "strategy", "sid"]).reset_index(drop=True)

    # 輸出
    print(f"\n=== 總計 {len(df)} 筆訊號 ===\n")
    summary = df.groupby(["date", "strategy"]).size().unstack(fill_value=0)
    print(summary)
    print()

    # 詳細按日列出
    for d in sorted(df["date"].unique()):
        day = df[df["date"] == d]
        print(f"\n─── {d}（{len(day)} 訊號）───")
        for strat in ["S4", "S5", "S6", "S7"]:
            rows = day[day["strategy"] == strat]
            if rows.empty: continue
            names = [f"{r['sid']} {r['name']}({r['close']:.0f})" for _, r in rows.iterrows()]
            print(f"  {strat} [{len(rows)}]: {', '.join(names)}")

    # 存 CSV
    out_path = f"reports/backfill_signals_{start}_{end}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n→ 已輸出 {out_path}")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-06-01"
    end = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")
    backfill(start, end)
