"""一次性種子化：把最近 N 個交易日的 S4-S7 訊號寫進 reports/open_signals.csv，
讓「訊號出場監控」一上線就能追蹤近期訊號（而不必等 N 天累積）。

用法：python -m scripts.seed_open_signals [N交易日，預設20]
"""
import sys
import logging

import pandas as pd
from tqdm import tqdm

from config import DATA_START
from data.cache import (init_db, load_prices, load_institutional,
                        load_monthly_revenue, load_per, load_shareholding_latest)
from data.universe import build_universe
from backtest.run_backtest import build_market_filter
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (signal_longterm_quality_entry, signal_revenue_momentum,
                               signal_growth_breakout, signal_accumulation_eve)
from screener.daily_run import (_S4_INST_THR, _S4_RETAIL_MAX, _S6_INST_THR,
                                _S6_REV_MIN, _S7_INST_THR, _S7_AQS_MIN, TAIEX_PROXY)
from notify.exit_monitor import _load, _save, _name_map

logging.basicConfig(level=logging.INFO, format="%(message)s")


def seed(n_trading_days: int = 20) -> None:
    init_db()
    uni = build_universe()
    names = _name_map()
    # 用 0050 行事曆取「N 個交易日前」作為起點
    cal = load_prices(TAIEX_PROXY, start="2025-01-01")["date"].sort_values()
    start_ts = cal.iloc[-n_trading_days] if len(cal) >= n_trading_days else cal.iloc[0]
    end_ts = cal.iloc[-1]
    print(f"種子化區間 {start_ts.date()} ~ {end_ts.date()}（最近 {n_trading_days} 交易日）")

    mf = build_market_filter(start=DATA_START, end=str(end_ts.date()))
    strict_mf = build_market_filter(start=DATA_START, end=str(end_ts.date()), strict=True)
    fund = batch_fundamentals(uni["stock_id"].tolist())
    fund_ok = set(fund[fund["passes_filter"]]["stock_id"])
    retail_ok = None
    if _S4_RETAIL_MAX is not None:
        sh = load_shareholding_latest()
        if not sh.empty:
            retail_ok = set(sh[sh["retail_pct"] <= _S4_RETAIL_MAX]["stock_id"])

    hits: dict[tuple, dict] = {}   # (sid, strat) → 最早一筆

    def add(sid, strat, r):
        key = (sid, strat)
        if key not in hits:   # 視窗內第一次出現 = 進場
            hits[key] = {"entry_date": str(r["date"].date()), "stock_id": sid,
                         "name": names.get(sid, ""), "strategy": strat,
                         "entry_price": float(r["close"]), "status": "open",
                         "alert_level": "none", "exit_date": "", "exit_reason": "",
                         "pnl_pct": ""}

    for sid in tqdm(uni["stock_id"].tolist(), desc="Seed"):
        price = load_prices(sid, start="2020-01-01")
        if len(price) < 60 or price["date"].max() < start_ts:
            continue
        inst = load_institutional(sid, start="2020-01-01")
        ia = inst if not inst.empty else None
        rev = load_monthly_revenue(sid)
        ra = rev if not rev.empty else None
        per = load_per(sid, start="2020-01-01")
        pa = per if not per.empty else None

        def win(df, col):
            return df[df[col] & (df["date"] >= start_ts) & (df["date"] <= end_ts)]

        try:
            if sid in fund_ok and (retail_ok is None or sid in retail_ok):
                for _, r in win(signal_longterm_quality_entry(price, ia, per_df=pa, market_filter=strict_mf, inst_threshold=_S4_INST_THR), "signal_long").iterrows():
                    add(sid, "S4", r)
            for _, r in win(signal_revenue_momentum(price, ia, ra, per_df=pa, market_filter=mf), "signal_rev").iterrows():
                add(sid, "S5", r)
            if sid in fund_ok:
                for _, r in win(signal_growth_breakout(price, ia, ra, market_filter=mf, inst_threshold=_S6_INST_THR, rev_growth_min=_S6_REV_MIN), "signal_growth").iterrows():
                    add(sid, "S6", r)
                for _, r in win(signal_accumulation_eve(price, ia, market_filter=mf, inst_threshold=_S7_INST_THR, aqs_min=_S7_AQS_MIN), "signal_accum").iterrows():
                    add(sid, "S7", r)
        except Exception:
            pass

    log = _load()
    open_keys = set(zip(log.loc[log.status == "open", "stock_id"],
                        log.loc[log.status == "open", "strategy"]))
    new = [v for k, v in hits.items() if k not in open_keys]
    if new:
        _save(pd.concat([log, pd.DataFrame(new)], ignore_index=True))
    print(f"\n種子化完成：新增 {len(new)} 筆追蹤訊號（依策略）：")
    if new:
        print(pd.DataFrame(new)["strategy"].value_counts().to_string())


if __name__ == "__main__":
    seed(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
