"""回測用的策略訊號產生器：所有「選股濾網」都用訊號當天已知的資料（point-in-time）。

舊回測的前視偏差（2026-10 修正）：
- 基本面濾網用「最新財報」判斷整段歷史 → 改 fundamental_pass_timeline（法定公告期限後才生效）
- S4 散戶比例濾網用「最新一週」集保快照 → 改用當週快照；2026-05 以前沒有集保資料 → 不過濾
"""
from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from config import DATA_START
from data.cache import (load_prices, load_institutional, load_monthly_revenue, load_per,
                        load_shareholding)
from fundamental.quality_filter import fundamental_pass_timeline, pit_mask

logger = logging.getLogger(__name__)

# 多個策略共用同一檔的基本面時間軸（回測流程內財報不會變）
_fund_timeline = lru_cache(maxsize=4096)(fundamental_pass_timeline)


def retail_pit_mask(stock_id: str, dates: pd.Series, retail_max: float) -> pd.Series:
    """當週集保散戶比例 ≤ retail_max；該股第一筆集保資料之前一律放行（沒有資料可判斷）。"""
    sh = load_shareholding(stock_id)
    if sh.empty:
        return pd.Series(True, index=dates.index)
    tl = pd.Series((sh["retail_pct"] <= retail_max).to_numpy(), index=pd.to_datetime(sh["date"]))
    ok = pit_mask(dates, tl)
    before = pd.to_datetime(dates) < tl.index.min()
    return ok | before


def strategy_signals(strategy: dict, stocks: list[str], end: str,
                     market_filter: pd.Series | None, strict_market_filter: pd.Series | None,
                     gate_fundamental: bool | None = None) -> dict[str, pd.DataFrame]:
    """對每檔股票算出 strategy 的訊號（已套 point-in-time 基本面／散戶濾網）。

    gate_fundamental=None → 依 strategy["needs_fundamental"]。
    """
    col = strategy["signal_col"]
    gate = strategy.get("needs_fundamental", False) if gate_fundamental is None else gate_fundamental
    mf = strict_market_filter if strategy.get("strict_market") else market_filter
    retail_max = strategy.get("retail_max_pct")
    extra_keys = ("inst_threshold", "rev_growth_min", "aqs_min")
    out: dict[str, pd.DataFrame] = {}
    for sid in stocks:
        price = load_prices(sid, start=DATA_START, end=end)
        if len(price) < 60:
            continue
        inst = load_institutional(sid, start=DATA_START)
        kw: dict = {k: strategy[k] for k in extra_keys if k in strategy}
        if strategy.get("needs_revenue"):
            rev = load_monthly_revenue(sid)
            kw["rev_df"] = rev if not rev.empty else None
        if strategy.get("needs_per"):
            per = load_per(sid, start=DATA_START, end=end)
            kw["per_df"] = per if not per.empty else None
        try:
            df = strategy["signal_fn"](price, inst_df=inst if not inst.empty else None,
                                       market_filter=mf if mf is not None and not mf.empty else None, **kw)
        except Exception as e:
            logger.debug(f"{sid} signal error: {e}")
            continue
        if not df[col].any():
            out[sid] = df
            continue
        if gate:
            df[col] = df[col] & pit_mask(df["date"], _fund_timeline(sid))
        if retail_max is not None:
            df[col] = df[col] & retail_pit_mask(sid, df["date"], retail_max)
        out[sid] = df
    return out
