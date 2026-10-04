"""進場候選的最終名單：通知顯示什麼，出場監控就追蹤什麼（同一份）。

2026-10 起因：10/01 S5 一次放出 73 檔，舊通知只顯示前 10，出場監控卻 73 檔全記
→ 隔天一堆「從沒出現在進場名單」的股票發出場警報。

規則（依序）：
  1. AQS < AQS_MIN_CANDIDATE 剔除（回測：AQS<50 每筆 +7.5% vs ≥70 +21.9%）
  2. 單策略單日上限（S5 營收公布日整批觸發 → 依動能取前 N）
  3. 跨策略合併後依動能取前 MAX_CANDIDATES 檔
"""
from __future__ import annotations

import pandas as pd

from notify.telegram_bot import _mom20

STRAT_KEYS = ("long", "revenue", "growth", "accum", "combo_47")
AQS_MIN_CANDIDATE = 50
MAX_PER_STRATEGY_DAY = {"revenue": 10}
MAX_CANDIDATES = 15


def _with_mom(df: pd.DataFrame, date: str, cache: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["_mom20"] = [cache.setdefault(str(s), _mom20(str(s), date)) for s in df["stock_id"]]
    return df.sort_values("_mom20", ascending=False, na_position="last")


def finalize_candidates(signals: dict[str, pd.DataFrame], date: str) -> dict[str, pd.DataFrame]:
    out = dict(signals)
    mom: dict[str, float] = {}
    for k in STRAT_KEYS:
        df = out.get(k)
        if df is None or df.empty:
            continue
        if "aqs_score" in df.columns:
            df = df[~(df["aqs_score"] < AQS_MIN_CANDIDATE)]  # NaN（算不出）保留
        df = _with_mom(df, date, mom)
        cap = MAX_PER_STRATEGY_DAY.get(k)
        if cap:
            df = df.head(cap)
        out[k] = df.reset_index(drop=True)

    ids = pd.Series({s: m for k in STRAT_KEYS
                     if out.get(k) is not None and not out[k].empty
                     for s, m in zip(out[k]["stock_id"].astype(str), out[k]["_mom20"])})
    if len(ids) > MAX_CANDIDATES:
        keep = set(ids.sort_values(ascending=False, na_position="last").index[:MAX_CANDIDATES])
        for k in STRAT_KEYS:
            df = out.get(k)
            if df is not None and not df.empty:
                out[k] = df[df["stock_id"].astype(str).isin(keep)].reset_index(drop=True)
    return out
