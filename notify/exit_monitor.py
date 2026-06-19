"""訊號出場監控 — 只追蹤「系統發過進場訊號」的股票（S4-S7），不碰個人 portfolio。

設計：論點破壞才通知（籌碼/量價惡化），不做到價停利 / 到天數的硬出場。
  🚨 出場：外資倒貨給散戶（外資賣超 + 散戶接手/法人紅旗）或 AQS 籌碼崩壞（派發）
  ⚠️ 注意：買力轉弱（量價同向↓ / 近5日法人轉賣 / AQS 退化到末段）

狀態存在 reports/open_signals.csv（append-only + status 更新），逐日去重：
  🚨 發一次就標 exited、不再評估；⚠️ 只在 none→warn 首次轉換時發，不每日洗版。
"""
from __future__ import annotations

import logging
import pandas as pd

from data.cache import (load_prices, load_institutional, load_shareholding,
                        load_universe, load_open_signals, save_open_signals)
from analysis.aqs import compute_aqs

logger = logging.getLogger(__name__)

_COLS = ["entry_date", "stock_id", "name", "strategy", "entry_price",
         "status", "alert_level", "exit_date", "exit_reason", "pnl_pct"]

# signals dict key → 策略標籤
_TF2STRAT = {"long": "S4", "revenue": "S5", "growth": "S6",
             "accum": "S7", "combo_47": "S4∩S7"}

# ── 門檻（先用預設，之後可調）─────────────────────────────────────────────────
FOREIGN_SELL_10D = -1_000_000   # 外資 10 日淨賣超 ≤ 此（股）視為大幅（=1,000 張）
SELLDAYS_MIN     = 6           # 近 10 日「外資賣超天數 ≥ 此」才算「持續撤離」（非單日事件）
INST_SELL_5D     = -500_000    # 近 5 日法人淨賣超 ≤ 此（股）才算轉賣（=500 張，濾掉零星）
DIM1_WEAK        = 8.0          # AQS 量價同向 < 此 → 買力轉弱
AQS_TRAP_SCORE   = 50          # score < 此且 dim4<0 → 派發 trap
AGE_OUT_DAYS     = 365         # 追蹤上限（純衛生，靜默不通知）


def _load() -> pd.DataFrame:
    df = load_open_signals()
    if df.empty:
        return pd.DataFrame(columns=_COLS)
    for c in _COLS:
        if c not in df.columns:
            df[c] = ""
    # 轉 object：SQLite 來的欄位可能是 str dtype，後續要寫入 float/str 混合
    return df[_COLS].astype(object)


def _save(df: pd.DataFrame) -> None:
    save_open_signals(df[_COLS])


def _name_map() -> dict[str, str]:
    uni = load_universe()
    if uni.empty:
        return {}
    return dict(zip(uni["stock_id"].astype(str), uni["stock_name"]))


def record_today(signals: dict[str, pd.DataFrame], date: str) -> None:
    """把今日各策略訊號記入追蹤 log（同 (stock_id, strategy) 已 open 則略過）。"""
    log = _load()
    open_keys = set(zip(log.loc[log.status == "open", "stock_id"],
                        log.loc[log.status == "open", "strategy"]))
    names = _name_map()
    new = []
    for tf, strat in _TF2STRAT.items():
        df = signals.get(tf)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sid = str(r["stock_id"])
            if (sid, strat) in open_keys:
                continue
            open_keys.add((sid, strat))
            new.append({"entry_date": date, "stock_id": sid,
                        "name": names.get(sid, ""), "strategy": strat,
                        "entry_price": float(r.get("close", 0) or 0),
                        "status": "open", "alert_level": "none",
                        "exit_date": "", "exit_reason": "", "pnl_pct": ""})
    if new:
        _save(pd.concat([log, pd.DataFrame(new)], ignore_index=True))
        logger.info(f"Exit monitor: recorded {len(new)} new signals to track")


def classify(aqs: dict | None, foreign_10d: float | None, foreign_selldays: float | None,
             inst_5d: float | None, retail_rising: bool) -> tuple[str, list[str]]:
    """純函式：依籌碼/量價狀態判斷 🚨 出場 / ⚠️ 注意 / ✅ 持有。"""
    score = aqs.get("score") if aqs else None
    dim4 = aqs.get("dim4_inst_price_align") if aqs else None
    dim1 = aqs.get("dim1_volprice") if aqs else None
    stage = aqs.get("stage", "") if aqs else ""

    # 🚨 出場：資金「持續」撤離（外資10日大幅賣超 + 賣超天數≥門檻 → 非單日事件）+ 散戶接手/紅旗
    if (foreign_10d is not None and foreign_10d <= FOREIGN_SELL_10D
            and (foreign_selldays is None or foreign_selldays >= SELLDAYS_MIN)
            and (retail_rising or (dim4 is not None and dim4 < 0))):
        why = "散戶比例上升(接手)" if retail_rising else "AQS紅旗(法人賣股價撐)"
        days = f"近10日{int(foreign_selldays)}天賣、" if foreign_selldays is not None else ""
        return "🚨 出場", [f"資金持續撤離（{days}外資賣超 {abs(foreign_10d) // 1000:,.0f} 張）、{why}"]
    # 🚨 出場：AQS 籌碼崩壞 / 派發
    if "派發" in stage or (score is not None and score < AQS_TRAP_SCORE
                          and dim4 is not None and dim4 < 0):
        return "🚨 出場", [f"AQS籌碼崩壞（{stage}，score {score:.0f}）"]

    # ⚠️ 注意：買力轉弱（只看真正的買力訊號；「末段」太常見、不算買力減弱，不納入）
    reasons = []
    if dim1 is not None and dim1 < DIM1_WEAK:
        reasons.append(f"買力轉弱（量價同向 {dim1:.0f}/20）")
    if inst_5d is not None and inst_5d <= INST_SELL_5D:
        reasons.append(f"近5日法人轉賣 {abs(inst_5d) // 1000:,.0f} 張")
    if reasons:
        return "⚠️ 注意", reasons
    return "✅ 持有", []


def _metrics(sid: str) -> dict:
    """抓單檔現況：現價、外資10日、法人5日、散戶趨勢、AQS。"""
    px = load_prices(sid, start="2025-09-01")
    if px.empty:
        return {}
    inst = load_institutional(sid, start="2025-12-01")
    foreign_10d = inst_5d = foreign_selldays = None
    if not inst.empty:
        inst = inst.sort_values("date")
        f10 = inst["foreign_"].fillna(0).tail(10)
        foreign_10d = float(f10.sum())
        foreign_selldays = float((f10 < 0).sum())   # 近10日外資賣超天數（持續性）
        net = inst["foreign_"].fillna(0) + (inst["trust"].fillna(0) if "trust" in inst else 0)
        inst_5d = float(net.tail(5).sum())
    sh = load_shareholding(sid, start="2025-01-01")
    retail_rising = False
    if not sh.empty and len(sh) >= 2 and "retail_pct" in sh:
        sh = sh.sort_values("date")
        retail_rising = float(sh["retail_pct"].iloc[-1]) > float(sh["retail_pct"].iloc[0])
    return {"close": float(px.iloc[-1]["close"]), "foreign_10d": foreign_10d,
            "foreign_selldays": foreign_selldays, "inst_5d": inst_5d,
            "retail_rising": retail_rising, "aqs": compute_aqs(sid)}


def evaluate(date: str) -> pd.DataFrame:
    """評估所有 open 訊號，更新 log，回傳「今天要通知」的列（新 🚨 + 新 ⚠️）。"""
    log = _load()
    if log.empty:
        return pd.DataFrame()
    today = pd.Timestamp(date)
    out = []
    for i, row in log[log.status == "open"].iterrows():
        sid = str(row["stock_id"])
        # 衛生：追蹤過久靜默移除
        if (today - pd.Timestamp(row["entry_date"])).days > AGE_OUT_DAYS:
            log.at[i, "status"] = "aged_out"
            continue
        m = _metrics(sid)
        if not m:
            continue
        entry = float(row["entry_price"]) or m["close"]
        pnl = (m["close"] / entry - 1) * 100 if entry else 0.0
        level, reasons = classify(m["aqs"], m["foreign_10d"], m["foreign_selldays"],
                                  m["inst_5d"], m["retail_rising"])
        rec = {"level": level, "stock_id": sid, "name": row["name"],
               "strategy": row["strategy"], "entry_date": row["entry_date"],
               "entry_price": entry, "close": m["close"], "pnl_pct": round(pnl, 1),
               "reason": "；".join(reasons)}
        if level == "🚨 出場":
            log.at[i, "status"] = "exited"
            log.at[i, "exit_date"] = date
            log.at[i, "exit_reason"] = rec["reason"]
            log.at[i, "pnl_pct"] = round(pnl, 1)
            out.append(rec)
        elif level == "⚠️ 注意":
            if row["alert_level"] != "warn":   # 只在首次轉入 ⚠️ 時通知
                log.at[i, "alert_level"] = "warn"
                out.append(rec)
        else:  # ✅ 持有：籌碼回穩則重置 ⚠️ 狀態，之後再惡化可再提醒
            if row["alert_level"] == "warn":
                log.at[i, "alert_level"] = "none"
    _save(log)
    return pd.DataFrame(out)
