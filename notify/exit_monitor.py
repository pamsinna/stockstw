"""訊號出場監控 — 只追蹤「系統發過進場訊號」的股票（S4-S7），不碰個人 portfolio。

設計（2026-10 改版）：出場＝該策略回測用的同一套規則，直接用回測引擎判斷：
  🚨 出場：觸及停損／停利／移動停利，或持有滿最長天數（次日開盤出）
  📈 移動停利啟動：漲幅達 trail_trigger，之後改用「高點回落 trail_pct」出場（提醒一次）
進場價 = 訊號日隔天開盤 + 滑價（同回測）。

舊版用籌碼（AQS 派發／外資撤離／法人轉賣／買力轉弱）觸發出場；回測 851 筆
S4～S7（2023-03～2026-09）疊加在策略原出場上，每一種都讓平均報酬掉 6～15pp
（砍掉少數大贏家）→ 降為出場訊息裡的「籌碼參考」，不再觸發出場。

狀態存在 open_signals 表（append-only + status 更新）；🚨 發一次就標 exited。
"""
from __future__ import annotations

import logging
import pandas as pd

from data.cache import (load_prices, load_institutional, load_shareholding,
                        load_universe, load_open_signals, save_open_signals)
from analysis.aqs import compute_aqs
from technical.signals import RULES_VERSION

logger = logging.getLogger(__name__)

_COLS = ["entry_date", "stock_id", "name", "strategy", "entry_price",
         "status", "alert_level", "exit_date", "exit_reason", "pnl_pct", "rules_version"]

# signals dict key → 策略標籤
_TF2STRAT = {"long": "S4", "revenue": "S5", "growth": "S6",
             "accum": "S7", "combo_47": "S4∩S7"}

# ── 門檻（先用預設，之後可調）─────────────────────────────────────────────────
FOREIGN_SELL_10D = -1_000_000   # 外資 10 日淨賣超 ≤ 此（股）視為大幅（絕對 floor，濾薄量股）
FOREIGN_SELL_10D_RATIO = -0.05  # 外資 10 日淨額 ÷ 10 日量 ≤ -5% 才算「大幅」（正規化，免被大型股嚇到）
SELLDAYS_MIN     = 6           # 近 10 日「外資賣超天數 ≥ 此」才算「持續撤離」（非單日事件）
INST_SELL_5D     = -500_000    # 近 5 日法人淨賣超 ≤ 此（股）才算轉賣（=500 張，濾掉零星）
SELL_RATIO_5D    = -0.10        # 5 日(外資+投信)淨額 ÷ 5 日量 ≤ -10% 才算重（正規化）
DIM1_WEAK        = 8.0          # AQS 量價同向 < 此 → 買力轉弱
AQS_TRAP_SCORE   = 50          # score < 此且 dim4<0 → 派發 trap
AGE_OUT_DAYS     = 365         # 追蹤上限（純衛生，靜默不通知）
RECENT_EXIT_DAYS = 5           # 出場日早於此（日曆天）→ 靜默結案（改版首跑會清出一批舊單）
RETAIL_RISE_MIN  = 0.30         # 散戶比例近月須上升 ≥0.3pp 才算「接手」（濾掉 ±0.1pp 噪音）


def retail_rising_recent(sh: pd.DataFrame | None) -> bool:
    """近月散戶比例是否「明顯」上升（≥RETAIL_RISE_MIN pp）。

    取最近 ~8 週（TDCC 週報）比較最新與起點，要求達門檻才算接手；
    資料 < 2 點或欄位缺則 False。修正舊版「現在 vs 視窗第一筆」對微幅噪音也回 True 的問題。
    """
    if sh is None or sh.empty or "retail_pct" not in sh.columns:
        return False
    s = sh.sort_values("date")["retail_pct"].astype(float).tail(8)
    return bool(len(s) >= 2 and (s.iloc[-1] - s.iloc[0]) >= RETAIL_RISE_MIN)


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
                        "exit_date": "", "exit_reason": "", "pnl_pct": "",
                        "rules_version": RULES_VERSION})
    if new:
        _save(pd.concat([log, pd.DataFrame(new)], ignore_index=True))
        logger.info(f"Exit monitor: recorded {len(new)} new signals to track")


# 舊通知每個策略區塊最多顯示 10 檔（S6 15 檔）；超過的根本沒出現在進場名單上
_DISPLAY_CAP = {"S6": 15}
_DISPLAY_CAP_DEFAULT = 10


def prune_untracked() -> int:
    """把「同日同策略超過通知顯示上限」的 open 訊號標成 not_notified、停止追蹤。

    2026-10-01 S5 一次 73 檔，通知只顯示動能前 10，出場監控卻全記 → 隔天
    對沒看過的股票發出場警報。依進場日當時的 20 日動能保留前 N（同舊通知排序），
    其餘標記。冪等：之後候選名單已有上限（screener.candidates），不會再超過。
    """
    from notify.telegram_bot import _mom20
    log = _load()
    if log.empty:
        return 0
    n = 0
    for (d, strat), g in log.groupby(["entry_date", "strategy"]):
        cap = _DISPLAY_CAP.get(strat, _DISPLAY_CAP_DEFAULT)
        if len(g) <= cap:
            continue
        mom = pd.Series({i: _mom20(str(r.stock_id), str(d)) for i, r in g.iterrows()})
        keep = set(mom.sort_values(ascending=False, na_position="last").index[:cap])
        drop = [i for i in g.index if i not in keep and log.at[i, "status"] == "open"]
        log.loc[drop, "status"] = "not_notified"
        n += len(drop)
    if n:
        _save(log)
        logger.info(f"Exit monitor: {n} signals beyond display cap marked not_notified")
    return n


def classify(aqs: dict | None, foreign_10d: float | None, foreign_selldays: float | None,
             inst_5d: float | None, retail_rising: bool,
             foreign_10d_ratio: float | None = None) -> tuple[str, list[str]]:
    """純函式：依籌碼/量價狀態判斷 🚨 出場 / ⚠️ 注意 / ✅ 持有。

    foreign_10d_ratio = 外資10日淨額 ÷ 10日量；有傳就加量能正規化（免被大型股絕對值嚇到）。
    """
    score = aqs.get("score") if aqs else None
    dim4 = aqs.get("dim4_inst_price_align") if aqs else None
    dim1 = aqs.get("dim1_volprice") if aqs else None
    stage = aqs.get("stage", "") if aqs else ""

    # 🚨 出場：資金「持續」撤離（外資10日大幅賣超 + 賣超天數≥門檻 → 非單日事件）+ 散戶接手/紅旗
    #   大幅 = 絕對額過 floor（濾薄量股）且（若有量資料）佔10日量 ≤ -5%（濾大型股）
    ratio_heavy = foreign_10d_ratio is None or foreign_10d_ratio <= FOREIGN_SELL_10D_RATIO
    if (foreign_10d is not None and foreign_10d <= FOREIGN_SELL_10D and ratio_heavy
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
    foreign_10d = inst_5d = foreign_selldays = foreign_10d_ratio = None
    if not inst.empty:
        inst = inst.sort_values("date")
        f10 = inst["foreign_"].fillna(0).tail(10)
        foreign_10d = float(f10.sum())
        foreign_selldays = float((f10 < 0).sum())   # 近10日外資賣超天數（持續性）
        net = inst["foreign_"].fillna(0) + (inst["trust"].fillna(0) if "trust" in inst else 0)
        inst_5d = float(net.tail(5).sum())
        vol_10d = float(px.sort_values("date")["volume"].tail(10).sum()) if "volume" in px else 0.0
        foreign_10d_ratio = foreign_10d / vol_10d if vol_10d else None
    retail_rising = retail_rising_recent(load_shareholding(sid, start="2025-01-01"))
    return {"close": float(px.iloc[-1]["close"]), "foreign_10d": foreign_10d,
            "foreign_selldays": foreign_selldays, "inst_5d": inst_5d,
            "foreign_10d_ratio": foreign_10d_ratio,
            "retail_rising": retail_rising, "aqs": compute_aqs(sid)}


_STRAT_CFG_KEY = {"S4": "long", "S5": "revenue", "S6": "growth", "S7": "accum",
                  "S4∩S7": "long"}   # 高信心組合沿用主力 S4 的出場規則


def _strategy_cfg(label: str) -> dict:
    from technical.signals import STRATEGIES
    key = _STRAT_CFG_KEY.get(label, "long")
    return next(s for s in STRATEGIES if s["timeframe"] == key)


def rule_status(sid: str, label: str, signal_date: str, px: pd.DataFrame | None = None) -> dict | None:
    """用回測引擎判斷這筆訊號到今天為止的狀態（與回測同一套出場規則）。

    回傳 {state: open|exit|pending, entry_price, close, pnl_pct, reason, exit_date, exit_price,
          trail_active, trail_level}；資料不足回 None。
    引擎只在「有下一根 K」時檢查出場 → 補一根明日佔位 K，讓今天的停損/停利也被檢查到；
    持有到期這類「次日開盤出」的出場落在佔位 K 上 = 明日開盤出場。
    """
    from backtest.engine import run_backtest
    st = _strategy_cfg(label)
    sd = pd.Timestamp(signal_date)
    if px is None:
        px = load_prices(sid, start=(sd - pd.Timedelta(days=40)).strftime("%Y-%m-%d"))
    if px is None or px.empty or not (px["date"] == sd).any():
        return None
    px = px.sort_values("date").reset_index(drop=True)
    last = px.iloc[-1]
    if last["date"] == sd:   # 訊號今天才發，還沒進場
        return {"state": "pending"}
    stub = {**last.to_dict(), "date": last["date"] + pd.Timedelta(days=1),
            "open": last["close"], "high": last["close"], "low": last["close"], "volume": 0.0}
    df = pd.concat([px, pd.DataFrame([stub])], ignore_index=True)
    df["_sig"] = df["date"] == sd
    res = run_backtest(df, "_sig", st["default_tp"], st["default_sl"], st["default_hold"],
                       df["date"].min().strftime("%Y-%m-%d"), stub["date"].strftime("%Y-%m-%d"),
                       sid, trail_trigger=st.get("trail_trigger"), trail_pct=st.get("trail_pct", 0.15))
    if not res.trades:
        return None
    t = res.trades[0]
    held = px[px["date"] >= t.entry_date]
    peak = float(max(t.entry_price, held["high"].max())) if not held.empty else t.entry_price
    trail = st.get("trail_trigger")
    out = {"entry_price": t.entry_price, "entry_date": t.entry_date, "close": float(last["close"]),
           "trail_active": bool(trail and peak >= t.entry_price * (1 + trail)),
           "trail_level": peak * (1 - st.get("trail_pct", 0.15)), "peak": peak,
           "stop_price": t.entry_price * (1 - st["default_sl"]), "cfg": st}
    if t.exit_reason == "end_of_period":
        out.update(state="open", pnl_pct=(last["close"] / t.entry_price - 1) * 100)
        return out
    tomorrow = t.exit_date == stub["date"]
    out.update(state="exit", reason=t.exit_reason, exit_price=float(t.exit_price),
               exit_date=None if tomorrow else t.exit_date, pnl_pct=t.pnl_pct * 100)
    return out


def _exit_reason_text(r: dict) -> str:
    from notify.telegram_bot import _px
    st, why = r["cfg"], r["reason"]
    when = "明日開盤" if r["exit_date"] is None else f"{r['exit_date']:%m/%d}"
    if why == "stop_loss":
        return f"觸及停損 −{st['default_sl']:.0%}（{when} {_px(r['exit_price'])}）"
    if why == "take_profit":
        return f"達停利 +{st['default_tp']:.0%}（{when} {_px(r['exit_price'])}）"
    if why == "trailing_stop":
        return (f"移動停利：高點 {_px(r['peak'])} 回落 {st.get('trail_pct', 0.15):.0%}"
                f"（{when} {_px(r['exit_price'])}）")
    if why == "max_hold":
        return f"持有滿 {st['default_hold']} 天 → {when}出場"
    return why


def _chip_note(sid: str) -> str:
    """籌碼狀況只當參考（回測：照籌碼出場會少賺 6～15pp）。"""
    try:
        m = _metrics(sid)
        if not m:
            return ""
        level, reasons = classify(m["aqs"], m["foreign_10d"], m["foreign_selldays"],
                                  m["inst_5d"], m["retail_rising"],
                                  foreign_10d_ratio=m.get("foreign_10d_ratio"))
        return "" if level == "✅ 持有" else "籌碼參考：" + "；".join(reasons)
    except Exception:
        return ""


def evaluate(date: str) -> pd.DataFrame:
    """評估所有 open 訊號（策略原規則），更新 log，回傳今天要通知的列。"""
    log = _load()
    if log.empty:
        return pd.DataFrame()
    today = pd.Timestamp(date)
    out = []
    n_stale = 0
    for i, row in log[log.status == "open"].iterrows():
        sid = str(row["stock_id"])
        if (today - pd.Timestamp(row["entry_date"])).days > AGE_OUT_DAYS:
            log.at[i, "status"] = "aged_out"
            continue
        r = rule_status(sid, str(row["strategy"]), str(row["entry_date"]))
        if not r or r["state"] == "pending":
            continue
        base = {"stock_id": sid, "name": row["name"], "strategy": row["strategy"],
                "entry_date": row["entry_date"], "entry_price": r["entry_price"],
                "close": r["close"], "pnl_pct": round(r["pnl_pct"], 1)}
        if r["state"] == "exit":
            reason = _exit_reason_text(r)
            stale = r["exit_date"] is not None and (today - r["exit_date"]).days > RECENT_EXIT_DAYS
            note = _chip_note(sid)
            log.at[i, "status"] = "exited"
            log.at[i, "exit_date"] = (r["exit_date"].strftime("%Y-%m-%d") if r["exit_date"] is not None
                                      else "next_open")
            log.at[i, "exit_reason"] = reason
            log.at[i, "pnl_pct"] = round(r["pnl_pct"], 1)
            if stale:   # 改版前就已觸及（例：舊籌碼規則時代的單），靜默結案不洗版
                n_stale += 1
                continue
            out.append({**base, "close": r["exit_price"], "level": "🚨 出場",
                        "reason": reason + (f"\n   {note}" if note else "")})
        elif r["trail_active"] and row["alert_level"] != "trail":
            from notify.telegram_bot import _px
            log.at[i, "alert_level"] = "trail"
            out.append({**base, "level": "📈 移動停利啟動",
                        "reason": f"跌破 {_px(r['trail_level'])} 出場（高點 {_px(r['peak'])} 回落 "
                                  f"{r['cfg'].get('trail_pct', 0.15):.0%}）"})
    _save(log)
    if n_stale:
        logger.info(f"Exit monitor: {n_stale} signals had already exited > {RECENT_EXIT_DAYS}d ago "
                    "(closed silently)")
    return pd.DataFrame(out)
