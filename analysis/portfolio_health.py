"""
持股健檢系統 — 對 portfolio.yml 裡每支股票做籌碼健診 + 賣出建議

決定邏輯：advisory only（給建議，不會自動賣）
規則層級：
  🚨 立即減碼: 已觸發停損 OR AQS<40 OR 近 5 日法人狂賣超門檻
  ⚠️ 觀察減碼: AQS<60 OR 法人近期轉賣 OR 跌破 MA60
  ✅ 繼續持有: AQS≥60 + 法人健康 + 趨勢未轉

CLI（純 terminal，不推 Telegram）:
  python -m analysis.portfolio_health
  python -m analysis.portfolio_health --portfolio my_holdings.yml
"""
from __future__ import annotations
import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from data.cache import init_db, load_prices, load_institutional
from data.universe import build_universe
from analysis.aqs import compute_aqs

_TZ = ZoneInfo("Asia/Taipei")

# 賣出規則閾值（可調）
SL_PCT             = -0.15   # 停損：跌破進場價 -15%
TRAIL_PCT          = -0.15   # trailing：從峰值跌 15%
TRAIL_TRIGGER      = 0.20    # 漲到 +20% 才啟動 trailing
AQS_DANGER         = 40      # AQS < 40 強烈警告
AQS_WARN           = 60      # AQS < 60 觀察
INST_SELL_5D_DANGER = -1_000_000   # 近 5 日三大法人賣超 < -1M 強烈警告
INST_SELL_10D_WARN  = -2_000_000   # 近 10 日 < -2M 觀察


def load_portfolio(path: str = "portfolio.yml") -> list[dict]:
    """讀取 portfolio.yml；若不存在 print 提示"""
    if not Path(path).exists():
        print(f"❌ 找不到 {path}")
        print("   請從 portfolio.yml.example 複製：cp portfolio.yml.example portfolio.yml")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("holdings", []) or []


def estimate_entry_date(price_df: pd.DataFrame, entry_price: float) -> tuple[pd.Timestamp, bool]:
    """根據 entry_price 反查最近的歷史交易日（用收盤價最接近的那天）。
    回傳 (estimated_date, is_estimated)"""
    if price_df.empty:
        return pd.Timestamp.today(), True
    p = price_df.copy()
    p["diff"] = (p["close"] - entry_price).abs()
    # 取最接近 + 較近期的（避免抓到很久以前的相近價）
    best = p.sort_values(["diff", "date"], ascending=[True, False]).iloc[0]
    return pd.to_datetime(best["date"]), True


def check_one(stock_id: str, entry_date: str | None, entry_price: float) -> dict:
    """對單一持股做健診，回傳 dict。entry_date 留空時自動估算。"""
    price_df = load_prices(stock_id, start="2025-01-01")
    inst_df = load_institutional(stock_id, start="2025-01-01")
    if price_df.empty:
        return {"stock_id": stock_id, "error": "無價格資料"}

    last = price_df.iloc[-1]
    current = float(last["close"])
    last_date = pd.to_datetime(last["date"]).date()

    # 進場日：使用者填的優先；留空則自動估算
    is_estimated = False
    if entry_date is None or str(entry_date).strip() in ("", "auto", "?"):
        entry_dt, is_estimated = estimate_entry_date(price_df, entry_price)
    else:
        entry_dt = pd.to_datetime(entry_date)
    held = price_df[price_df["date"] >= entry_dt]
    peak = float(held["close"].max()) if not held.empty else current

    pnl_pct = (current / entry_price - 1) * 100
    days_held = (last_date - entry_dt.date()).days

    # AQS
    aqs = compute_aqs(stock_id) or {}

    # 法人近 5 / 10 / 30 日
    if not inst_df.empty:
        inst_df = inst_df.sort_values("date")
        inst_df["all"] = (inst_df["foreign_"].fillna(0) +
                          inst_df["trust"].fillna(0) +
                          inst_df["dealer"].fillna(0))
        inst_5d = int(inst_df.tail(5)["all"].sum())
        inst_10d = int(inst_df.tail(10)["all"].sum())
        inst_30d = int(inst_df.tail(30)["all"].sum())
        f_5d = int(inst_df.tail(5)["foreign_"].sum())
    else:
        inst_5d = inst_10d = inst_30d = f_5d = 0

    # 趨勢面：跌破 MA60？
    if len(price_df) >= 60:
        ma60 = price_df["close"].tail(60).mean()
        below_ma60 = current < ma60
    else:
        ma60 = 0
        below_ma60 = False

    # 賣出規則
    actions = []
    severity = "✅ 繼續持有"

    # 🚨 立即減碼類
    if pnl_pct <= SL_PCT * 100:
        actions.append(f"🚨 跌破停損 {SL_PCT*100:.0f}%（虧損 {pnl_pct:.1f}%）")
        severity = "🚨 立即出"

    # Trailing 觸發
    if peak > entry_price * (1 + TRAIL_TRIGGER):
        trail_level = peak * (1 + TRAIL_PCT)
        if current <= trail_level:
            actions.append(f"🚨 trailing 觸發：峰值 {peak:.0f} 跌 15% 至 {trail_level:.0f}，現價 {current:.0f}")
            severity = "🚨 立即出（鎖利）"

    aqs_score = aqs.get("score")
    if aqs_score is not None:
        if aqs_score < AQS_DANGER:
            actions.append(f"🚨 AQS {aqs_score:.0f} < {AQS_DANGER}（疑似派發/籌碼差）")
            if severity == "✅ 繼續持有":
                severity = "🚨 立即減碼"

    if inst_5d < INST_SELL_5D_DANGER:
        actions.append(f"🚨 近 5 日三大賣超 {inst_5d:,}（門檻 {INST_SELL_5D_DANGER:,}）")
        if severity == "✅ 繼續持有":
            severity = "🚨 立即減碼"

    # ⚠️ 觀察減碼類
    if aqs_score is not None and AQS_DANGER <= aqs_score < AQS_WARN:
        actions.append(f"⚠️ AQS {aqs_score:.0f} 偏低（< {AQS_WARN}）")
        if severity == "✅ 繼續持有":
            severity = "⚠️ 觀察"

    if INST_SELL_5D_DANGER <= inst_5d < 0 and inst_10d < INST_SELL_10D_WARN:
        actions.append(f"⚠️ 法人近期轉賣（5d {inst_5d:,}, 10d {inst_10d:,}）")
        if severity == "✅ 繼續持有":
            severity = "⚠️ 觀察"

    if below_ma60:
        actions.append(f"⚠️ 跌破 MA60（{ma60:.0f}）")
        if severity == "✅ 繼續持有":
            severity = "⚠️ 觀察"

    # AQS verdict 有「派發 trap」
    aqs_verdict = aqs.get("verdict", "")
    if "trap" in aqs_verdict:
        actions.append(f"⚠️ AQS verdict: {aqs_verdict}")
        if severity == "✅ 繼續持有":
            severity = "⚠️ 觀察減碼"

    return {
        "stock_id": stock_id,
        "current": current,
        "last_date": last_date,
        "entry_price": entry_price,
        "entry_date": entry_dt.date(),
        "entry_estimated": is_estimated,
        "days_held": days_held,
        "peak": peak,
        "pnl_pct": pnl_pct,
        "aqs_score": aqs_score,
        "aqs_stage": aqs.get("stage", ""),
        "aqs_verdict": aqs_verdict,
        "inst_5d": inst_5d,
        "inst_10d": inst_10d,
        "inst_30d": inst_30d,
        "f_5d": f_5d,
        "below_ma60": below_ma60,
        "ma60": ma60,
        "actions": actions,
        "severity": severity,
    }


def format_report(results: list[dict], name_map: dict) -> str:
    """產生報告文字（terminal 用）"""
    today = datetime.now(_TZ).strftime("%Y-%m-%d %H:%M")
    lines = [f"\n📊 持股健檢報告  {today}", "=" * 70]

    # 按 severity 分組
    groups = {"🚨 立即出": [], "🚨 立即減碼": [], "🚨 立即出（鎖利）": [],
              "⚠️ 觀察": [], "⚠️ 觀察減碼": [], "✅ 繼續持有": [],
              "ERROR": []}
    for r in results:
        if "error" in r:
            groups["ERROR"].append(r)
        else:
            groups[r["severity"]].append(r)

    # 印每組
    for sev in ["🚨 立即出", "🚨 立即出（鎖利）", "🚨 立即減碼",
                "⚠️ 觀察", "⚠️ 觀察減碼", "✅ 繼續持有"]:
        if not groups[sev]:
            continue
        lines.append(f"\n【{sev}】({len(groups[sev])} 支)")
        for r in groups[sev]:
            sid = r["stock_id"]
            name = name_map.get(sid, "?")
            pnl_sign = "+" if r["pnl_pct"] >= 0 else ""
            est_mark = "≈" if r.get("entry_estimated") else ""
            head = (f"  {sid} {name}  ${r['current']:.0f}  "
                    f"{pnl_sign}{r['pnl_pct']:.1f}% ({r['days_held']}d{est_mark})  "
                    f"AQS {r['aqs_score']:.0f}" if r['aqs_score'] is not None else
                    f"  {sid} {name}  ${r['current']:.0f}  {pnl_sign}{r['pnl_pct']:.1f}% AQS N/A")
            if r.get("aqs_stage"):
                head += f" {r['aqs_stage']}"
            lines.append(head)
            lines.append(f"    法人 5d {r['inst_5d']:+,}  10d {r['inst_10d']:+,}  30d {r['inst_30d']:+,}")
            if r["actions"]:
                for a in r["actions"]:
                    lines.append(f"    └ {a}")
            else:
                lines.append("    └ 籌碼面健康、無賣出訊號")

    if groups["ERROR"]:
        lines.append("\n【⚠️ 資料錯誤】")
        for r in groups["ERROR"]:
            lines.append(f"  {r['stock_id']}: {r.get('error', '?')}")

    # 總結
    n_critical = len(groups["🚨 立即出"]) + len(groups["🚨 立即出（鎖利）"]) + len(groups["🚨 立即減碼"])
    n_warn = len(groups["⚠️ 觀察"]) + len(groups["⚠️ 觀察減碼"])
    n_safe = len(groups["✅ 繼續持有"])
    lines.append(f"\n總結: 🚨 {n_critical} 支 / ⚠️ {n_warn} 支 / ✅ {n_safe} 支")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="portfolio.yml", help="持股檔路徑")
    args = ap.parse_args()

    init_db()
    universe = build_universe()
    name_map = dict(zip(universe["stock_id"], universe["stock_name"])) if not universe.empty else {}

    holdings = load_portfolio(args.portfolio)
    if not holdings:
        print(f"❌ {args.portfolio} 沒有持股")
        return 1

    results = []
    for h in holdings:
        sid = str(h["stock_id"])
        ep = float(h["entry_price"])
        ed = h.get("entry_date")  # 可能是 None / "auto" / 日期字串
        try:
            r = check_one(sid, ed, ep)
        except Exception as e:
            r = {"stock_id": sid, "error": str(e)}
        results.append(r)

    report = format_report(results, name_map)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
