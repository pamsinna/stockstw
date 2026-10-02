"""系統性 audit：找出所有可能讓「資料沒讀到 → 訊號沒發」的失敗點。

兩種用法：
  python scripts/system_audit.py           # 印詳細報告（人讀）
  python scripts/system_audit.py --telegram # 只推 critical 摘要到 Telegram

可程式化呼叫：collect_findings() 回傳 list[dict]，每項含
  severity: "critical" | "warning" | "ok"
  key: 短英文 ID（telegram 摘要會用）
  message: 中文描述
"""
from __future__ import annotations
import sys
import sqlite3
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import pandas as pd

from data.cache import init_db
from data.universe import EXCLUDED_INDUSTRIES


DB_PATH = "data/cache.db"


def collect_findings() -> list[dict]:
    """跑一輪 audit，回傳 [{severity, key, message, value}]"""
    init_db()
    con = sqlite3.connect(DB_PATH)
    today = date.today()
    price_cutoff = today - timedelta(days=7)

    findings: list[dict] = []

    def add(sev, key, message, value=None):
        findings.append({"severity": sev, "key": key,
                         "message": message, "value": value})

    # 只看實際追蹤的 universe（扣除排除產業）：沒扣時 ~800 檔排除產業的股票本來就
    # 不更新價格，「價格 stale 795 支」天天 critical → 狼來了，真警報被淹沒。
    uni_sids = {r[0] for r in con.execute("SELECT stock_id, industry FROM stock_universe").fetchall()
                if r[1] not in EXCLUDED_INDUSTRIES}

    # 1. fetch_log 9999 殘留
    n = con.execute("SELECT COUNT(*) FROM fetch_log WHERE last_date='9999-12-31'").fetchone()[0]
    if n > 0:
        add("critical", "9999_residue", f"fetch_log 9999 殘留 {n} 筆", n)
    else:
        add("ok", "9999_residue", "fetch_log 9999 已清", 0)

    # 2. 價格 stale
    fresh = {r[0] for r in con.execute(
        "SELECT stock_id FROM fetch_log WHERE dataset='price' AND last_date >= ? "
        "AND last_date < '9999-01-01'", (price_cutoff.isoformat(),)).fetchall()}
    n_stale_price = len(uni_sids - fresh)
    if n_stale_price > 200:
        add("critical", "stale_price", f"價格 stale {n_stale_price} 支（> 200 表示 CI fetch 大量失敗）", n_stale_price)
    elif n_stale_price > 50:
        add("warning", "stale_price", f"價格 stale {n_stale_price} 支", n_stale_price)
    else:
        add("ok", "stale_price", f"價格 stale {n_stale_price} 支", n_stale_price)

    # 3. 沒法人 / 法人 stale
    have_inst = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM institutional").fetchall()}
    no_inst = len(uni_sids - have_inst)
    if no_inst > 200:
        add("warning", "no_institutional", f"完全沒法人資料 {no_inst} 支", no_inst)

    # 4. 沒財報（影響 passes_filter）
    have_fin = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM financial").fetchall()}
    no_fin = len(uni_sids - have_fin)
    if no_fin > 100:
        add("warning", "no_financial",
            f"完全沒財報 {no_fin} 支 → passes_filter=False → S4/S6/S7 silently skip", no_fin)

    # 5. 0050 (大盤代理) 新鮮度
    r = con.execute("SELECT MAX(date) FROM daily_price WHERE stock_id='0050'").fetchone()
    if not r[0]:
        add("critical", "stale_0050", "0050 完全沒資料 → market_filter 失能", None)
    else:
        days = (today - pd.to_datetime(r[0]).date()).days
        if days > 4:
            add("critical", "stale_0050", f"0050 距今 {days} 天（market_filter 用過舊資料）", days)
        elif days > 2:
            add("warning", "stale_0050", f"0050 距今 {days} 天", days)

    # 6. 0056 / TX 期貨（regime gauge）
    for sid, tname in [("0056", "0056 (高股息 ETF)"), ("TX", "TX 期貨外資未平倉")]:
        if sid == "TX":
            r = con.execute("SELECT MAX(date) FROM futures_inst WHERE futures_id='TX'").fetchone()
        else:
            r = con.execute("SELECT MAX(date) FROM daily_price WHERE stock_id=?", (sid,)).fetchone()
        if not r[0]:
            add("critical", f"missing_{sid.lower()}", f"{tname} 完全沒資料 → regime gauge 失能", None)
        else:
            days = (today - pd.to_datetime(r[0]).date()).days
            if days > 4:
                add("critical", f"stale_{sid.lower()}", f"{tname} 距今 {days} 天", days)

    # 7. AQS 算不出（2026 年資料 < 60 日）
    enough = {r[0] for r in con.execute(
        "SELECT stock_id FROM daily_price WHERE date >= '2026-01-01' "
        "GROUP BY stock_id HAVING COUNT(*) >= 60").fetchall()}
    aqs_short = len(uni_sids - enough)
    if aqs_short > 300:
        add("warning", "aqs_unavailable", f"AQS 算不出 {aqs_short} 支（2026 資料 < 60 日）", aqs_short)

    # 8. 集保 shareholding 新鮮度
    r = con.execute("SELECT MAX(date) FROM shareholding").fetchone()
    if not r[0]:
        add("warning", "no_shareholding", "集保 shareholding 完全沒資料 → S4 retail filter 失能", None)
    else:
        days = (today - pd.to_datetime(r[0]).date()).days
        if days > 14:
            add("warning", "stale_shareholding", f"集保資料 {days} 天前 → retail filter 過舊", days)

    # 9. 本益比「全市場」新鮮度（不能看 MAX(date)：零星個股會掩蓋全市場凍結；
    #    2026-05～09 PER 全市場凍結 5 個月沒人發現）
    r = con.execute("""SELECT MAX(date) FROM (SELECT date FROM daily_per GROUP BY date
                       HAVING COUNT(*) >= 500)""").fetchone()
    p0050 = con.execute("SELECT MAX(date) FROM daily_price WHERE stock_id='0050'").fetchone()[0]
    if not r[0]:
        add("critical", "stale_per", "本益比全市場無資料 → S4/S5 PER 過濾失能", None)
    elif p0050 and r[0] < p0050:
        lag = (pd.to_datetime(p0050) - pd.to_datetime(r[0])).days
        sev = "critical" if lag > 4 else "warning"
        add(sev, "stale_per", f"本益比全市場最新 {r[0]}，落後價格 {lag} 天 → S4/S5 用舊 PER", lag)

    # 10. 月營收覆蓋率：上一個「已過法定期限」的申報月筆數 vs universe
    #     （2026-06 申報月整月只剩 1 筆、opendata 每月少 ~800 檔都沒被抓到）
    d = today.replace(day=1)
    if today.day <= 12:  # 本月申報期還沒過 → 檢查上個月
        d = (d - timedelta(days=1)).replace(day=1)
    label = d.isoformat()
    have_rev = {r[0] for r in con.execute(
        "SELECT stock_id FROM monthly_revenue WHERE date=?", (label,)).fetchall()}
    n_rev = len(uni_sids & have_rev)
    ratio = n_rev / max(len(uni_sids), 1)
    if ratio < 0.8:
        add("critical", "revenue_coverage",
            f"月營收申報月 {label} 只有 {n_rev}/{len(uni_sids)} 檔 → S5/S6 營收因子缺資料", n_rev)

    # 11. 財報季新鮮度：法定期限 +5 天後，最新季覆蓋率
    y = today.year
    due = [(date(y, 11, 19), f"{y}-09-30"), (date(y, 8, 19), f"{y}-06-30"),
           (date(y, 5, 20), f"{y}-03-31"), (date(y, 4, 5), f"{y-1}-12-31")]
    target = next((q for dd, q in due if today >= dd), f"{y-1}-09-30")
    n_fin = con.execute("SELECT COUNT(DISTINCT stock_id) FROM financial WHERE type='EPS' AND date>=?",
                        (target,)).fetchone()[0]
    n_any = con.execute("SELECT COUNT(DISTINCT stock_id) FROM financial WHERE type='EPS'").fetchone()[0]
    fin_ratio = n_fin / max(n_any, 1)
    if fin_ratio < 0.5:
        add("warning", "stale_financial",
            f"財報 {target} 只有 {n_fin}/{n_any} 檔 → 基本面濾網用舊財報（滾動刷新中）", n_fin)

    return findings


def print_report(findings: list[dict]) -> None:
    """人讀詳細報告"""
    print("\n" + "=" * 70)
    print(f" 系統 Audit  ({date.today()})")
    print("=" * 70)

    by_sev = {"critical": [], "warning": [], "ok": []}
    for f in findings:
        by_sev[f["severity"]].append(f)

    for sev, emoji in [("critical", "🚨"), ("warning", "⚠️"), ("ok", "✅")]:
        items = by_sev.get(sev, [])
        if not items:
            continue
        print(f"\n{emoji} {sev.upper()} ({len(items)})")
        for f in items:
            print(f"   • {f['message']}")

    n_crit = len(by_sev["critical"])
    n_warn = len(by_sev["warning"])
    print(f"\n總結: {n_crit} critical, {n_warn} warning, {len(by_sev['ok'])} ok")
    print("=" * 70)


def telegram_summary(findings: list[dict]) -> str | None:
    """產生簡短 Telegram 摘要；無 critical/warning 時回 None。"""
    crits = [f for f in findings if f["severity"] == "critical"]
    warns = [f for f in findings if f["severity"] == "warning"]
    if not crits and not warns:
        return None

    lines = [f"🩺 <b>系統 Audit {date.today()}</b>"]
    if crits:
        lines.append(f"\n🚨 <b>Critical ({len(crits)})</b>")
        for f in crits:
            lines.append(f"  • {f['message']}")
    if warns:
        lines.append(f"\n⚠️ <b>Warning ({len(warns)})</b>")
        for f in warns:
            lines.append(f"  • {f['message']}")
    lines.append("\n<i>跑 <code>python scripts/system_audit.py</code> 看詳細</i>")
    return "\n".join(lines)


def main() -> int:
    findings = collect_findings()
    if "--telegram" in sys.argv:
        msg = telegram_summary(findings)
        if msg is None:
            print("✅ 全綠燈，不推 Telegram")
            return 0
        from notify.telegram_bot import send_message
        ok = send_message(msg)
        print(f"Telegram audit summary {'sent' if ok else 'FAILED'}")
        return 0 if ok else 1
    else:
        print_report(findings)
        n_crit = sum(1 for f in findings if f["severity"] == "critical")
        return 1 if n_crit > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
