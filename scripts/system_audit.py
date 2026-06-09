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

    uni_sids = {r[0] for r in con.execute("SELECT stock_id FROM stock_universe").fetchall()}

    # 1. fetch_log 9999 殘留
    n = con.execute("SELECT COUNT(*) FROM fetch_log WHERE last_date='9999-12-31'").fetchone()[0]
    if n > 0:
        add("critical", "9999_residue", f"fetch_log 9999 殘留 {n} 筆", n)
    else:
        add("ok", "9999_residue", "fetch_log 9999 已清", 0)

    # 2. 價格 stale
    q = """SELECT COUNT(*) FROM stock_universe u
           WHERE NOT EXISTS (SELECT 1 FROM fetch_log f WHERE f.stock_id=u.stock_id
                             AND f.dataset='price' AND f.last_date >= ?
                             AND f.last_date < '9999-01-01')"""
    n_stale_price = con.execute(q, (price_cutoff.isoformat(),)).fetchone()[0]
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
    q = """SELECT COUNT(*) FROM stock_universe u
           LEFT JOIN (SELECT stock_id, COUNT(*) c FROM daily_price
                      WHERE date >= '2026-01-01' GROUP BY stock_id) p
           ON u.stock_id=p.stock_id WHERE COALESCE(p.c, 0) < 60"""
    aqs_short = con.execute(q).fetchone()[0]
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
