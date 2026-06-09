"""系統性 audit：找出所有可能讓「資料沒讀到 → 訊號沒發」的失敗點。

針對每個失敗模式給出具體數字 + 哪幾支股票受影響。
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import sqlite3
from datetime import date, timedelta
import pandas as pd

from data.cache import init_db, load_prices, load_institutional, load_monthly_revenue, load_per

init_db()
con = sqlite3.connect("data/cache.db")

DB_PATH = "data/cache.db"
TODAY = date.today()
PRICE_CUTOFF = TODAY - timedelta(days=7)
REV_CUTOFF = TODAY - timedelta(days=60)
PER_CUTOFF = TODAY - timedelta(days=7)
FIN_CUTOFF = TODAY - timedelta(days=120)  # 季報延遲合理 120 天
SH_CUTOFF = TODAY - timedelta(days=14)

print("=" * 78)
print(f" 系統 Audit  ({TODAY})")
print("=" * 78)

# Universe size
uni = con.execute("SELECT stock_id, market FROM stock_universe").fetchall()
uni_sids = {r[0] for r in uni}
print(f"\nUniverse: {len(uni_sids)} 支")

# ─── 1. fetch_log 9999 殘留 ─────────────────────────────────────────────────
n = con.execute("SELECT COUNT(*) FROM fetch_log WHERE last_date='9999-12-31'").fetchone()[0]
print(f"\n[1] fetch_log 9999 殘留: {n} 個 {'❌' if n else '✓'}")
if n:
    rows = con.execute("SELECT stock_id, dataset, COUNT(*) FROM fetch_log "
                      "WHERE last_date='9999-12-31' GROUP BY dataset").fetchall()
    for sid, ds, cnt in rows:
        print(f"     {ds}: {cnt}")

# ─── 2. 價格資料新鮮度 ───────────────────────────────────────────────────────
q = """SELECT u.stock_id FROM stock_universe u
       WHERE NOT EXISTS (SELECT 1 FROM fetch_log f WHERE f.stock_id=u.stock_id
                         AND f.dataset='price' AND f.last_date >= ?
                         AND f.last_date < '9999-01-01')"""
stale_price = [r[0] for r in con.execute(q, (PRICE_CUTOFF.isoformat(),)).fetchall()]
print(f"\n[2] 價格 stale (last < {PRICE_CUTOFF}): {len(stale_price)} 支 {'❌' if len(stale_price)>50 else '⚠️' if stale_price else '✓'}")
if stale_price[:5]:
    print(f"    sample: {', '.join(stale_price[:5])}{' ...' if len(stale_price)>5 else ''}")

# ─── 3. 法人資料缺失 ─────────────────────────────────────────────────────────
have_inst = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM institutional").fetchall()}
no_inst = uni_sids - have_inst
print(f"\n[3] 完全沒有法人資料: {len(no_inst)} 支 {'⚠️' if no_inst else '✓'}")
print("    影響: S4/S5/S6/S7 都用法人條件 — 沒法人資料的股票自動失去重要 filter")
if list(no_inst)[:5]:
    print(f"    sample: {', '.join(list(no_inst)[:5])}")

# 法人資料 stale
q = """SELECT u.stock_id FROM stock_universe u
       WHERE NOT EXISTS (SELECT 1 FROM fetch_log f WHERE f.stock_id=u.stock_id
                         AND f.dataset='institutional' AND f.last_date >= ?
                         AND f.last_date < '9999-01-01')"""
stale_inst = [r[0] for r in con.execute(q, (PRICE_CUTOFF.isoformat(),)).fetchall()]
print(f"    法人 stale (last < {PRICE_CUTOFF}): {len(stale_inst)} 支")

# ─── 4. 月營收缺失 / 過期 ────────────────────────────────────────────────────
have_rev = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM monthly_revenue").fetchall()}
no_rev = uni_sids - have_rev
print(f"\n[4] 完全沒月營收: {len(no_rev)} 支 {'⚠️' if len(no_rev)>100 else '✓'}")
print("    影響: S5 (需要 yoy + 連續性) 跟 S6 (需 3M sum) 都會 silently skip")

# 月營收最新一筆是 4 月以前的（資料過舊）
q = """SELECT u.stock_id, MAX(r.date) FROM stock_universe u
       LEFT JOIN monthly_revenue r ON u.stock_id=r.stock_id
       GROUP BY u.stock_id HAVING MAX(r.date) < '2026-04-01' OR MAX(r.date) IS NULL"""
old_rev = con.execute(q).fetchall()
no_rev_at_all = [s for s, d in old_rev if d is None]
old_rev_only = [(s, d) for s, d in old_rev if d is not None]
print(f"    其中最後一筆 < 2026-04 的: {len(old_rev_only)} 支")
if old_rev_only[:3]:
    for s, d in old_rev_only[:3]:
        print(f"      {s}: last={d}")

# ─── 5. revenue_yoy 缺漏（已修但 audit ） ─────────────────────────────────────
n_na = con.execute("SELECT COUNT(*) FROM monthly_revenue WHERE revenue_yoy IS NULL").fetchone()[0]
n_tot = con.execute("SELECT COUNT(*) FROM monthly_revenue").fetchone()[0]
print(f"\n[5] revenue_yoy NaN 比例: {n_na}/{n_tot} ({n_na/n_tot*100:.1f}%) — 用 computed fallback OK ✓")

# ─── 6. PER 缺失 ────────────────────────────────────────────────────────────
have_per = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM daily_per").fetchall()}
no_per = uni_sids - have_per
print(f"\n[6] 沒 PER 資料: {len(no_per)} 支 {'⚠️' if len(no_per)>200 else '✓'}")
print("    影響: S4 PER<20 過濾條件無法套用 — 沒 PER 資料的股票 PER condition 放行")
print("           (這代表會多抓一些高 PE 飆股 — 不一定是 bug，看你怎麼想)")

q = """SELECT u.stock_id FROM stock_universe u
       WHERE NOT EXISTS (SELECT 1 FROM fetch_log f WHERE f.stock_id=u.stock_id
                         AND f.dataset='per' AND f.last_date >= ?
                         AND f.last_date < '9999-01-01')"""
stale_per = [r[0] for r in con.execute(q, (PER_CUTOFF.isoformat(),)).fetchall()]
print(f"    PER stale (last < {PER_CUTOFF}): {len(stale_per)} 支")

# ─── 7. 財報資料缺失 ─────────────────────────────────────────────────────────
have_fin = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM financial").fetchall()}
no_fin = uni_sids - have_fin
print(f"\n[7] 沒財報資料: {len(no_fin)} 支 {'❌' if len(no_fin)>50 else '⚠️' if no_fin else '✓'}")
print("    影響: passes_filter=False → S4/S6/S7 通通 silently skip")
print("           (這是國巨 vs 雷科那種「明明很強卻沒抓」的核心原因)")
if list(no_fin)[:5]:
    print(f"    sample: {', '.join(list(no_fin)[:5])}")

# ─── 8. TDCC 集保 shareholding 新鮮度 ────────────────────────────────────────
sh = con.execute("SELECT MAX(date), COUNT(DISTINCT stock_id) FROM shareholding").fetchone()
print(f"\n[8] 集保 shareholding: max date={sh[0]}, {sh[1]} 支")
if sh[0]:
    d = pd.to_datetime(sh[0]).date()
    days = (TODAY - d).days
    status = "✓" if days <= 14 else "⚠️" if days <= 30 else "❌"
    print(f"    距今 {days} 天 {status}  (S4 的 retail filter 用這個 — 過舊→S4 retail filter 失效)")

# ─── 9. 大盤代理 0050 新鮮度 ─────────────────────────────────────────────────
r = con.execute("SELECT MAX(date) FROM daily_price WHERE stock_id='0050'").fetchone()
last_50 = r[0]
days = (TODAY - pd.to_datetime(last_50).date()).days if last_50 else 999
print(f"\n[9] 0050 (大盤代理): last={last_50}, 距今 {days} 天 {'✓' if days<=4 else '⚠️' if days<=7 else '❌'}")
print("    影響: market_filter 用 0050 算 — 0050 新鮮度落後會讓所有訊號 mismatch")
print("    note: screen_today 用 0050 last date 當「本日交易日」基準 — 如果 0050 比個股舊，")
print("           會放掉一堆個股的最新訊號")

# ─── 10. fetched_date 異常（影響 S5 publish_date 邏輯）──────────────────────
print(f"\n[10] monthly_revenue fetched_date 分布:")
q = """SELECT MIN(fetched_date), MAX(fetched_date),
       SUM(CASE WHEN fetched_date IS NULL THEN 1 ELSE 0 END) FROM monthly_revenue"""
r = con.execute(q).fetchone()
print(f"     fetched_date 範圍: {r[0]} ~ {r[1]}, NULL 數: {r[2]}")
print("     若太多 NULL → S5 公布日退回「次月 10 日」保守估計（不是 bug 但 conservative）")

# ─── 11. STRATEGIES 設定缺漏 ────────────────────────────────────────────────
from technical.signals import STRATEGIES
print(f"\n[11] STRATEGIES 設定檢查:")
required = ["name", "signal_fn", "default_tp", "default_sl", "default_hold"]
for s in STRATEGIES:
    miss = [k for k in required if k not in s]
    status = "✓" if not miss else f"❌ missing {miss}"
    print(f"     {s.get('name', '?')}: {status}")

# ─── 12. AQS 資料不足 ──────────────────────────────────────────────────────
# AQS 需要 60+ 日資料；新股 IPO < 60 日無法算
q = """SELECT u.stock_id FROM stock_universe u
       LEFT JOIN (SELECT stock_id, COUNT(*) c FROM daily_price
                  WHERE date >= '2026-01-01' GROUP BY stock_id) p
       ON u.stock_id=p.stock_id WHERE COALESCE(p.c, 0) < 60"""
aqs_short = [r[0] for r in con.execute(q).fetchall()]
print(f"\n[12] 2026 年資料 < 60 日（AQS 算不出來）: {len(aqs_short)} 支")
print("     影響: S7 需要 AQS≥70 — AQS=None 的股票 S7 silently skip")

# ─── 13. CI 跑完是否真的把 0050 + 0056 + futures 也更新了 ─────────────────────
print(f"\n[13] 衍生資料源（regime gauge 用的）新鮮度:")
for sid in ("0050", "0056"):
    r = con.execute("SELECT MAX(date) FROM daily_price WHERE stock_id=?", (sid,)).fetchone()
    if r[0]:
        d = pd.to_datetime(r[0]).date()
        days = (TODAY - d).days
        print(f"     {sid}: last={r[0]} ({days}d) {'✓' if days<=4 else '❌'}")
    else:
        print(f"     {sid}: 沒資料 ❌")

r = con.execute("SELECT MAX(date) FROM futures_inst").fetchone()
if r[0]:
    d = pd.to_datetime(r[0]).date()
    days = (TODAY - d).days
    print(f"     TX 期貨外資未平倉: last={r[0]} ({days}d) {'✓' if days<=4 else '❌'}")
else:
    print(f"     TX 期貨外資未平倉: 沒資料（regime gauge 失能）❌")

# ─── 14. 月營收 fetched_date vs 預期公布窗口 ────────────────────────────────
print(f"\n[14] 月營收 fetched_date 是否在 S5 採用窗口 (±20 天) 內:")
q = """SELECT date, fetched_date FROM monthly_revenue
       WHERE fetched_date IS NOT NULL AND date >= '2026-01-01'"""
rows = con.execute(q).fetchall()
out_of_window = 0
for rev_date, fetched in rows:
    rev_dt = pd.to_datetime(rev_date)
    fetched_dt = pd.to_datetime(fetched)
    expected = pd.Timestamp(rev_dt.year, rev_dt.month + 1 if rev_dt.month < 12 else 1, 10) \
               + pd.DateOffset(years=(0 if rev_dt.month < 12 else 1))
    if abs((fetched_dt - expected).days) > 20:
        out_of_window += 1
print(f"     範圍外的: {out_of_window} 筆 — 這些會退回「次月 10 日」保守估計")

print("\n" + "=" * 78)
