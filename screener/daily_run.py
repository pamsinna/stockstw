"""
每日選股器：
1. 增量更新今日資料（只更新 DB 中已有資料的股票）
2. 對每個通過基本面的股票算技術訊號
3. 分三個時間框架輸出當日訊號清單，並套用大盤過濾
"""
import os
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from config import DATA_START
from data.cache import (
    init_db, load_prices, load_institutional, load_monthly_revenue, load_per,
    save_prices, save_prices_bulk, save_institutional_bulk,
    save_monthly_revenue_bulk, save_per_bulk, save_financial_bulk, financial_coverage,
    last_full_market_date, revenue_month_counts,
    last_price_date, earliest_last_date_since,
    load_shareholding_latest, load_shareholding, save_radar_log,
    get_meta, set_meta, inst_coverage_on,
    save_shareholding, last_shareholding_date,
    save_futures_inst, last_futures_inst_date,
)
from data.universe import build_universe
from data.fetcher import (fetch_tdcc_shareholding, fetch_taifex_futures_inst,
                          fetch_stock_history, fetch_mops_financials,
                          fetch_all_prices_by_date, fetch_all_inst_by_date,
                          fetch_mops_monthly_revenue, fetch_all_per_by_date)
from backtest.run_backtest import build_market_filter
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (
    signal_longterm_quality_entry,
    signal_revenue_momentum,
    signal_growth_breakout,
    signal_accumulation_eve,
    signal_revenue_burst,
    layout_radar_today,
    STRATEGIES,
)
from analysis.aqs import compute_aqs

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("Asia/Taipei")
TAIEX_PROXY = "0050"
# Regime gauge 用的衍生資料源（非 universe 內的個股）
_AUX_PRICE_IDS = ["0056"]              # 高股息 ETF（防禦輪動指標）
_AUX_FUTURES_IDS = ["TX"]              # 台指期（外資未平倉指標）

_S4 = next(s for s in STRATEGIES if s["name"] == "中長線_品質股低接")
_S4_INST_THR = _S4.get("inst_threshold", 0)
_S4_RETAIL_MAX = _S4.get("retail_max_pct")

_S6 = next(s for s in STRATEGIES if s["name"] == "高成長突破")
_S6_INST_THR = _S6.get("inst_threshold", 0)
_S6_REV_MIN = _S6.get("rev_growth_min", 10.0)

_S7 = next(s for s in STRATEGIES if s["name"] == "累積前夕")
_S7_INST_THR = _S7.get("inst_threshold", 3_000_000)
_S7_AQS_MIN = _S7.get("aqs_min", 70.0)

# 官方 bulk 補資料的回看窗（日曆天）：足以涵蓋連假／漏跑；超過此窗仍落後的
# 個股（新股 / bootstrap）才退回 FinMind 逐檔深歷史。
BULK_LOOKBACK_DAYS = 30
# 每次至少重抓最近這幾天（即使全市場都已最新）：讓偶發單日缺口下次跑自動補回。
MIN_REFETCH_DAYS = 7
# 絕對日曆防呆：代理 0050 最新資料落後現實超過這天數 → 視為資料管線壞掉，中止
# 選股不發訊號（避免拿舊價當「今日」，見 2026-06 FinMind 額度爆掉事件）。設 7：
# 涵蓋一般連假，真凍結（會逐日擴大）一週內必觸發；長假誤觸也只是「無新資料不選股」。
MAX_PROXY_STALE_DAYS = 7
# 月營收整月缺漏回補：檢查最近這幾個申報月，筆數 < 最多那月 × 比例 → 重抓該月。
REV_GAPFILL_MONTHS = 6
REV_GAPFILL_RATIO = 0.8
# 本益比 bulk：從「最後一個全市場都有 PER 的日期」補起，最多回補這麼多日曆天。
PER_BACKFILL_MAX_DAYS = 200
# 期限已過的最新一季：覆蓋率達此比例就不再補抓（剩下的多是延遲申報／無財報）
FIN_COVERAGE_OK = 0.9
# 新股深歷史從這天起補（官方個股頁逐月查，一個月一個請求）
DEEP_HISTORY_START = "2024-01-01"
# 觀察名單（營收爆發）回看窗：近 20 交易日內觸發過都列，今日新觸發標 🆕
WATCH_WINDOW = 20
# 多排程（16:37 / 18:07 / 21:07）：法人入庫比例達此才發通知；本地時間過 FINAL_RUN_HOUR
# 的那次無論如何都發（附註資料未齊）。LAST_NOTIFIED_KEY 記最後發過的交易日，避免重發。
INST_READY_RATIO = 0.9
FINAL_RUN_HOUR = 20
LAST_NOTIFIED_KEY = "last_notified_trade_date"
# 法人佈局雷達顯示／記錄上限（實測近 40 交易日平均每天 ~1 檔、最多 4 檔）
RADAR_MAX = 10


def update_monthly_revenue(today, keep: set[str]) -> None:
    """月營收：每天抓 MOPS t21sc03（即時頁），INSERT OR IGNORE 保留首次抓到日
    = 實際公布日。

    舊版只在 1～10 日抓 opendata，但 opendata 約 17 日才換月 → 永遠抓到上一期，
    整個營收因子晚 3～4 週；遷移那個月還整月漏掉（2026-06 申報月只剩 1 筆）。
    """
    # 1) 最新營收月（上個月）：每天抓，公司陸續申報陸續進來
    ry, rm = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    rev = fetch_mops_monthly_revenue(ry, rm)
    if not rev.empty:
        rev = rev[rev["stock_id"].isin(keep)]
        save_monthly_revenue_bulk(rev)
        logger.info(f"Monthly revenue (MOPS) {ry}-{rm:02d}: {len(rev)} rows.")
    else:
        logger.warning(f"Monthly revenue (MOPS) {ry}-{rm:02d} returned empty")

    # 2) 整月缺漏回補（不含當月：當月本來就還在陸續申報）
    labels = []
    y, m = today.year, today.month
    for _ in range(REV_GAPFILL_MONTHS):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        labels.append((y, m))  # 申報月 (y, m) ↔ 營收月 = 前一個月
    counts = revenue_month_counts(f"{labels[-1][0]:04d}-{labels[-1][1]:02d}-01")
    full = max(counts.values(), default=0)
    for ly, lm in labels:
        label = f"{ly:04d}-{lm:02d}-01"
        if full and counts.get(label, 0) >= full * REV_GAPFILL_RATIO:
            continue
        vy, vm = (ly - 1, 12) if lm == 1 else (ly, lm - 1)
        gap = fetch_mops_monthly_revenue(vy, vm)
        if gap.empty:
            continue
        gap = gap[gap["stock_id"].isin(keep)]
        save_monthly_revenue_bulk(gap, fetched_date=None)  # 非即時抓到，公布日交給訊號端估計
        logger.info(f"Revenue gap-fill {label}: had {counts.get(label, 0)}, fetched {len(gap)}.")


def update_per(today, keep: set[str]) -> None:
    """本益比：官方 bulk（TWSE BWIBBU_d + TPEx peQryDate）逐日補。

    遷移到 bulk 時 PER 沒接替代來源，2026-05-05 起全市場凍結 → S4/S5 一直拿
    舊 PER 過濾。起點用「最後一個全市場都有 PER 的日期」，自動回補整段缺口。
    """
    last_full = last_full_market_date("daily_per")
    floor = today - timedelta(days=PER_BACKFILL_MAX_DAYS)
    start = datetime.fromisoformat(last_full).date() + timedelta(days=1) if last_full else floor
    start = max(min(start, today - timedelta(days=MIN_REFETCH_DAYS)), floor)
    n = 0
    for offset in range((today - start).days + 1):
        dt = start + timedelta(days=offset)
        if dt.weekday() >= 5:
            continue
        df = fetch_all_per_by_date(dt.isoformat())
        if not df.empty:
            df = df[df["stock_id"].isin(keep)]
            save_per_bulk(df)
            n += len(df)
        time.sleep(0.5)
    logger.info(f"PER bulk fill {start}..{today}: {n} rows.")


_Q_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def _filing_deadline(y: int, q: int):
    """法定公告期限：Q1→5/15、Q2→8/14、Q3→11/14、年報→隔年 3/31。"""
    return {1: datetime(y, 5, 15), 2: datetime(y, 8, 14), 3: datetime(y, 11, 14),
            4: datetime(y + 1, 3, 31)}[q].date()


def _latest_due_quarter(today) -> str:
    """法定期限已過（+5 天緩衝）的最新一季季底日。"""
    y = today.year
    candidates = [
        (datetime(y, 11, 19).date(), f"{y}-09-30"),
        (datetime(y, 8, 19).date(),  f"{y}-06-30"),
        (datetime(y, 5, 20).date(),  f"{y}-03-31"),
        (datetime(y, 4, 5).date(),   f"{y - 1}-12-31"),
    ]
    for due, q in candidates:
        if today >= due:
            return q
    return f"{y - 1}-09-30"


def _quarters_to_refresh(today) -> list[tuple[int, int]]:
    """(1) 申報期間中的季（季底後 ～ 期限 +5 天）：每天抓，早申報的先進來；
    (2) 期限已過的最新季：覆蓋率不足才補抓（見 refresh_financials）。"""
    out = []
    for y in (today.year - 1, today.year):
        for q in (1, 2, 3, 4):
            m, d = _Q_END[q]
            q_end = datetime(y, m, d).date()
            if q_end < today <= _filing_deadline(y, q) + timedelta(days=5):
                out.append((y, q))
    due = pd.Timestamp(_latest_due_quarter(today))
    out.append((due.year, (due.month - 1) // 3 + 1))
    return list(dict.fromkeys(out))


def refresh_financials(today, all_stocks: list[str]) -> None:
    """財報：MOPS 彙總報表（每季上市＋上櫃各 4 個請求涵蓋全市場，含 KY）。

    取代 FinMind 逐檔抓（舊版每天 150 檔 × 3 表 × 6 秒）。申報期間每天重抓該季，
    INSERT OR IGNORE → 早申報的公司先進來、已存在的不覆蓋。
    """
    keep = set(all_stocks)
    due = _latest_due_quarter(today)
    for y, q in _quarters_to_refresh(today):
        m, d = _Q_END[q]
        date = f"{y}-{m:02d}-{d:02d}"
        if date == due and financial_coverage(date, keep) >= FIN_COVERAGE_OK:
            continue
        df = fetch_mops_financials(y, q)
        if df.empty:
            # 申報期間初期還沒人申報是正常的；只有「期限已過」的季抓不到才算異常
            (logger.warning if date == due else logger.info)(f"MOPS financials {y}Q{q} returned empty")
            continue
        df = df[df["stock_id"].isin(keep)]
        save_financial_bulk(df)
        logger.info(f"MOPS financials {y}Q{q}: {df['stock_id'].nunique()} stocks; "
                    f"coverage {financial_coverage(date, keep):.0%}")


def _bulk_fill(start, end, keep: set[str], with_per: bool = False) -> None:
    """官方 bulk 逐日補全市場價量 + 三大法人（with_per 也補本益比）。"""
    n_price = n_inst = 0
    for offset in tqdm(range((end - start).days + 1), desc="Bulk"):
        dt = start + timedelta(days=offset)
        if dt.weekday() >= 5:  # 週末必非交易日，省一次請求
            continue
        diso = dt.isoformat()
        pdf = fetch_all_prices_by_date(diso)
        if not pdf.empty:
            pdf = pdf[pdf["stock_id"].isin(keep)]
            save_prices_bulk(pdf)
            n_price += len(pdf)
        idf = fetch_all_inst_by_date(diso)
        if not idf.empty:
            idf = idf[idf["stock_id"].isin(keep)]
            save_institutional_bulk(idf)
            n_inst += len(idf)
        if with_per and not pdf.empty:
            per = fetch_all_per_by_date(diso)
            if not per.empty:
                save_per_bulk(per[per["stock_id"].isin(keep)])
        time.sleep(0.5)  # 對官方站點客氣一點
    logger.info(f"Bulk fill done: {n_price} price rows, {n_inst} inst rows.")


def bootstrap_official(universe: pd.DataFrame, start: str) -> None:
    """從零重建資料庫（全部官方來源，免 token）：價量／法人／本益比逐日 bulk、
    月營收 MOPS 逐月、財報 MOPS 逐季、台指期法人 期交所。約每交易日 4 個請求，
    2019 起約需 2～3 小時。已存在的資料不覆蓋，可中斷後重跑接續。"""
    today = datetime.now(_TZ).date()
    s0 = datetime.fromisoformat(start).date()
    keep = set(universe["stock_id"]) | {TAIEX_PROXY} | set(_AUX_PRICE_IDS)
    earliest = earliest_last_date_since("price", start)
    resume = datetime.fromisoformat(earliest).date() if earliest else s0
    _bulk_fill(max(s0, resume), today, keep, with_per=True)
    for m in pd.period_range(pd.Timestamp(s0) - pd.DateOffset(months=24), pd.Timestamp(today), freq="M"):
        rev = fetch_mops_monthly_revenue(m.year, m.month)
        if not rev.empty:
            save_monthly_revenue_bulk(rev[rev["stock_id"].isin(keep)], fetched_date=None)
    for y in range(s0.year - 2, today.year + 1):
        for q in (1, 2, 3, 4):
            m_, d_ = _Q_END[q]
            if datetime(y, m_, d_).date() >= today:
                continue
            fin = fetch_mops_financials(y, q)
            if not fin.empty:
                save_financial_bulk(fin[fin["stock_id"].isin(keep)])
                logger.info(f"Bootstrap financials {y}Q{q}: {fin['stock_id'].nunique()} stocks")
    for fid in _AUX_FUTURES_IDS:
        for y in range(s0.year, today.year + 1):
            df = fetch_taifex_futures_inst(fid, f"{y}-01-01", f"{y}-12-31")
            if not df.empty:
                save_futures_inst(fid, df)


def incremental_update(universe: pd.DataFrame) -> None:
    """
    更新所有 universe 內的股票：
    - 價量 + 三大法人：用 TWSE/TPEx 官方 bulk（單一請求回傳全市場單日）補最近
      BULK_LOOKBACK_DAYS 天 → 免 token、無 600/hr 限流、整批一致。
    - 仍落後超過 bulk 窗的個股（新股）→ 證交所／櫃買個股歷史頁逐月補價量。
    - 0050（大盤代理）含在 keep 內，確保 last_trading_day 永遠跟上。
    """
    today = datetime.now(_TZ).date()
    today_str = today.strftime("%Y-%m-%d")

    all_stocks = universe["stock_id"].tolist()
    keep = set(all_stocks) | {TAIEX_PROXY} | set(_AUX_PRICE_IDS)

    # ── 1) 官方 bulk：補最近 BULK_LOOKBACK_DAYS 天的全市場價量 + 法人 ───────────
    bulk_floor = today - timedelta(days=BULK_LOOKBACK_DAYS)
    min_refetch = today - timedelta(days=MIN_REFETCH_DAYS)
    # 起點 = min(最舊活躍 last_date, 最近 MIN_REFETCH_DAYS 天)，但不早於 bulk_floor。
    # 取「最近 N 天」這層保證即使全市場都最新，仍會重抓近窗 → 偶發單日缺口冪等補回。
    earliest = earliest_last_date_since("price", bulk_floor.isoformat())
    start_active = datetime.fromisoformat(earliest).date() if earliest else bulk_floor
    start = max(min(start_active, min_refetch), bulk_floor)
    logger.info(f"Bulk fill (TWSE/TPEx official) {start}..{today} "
                f"for {len(keep)} tracked stocks...")
    _bulk_fill(start, today, keep)

    # ── 2) 官方個股歷史：只補落後超過 bulk 窗的個股（新股），法人歷史不補（只能逐日 bulk）
    floor_str = bulk_floor.isoformat()
    market_map = dict(zip(universe["stock_id"], universe["market"]))
    deep = [sid for sid in ([TAIEX_PROXY] + all_stocks)
            if (last_price_date(sid) or DATA_START) < floor_str]
    if deep:
        logger.info(f"Official per-stock history for {len(deep)} stocks behind {floor_str}...")
    for sid in tqdm(deep, desc="Backfill"):
        last = last_price_date(sid) or DEEP_HISTORY_START
        df = fetch_stock_history(sid, market_map.get(sid, "TWSE"), max(last, DEEP_HISTORY_START))
        if not df.empty:
            save_prices(sid, df)

    update_monthly_revenue(today, set(all_stocks))
    update_per(today, set(all_stocks))
    refresh_financials(today, all_stocks)

    # Regime gauge：0056 已在 bulk（keep 含 _AUX_PRICE_IDS）；TX 期貨用期交所官方
    for fid in _AUX_FUTURES_IDS:
        last = last_futures_inst_date(fid) or DATA_START
        if last < today_str:
            df = fetch_taifex_futures_inst(fid, last)
            if not df.empty:
                save_futures_inst(fid, df)
                logger.info(f"Aux futures_inst {fid}: updated to {df['date'].max()}")

    # TDCC 千張大戶週報：超過 7 天才更新（一週公布一次）
    last_sh = last_shareholding_date()
    today_dt = datetime.now(_TZ).date()
    if last_sh is None or (today_dt - datetime.fromisoformat(last_sh).date()).days >= 7:
        logger.info("Refreshing TDCC shareholding (weekly snapshot)...")
        try:
            sh_df = fetch_tdcc_shareholding()
            if not sh_df.empty:
                n = save_shareholding(sh_df)
                logger.info(f"  TDCC saved: {n} rows for week {sh_df['date'].iloc[0]}")
            else:
                logger.warning("TDCC shareholding fetch returned empty")
        except Exception as e:
            logger.warning(f"TDCC fetch failed (non-fatal): {e}")


def screen_today(universe: pd.DataFrame,
                 use_fundamental_filter: bool = True) -> dict[str, pd.DataFrame]:
    """
    回傳 {timeframe: DataFrame of signals today}
    timeframe: "short", "swing", "long"
    """
    results: dict[str, list] = {"long": [], "revenue": [], "growth": [], "accum": [], "combo_47": [],
                                "watch": [], "radar": []}
    market_map = dict(zip(universe["stock_id"], universe["market"]))
    industry_map = (dict(zip(universe["stock_id"], universe["industry"]))
                    if "industry" in universe.columns else {})

    # 回測規則：S4 ∩ S7 在 20 交易日內接力 — 60 日勝率 66.3%、平均 +11.33%
    # （vs 60d window 的 60.4% / +10.37%，vs S7 only 56.9% / +8.40%）。
    # 20d 是甜蜜點：兩 leg 隔太久反而把弱訊號也算進來，剛接力的最強。
    COMBO_WINDOW = 20

    # 大盤過濾：今天是否多頭趨勢
    today_str = datetime.now(_TZ).strftime("%Y-%m-%d")
    market_filter = build_market_filter(start=DATA_START, end=today_str)
    strict_market_filter = build_market_filter(start=DATA_START, end=today_str, strict=True)
    if market_filter.empty:
        logger.warning("Market filter unavailable — running without it")
        market_filter = None
        strict_market_filter = None
    else:
        avail = market_filter[market_filter.index <= pd.Timestamp(today_str)]
        if not avail.empty:
            latest_mf = avail.iloc[-1]
            logger.info(f"Market filter (latest): {'多頭' if latest_mf else '空頭'} "
                        f"({avail.index[-1].date()})")

    # 基本面篩選（有財報資料時才有意義）
    fund_ok: set[str] = set(universe["stock_id"])
    if use_fundamental_filter:
        logger.info("Running fundamental filter...")
        fund_df = batch_fundamentals(universe["stock_id"].tolist())
        fund_ok = set(fund_df[fund_df["passes_filter"]]["stock_id"])
        logger.info(f"Fundamental pass: {len(fund_ok)} / {len(universe)}")

    # 散戶比例 filter（用 TDCC 最新一週快照；只套用於 S4，從 STRATEGIES 讀）
    retail_ok_s4: set[str] | None = None
    if _S4_RETAIL_MAX is not None:
        sh = load_shareholding_latest()
        if not sh.empty:
            retail_ok_s4 = set(sh[sh["retail_pct"] <= _S4_RETAIL_MAX]["stock_id"])
            logger.info(f"S4 retail filter ≤ {_S4_RETAIL_MAX}%: {len(retail_ok_s4)} stocks")
        else:
            logger.warning("Shareholding data unavailable — S4 retail filter disabled")

    logger.info("Generating signals...")
    mf = market_filter
    strict_mf = strict_market_filter

    # 用 0050 最後資料日當「本日交易日」基準：只對資料已更新至此日的股票產生訊號
    taiex_price = load_prices(TAIEX_PROXY, start="2024-01-01")
    bench_close = taiex_price.set_index("date")["close"] if not taiex_price.empty else pd.Series(dtype=float)
    last_trading_day = taiex_price["date"].max() if not taiex_price.empty else pd.Timestamp("2000-01-01")
    logger.info(f"Latest trading day (0050): {last_trading_day.date()}")

    # 絕對日曆 freshness gate（belt-and-suspenders）：last_trading_day 來自 0050 自身，
    # 若整條管線壞掉、0050 也凍結，原本的 last_date < last_trading_day 守門就形同虛設
    # （拿舊價當今日）。這裡用 wall-clock 絕對比對，落後太多就中止選股、不發訊號。
    proxy_stale_days = (pd.Timestamp(datetime.now(_TZ).date()) - last_trading_day).days
    if proxy_stale_days > MAX_PROXY_STALE_DAYS:
        logger.error(
            f"🚨 資料過期：代理 {TAIEX_PROXY} 最新 {last_trading_day.date()}，落後現實 "
            f"{proxy_stale_days} 天（> {MAX_PROXY_STALE_DAYS}）— 中止選股，不發訊號"
        )
        out = {k: pd.DataFrame() for k in ("long", "revenue", "growth", "accum", "combo_47", "watch", "radar")}
        out["_meta"] = pd.DataFrame([{
            "regime_label": f"🚨 資料過期 {proxy_stale_days} 天，已暫停選股",
            "regime_60d_return": 0.0,
            "data_stale_days": proxy_stale_days,
        }])
        return out

    # S5 regime gauge：用 0050 過去 60 日報酬率判斷市場熱度
    # 回測：S5 在 0050 60d 勝率 >= 65% 時 80% 勝率 +42% avg；< 50% 時 -0.13%
    regime_label = "🟡 中性"
    regime_60d_return = 0.0
    if len(taiex_price) >= 65:
        recent = taiex_price.sort_values("date").tail(65).reset_index(drop=True)
        # 60 日報酬率
        regime_60d_return = float(recent.iloc[-1]["close"] / recent.iloc[-61]["close"] - 1)
        if regime_60d_return >= 0.05:
            regime_label = "🔥 多頭"
        elif regime_60d_return <= -0.05:
            regime_label = "🥶 空頭"
        else:
            regime_label = "🟡 中性"
    logger.info(f"Regime gauge: {regime_label}  0050 60d return={regime_60d_return*100:+.1f}%")

    stale_cutoff = pd.Timestamp(datetime.now(_TZ).date()) - pd.Timedelta(days=15)  # ~10 交易日
    signal_errors: dict[str, int] = {}  # exception class → count

    for sid in tqdm(universe["stock_id"], desc="Screen"):
        price = load_prices(sid, start="2020-01-01")
        if len(price) < 60:
            continue
        last_date = price["date"].max()
        # 下市或長期停牌
        if last_date < stale_cutoff:
            continue
        # 資料未更新至最後交易日：跳過，避免用舊資料產生訊號
        if last_date < last_trading_day:
            continue
        inst = load_institutional(sid, start="2020-01-01")
        inst_arg = inst if not inst.empty else None
        market = market_map.get(sid, "TWSE")

        per = load_per(sid, start="2020-01-01")
        per_arg = per if not per.empty else None

        try:
            s4_ok = sid in fund_ok and (retail_ok_s4 is None or sid in retail_ok_s4)
            df_l = None
            df_a = None
            s4_today = False
            s7_today = False

            if s4_ok:
                df_l = signal_longterm_quality_entry(price, inst_arg, per_df=per_arg, market_filter=strict_mf, inst_threshold=_S4_INST_THR)
                s4_today = bool(df_l.iloc[-1]["signal_long"])
                if s4_today:
                    results["long"].append(_summary_row(sid, market, df_l, "long"))

            # 策略五：月營收動能（每月 10 日後第一個交易日才會有訊號）
            rev = load_monthly_revenue(sid)
            rev_arg = rev if not rev.empty else None
            df_rv = signal_revenue_momentum(price, inst_arg, rev_arg, per_df=per_arg, market_filter=mf)
            if bool(df_rv.iloc[-1]["signal_rev"]):
                results["revenue"].append(_summary_row(sid, market, df_rv, "revenue"))

            # 觀察名單：營收爆發＋突破（不分基本面，參考用，不是進場訊號）
            df_b = signal_revenue_burst(price, rev_arg, market_filter=mf)
            w = _watch_row(sid, market, industry_map.get(sid, ""), df_b, inst)
            if w:
                results["watch"].append(w)

            # 法人佈局雷達（前瞻追蹤，不是進場訊號；條件凍結 LAYOUT_RULES_VERSION）
            rd = layout_radar_today(price, inst_arg, rev_arg, bench_close, load_shareholding(sid))
            if rd:
                results["radar"].append({"stock_id": sid, "market": market,
                                         "industry": industry_map.get(sid, ""), "vol_ratio": 0.0, **rd})

            # 策略六：高成長突破（需基本面 pass，loose market filter）
            if sid in fund_ok:
                df_g = signal_growth_breakout(price, inst_arg, rev_arg,
                    market_filter=mf, inst_threshold=_S6_INST_THR,
                    rev_growth_min=_S6_REV_MIN)
                if bool(df_g.iloc[-1].get("signal_growth", False)):
                    results["growth"].append(_summary_row(sid, market, df_g, "growth"))

            # 策略七：累積前夕（需基本面 pass，loose market filter）
            if sid in fund_ok:
                df_a = signal_accumulation_eve(price, inst_arg,
                    market_filter=mf, inst_threshold=_S7_INST_THR,
                    aqs_min=_S7_AQS_MIN)
                s7_today = bool(df_a.iloc[-1].get("signal_accum", False))
                if s7_today:
                    results["accum"].append(_summary_row(sid, market, df_a, "accum"))

            # S4 ∩ S7 combo：今日有 S4 或 S7，且另一邊在過去 COMBO_WINDOW 交易日
            # 內也曾觸發 → 高信心進場
            if s4_ok and df_l is not None and df_a is not None and (s4_today or s7_today):
                recent_s4 = bool(df_l["signal_long"].tail(COMBO_WINDOW + 1).any())
                recent_s7 = bool(df_a["signal_accum"].tail(COMBO_WINDOW + 1).any())
                if recent_s4 and recent_s7:
                    # 用今日觸發那邊的 dataframe 取數值；S4 優先（含 PER 等較完整）
                    primary_df = df_l if s4_today else df_a
                    row = _summary_row(sid, market, primary_df, "combo_47")
                    row["s4_today"] = s4_today
                    row["s7_today"] = s7_today
                    results["combo_47"].append(row)

        except Exception as e:
            cls = type(e).__name__
            signal_errors[cls] = signal_errors.get(cls, 0) + 1
            logger.debug(f"{sid}: {cls}: {e}")

    if signal_errors:
        total = sum(signal_errors.values())
        breakdown = ", ".join(f"{cls}={n}" for cls, n in sorted(signal_errors.items()))
        logger.warning(f"Signal computation failed for {total} stocks ({breakdown})")

    # 對每個訊號補上 AQS（累積品質分）+ stage + verdict
    # S4～S7、combo_47 都加 AQS（候選篩選 AQS<50 剔除、出場監控進場快照都用）
    for tf in ("long", "revenue", "growth", "accum", "combo_47"):
        for row in results[tf]:
            sid = row["stock_id"]
            try:
                aqs = compute_aqs(sid)
                if aqs is not None:
                    row["aqs_score"] = aqs["score"]
                    row["aqs_stage"] = aqs["stage"]
                    row["aqs_verdict"] = aqs["verdict"]
            except Exception as e:
                logger.debug(f"AQS compute failed for {sid}: {e}")

    out = {
        k: pd.DataFrame(v).sort_values("vol_ratio", ascending=False)
        if v else pd.DataFrame()
        for k, v in results.items()
    }
    # 把 regime 訊息塞進結果，供 notify 顯示
    out["_meta"] = pd.DataFrame([{
        "regime_label": regime_label,
        "regime_60d_return": regime_60d_return,
        "trade_date": last_trading_day.strftime("%Y-%m-%d"),
    }])
    return out


def _watch_row(stock_id: str, market: str, industry: str,
               df: pd.DataFrame, inst: pd.DataFrame) -> dict | None:
    """近 WATCH_WINDOW 交易日內觸發過營收爆發 → 觀察名單一列（取最近一次觸發）。"""
    recent = df.tail(WATCH_WINDOW)
    hits = recent[recent["signal_burst"]]
    if hits.empty:
        return None
    h = hits.iloc[-1]
    close = float(df.iloc[-1]["close"])
    f60 = float(inst["foreign_"].tail(60).sum()) if inst is not None and not inst.empty else float("nan")
    return {
        "stock_id": stock_id, "market": market, "industry": industry or "未分類",
        "trigger_date": pd.Timestamp(h["date"]).strftime("%Y-%m-%d"),
        "trigger_close": float(h["close"]), "close": close,
        "since_pct": (close / float(h["close"]) - 1) * 100,
        "is_new": bool(df.iloc[-1]["signal_burst"]),
        "burst_yoy": float(h["burst_yoy"]), "burst_g3": float(h["burst_g3"]),
        "f_60d": f60, "vol_ratio": 0.0,
    }


def _summary_row(stock_id: str, market: str,
                  df: pd.DataFrame, timeframe: str) -> dict:
    last = df.iloc[-1]
    return {
        "stock_id":  stock_id,
        "market":    market,
        "timeframe": timeframe,
        "close":     round(float(last.get("close", 0)), 2),
        "volume":    float(last.get("volume", 0)),
        "vol_ratio": round(float(last.get("vol_ratio", 0)), 2),
        "bb_pct":    round(float(last.get("bb_pct", float("nan"))), 3),
        "kd_k":      round(float(last.get("kd_k", 0)), 1),
        "rsi":       round(float(last.get("rsi", 0)), 1),
        "ma_aligned":bool(last.get("ma_aligned", False)),
        "above_ma20": bool(last.get("close", 0) > last.get("ma20", float("inf"))),
        "inst_total":float(last.get("inst_total", 0)),
        "per":         float(last.get("per", float("nan"))),
        "f_60d":       float(last.get("f_60d", 0.0)),
        "t_60d":       float(last.get("t_60d", 0.0)),
        "f_20d":       float(last.get("f_20d", float("nan"))),
        "revenue_yoy": float(last.get("revenue_yoy", float("nan"))),
    }


def run_daily(notify_fn=None) -> dict | None:
    """GitHub Actions 呼叫的入口"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    init_db()
    universe = build_universe()
    if universe.empty:
        logger.error("Empty universe")
        return None

    incremental_update(universe)

    # 一天排多個 cron（GitHub 排程常延遲數小時）：第一個「資料齊全」的 run 發通知，
    # 之後的 run 只更新資料。假日也不會再把前一交易日的訊號重發一次。
    trade_date = last_price_date(TAIEX_PROXY) or ""
    force = os.getenv("FORCE_NOTIFY") == "1"
    if notify_fn and not force and trade_date and get_meta(LAST_NOTIFIED_KEY) == trade_date:
        logger.info(f"Already notified for {trade_date} — data updated, skipping screen/notify.")
        return None
    inst_ready = inst_coverage_on(trade_date, set(universe["stock_id"])) >= INST_READY_RATIO
    final_run = datetime.now(_TZ).hour >= FINAL_RUN_HOUR
    if notify_fn and not force and not inst_ready and not final_run:
        logger.warning(f"三大法人 {trade_date} 尚未公布齊全 — 不發通知，等下一個排程。")
        return None

    signals = screen_today(universe)

    # 報告日 = 資料最後交易日（cron 常延遲、跑過午夜會變成隔天甚至週六）
    meta = signals.get("_meta", pd.DataFrame())
    today = (meta.iloc[0]["trade_date"] if not meta.empty and "trade_date" in meta.columns
             else datetime.now(_TZ).strftime("%Y-%m-%d"))
    if not inst_ready and not meta.empty:
        signals["_meta"]["regime_label"] = (str(meta.iloc[0].get("regime_label", ""))
                                            + "｜⚠️ 當日法人資料未齊，籌碼條件可能少算一天")

    # 最終候選名單：通知顯示什麼、出場監控就追蹤什麼（AQS<50 剔除、S5 單日上限、總數上限）
    from screener.candidates import finalize_candidates
    signals = finalize_candidates(signals, today)

    # 法人佈局雷達：顯示前 RADAR_MAX 檔（投信 20 日買超佔股本由高到低），顯示什麼就記什麼
    radar = signals.get("radar", pd.DataFrame())
    if radar is not None and not radar.empty:
        radar = (radar.sort_values("trust_20d_pct_shares", ascending=False, na_position="last")
                 .head(RADAR_MAX).reset_index(drop=True))
        signals["radar"] = radar
        try:
            save_radar_log(radar.assign(date=today))
        except Exception as e:
            logger.warning(f"Radar log failed: {e}")

    # 訊號出場監控：記今日訊號 + 評估既有 open 訊號的籌碼出場（論點破壞才提醒）
    try:
        from notify.exit_monitor import record_today, evaluate, _load, prune_untracked
        if _load().empty:   # 首次上線：自動回填近 20 交易日，之後每日累積維護
            logger.info("Exit monitor: empty state → seeding last 20 trading days...")
            from scripts.seed_open_signals import seed
            seed(20)
        prune_untracked()
        record_today(signals, today)
        signals["exits"] = evaluate(today)
    except Exception as e:
        logger.warning(f"Exit monitor failed: {e}")

    # 美國信用壓力溫度（FRED HY OAS）→ 塞進 _meta 給通知顯示（固收領先股市的 risk-off 溫度）
    try:
        from data.fetcher import us_credit_stress_summary
        cs = us_credit_stress_summary()
        if cs and "_meta" in signals and not signals["_meta"].empty:
            signals["_meta"]["credit_stress"] = cs
    except Exception as e:
        logger.warning(f"Credit stress fetch failed: {e}")

    # 電信三雄法人資金流（純參考）→ 塞進 _meta 給通知顯示
    try:
        from data.cache import telecom_flow_summary
        tf = telecom_flow_summary()
        if tf and "_meta" in signals and not signals["_meta"].empty:
            signals["_meta"]["telecom_flow"] = tf
    except Exception as e:
        logger.warning(f"Telecom flow summary failed: {e}")

    for tf, df in signals.items():
        n = len(df)
        logger.info(f"[{tf}] {n} signals today")
        if not df.empty:
            df.to_csv(f"reports/signals_{tf}_{today}.csv", index=False)

    if notify_fn:
        notify_fn(signals)
        set_meta(LAST_NOTIFIED_KEY, today)

    return signals
