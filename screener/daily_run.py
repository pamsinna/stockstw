"""
每日選股器：
1. 增量更新今日資料（只更新 DB 中已有資料的股票）
2. 對每個通過基本面的股票算技術訊號
3. 分三個時間框架輸出當日訊號清單，並套用大盤過濾
"""
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from config import DATA_START
from data.cache import (
    init_db, load_prices, load_institutional, load_monthly_revenue, load_per,
    save_prices, save_institutional, save_prices_bulk, save_institutional_bulk,
    last_price_date, last_institutional_date, earliest_last_date_since,
    last_revenue_date, mark_fetch_skip, load_shareholding_latest,
    save_shareholding, last_shareholding_date,
    save_futures_inst, last_futures_inst_date,
)
from data.universe import build_universe
from data.fetcher import (fetch_price, fetch_institutional, fetch_monthly_revenue,
                          fetch_tdcc_shareholding, fetch_futures_inst,
                          fetch_all_prices_by_date, fetch_all_inst_by_date)
from backtest.run_backtest import build_market_filter, _normalize_and_save_revenue
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (
    signal_longterm_quality_entry,
    signal_revenue_momentum,
    signal_growth_breakout,
    signal_accumulation_eve,
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


def incremental_update(universe: pd.DataFrame) -> None:
    """
    更新所有 universe 內的股票：
    - 價量 + 三大法人：用 TWSE/TPEx 官方 bulk（單一請求回傳全市場單日）補最近
      BULK_LOOKBACK_DAYS 天 → 免 token、無 600/hr 限流、整批一致。
    - 仍落後超過 bulk 窗的個股（新股 / bootstrap）→ 退回 FinMind 逐檔深歷史。
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
    n_days = (today - start).days + 1
    logger.info(f"Bulk fill (TWSE/TPEx official) {start}..{today} "
                f"for {len(keep)} tracked stocks...")
    n_price = n_inst = 0
    for offset in tqdm(range(n_days), desc="Bulk"):
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
        time.sleep(0.5)  # 對官方站點客氣一點
    logger.info(f"Bulk fill done: {n_price} price rows, {n_inst} inst rows.")

    # ── 2) FinMind fallback：只補落後超過 bulk 窗的個股（新股 / bootstrap 深歷史）─
    floor_str = bulk_floor.isoformat()
    deep = [sid for sid in ([TAIEX_PROXY] + all_stocks)
            if (last_price_date(sid) or DATA_START) < floor_str]
    if deep:
        logger.info(f"FinMind deep-history fallback for {len(deep)} stocks "
                    f"behind {floor_str}...")
    for sid in tqdm(deep, desc="Backfill"):
        last = last_price_date(sid) or DATA_START
        price = fetch_price(sid, last)  # rate-limited inside _finmind()
        if price is None:
            mark_fetch_skip(sid, "price")
        elif not price.empty:
            save_prices(sid, price)

        last_inst = last_institutional_date(sid) or DATA_START
        if last_inst < floor_str:
            inst = fetch_institutional(sid, last_inst)  # rate-limited inside _finmind()
            if inst is None:
                mark_fetch_skip(sid, "institutional")
            elif not inst.empty:
                save_institutional(sid, inst)

    # 月營收：每月 1～10 號才抓（法規要求 10 號前公布，提早抓以第一時間收到）
    if today.day <= 10:
        # 以「上個月1日」為 stale 基準：避免 today-35天 比月初早一天造成全量重跑
        # 例如：5月7日 → stale_before = 2026-04-01；有四月營收的股票全部跳過
        # 6月1日 → stale_before = 2026-05-01；四月營收股票重跑（找五月資料）← 正確
        y, m = today.year, today.month - 1
        if m == 0:
            y, m = y - 1, 12
        stale_before = f"{y}-{m:02d}-01"
        rev_targets = [
            sid for sid in all_stocks
            if (last_revenue_date(sid) or DATA_START) < stale_before
        ]
        if rev_targets:
            logger.info(f"Refreshing monthly revenue for {len(rev_targets)} stocks "
                        f"(stale before {stale_before})...")
            for sid in tqdm(rev_targets, desc="Revenue"):
                fetch_start = last_revenue_date(sid) or DATA_START
                rev = fetch_monthly_revenue(sid, fetch_start)  # rate-limited inside _finmind()
                if rev is None:
                    mark_fetch_skip(sid, "revenue")
                elif not rev.empty:
                    _normalize_and_save_revenue(sid, rev)

    # Regime gauge 用的衍生資料（0056 + TX 期貨）— sync_db 會覆蓋 local，
    # 因此每次 incremental_update 都要重新補齊
    for sid in _AUX_PRICE_IDS:
        last = last_price_date(sid) or DATA_START
        if last < today_str:
            df = fetch_price(sid, last)
            if df is None:
                mark_fetch_skip(sid, "price")
            elif not df.empty:
                save_prices(sid, df)
                logger.info(f"Aux price {sid}: updated to {df['date'].max()}")

    for fid in _AUX_FUTURES_IDS:
        last = last_futures_inst_date(fid) or DATA_START
        if last < today_str:
            df = fetch_futures_inst(fid, last)
            if df is not None and not df.empty:
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
    results: dict[str, list] = {"long": [], "revenue": [], "growth": [], "accum": [], "combo_47": []}
    market_map = dict(zip(universe["stock_id"], universe["market"]))

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
    last_trading_day = taiex_price["date"].max() if not taiex_price.empty else pd.Timestamp("2000-01-01")
    logger.info(f"Latest trading day (0050): {last_trading_day.date()}")

    # S5 regime gauge：用 0050 過去 60 日報酬率判斷市場熱度
    # 回測：S5 在 0050 60d 勝率 >= 65% 時 80% 勝率 +42% avg；< 50% 時 -0.13%
    regime_label = "🟡 中性"
    regime_60d_return = 0.0
    if len(taiex_price) >= 65:
        recent = taiex_price.sort_values("date").tail(65).reset_index(drop=True)
        # 60 日報酬率
        regime_60d_return = float(recent.iloc[-1]["close"] / recent.iloc[-61]["close"] - 1)
        if regime_60d_return >= 0.05:
            regime_label = "🔥 多頭（S5 升級主力）"
        elif regime_60d_return <= -0.05:
            regime_label = "🥶 空頭（S5 暫停/減半）"
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
    # S4 (long), S6 (growth), S7 (accum), combo_47 都加 AQS 二次確認
    for tf in ("long", "growth", "accum", "combo_47"):
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
    }])
    return out


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
    signals = screen_today(universe)

    today = datetime.now(_TZ).strftime("%Y-%m-%d")
    for tf, df in signals.items():
        n = len(df)
        logger.info(f"[{tf}] {n} signals today")
        if not df.empty:
            df.to_csv(f"reports/signals_{tf}_{today}.csv", index=False)

    if notify_fn:
        notify_fn(signals)

    return signals
