"""
每日選股器：
1. 增量更新今日資料（只更新 DB 中已有資料的股票）
2. 對每個通過基本面的股票算技術訊號
3. 分三個時間框架輸出當日訊號清單，並套用大盤過濾
"""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from data.cache import (
    init_db, load_universe, load_prices, load_institutional, load_monthly_revenue,
    save_prices, save_institutional, last_price_date,
)
from data.universe import build_universe
from data.fetcher import fetch_price, fetch_institutional
from backtest.run_backtest import build_market_filter
from fundamental.quality_filter import batch_fundamentals
from technical.signals import (
    signal_short_vol_breakout,
    signal_swing_ma_kd_inst,
    signal_swing_dual_inst,
    signal_longterm_quality_entry,
    signal_revenue_momentum,
)

logger = logging.getLogger(__name__)

TAIEX_PROXY = "0050"


def incremental_update(universe: pd.DataFrame, sleep_sec: float = 6.0) -> None:
    """
    只更新 DB 中已有歷史資料的股票 + 0050（大盤代理）
    新股第一次下載需執行 download 模式
    """
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    all_stocks = universe["stock_id"].tolist()
    stocks_to_update = [
        sid for sid in all_stocks if last_price_date(sid) is not None
    ]
    if TAIEX_PROXY not in stocks_to_update:
        stocks_to_update.insert(0, TAIEX_PROXY)

    logger.info(f"Incremental update for {len(stocks_to_update)} stocks "
                f"(out of {len(all_stocks)} in universe)...")

    for sid in tqdm(stocks_to_update, desc="Update"):
        last = last_price_date(sid) or "2018-01-01"
        if last >= yesterday:
            continue

        price = fetch_price(sid, last)
        if not price.empty:
            save_prices(sid, price)
        time.sleep(sleep_sec)

        inst = fetch_institutional(sid, last)
        if not inst.empty:
            save_institutional(sid, inst)
        time.sleep(sleep_sec)


def screen_today(universe: pd.DataFrame,
                 use_fundamental_filter: bool = True) -> dict[str, pd.DataFrame]:
    """
    回傳 {timeframe: DataFrame of signals today}
    timeframe: "short", "swing", "long"
    """
    results: dict[str, list] = {"short": [], "swing": [], "long": [], "revenue": []}
    market_map = dict(zip(universe["stock_id"], universe["market"]))

    # 大盤過濾：今天是否多頭趨勢
    today_str = datetime.today().strftime("%Y-%m-%d")
    market_filter = build_market_filter(start="2018-01-01", end=today_str)
    if market_filter.empty:
        logger.warning("Market filter unavailable — running without it")
        market_filter = None
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

    logger.info("Generating signals...")
    mf = market_filter

    stale_cutoff = pd.Timestamp.today() - pd.Timedelta(days=15)  # ~10 交易日

    for sid in tqdm(universe["stock_id"], desc="Screen"):
        price = load_prices(sid, start="2020-01-01")
        if len(price) < 60:
            continue
        # 過濾下市或長期停牌（最後交易日超過 15 天視為非活躍）
        if price["date"].max() < stale_cutoff:
            continue
        inst = load_institutional(sid, start="2020-01-01")
        inst_arg = inst if not inst.empty else None
        market = market_map.get(sid, "TWSE")

        try:
            df_s = signal_short_vol_breakout(price, inst_arg, market_filter=mf)
            if bool(df_s.iloc[-1]["signal_short"]):
                results["short"].append(_summary_row(sid, market, df_s, "short"))

            if sid in fund_ok:
                df_sw = signal_swing_ma_kd_inst(price, inst_arg, market_filter=mf)
                df_di = signal_swing_dual_inst(price, inst_arg, market_filter=mf)
                if bool(df_sw.iloc[-1]["signal_swing"]) or bool(df_di.iloc[-1]["signal_dual_inst"]):
                    base_df = df_di if bool(df_di.iloc[-1]["signal_dual_inst"]) else df_sw
                    results["swing"].append(_summary_row(sid, market, base_df, "swing"))

            if sid in fund_ok:
                df_l = signal_longterm_quality_entry(price, inst_arg, market_filter=mf)
                if bool(df_l.iloc[-1]["signal_long"]):
                    results["long"].append(_summary_row(sid, market, df_l, "long"))

            # 策略五：月營收動能（每月 10 日後第一個交易日才會有訊號）
            rev = load_monthly_revenue(sid)
            rev_arg = rev if not rev.empty else None
            df_rv = signal_revenue_momentum(price, inst_arg, rev_arg, market_filter=mf)
            if bool(df_rv.iloc[-1]["signal_rev"]):
                results["revenue"].append(_summary_row(sid, market, df_rv, "revenue"))

        except Exception as e:
            logger.debug(f"{sid}: {e}")

    return {
        k: pd.DataFrame(v).sort_values("vol_ratio", ascending=False)
        if v else pd.DataFrame()
        for k, v in results.items()
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
        "inst_total":float(last.get("inst_total", 0)),
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

    today = datetime.today().strftime("%Y-%m-%d")
    for tf, df in signals.items():
        n = len(df)
        logger.info(f"[{tf}] {n} signals today")
        if not df.empty:
            df.to_csv(f"reports/signals_{tf}_{today}.csv", index=False)

    if notify_fn:
        notify_fn(signals)

    return signals
