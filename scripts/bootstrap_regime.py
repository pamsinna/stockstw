"""一次性 bootstrap：抓 0056 歷史價格 + TX 期貨外資未平倉，
給 analysis/market_regime.py 用。

用法：
    python scripts/bootstrap_regime.py             # 從 2019-01-01 抓
    python scripts/bootstrap_regime.py 2024-01-01  # 從指定日期抓
"""
from __future__ import annotations
import sys
import logging

from data.cache import (
    init_db,
    save_prices,
    save_futures_inst,
    last_price_date,
    last_futures_inst_date,
)
from data.fetcher import fetch_price, fetch_futures_inst

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def bootstrap(start: str = "2019-01-01") -> None:
    init_db()

    # 0056 (高股息 ETF)
    last = last_price_date("0056") or start
    fetch_from = max(last, start)
    logger.info(f"Fetching 0056 prices from {fetch_from}...")
    df = fetch_price("0056", fetch_from)
    if df is not None and not df.empty:
        save_prices("0056", df)
        logger.info(f"  0056: saved {len(df)} rows, last={df['date'].max()}")
    else:
        logger.warning("  0056: no data")

    # TX 台指期 外資/投信/自營商 未平倉
    last = last_futures_inst_date("TX") or start
    fetch_from = max(last, start)
    logger.info(f"Fetching TX futures_inst from {fetch_from}...")
    df = fetch_futures_inst("TX", fetch_from)
    if df is not None and not df.empty:
        save_futures_inst("TX", df)
        logger.info(f"  TX: saved {len(df)} rows, last={df['date'].max()}")
    else:
        logger.warning("  TX: no data")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    bootstrap(start)
