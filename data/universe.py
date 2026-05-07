"""
股票池管理：以 TWSE ISIN 官方頁面為來源（上市 + 上櫃普通股）
解析 section header 確保只保留「股票」區塊，排除 ETF、受益憑證等。
"""

EXCLUDED_INDUSTRIES: set[str] = {
    "食品工業",
}
import logging
import pandas as pd
from data.fetcher import fetch_twse_stock_list, fetch_tpex_stock_list
from data.cache import save_universe, load_universe

logger = logging.getLogger(__name__)


def build_universe(force_refresh: bool = False) -> pd.DataFrame:
    existing = load_universe()
    if not existing.empty and not force_refresh:
        existing = existing[~existing["industry"].isin(EXCLUDED_INDUSTRIES)].reset_index(drop=True)
        logger.info(f"Universe loaded from cache: {len(existing)} stocks")
        return existing

    logger.info("Fetching stock universe from TWSE/TPEx ISIN pages...")
    twse = fetch_twse_stock_list()
    tpex = fetch_tpex_stock_list()

    if twse.empty and tpex.empty:
        logger.error("Both TWSE and TPEx stock lists empty — check network")
        return pd.DataFrame()

    df = pd.concat([twse, tpex], ignore_index=True)

    if df.empty:
        logger.error("Universe empty after merge")
        return pd.DataFrame()

    before = len(df)
    df = df[~df["industry"].isin(EXCLUDED_INDUSTRIES)].reset_index(drop=True)
    excluded = before - len(df)
    if excluded:
        logger.info(f"Excluded {excluded} stocks from industries: {EXCLUDED_INDUSTRIES}")

    save_universe(df)
    counts = df["market"].value_counts().to_dict()
    logger.info(f"Universe saved: {len(df)} stocks — {counts}")
    return df


def get_stock_market(stock_id: str, universe: pd.DataFrame) -> str:
    row = universe[universe["stock_id"] == stock_id]
    if row.empty:
        return "TWSE"
    return row.iloc[0]["market"]


def filter_by_market(universe: pd.DataFrame,
                     include: list[str] | None = None) -> pd.DataFrame:
    if include is None:
        return universe
    return universe[universe["market"].isin(include)].reset_index(drop=True)
