"""
股票池管理：以 TWSE ISIN 官方頁面為來源（上市 + 上櫃普通股）
解析 section header 確保只保留「股票」區塊，排除 ETF、受益憑證等。
"""

EXCLUDED_INDUSTRIES: set[str] = {
    # 原始排除（低品質/非目標產業）
    "食品工業", "紡織纖維", "觀光餐旅", "鋼鐵工業", "汽車工業",
    "綠能環保", "居家生活", "運動休閒", "文化創意業", "水泥工業",
    "造紙工業", "玻璃陶瓷", "農業科技業",
    # 策略不適用（2026-05-08 新增）
    "生技醫療業",   # 股價與月營收脫鉤，Pipeline 邏輯不同
    "金融保險業",   # 財報結構不同，無月營收概念
    "航運業",       # 極端景氣循環，YoY 爆衝多為基期效果非 alpha
    "建材營造業",   # 完工認列時間不規律，YoY 動能邏輯失效
    "油電燃氣業",   # 營收認列不穩定（台汽電 +599% 案例）
    "其他業",       # 未定義雜項，商業模式混雜
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
