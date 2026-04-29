"""
股票池管理：以 FinMind TaiwanStockInfo 為主要來源（涵蓋上市+上櫃+興櫃）
HTML 解析方式不穩定，改用 API 直接取得結構化資料
"""
import logging
import pandas as pd
from data.fetcher import fetch_stock_list_finmind
from data.cache import save_universe, load_universe

logger = logging.getLogger(__name__)


def build_universe(force_refresh: bool = False) -> pd.DataFrame:
    existing = load_universe()
    if not existing.empty and not force_refresh:
        logger.info(f"Universe loaded from cache: {len(existing)} stocks")
        return existing

    logger.info("Fetching stock universe from FinMind TaiwanStockInfo...")
    raw = fetch_stock_list_finmind()

    if raw.empty:
        logger.error("FinMind TaiwanStockInfo returned empty data — check FINMIND_TOKEN")
        return pd.DataFrame()

    logger.debug(f"FinMind columns: {raw.columns.tolist()}")
    logger.debug(f"FinMind sample:\n{raw.head(3).to_string()}")

    df = _normalize(raw)

    if df.empty:
        logger.error("Universe normalization failed")
        return pd.DataFrame()

    save_universe(df)
    counts = df["market"].value_counts().to_dict()
    logger.info(f"Universe saved: {len(df)} stocks — {counts}")
    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # FinMind 欄位名稱可能是 stock_id / industry_category / type
    rename = {"industry_category": "industry"}
    df = df.rename(columns=rename)

    # 確保必要欄位存在
    if "stock_id" not in df.columns:
        logger.error(f"stock_id column missing, available: {df.columns.tolist()}")
        return pd.DataFrame()

    # 市場分類：依 type 欄位判斷
    if "type" in df.columns:
        df["market"] = df["type"].apply(_map_market)
    else:
        df["market"] = "TWSE"

    if "industry" not in df.columns:
        df["industry"] = ""

    if "stock_name" not in df.columns:
        df["stock_name"] = df.get("stock_name", "")

    # 只保留 4 碼純數字股票（排除 ETF、權證、KY 特殊格式等）
    df = df[df["stock_id"].astype(str).str.match(r"^\d{4}$", na=False)].copy()

    # 只保留有在交易的市場
    df = df[df["market"].isin(["TWSE", "TPEx", "Emerging"])].copy()

    return df[["stock_id", "stock_name", "market", "industry"]].reset_index(drop=True)


def _map_market(t: str) -> str:
    t = str(t).lower()
    if t == "twse" or "上市" in t:
        return "TWSE"
    if t == "tpex" or "上櫃" in t:
        return "TPEx"
    if "興櫃" in t or "emg" in t:
        return "Emerging"
    return "OTHER"


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
